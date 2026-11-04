/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Minimal 64-bit Mach-O image loader for the isolated A64 runtime.
//! Dyld fixups and LC_UNIXTHREAD remain explicit follow-up work.

use super::memory::{GuestAddr, GuestMemory, MemoryError};
use mach_object::{LoadCommand, MachCommand, OFile, CPU_TYPE_ARM64};
use std::io::Cursor;

#[derive(Clone, Debug)]
pub struct Segment {
    pub name: String,
    pub address: GuestAddr,
    pub memory_size: u64,
    pub file_offset: u64,
    pub file_size: u64,
    pub initial_protection: u32,
}

#[derive(Clone, Debug)]
pub struct MachOImage {
    pub segments: Vec<Segment>,
    pub entry_point: GuestAddr,
    pub dynamic_libraries: Vec<String>,
    /// Raw LC_DYLD_CHAINED_FIXUPS payload. Parsed separately by the A64 dyld layer.
    pub chained_fixups: Option<Vec<u8>>,
    pub text_base: GuestAddr,
    pub end_address: GuestAddr,
}

#[derive(Debug)]
pub enum MachOError {
    Parse, WrongArchitecture, UnsupportedSubtype, MissingTextSegment,
    DuplicateTextSegment, DuplicateChainedFixups, MissingEntryPoint, EntryPointOutsideText,
    InvalidSegmentRange, OverlappingSegments, AddressOverflow, Memory(MemoryError),
}
impl From<MemoryError> for MachOError {
    fn from(value: MemoryError) -> Self { Self::Memory(value) }
}

fn choose_arm64_slice(bytes: &[u8]) -> Result<&[u8], MachOError> {
    match OFile::parse(&mut Cursor::new(bytes)).map_err(|_| MachOError::Parse)? {
        OFile::MachFile { header, .. } => {
            if header.cputype != CPU_TYPE_ARM64 { return Err(MachOError::WrongArchitecture); }
            if header.cpusubtype != mach_object::CPU_SUBTYPE_ARM64_ALL
                && header.cpusubtype != mach_object::CPU_SUBTYPE_ARM64_V8 {
                return Err(MachOError::UnsupportedSubtype);
            }
            Ok(bytes)
        }
        OFile::FatFile { files, .. } => {
            let mut best: Option<(&[u8], u8)> = None;
            for (arch, _) in files {
                if arch.cputype != CPU_TYPE_ARM64 { continue; }
                let priority = match arch.cpusubtype {
                    mach_object::CPU_SUBTYPE_ARM64_V8 => 2,
                    mach_object::CPU_SUBTYPE_ARM64_ALL => 1,
                    _ => continue,
                };
                let start = usize::try_from(arch.offset).map_err(|_| MachOError::InvalidSegmentRange)?;
                let size = usize::try_from(arch.size).map_err(|_| MachOError::InvalidSegmentRange)?;
                let end = start.checked_add(size).ok_or(MachOError::InvalidSegmentRange)?;
                let slice = bytes.get(start..end).ok_or(MachOError::InvalidSegmentRange)?;
                match OFile::parse(&mut Cursor::new(slice)).map_err(|_| MachOError::Parse)? {
                    OFile::MachFile { header, .. }
                        if header.cputype == CPU_TYPE_ARM64 && header.cpusubtype == arch.cpusubtype => {
                            if best.map_or(true, |(_, old)| priority > old) { best = Some((slice, priority)); }
                        }
                    _ => continue,
                }
            }
            best.map(|(slice, _)| slice).ok_or(MachOError::WrongArchitecture)
        }
    }
}

/// LC_MAIN entryoff is a file offset (normally __TEXT.fileoff is zero).
/// Convert it using the file-to-VM delta, then apply the requested ASLR slide.
fn resolve_main_entry(text: &Segment, entryoff: u64, slide: u64) -> Result<GuestAddr, MachOError> {
    let text_file_end = text.file_offset.checked_add(text.file_size)
        .ok_or(MachOError::InvalidSegmentRange)?;
    if entryoff < text.file_offset || entryoff >= text_file_end || text.initial_protection & 0x4 == 0 {
        return Err(MachOError::EntryPointOutsideText);
    }
    let delta = entryoff - text.file_offset;
    if delta >= text.memory_size { return Err(MachOError::EntryPointOutsideText); }
    text.address.checked_add(delta).and_then(|a| a.checked_add(slide))
        .ok_or(MachOError::AddressOverflow)
}

impl MachOImage {
    /// Parse an ARM64 Mach-O executable and map its 64-bit segments.
    /// slide is added to every guest virtual address.
    pub fn load(bytes: &[u8], memory: &mut GuestMemory, slide: GuestAddr) -> Result<Self, MachOError> {
        let image_bytes = choose_arm64_slice(bytes)?;
        let (header, commands) = match OFile::parse(&mut Cursor::new(image_bytes)).map_err(|_| MachOError::Parse)? {
            OFile::MachFile { header, commands } => (header, commands),
            OFile::FatFile { .. } => return Err(MachOError::Parse),
        };
        if header.cputype != CPU_TYPE_ARM64 || !header.is_64bit() { return Err(MachOError::WrongArchitecture); }

        let mut segments = Vec::new();
        let mut text_index = None;
        let mut entry_offset = None;
        let mut dynamic_libraries = Vec::new();
        let mut chained_fixups = None;
        for MachCommand(command, _) in &commands {
            match command {
                LoadCommand::Segment64 { segname, vmaddr, vmsize, fileoff, filesize, initprot, .. } => {
                    let address = u64::try_from(*vmaddr).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let memory_size = u64::try_from(*vmsize).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let file_offset = u64::try_from(*fileoff).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let file_size = u64::try_from(*filesize).map_err(|_| MachOError::InvalidSegmentRange)?;
                    if file_size > memory_size { return Err(MachOError::InvalidSegmentRange); }
                    address.checked_add(memory_size).ok_or(MachOError::AddressOverflow)?;
                    let file_end = file_offset.checked_add(file_size).ok_or(MachOError::InvalidSegmentRange)?;
                    if file_end > image_bytes.len() as u64 { return Err(MachOError::InvalidSegmentRange); }
                    if segname == "__TEXT" && text_index.replace(segments.len()).is_some() {
                        return Err(MachOError::DuplicateTextSegment);
                    }
                    let loaded_address = address.checked_add(slide).ok_or(MachOError::AddressOverflow)?;
                    segments.push(Segment { name: segname.clone(), address: loaded_address, memory_size, file_offset, file_size, initial_protection: *initprot as u32 });
                }
                LoadCommand::EntryPoint { entryoff, .. } => {
                    if entry_offset.replace(*entryoff).is_some() { return Err(MachOError::MissingEntryPoint); }
                }
                LoadCommand::DyldChainedFixups(data) => {
                    if chained_fixups.is_some() { return Err(MachOError::DuplicateChainedFixups); }
                    let start = usize::try_from(data.off).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let size = usize::try_from(data.size).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let finish = start.checked_add(size).ok_or(MachOError::InvalidSegmentRange)?;
                    let payload = image_bytes.get(start..finish).ok_or(MachOError::InvalidSegmentRange)?;
                    if size < 28 { return Err(MachOError::InvalidSegmentRange); }
                    chained_fixups = Some(payload.to_vec());
                }
                LoadCommand::LoadDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                LoadCommand::LoadWeakDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                LoadCommand::ReexportDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                _ => {}
            }
        }

        let text = segments.get(text_index.ok_or(MachOError::MissingTextSegment)?)
            .ok_or(MachOError::MissingTextSegment)?;
        let text_base = text.address;
        let entry_point = resolve_main_entry(text, entry_offset.ok_or(MachOError::MissingEntryPoint)?, 0)?;

        // Validate all address ranges and reject overlapping VM segments before
        // mutating guest memory. Silently merging two segment mappings would
        // hide malformed images and can overwrite another segment's contents.
        let mut ranges = Vec::with_capacity(segments.len());
        let mut end_address = 0;
        for segment in &segments {
            let start = segment.address;
            let end = start.checked_add(segment.memory_size).ok_or(MachOError::AddressOverflow)?;
            if segment.memory_size != 0 {
                ranges.push((start, end));
            }
            end_address = end_address.max(end);
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(MachOError::OverlappingSegments);
            }
        }
        for segment in &segments {
            let loaded_address = segment.address;
            // __PAGEZERO reserves an inaccessible range; allocating its often
            // multi-gigabyte size in the sparse guest map would be disastrous.
            if segment.memory_size == 0 || (segment.file_size == 0 && segment.initial_protection == 0) { continue; }
            memory.map_zeroed(loaded_address, segment.memory_size)?;
            if segment.file_size != 0 {
                let start = usize::try_from(segment.file_offset).map_err(|_| MachOError::InvalidSegmentRange)?;
                let file_end = segment.file_offset.checked_add(segment.file_size).ok_or(MachOError::InvalidSegmentRange)?;
                let finish = usize::try_from(file_end).map_err(|_| MachOError::InvalidSegmentRange)?;
                memory.write(loaded_address, &image_bytes[start..finish])?;
            }
        }
        Ok(Self { segments, entry_point, dynamic_libraries, chained_fixups, text_base, end_address })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(file_offset: u64, file_size: u64, vmaddr: u64, vmsize: u64) -> Segment {
        Segment { name: "__TEXT".into(), address: vmaddr, memory_size: vmsize, file_offset, file_size, initial_protection: 0x5 }
    }
    #[test]
    fn lc_main_resolves_file_offset_and_slide() {
        assert_eq!(resolve_main_entry(&text(0, 0x3000, 0x1_0000_0000, 0x3000), 0x1234, 0x4000).unwrap(), 0x1_0000_5234);
    }
    #[test]
    fn lc_main_handles_nonzero_text_file_offset() {
        assert_eq!(resolve_main_entry(&text(0x1000, 0x3000, 0x1_0000_0000, 0x3000), 0x1234, 0).unwrap(), 0x1_0000_0234);
    }
    #[test]
    fn lc_main_rejects_offsets_outside_file_backed_executable_text() {
        let t = text(0, 0x1000, 0x1_0000_0000, 0x3000);
        assert!(matches!(resolve_main_entry(&t, 0x1000, 0), Err(MachOError::EntryPointOutsideText)));
        assert!(matches!(resolve_main_entry(&t, 0x2000, 0), Err(MachOError::EntryPointOutsideText)));
    }
    #[test]
    fn lc_main_rejects_non_executable_text() {
        let mut t = text(0, 0x3000, 0x1_0000_0000, 0x3000);
        t.initial_protection = 0x3;
        assert!(matches!(resolve_main_entry(&t, 0x100, 0), Err(MachOError::EntryPointOutsideText)));
    }
}

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! 64-bit Mach-O image loader for the isolated A64 runtime.
//! Generic DYLD_CHAINED_PTR_64_OFFSET, ARM64E_USERLAND, and
//! ARM64E_USERLAND24 chained rebases/binds are supported, including
//! multi-start pages. Authenticated
//! pointers are emitted unsigned; PAC signing and LC_UNIXTHREAD remain TODO.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DylibDependencyKind {
    Required,
    Weak,
    Reexport,
    Upward,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DylibDependency {
    pub install_name: String,
    pub kind: DylibDependencyKind,
}

#[derive(Clone, Debug)]
pub struct MachOImage {
    pub segments: Vec<Segment>,
    pub entry_point: GuestAddr,
    /// Ordered install names retained for compatibility with earlier callers.
    pub dynamic_libraries: Vec<String>,
    /// Ordered dependencies with their original load-command semantics.
    pub dependencies: Vec<DylibDependency>,
    /// Ordered LC_RPATH entries used to resolve @rpath dependencies.
    pub rpaths: Vec<String>,
    /// Raw LC_DYLD_CHAINED_FIXUPS payload. Parsed separately by the A64 dyld layer.
    pub chained_fixups: Option<Vec<u8>>,
    /// Raw LC_DYLD_EXPORTS_TRIE payload, decoded by the A64 dyld layer.
    pub exports_trie: Option<Vec<u8>>,
    pub text_base: GuestAddr,
    pub end_address: GuestAddr,
}

#[derive(Debug)]
pub enum MachOError {
    Parse, WrongArchitecture, UnsupportedSubtype, MissingTextSegment,
    DuplicateTextSegment, DuplicateChainedFixups, DuplicateExportsTrie, ConflictingExportsTrieCommands, MissingEntryPoint, EntryPointOutsideText,
    InvalidSegmentRange, InvalidChainedFixups, InvalidExportsTrie, UnsupportedPointerFormat, UnsupportedChainedBind, OverlappingSegments,
    InvalidImportOrdinal, UnresolvedStrongImport(String), BindAddendOverflow, DylibRegistry(super::dyld::DylibRegistryError), AddressOverflow, Memory(MemoryError),
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

/// Validate the offset tables in an LC_DYLD_CHAINED_FIXUPS payload.
/// This deliberately does not decode or apply pointer chains.
fn validate_chained_fixups(data: &[u8]) -> Result<(), MachOError> {
    fn u16_at(data: &[u8], offset: usize) -> Result<u16, MachOError> {
        let bytes = data.get(offset..offset.checked_add(2).ok_or(MachOError::InvalidChainedFixups)?)
            .ok_or(MachOError::InvalidChainedFixups)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    fn u32_at(data: &[u8], offset: usize) -> Result<u32, MachOError> {
        let bytes = data.get(offset..offset.checked_add(4).ok_or(MachOError::InvalidChainedFixups)?)
            .ok_or(MachOError::InvalidChainedFixups)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    fn checked_range(start: usize, count: usize, width: usize, limit: usize) -> Result<std::ops::Range<usize>, MachOError> {
        let size = count.checked_mul(width).ok_or(MachOError::InvalidChainedFixups)?;
        let end = start.checked_add(size).ok_or(MachOError::InvalidChainedFixups)?;
        if start > limit || end > limit { return Err(MachOError::InvalidChainedFixups); }
        Ok(start..end)
    }

    const HEADER_SIZE: usize = 28;
    const STARTS_SEGMENT_HEADER_SIZE: usize = 22;
    if data.len() < HEADER_SIZE || u32_at(data, 0)? != 0 {
        return Err(MachOError::InvalidChainedFixups);
    }
    let starts_offset = usize::try_from(u32_at(data, 4)?).map_err(|_| MachOError::InvalidChainedFixups)?;
    let imports_offset = usize::try_from(u32_at(data, 8)?).map_err(|_| MachOError::InvalidChainedFixups)?;
    let symbols_offset = usize::try_from(u32_at(data, 12)?).map_err(|_| MachOError::InvalidChainedFixups)?;
    let imports_count = usize::try_from(u32_at(data, 16)?).map_err(|_| MachOError::InvalidChainedFixups)?;
    let imports_format = u32_at(data, 20)?;
    let symbols_format = u32_at(data, 24)?;

    let import_width = match imports_format {
        1 => 4,
        2 => 8,
        3 => 16,
        _ => return Err(MachOError::InvalidChainedFixups),
    };
    if symbols_format > 1 || starts_offset < HEADER_SIZE || symbols_offset > data.len() {
        return Err(MachOError::InvalidChainedFixups);
    }
    let imports = checked_range(imports_offset, imports_count, import_width, data.len())?;
    if imports.end > symbols_offset || imports_offset < HEADER_SIZE {
        return Err(MachOError::InvalidChainedFixups);
    }

    // starts_in_image contains seg_count followed by seg_count u32 offsets.
    let seg_count = usize::try_from(u32_at(data, starts_offset)?).map_err(|_| MachOError::InvalidChainedFixups)?;
    let offsets_start = starts_offset.checked_add(4).ok_or(MachOError::InvalidChainedFixups)?;
    let offsets = checked_range(offsets_start, seg_count, 4, data.len())?;
    if offsets.end > imports_offset { return Err(MachOError::InvalidChainedFixups); }
    let minimum_segment_offset = offsets.end.checked_sub(starts_offset).ok_or(MachOError::InvalidChainedFixups)?;
    for index in 0..seg_count {
        let relative = usize::try_from(u32_at(data, offsets.start + index * 4)?)
            .map_err(|_| MachOError::InvalidChainedFixups)?;
        if relative == 0 { continue; }
        if relative < minimum_segment_offset { return Err(MachOError::InvalidChainedFixups); }
        let segment_start = starts_offset.checked_add(relative).ok_or(MachOError::InvalidChainedFixups)?;
        let size = usize::try_from(u32_at(data, segment_start)?)
            .map_err(|_| MachOError::InvalidChainedFixups)?;
        if size < STARTS_SEGMENT_HEADER_SIZE { return Err(MachOError::InvalidChainedFixups); }
        let segment_end = segment_start.checked_add(size).ok_or(MachOError::InvalidChainedFixups)?;
        if segment_end > data.len() || segment_end > imports_offset {
            return Err(MachOError::InvalidChainedFixups);
        }
        let page_count = usize::from(u16_at(data, segment_start + 20)?);
        let page_array_end = STARTS_SEGMENT_HEADER_SIZE.checked_add(page_count.checked_mul(2)
            .ok_or(MachOError::InvalidChainedFixups)?).ok_or(MachOError::InvalidChainedFixups)?;
        if page_array_end > size { return Err(MachOError::InvalidChainedFixups); }
    }
    Ok(())
}

/// Decode a page's chain starts, including its START_MULTI overflow list.
fn chained_page_starts(payload: &[u8], info: usize, info_size: usize, page_count: usize, value: u16) -> Result<Vec<u16>, MachOError> {
    if value == 0xFFFF { return Ok(Vec::new()); }
    if value & 0x8000 == 0 { return Ok(vec![value]); }
    let mut index = usize::from(value & 0x7FFF);
    let capacity = info_size.checked_sub(22).ok_or(MachOError::InvalidChainedFixups)? / 2;
    if index < page_count || index >= capacity { return Err(MachOError::InvalidChainedFixups); }
    let mut starts = Vec::new();
    loop {
        if index >= capacity || starts.len() >= capacity { return Err(MachOError::InvalidChainedFixups); }
        let at = info.checked_add(22).and_then(|v| v.checked_add(index.checked_mul(2)?)).ok_or(MachOError::InvalidChainedFixups)?;
        let end = at.checked_add(2).ok_or(MachOError::InvalidChainedFixups)?;
        let raw = u16::from_le_bytes(payload.get(at..end).ok_or(MachOError::InvalidChainedFixups)?.try_into().map_err(|_| MachOError::InvalidChainedFixups)?);
        starts.push(raw & 0x7FFF);
        if raw & 0x8000 != 0 { break; }
        index = index.checked_add(1).ok_or(MachOError::InvalidChainedFixups)?;
    }
    Ok(starts)
}

/// Apply rebases for DYLD_CHAINED_PTR_64_OFFSET (generic arm64 format).
/// Generic and ARM64E_USERLAND rebases are decoded to guest pointers.
/// Authenticated ARM64e pointers lose PAC metadata; signing is not modeled.
fn apply_chained_rebases(payload: &[u8], segments: &[Segment], image_base: GuestAddr, slide: GuestAddr, memory: &mut GuestMemory) -> Result<(), MachOError> {
    fn u16_at(d: &[u8], p: usize) -> Result<u16, MachOError> { let b=d.get(p..p.checked_add(2).ok_or(MachOError::InvalidChainedFixups)?).ok_or(MachOError::InvalidChainedFixups)?; Ok(u16::from_le_bytes([b[0],b[1]])) }
    fn u32_at(d: &[u8], p: usize) -> Result<u32, MachOError> { let b=d.get(p..p.checked_add(4).ok_or(MachOError::InvalidChainedFixups)?).ok_or(MachOError::InvalidChainedFixups)?; Ok(u32::from_le_bytes([b[0],b[1],b[2],b[3]])) }
    let starts=usize::try_from(u32_at(payload,4)?).map_err(|_|MachOError::InvalidChainedFixups)?;
    let count=usize::try_from(u32_at(payload,starts)?).map_err(|_|MachOError::InvalidChainedFixups)?;
    if count != segments.len() { return Err(MachOError::InvalidChainedFixups); }
    let mut patches: Vec<(GuestAddr, [u8; 8])> = Vec::new();
    for (i,seg) in segments.iter().enumerate() {
        let rel_pos=starts.checked_add(4).and_then(|v|v.checked_add(i.checked_mul(4)?)).ok_or(MachOError::InvalidChainedFixups)?;
        let rel=usize::try_from(u32_at(payload,rel_pos)?).map_err(|_|MachOError::InvalidChainedFixups)?;
        if rel==0 { continue; }
        let info=starts.checked_add(rel).ok_or(MachOError::InvalidChainedFixups)?;
        let size=usize::try_from(u32_at(payload,info)?).map_err(|_|MachOError::InvalidChainedFixups)?;
        let page_size=u64::from(u16_at(payload,info+4)?);
        let format=u16_at(payload,info+6)?;
        let seg_off=u64::from_le_bytes(payload.get(info+8..info+16).ok_or(MachOError::InvalidChainedFixups)?.try_into().map_err(|_|MachOError::InvalidChainedFixups)?);
        let pages=usize::from(u16_at(payload,info+20)?);
        if format!=6 && format!=9 && format!=12 { return Err(MachOError::UnsupportedPointerFormat); }
        let page_array_size=pages.checked_mul(2).and_then(|v|v.checked_add(22)).ok_or(MachOError::InvalidChainedFixups)?;
        if (page_size!=0x1000 && page_size!=0x4000) || size<page_array_size { return Err(MachOError::InvalidChainedFixups); }
        let expected=seg.address.checked_sub(slide).and_then(|a|a.checked_sub(image_base)).ok_or(MachOError::InvalidChainedFixups)?;
        if seg_off!=expected { return Err(MachOError::InvalidChainedFixups); }
        for page in 0..pages {
            let start=u16_at(payload,info+22+page*2)?;
            let starts_for_page=chained_page_starts(payload,info,size,pages,start)?;
            if starts_for_page.is_empty() { continue; }
            let seg_end=seg.address.checked_add(seg.memory_size).ok_or(MachOError::AddressOverflow)?;
            let page_base=seg.address.checked_add((page as u64).checked_mul(page_size).ok_or(MachOError::InvalidChainedFixups)?).ok_or(MachOError::AddressOverflow)?;
            let page_end=page_base.checked_add(page_size).ok_or(MachOError::AddressOverflow)?.min(seg_end);
            for chain_start in starts_for_page {
            let stride = if format == 6 { 4 } else { 8 };
            if u64::from(chain_start).checked_add(8).ok_or(MachOError::AddressOverflow)? > page_size
                || u64::from(chain_start) % stride != 0 { return Err(MachOError::InvalidChainedFixups); }
            let mut loc=page_base.checked_add(u64::from(chain_start)).ok_or(MachOError::AddressOverflow)?;
            loop {
                if loc.checked_add(8).ok_or(MachOError::AddressOverflow)?>page_end { return Err(MachOError::InvalidChainedFixups); }
                let mut bytes=[0u8;8]; memory.read(loc,&mut bytes)?;
                let raw=u64::from_le_bytes(bytes);
                let is_arm64e = format == 9 || format == 12;
                let next = if is_arm64e { (raw >> 51) & 0x7FF } else { (raw >> 51) & 0xFFF };
                let is_bind = if is_arm64e {
                    raw & (1u64 << 62) != 0
                } else {
                    raw & (1u64 << 63) != 0
                };
                let authenticated = is_arm64e && raw & (1u64 << 63) != 0;
                if is_arm64e {
                    // Bind pointers are resolved in the later registry-aware pass.
                    if is_bind {
                        // Do not patch this word, but continue following its chain.
                    } else {
                    // ARM64E_USERLAND rebase targets are image-relative offsets.
                    // Authenticated pointers retain their PAC metadata in the file,
                    // but this portable emulator writes the unsigned guest address.
                    let target = if authenticated { raw & 0xFFFF_FFFF } else { raw & ((1u64<<43)-1) };
                    let resolved=image_base.checked_add(target).and_then(|v|v.checked_add(slide)).ok_or(MachOError::AddressOverflow)?;
                    if resolved>0x00FF_FFFF_FFFF_FFFF { return Err(MachOError::AddressOverflow); }
                    let value = if authenticated { resolved } else {
                        let high8=(raw>>43)&0xFF;
                        (resolved & 0x00FF_FFFF_FFFF_FFFF)|(high8<<56)
                    };
                    patches.push((loc,value.to_le_bytes()));
                    }
                } else {
                    if is_bind {
                        // Generic binds are handled after dependency resolution.
                    } else {
                    if ((raw >> 44) & 0x7F) != 0 { return Err(MachOError::InvalidChainedFixups); }
                    let target=raw & ((1u64<<36)-1);
                    let high8=(raw>>36)&0xFF;
                    let resolved=image_base.checked_add(target).and_then(|v|v.checked_add(slide)).ok_or(MachOError::AddressOverflow)?;
                    if resolved>0x00FF_FFFF_FFFF_FFFF { return Err(MachOError::AddressOverflow); }
                    let value=(resolved & 0x00FF_FFFF_FFFF_FFFF)|(high8<<56);
                    patches.push((loc,value.to_le_bytes()));
                    }
                }
                if next==0 { break; }
                loc=loc.checked_add(next.checked_mul(stride).ok_or(MachOError::AddressOverflow)?).ok_or(MachOError::AddressOverflow)?;
            }
            }
        }
    }
    for (address, bytes) in patches { memory.write(address, &bytes)?; }
    Ok(())
}


/// Resolve generic 64-bit chained binds. The resolver returns the absolute guest
/// address of an imported symbol. Writes are staged so a missing strong symbol
/// or malformed chain cannot leave guest memory half-patched.
/// Apply binds using the shared A64 dylib registry and this image's ID.
pub fn apply_chained_binds_with_registry(
    image: &MachOImage,
    memory: &mut GuestMemory,
    registry: &super::dyld::DylibRegistry,
    importer_id: usize,
) -> Result<(), MachOError> {
    // Validate all import ordinals before patching anything. The low-level
    // binder accepts an Option resolver, so errors from the registry must not
    // be silently converted into unresolved weak imports.
    if let Some(payload) = image.chained_fixups.as_deref() {
        let imports = super::dyld::parse_chained_imports(payload)
            .map_err(|_| MachOError::InvalidChainedFixups)?;
        for import in &imports {
            registry.resolve(importer_id, import)
                .map_err(MachOError::DylibRegistry)?;
        }
    }
    apply_chained_binds(image, memory, |import| {
        registry.resolve(importer_id, import).ok().flatten()
    })
}

pub fn apply_chained_binds<F>(
    image: &MachOImage,
    memory: &mut GuestMemory,
    mut resolve: F,
) -> Result<(), MachOError>
where
    F: FnMut(&super::dyld::ChainedImport) -> Option<GuestAddr>,
{
    fn u16_at(data: &[u8], at: usize) -> Result<u16, MachOError> {
        let end = at.checked_add(2).ok_or(MachOError::InvalidChainedFixups)?;
        let bytes = data.get(at..end).ok_or(MachOError::InvalidChainedFixups)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    fn u32_at(data: &[u8], at: usize) -> Result<u32, MachOError> {
        let end = at.checked_add(4).ok_or(MachOError::InvalidChainedFixups)?;
        let bytes = data.get(at..end).ok_or(MachOError::InvalidChainedFixups)?;
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| MachOError::InvalidChainedFixups)?))
    }
    fn add_signed(base: u64, addend: i64) -> Result<u64, MachOError> {
        if addend >= 0 {
            base.checked_add(addend as u64)
        } else {
            base.checked_sub(addend.unsigned_abs())
        }.ok_or(MachOError::BindAddendOverflow)
    }

    let payload = match image.chained_fixups.as_deref() {
        Some(data) => data,
        None => return Ok(()),
    };
    let imports = super::dyld::parse_chained_imports(payload)
        .map_err(|_| MachOError::InvalidChainedFixups)?;
    let starts = usize::try_from(u32_at(payload, 4)?)
        .map_err(|_| MachOError::InvalidChainedFixups)?;
    let segment_count = usize::try_from(u32_at(payload, starts)?)
        .map_err(|_| MachOError::InvalidChainedFixups)?;
    if segment_count != image.segments.len() {
        return Err(MachOError::InvalidChainedFixups);
    }

    let mut patches: Vec<(GuestAddr, [u8; 8])> = Vec::new();
    for (segment_index, segment) in image.segments.iter().enumerate() {
        let offset_pos = starts.checked_add(4)
            .and_then(|v| v.checked_add(segment_index.checked_mul(4)?))
            .ok_or(MachOError::InvalidChainedFixups)?;
        let relative = usize::try_from(u32_at(payload, offset_pos)?)
            .map_err(|_| MachOError::InvalidChainedFixups)?;
        if relative == 0 { continue; }
        let info = starts.checked_add(relative).ok_or(MachOError::InvalidChainedFixups)?;
        let info_size = usize::try_from(u32_at(payload, info)?)
            .map_err(|_| MachOError::InvalidChainedFixups)?;
        let page_size = u64::from(u16_at(payload, info + 4)?);
        let pointer_format = u16_at(payload, info + 6)?;
        let segment_offset = u64::from_le_bytes(
            payload.get(info + 8..info + 16).ok_or(MachOError::InvalidChainedFixups)?
                .try_into().map_err(|_| MachOError::InvalidChainedFixups)?
        );
        let page_count = usize::from(u16_at(payload, info + 20)?);
        if pointer_format != 6 && pointer_format != 9 && pointer_format != 12 {
            return Err(MachOError::UnsupportedPointerFormat);
        }
        if page_size != 0x1000 && page_size != 0x4000 {
            return Err(MachOError::InvalidChainedFixups);
        }
        let array_end = 22usize.checked_add(page_count.checked_mul(2)
            .ok_or(MachOError::InvalidChainedFixups)?)
            .ok_or(MachOError::InvalidChainedFixups)?;
        if array_end > info_size { return Err(MachOError::InvalidChainedFixups); }
        let expected_offset = segment.address.checked_sub(image.text_base)
            .ok_or(MachOError::InvalidChainedFixups)?;
        if segment_offset != expected_offset { return Err(MachOError::InvalidChainedFixups); }

        let stride = if pointer_format == 6 { 4u64 } else { 8u64 };
        let segment_end = segment.address.checked_add(segment.memory_size)
            .ok_or(MachOError::AddressOverflow)?;
        for page_index in 0..page_count {
            let page_start = u16_at(payload, info + 22 + page_index * 2)?;
            let starts_for_page = chained_page_starts(payload, info, info_size, page_count, page_start)?;
            if starts_for_page.is_empty() { continue; }
            let page_base = segment.address.checked_add(
                (page_index as u64).checked_mul(page_size)
                    .ok_or(MachOError::AddressOverflow)?
            ).ok_or(MachOError::AddressOverflow)?;
            let page_end = page_base.checked_add(page_size)
                .ok_or(MachOError::AddressOverflow)?.min(segment_end);
            for chain_start in starts_for_page {
                if u64::from(chain_start).checked_add(8).ok_or(MachOError::AddressOverflow)? > page_size
                    || u64::from(chain_start) % stride != 0 {
                    return Err(MachOError::InvalidChainedFixups);
                }
                let mut location = page_base.checked_add(u64::from(chain_start))
                    .ok_or(MachOError::AddressOverflow)?;
                loop {
                    if location.checked_add(8).ok_or(MachOError::AddressOverflow)? > page_end {
                        return Err(MachOError::InvalidChainedFixups);
                    }
                    let mut raw_bytes = [0u8; 8];
                    memory.read(location, &mut raw_bytes)?;
                    let raw = u64::from_le_bytes(raw_bytes);
                    let is_arm64e = pointer_format == 9 || pointer_format == 12;
                    let (is_bind, authenticated, next) = if is_arm64e {
                        (raw & (1u64 << 62) != 0, raw & (1u64 << 63) != 0, (raw >> 51) & 0x7FF)
                    } else {
                        (raw & (1u64 << 63) != 0, false, (raw >> 51) & 0xFFF)
                    };

                    if is_bind {
                        let (ordinal, embedded_addend) = if is_arm64e {
                            let ordinal_bits = if pointer_format == 12 { 24 } else { 16 };
                            let ordinal_mask = (1u64 << ordinal_bits) - 1;
                            let ordinal = usize::try_from(raw & ordinal_mask)
                                .map_err(|_| MachOError::InvalidImportOrdinal)?;
                            let reserved_bits = if pointer_format == 12 { (raw >> 24) & 0xFF } else { (raw >> 16) & 0xFFFF };
                            if reserved_bits != 0 {
                                return Err(MachOError::InvalidChainedFixups);
                            }
                            let addend = if authenticated {
                                0
                            } else {
                                let bits = ((raw >> 32) & 0x7_FFFF) as i64;
                                if bits & 0x4_0000 != 0 { bits | !0x7_FFFF } else { bits }
                            };
                            (ordinal, Some(addend))
                        } else {
                            let ordinal = usize::try_from(raw & 0x00FF_FFFF)
                                .map_err(|_| MachOError::InvalidImportOrdinal)?;
                            let inline_addend = ((raw >> 24) & 0xFF) as u8;
                            let embedded_reserved = (raw >> 32) & 0x7_FFFF;
                            if embedded_reserved != 0 { return Err(MachOError::InvalidChainedFixups); }
                            // Generic 64-bit inline addend replaces the import addend.
                            (ordinal, if inline_addend != 0 { Some(i64::from(inline_addend)) } else { None })
                        };
                        let import = imports.get(ordinal).ok_or(MachOError::InvalidImportOrdinal)?;
                        let symbol = match resolve(import) {
                            Some(address) => address,
                            None if import.weak => 0,
                            None => return Err(MachOError::UnresolvedStrongImport(import.symbol.clone())),
                        };
                        let total_addend = if is_arm64e {
                            import.addend.checked_add(embedded_addend.unwrap_or(0))
                                .ok_or(MachOError::BindAddendOverflow)?
                        } else {
                            embedded_addend.unwrap_or(import.addend)
                        };
                        // Authenticated binds are emitted as unsigned guest pointers.
                        // PAC key/diversity/address-diversity metadata is not emulated.
                        let value = add_signed(symbol, total_addend)?;
                        patches.push((location, value.to_le_bytes()));
                    } else if is_arm64e && authenticated {
                        // Rebase is handled by apply_chained_rebases.
                    }

                    if next == 0 { break; }
                    location = location.checked_add(next.checked_mul(stride)
                        .ok_or(MachOError::AddressOverflow)?)
                        .ok_or(MachOError::AddressOverflow)?;
                }
            }
        }
    }
    for (address, bytes) in patches {
        memory.write(address, &bytes)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportResolveError {
    UnsupportedExportKind,
    AddressOverflow,
}

/// Resolve an export from this image. Re-exports are delegated to the caller
/// so the image loader does not guess how a library ordinal maps to a loaded image.
pub fn resolve_export<F>(
    image: &MachOImage,
    symbol_name: &str,
    mut resolve_reexport: F,
) -> Result<Option<GuestAddr>, ExportResolveError>
where
    F: FnMut(u64, &str) -> Option<GuestAddr>,
{
    let exports = image.exported_symbols().map_err(|_| ExportResolveError::UnsupportedExportKind)?;
    let symbol = match exports.iter().find(|entry| entry.name == symbol_name) {
        Some(symbol) => symbol,
        None => return Ok(None),
    };
    match &symbol.kind {
        super::dyld::ExportKind::Regular { address } => match symbol.flags & 0x03 {
            0 => image.text_base.checked_add(*address).map(Some).ok_or(ExportResolveError::AddressOverflow),
            2 => Ok(Some(*address)), // absolute symbol
            _ => Err(ExportResolveError::UnsupportedExportKind), // thread-local/reserved
        },
        super::dyld::ExportKind::StubAndResolver { stub, .. } => {
            image.text_base.checked_add(*stub).map(Some).ok_or(ExportResolveError::AddressOverflow)
        }
        super::dyld::ExportKind::Reexport { library_ordinal, imported_name } => {
            Ok(resolve_reexport(*library_ordinal, if imported_name.is_empty() { symbol_name } else { imported_name }))
        }
    }
}

/// Return the lowest non-empty ARM64 VM address in a Mach-O image before sliding.
pub fn preferred_base(bytes: &[u8]) -> Result<GuestAddr, MachOError> {
    let image_bytes = choose_arm64_slice(bytes)?;
    let (header, commands) = match OFile::parse(&mut Cursor::new(image_bytes)).map_err(|_| MachOError::Parse)? {
        OFile::MachFile { header, commands } => (header, commands),
        OFile::FatFile { .. } => return Err(MachOError::Parse),
    };
    if header.cputype != CPU_TYPE_ARM64 || !header.is_64bit() { return Err(MachOError::WrongArchitecture); }
    let mut base = None;
    for MachCommand(command, _) in &commands {
        if let LoadCommand::Segment64 { vmaddr, vmsize, .. } = command {
            if *vmsize != 0 {
                let address = u64::try_from(*vmaddr).map_err(|_| MachOError::InvalidSegmentRange)?;
                base = Some(base.map_or(address, |current: u64| current.min(address)));
            }
        }
    }
    base.ok_or(MachOError::MissingTextSegment)
}

impl MachOImage {
    /// Decode the image's exported symbols, if it contains an export trie.
    pub fn exported_symbols(
        &self,
    ) -> Result<Vec<super::dyld::ExportedSymbol>, super::dyld::ExportTrieError> {
        match self.exports_trie.as_deref() {
            Some(trie) => super::dyld::parse_export_trie(trie),
            None => Ok(Vec::new()),
        }
    }

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
        let mut dependencies = Vec::new();
        let mut rpaths = Vec::new();
        let mut chained_fixups = None;
        let mut exports_trie = None;
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
                    validate_chained_fixups(payload)?;
                    chained_fixups = Some(payload.to_vec());
                }
                LoadCommand::DyldExportsTrie(data) => {
                    if exports_trie.is_some() { return Err(MachOError::DuplicateExportsTrie); }
                    let start = usize::try_from(data.off).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let size = usize::try_from(data.size).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let finish = start.checked_add(size).ok_or(MachOError::InvalidSegmentRange)?;
                    let payload = image_bytes.get(start..finish).ok_or(MachOError::InvalidSegmentRange)?;
                    super::dyld::parse_export_trie(payload)
                        .map_err(|_| MachOError::InvalidExportsTrie)?;
                    exports_trie = Some(payload.to_vec());
                }
                LoadCommand::DyldInfo { export_off, export_size, .. } if *export_size != 0 => {
                    if exports_trie.is_some() { return Err(MachOError::ConflictingExportsTrieCommands); }
                    let start = usize::try_from(*export_off).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let size = usize::try_from(*export_size).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let finish = start.checked_add(size).ok_or(MachOError::InvalidSegmentRange)?;
                    let payload = image_bytes.get(start..finish).ok_or(MachOError::InvalidSegmentRange)?;
                    super::dyld::parse_export_trie(payload)
                        .map_err(|_| MachOError::InvalidExportsTrie)?;
                    exports_trie = Some(payload.to_vec());
                }
                LoadCommand::LoadDyLib(dylib) => {
                    let name = dylib.name.to_string();
                    dynamic_libraries.push(name.clone());
                    dependencies.push(DylibDependency { install_name: name, kind: DylibDependencyKind::Required });
                }
                LoadCommand::LoadWeakDyLib(dylib) => {
                    let name = dylib.name.to_string();
                    dynamic_libraries.push(name.clone());
                    dependencies.push(DylibDependency { install_name: name, kind: DylibDependencyKind::Weak });
                }
                LoadCommand::ReexportDyLib(dylib) => {
                    let name = dylib.name.to_string();
                    dynamic_libraries.push(name.clone());
                    dependencies.push(DylibDependency { install_name: name, kind: DylibDependencyKind::Reexport });
                }
                LoadCommand::LoadUpwardDylib(dylib) => {
                    let name = dylib.name.to_string();
                    dynamic_libraries.push(name.clone());
                    dependencies.push(DylibDependency { install_name: name, kind: DylibDependencyKind::Upward });
                }
                LoadCommand::LazyLoadDylib(dylib) => {
                    // Until lazy bind opcodes are implemented, load these eagerly
                    // so their symbols are available before the first call.
                    let name = dylib.name.to_string();
                    dynamic_libraries.push(name.clone());
                    dependencies.push(DylibDependency { install_name: name, kind: DylibDependencyKind::Required });
                }
                LoadCommand::Rpath(path) => rpaths.push(path.clone()),
                _ => {}
            }
        }

        let text = segments.get(text_index.ok_or(MachOError::MissingTextSegment)?)
            .ok_or(MachOError::MissingTextSegment)?;
        let text_base = text.address;
        // MH_EXECUTE (2) images use LC_MAIN. MH_DYLIB (6) images are entered
        // through their clients and normally have no LC_MAIN command.
        let entry_point = match entry_offset {
            Some(offset) => resolve_main_entry(text, offset, 0)?,
            None if header.filetype == 6 => 0,
            None => return Err(MachOError::MissingEntryPoint),
        };

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
        if let Some(payload) = chained_fixups.as_deref() {
            let image_base = text_base.checked_sub(slide).ok_or(MachOError::AddressOverflow)?;
            apply_chained_rebases(payload, &segments, image_base, slide, memory)?;
        }
        Ok(Self { segments, entry_point, dynamic_libraries, dependencies, rpaths, chained_fixups, exports_trie, text_base, end_address })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(file_offset: u64, file_size: u64, vmaddr: u64, vmsize: u64) -> Segment {
        Segment { name: "__TEXT".into(), address: vmaddr, memory_size: vmsize, file_offset, file_size, initial_protection: 0x5 }
    }
    fn minimal_fixups_payload() -> Vec<u8> {
        let mut data = vec![0u8; 28 + 8 + 22 + 2 + 4];
        data[4..8].copy_from_slice(&28u32.to_le_bytes()); // starts_offset
        data[8..12].copy_from_slice(&60u32.to_le_bytes()); // imports_offset
        data[12..16].copy_from_slice(&60u32.to_le_bytes()); // symbols_offset
        data[20..24].copy_from_slice(&1u32.to_le_bytes()); // imports_format
        data[24..28].copy_from_slice(&0u32.to_le_bytes()); // symbols_format
        data[28..32].copy_from_slice(&1u32.to_le_bytes()); // seg_count
        data[32..36].copy_from_slice(&8u32.to_le_bytes()); // seg_info_offset[0]
        data[36..40].copy_from_slice(&24u32.to_le_bytes()); // starts_in_segment.size
        data[56..58].copy_from_slice(&1u16.to_le_bytes()); // page_count
        data[60..64].copy_from_slice(&[0, 0, 0, 0]);
        data
    }

    #[test]
    fn chained_fixups_accepts_bounded_minimal_tables() {
        assert!(validate_chained_fixups(&minimal_fixups_payload()).is_ok());
    }

    #[test]
    fn chained_fixups_rejects_unknown_version_and_truncated_segment_table() {
        let mut data = minimal_fixups_payload();
        data[0..4].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(validate_chained_fixups(&data), Err(MachOError::InvalidChainedFixups)));
        let mut data = minimal_fixups_payload();
        data[56..58].copy_from_slice(&4u16.to_le_bytes());
        assert!(matches!(validate_chained_fixups(&data), Err(MachOError::InvalidChainedFixups)));
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
    fn bind_test_image() -> (MachOImage, GuestMemory) {
        let mut payload = vec![0u8; 68];
        payload[4..8].copy_from_slice(&28u32.to_le_bytes());
        payload[8..12].copy_from_slice(&60u32.to_le_bytes());
        payload[12..16].copy_from_slice(&64u32.to_le_bytes());
        payload[16..20].copy_from_slice(&1u32.to_le_bytes());
        payload[20..24].copy_from_slice(&1u32.to_le_bytes());
        payload[24..28].copy_from_slice(&0u32.to_le_bytes());
        payload[28..32].copy_from_slice(&1u32.to_le_bytes());
        payload[32..36].copy_from_slice(&8u32.to_le_bytes());
        payload[36..40].copy_from_slice(&24u32.to_le_bytes());
        payload[40..42].copy_from_slice(&0x1000u16.to_le_bytes());
        payload[42..44].copy_from_slice(&6u16.to_le_bytes());
        payload[48..56].copy_from_slice(&0u64.to_le_bytes());
        payload[56..58].copy_from_slice(&1u16.to_le_bytes());
        payload[58..60].copy_from_slice(&0u16.to_le_bytes());
        payload[60..64].copy_from_slice(&1u32.to_le_bytes());
        payload[64..68].copy_from_slice(b"sym\0");
        let segment = Segment {
            name: "__DATA".into(), address: 0x1_0000_0000, memory_size: 0x1000,
            file_offset: 0, file_size: 0x1000, initial_protection: 3,
        };
        let image = MachOImage {
            segments: vec![segment], entry_point: 0, dynamic_libraries: vec![], dependencies: vec![], rpaths: vec![],
            chained_fixups: Some(payload), exports_trie: None,
            text_base: 0x1_0000_0000, end_address: 0x1_0000_1000,
        };
        let mut memory = GuestMemory::new();
        memory.map_zeroed(0x1_0000_0000, 0x1000).unwrap();
        (image, memory)
    }

    #[test]
    fn applies_chained_bind_using_resolved_symbol_address() {
        let (image, mut memory) = bind_test_image();
        memory.write(0x1_0000_0000, &(1u64 << 63).to_le_bytes()).unwrap();
        apply_chained_binds(&image, &mut memory, |import| {
            assert_eq!(import.symbol, "sym");
            Some(0x1_2345_6780)
        }).unwrap();
        let mut actual = [0u8; 8];
        memory.read(0x1_0000_0000, &mut actual).unwrap();
        assert_eq!(u64::from_le_bytes(actual), 0x1_2345_6780);
    }

    #[test]
    fn chained_bind_walk_continues_after_rebase_and_stages_writes() {
        let (image, mut memory) = bind_test_image();
        memory.write(0x1_0000_0000, &(2u64 << 51).to_le_bytes()).unwrap();
        memory.write(0x1_0000_0008, &(1u64 << 63).to_le_bytes()).unwrap();
        assert!(matches!(
            apply_chained_binds(&image, &mut memory, |_| None),
            Err(MachOError::UnresolvedStrongImport(_))
        ));
        let mut unchanged = [0u8; 8];
        memory.read(0x1_0000_0008, &mut unchanged).unwrap();
        assert_eq!(u64::from_le_bytes(unchanged), 1u64 << 63);
        apply_chained_binds(&image, &mut memory, |_| Some(0x9000)).unwrap();
        let mut resolved = [0u8; 8];
        memory.read(0x1_0000_0008, &mut resolved).unwrap();
        assert_eq!(u64::from_le_bytes(resolved), 0x9000);
    }

    #[test]
    fn decodes_multiple_chained_starts_from_overflow_array() {
        // starts-in-segment: 22-byte header, two page entries, then two overflow entries.
        let mut payload = vec![0u8; 30];
        payload[0..4].copy_from_slice(&30u32.to_le_bytes());
        payload[22..24].copy_from_slice(&0x8002u16.to_le_bytes());
        payload[24..26].copy_from_slice(&0xFFFFu16.to_le_bytes());
        payload[26..28].copy_from_slice(&0x0010u16.to_le_bytes());
        payload[28..30].copy_from_slice(&0x8020u16.to_le_bytes());
        assert_eq!(chained_page_starts(&payload, 0, 30, 2, 0x8002).unwrap(), vec![0x10, 0x20]);
        assert!(matches!(chained_page_starts(&payload, 0, 26, 2, 0x8002), Err(MachOError::InvalidChainedFixups)));
        assert!(matches!(chained_page_starts(&payload, 0, 30, 2, 0x8001), Err(MachOError::InvalidChainedFixups)));
    }

}

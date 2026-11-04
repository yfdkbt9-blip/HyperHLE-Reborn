/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Minimal 64-bit Mach-O image loader for the isolated A64 runtime.
//!
//! This loader deliberately handles LC_MAIN images first. A64 dyld rebasing,
//! binding, chained fixups, and LC_UNIXTHREAD state are separate follow-up
//! work; they are not silently treated as if the existing A32 loader handled
//! them.

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
    pub text_base: GuestAddr,
    pub end_address: GuestAddr,
}

#[derive(Debug)]
pub enum MachOError {
    Parse,
    WrongArchitecture,
    UnsupportedSubtype,
    MissingTextSegment,
    MissingEntryPoint,
    InvalidSegmentRange,
    AddressOverflow,
    Memory(MemoryError),
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
                        if header.cputype == CPU_TYPE_ARM64 && header.cpusubtype == arch.cpusubtype =>
                    {
                        if best.map_or(true, |(_, old)| priority > old) { best = Some((slice, priority)); }
                    }
                    _ => continue,
                }
            }
            best.map(|(slice, _)| slice).ok_or(MachOError::WrongArchitecture)
        }
    }
}

impl MachOImage {
    /// Parse an ARM64 Mach-O executable and map its 64-bit segments.
    /// `slide` is added to every guest virtual address.
    pub fn load(bytes: &[u8], memory: &mut GuestMemory, slide: GuestAddr) -> Result<Self, MachOError> {
        let image_bytes = choose_arm64_slice(bytes)?;
        let (header, commands) = match OFile::parse(&mut Cursor::new(image_bytes)).map_err(|_| MachOError::Parse)? {
            OFile::MachFile { header, commands } => (header, commands),
            OFile::FatFile { .. } => return Err(MachOError::Parse),
        };
        if header.cputype != CPU_TYPE_ARM64 || !header.is_64bit() { return Err(MachOError::WrongArchitecture); }

        let mut segments = Vec::new();
        let mut text_vmaddr = None;
        let mut entry_offset = None;
        let mut dynamic_libraries = Vec::new();
        for MachCommand(command, _) in &commands {
            match command {
                LoadCommand::Segment64 { segname, vmaddr, vmsize, fileoff, filesize, initprot, .. } => {
                    let address = u64::try_from(*vmaddr).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let memory_size = u64::try_from(*vmsize).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let file_offset = u64::try_from(*fileoff).map_err(|_| MachOError::InvalidSegmentRange)?;
                    let file_size = u64::try_from(*filesize).map_err(|_| MachOError::InvalidSegmentRange)?;
                    if file_size > memory_size { return Err(MachOError::InvalidSegmentRange); }
                    let file_end = file_offset.checked_add(file_size).ok_or(MachOError::InvalidSegmentRange)?;
                    if file_end > image_bytes.len() as u64 { return Err(MachOError::InvalidSegmentRange); }
                    if segname == "__TEXT" { text_vmaddr = Some(address); }
                    segments.push(Segment {
                        name: segname.clone(),
                        address: address.checked_add(slide).ok_or(MachOError::AddressOverflow)?,
                        memory_size,
                        file_offset,
                        file_size,
                        initial_protection: *initprot as u32,
                    });
                }
                LoadCommand::EntryPoint { entryoff, .. } => {
                    if entry_offset.replace(*entryoff).is_some() { return Err(MachOError::MissingEntryPoint); }
                }
                LoadCommand::LoadDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                LoadCommand::LoadWeakDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                LoadCommand::ReexportDyLib(dylib) => dynamic_libraries.push(dylib.name.to_string()),
                _ => {}
            }
        }
        let text_base = text_vmaddr.ok_or(MachOError::MissingTextSegment)?
            .checked_add(slide).ok_or(MachOError::AddressOverflow)?;
        let entry_point = text_base.checked_add(entry_offset.ok_or(MachOError::MissingEntryPoint)?)
            .ok_or(MachOError::AddressOverflow)?;
        let mut end_address = 0;
        for segment in &segments {
            let end = segment.address.checked_add(segment.memory_size).ok_or(MachOError::AddressOverflow)?;
            end_address = end_address.max(end);
            memory.map_zeroed(segment.address, segment.memory_size)?;
            if segment.file_size != 0 {
                let start = usize::try_from(segment.file_offset).map_err(|_| MachOError::InvalidSegmentRange)?;
                let finish = usize::try_from(segment.file_offset + segment.file_size).map_err(|_| MachOError::InvalidSegmentRange)?;
                memory.write(segment.address, &image_bytes[start..finish])?;
            }
        }
        Ok(Self { segments, entry_point, dynamic_libraries, text_base, end_address })
    }
}

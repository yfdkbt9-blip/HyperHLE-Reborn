/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Reading of 64-bit (arm64) Mach-O executables.
//!
//! This is the 64-bit counterpart of `mach_o.rs`. It deliberately depends only
//! on `std` (not on the `mach_object` crate, whose segment types are built
//! around 32-bit addresses), and it only *parses*: it produces lists of
//! segments, rebases, binds and initializer addresses. Copying segments into
//! guest memory and resolving binds against the HLE function table is left to
//! the caller, via [GuestMemory64].
//!
//! Supported: thin or fat images with an arm64 slice, `LC_SEGMENT_64`,
//! classic `LC_DYLD_INFO(_ONLY)` rebase/bind/lazy-bind opcodes (what iOS SDKs
//! up to iOS 14 emit), `LC_MAIN`, `__mod_init_func` and `__init_offsets`.
//!
//! Not supported yet: chained fixups (`LC_DYLD_CHAINED_FIXUPS`, iOS 15+ SDKs),
//! arm64e pointer authentication, threaded binds.
//!
//! Verified against a reference implementation in Python on a real iOS 10 SDK
//! binary: 128424 rebases, 13010 binds, 889 lazy binds, 86 init functions.
//!
//! References: `<mach-o/loader.h>` and `<mach-o/fixup-chains.h>` from the
//! macOS SDK (opcode values below are the `REBASE_OPCODE_*` and
//! `BIND_OPCODE_*` constants).

use std::collections::HashMap;

pub const MH_MAGIC_64: u32 = 0xfeed_facf;
pub const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const FAT_MAGIC: u32 = 0xcafe_babe;

const LC_REQ_DYLD: u32 = 0x8000_0000;
const LC_SYMTAB: u32 = 0x2;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_LOAD_WEAK_DYLIB: u32 = 0x18 | LC_REQ_DYLD;
const LC_SEGMENT_64: u32 = 0x19;
const LC_REEXPORT_DYLIB: u32 = 0x1f | LC_REQ_DYLD;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;
const LC_DYLD_INFO: u32 = 0x22;
const LC_DYLD_INFO_ONLY: u32 = 0x22 | LC_REQ_DYLD;
const LC_LOAD_UPWARD_DYLIB: u32 = 0x23 | LC_REQ_DYLD;
const LC_MAIN: u32 = 0x28 | LC_REQ_DYLD;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x34 | LC_REQ_DYLD;

const SECTION_TYPE_MASK: u32 = 0xff;
const S_MOD_INIT_FUNC_POINTERS: u32 = 0x9;
const S_INIT_FUNC_OFFSETS: u32 = 0x16;

const N_STAB: u8 = 0xe0;
const N_EXT: u8 = 0x01;
const N_TYPE: u8 = 0x0e;
const N_SECT: u8 = 0x0e;

/// Sanity limit for "do N times" opcodes, so a corrupt stream can't make us
/// allocate gigabytes.
const MAX_REPEAT: u64 = 1 << 24;

#[derive(Debug, Clone)]
pub struct Section64 {
    pub segment_name: String,
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub file_offset: u32,
    pub flags: u32,
}

impl Section64 {
    pub fn section_type(&self) -> u32 {
        self.flags & SECTION_TYPE_MASK
    }
}

#[derive(Debug, Clone)]
pub struct Segment64 {
    pub name: String,
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    /// `VM_PROT_*` bits: 1 = read, 2 = write, 4 = execute.
    pub initprot: u32,
    pub sections: Vec<Section64>,
}

#[derive(Debug, Clone)]
pub struct DylibRef {
    pub path: String,
    pub weak: bool,
    pub reexport: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bind64 {
    /// Unslid address of the 8-byte pointer to be filled in.
    pub addr: u64,
    pub symbol: String,
    /// 1-based index into [MachO64::dylibs], or one of the special values
    /// 0 (this image), -1 (main executable), -2 (flat lookup).
    pub lib_ordinal: i64,
    pub addend: i64,
    pub weak_import: bool,
    /// Came from the lazy bind stream (`__la_symbol_ptr`) rather than the
    /// regular one. Real dyld resolves these on first call; we resolve them
    /// eagerly, so the caller can treat both kinds the same.
    pub lazy: bool,
}

#[derive(Debug)]
pub struct MachO64 {
    pub name: String,
    /// Offset of the arm64 slice inside the original file (non-zero only for
    /// fat files). `Segment64::fileoff` is relative to this.
    pub image_offset: usize,
    pub segments: Vec<Segment64>,
    /// In ordinal order, so `dylibs[lib_ordinal - 1]` is the library for a
    /// bind.
    pub dylibs: Vec<DylibRef>,
    /// Unslid addresses of 8-byte pointers to which the slide must be added.
    pub rebases: Vec<u64>,
    pub binds: Vec<Bind64>,
    /// Unslid addresses of static initializers, in execution order.
    pub init_funcs: Vec<u64>,
    /// Unslid address of the `main` function (from `LC_MAIN`).
    pub entry_point_pc: Option<u64>,
    /// Unslid address of the Mach-O header (start of `__TEXT`).
    pub text_base: u64,
    /// End address of the highest-addressed segment.
    pub last_segment_end: u64,
    /// All defined symbols (unslid addresses).
    pub exported_symbols: HashMap<String, u64>,
    /// Defined symbols marked external.
    pub external_symbols: HashMap<String, u64>,
}

/// Minimal interface to 64-bit guest memory, implemented by whatever memory
/// model the 64-bit port ends up with.
pub trait GuestMemory64 {
    /// Returns false if the range isn't mapped.
    fn write_bytes(&mut self, addr: u64, bytes: &[u8]) -> bool;
    /// Zero-fill. Returns false if the range isn't mapped.
    fn zero(&mut self, addr: u64, len: u64) -> bool;
    fn read_u64(&self, addr: u64) -> Option<u64>;
    fn write_u64(&mut self, addr: u64, value: u64) -> bool;
}

// ---------------------------------------------------------------------------
// Low-level readers

fn rd_u32(d: &[u8], off: usize) -> Result<u32, String> {
    let end = off.checked_add(4).ok_or("offset overflow")?;
    d.get(off..end)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| format!("read past end of file at {off:#x}"))
}

fn rd_u64(d: &[u8], off: usize) -> Result<u64, String> {
    let end = off.checked_add(8).ok_or("offset overflow")?;
    d.get(off..end)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| format!("read past end of file at {off:#x}"))
}

fn be_u32(d: &[u8], off: usize) -> Result<u32, String> {
    let end = off.checked_add(4).ok_or("offset overflow")?;
    d.get(off..end)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| format!("read past end of file at {off:#x}"))
}

/// Fixed-size NUL-padded name field (segment/section names are 16 bytes).
fn fixed_str(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// NUL-terminated string starting at `pos`; advances `pos` past the NUL.
fn read_cstr(d: &[u8], pos: &mut usize) -> Result<String, String> {
    let rest = d.get(*pos..).ok_or("string starts past end of data")?;
    let len = rest
        .iter()
        .position(|&c| c == 0)
        .ok_or("unterminated string")?;
    let s = String::from_utf8_lossy(&rest[..len]).into_owned();
    *pos += len + 1;
    Ok(s)
}

fn read_uleb(d: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *d.get(*pos).ok_or("truncated LEB128")?;
        *pos += 1;
        if shift < 64 {
            result |= ((byte & 0x7f) as u64) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
    }
}

fn read_sleb(d: &[u8], pos: &mut usize) -> Result<i64, String> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *d.get(*pos).ok_or("truncated LEB128")?;
        *pos += 1;
        if shift < 64 {
            result |= ((byte & 0x7f) as i64) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && byte & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Ok(result);
        }
    }
}

// ---------------------------------------------------------------------------
// dyld opcode streams

fn seg_addr(segs: &[Segment64], seg: usize, off: u64, what: &str) -> Result<u64, String> {
    segs.get(seg)
        .map(|s| s.vmaddr.wrapping_add(off))
        .ok_or_else(|| format!("{what}: bad segment index {seg}"))
}

/// Parse the rebase opcode stream (`REBASE_OPCODE_*`).
///
/// All offset arithmetic MUST wrap: compilers encode "step back by 8" as the
/// ULEB value 2^64 - 8.
fn parse_rebases(d: &[u8], segs: &[Segment64]) -> Result<Vec<u64>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let (mut seg, mut off, mut ty) = (0usize, 0u64, 1u8);
    while pos < d.len() {
        let byte = d[pos];
        pos += 1;
        let (op, imm) = (byte & 0xf0, byte & 0x0f);
        match op {
            0x00 => break, // DONE
            0x10 => ty = imm, // SET_TYPE_IMM
            0x20 => {
                // SET_SEGMENT_AND_OFFSET_ULEB
                seg = imm as usize;
                off = read_uleb(d, &mut pos)?;
            }
            0x30 => off = off.wrapping_add(read_uleb(d, &mut pos)?), // ADD_ADDR_ULEB
            0x40 => off = off.wrapping_add(imm as u64 * 8),          // ADD_ADDR_IMM_SCALED
            0x50 | 0x60 | 0x70 | 0x80 => {
                if ty != 1 {
                    return Err(format!("unsupported rebase type {ty} (expected pointer)"));
                }
                let (times, skip) = match op {
                    0x50 => (imm as u64, 0u64), // DO_REBASE_IMM_TIMES
                    0x60 => (read_uleb(d, &mut pos)?, 0), // DO_REBASE_ULEB_TIMES
                    0x70 => {
                        // DO_REBASE_ADD_ADDR_ULEB: one rebase, then skip.
                        (1, read_uleb(d, &mut pos)?)
                    }
                    _ => {
                        // DO_REBASE_ULEB_TIMES_SKIPPING_ULEB
                        let n = read_uleb(d, &mut pos)?;
                        (n, read_uleb(d, &mut pos)?)
                    }
                };
                if times > MAX_REPEAT {
                    return Err(format!("rebase repeat count {times} is implausible"));
                }
                for _ in 0..times {
                    out.push(seg_addr(segs, seg, off, "rebase")?);
                    off = off.wrapping_add(skip).wrapping_add(8);
                }
            }
            _ => return Err(format!("unknown rebase opcode {op:#x}")),
        }
    }
    Ok(out)
}

/// Parse a bind opcode stream (`BIND_OPCODE_*`). The lazy stream uses the same
/// opcodes, but its `DONE` opcodes separate entries instead of ending it.
fn parse_binds(d: &[u8], segs: &[Segment64], lazy: bool) -> Result<Vec<Bind64>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let (mut seg, mut off) = (0usize, 0u64);
    let mut symbol: Option<String> = None;
    let (mut ordinal, mut addend, mut weak_import) = (0i64, 0i64, false);

    // Emit one bind at the current position.
    macro_rules! emit {
        () => {{
            let symbol = symbol
                .clone()
                .ok_or("bind opcode used before a symbol was set")?;
            out.push(Bind64 {
                addr: seg_addr(segs, seg, off, "bind")?,
                symbol,
                lib_ordinal: ordinal,
                addend,
                weak_import,
                lazy,
            });
        }};
    }

    while pos < d.len() {
        let byte = d[pos];
        pos += 1;
        let (op, imm) = (byte & 0xf0, byte & 0x0f);
        match op {
            0x00 => {
                // DONE
                if lazy {
                    continue;
                }
                break;
            }
            0x10 => ordinal = imm as i64, // SET_DYLIB_ORDINAL_IMM
            0x20 => ordinal = read_uleb(d, &mut pos)? as i64, // SET_DYLIB_ORDINAL_ULEB
            0x30 => {
                // SET_DYLIB_SPECIAL_IMM: 0 = self, 0xf = -1, 0xe = -2
                ordinal = if imm == 0 {
                    0
                } else {
                    ((0xf0 | imm) as i8) as i64
                };
            }
            0x40 => {
                // SET_SYMBOL_TRAILING_FLAGS_IMM
                weak_import = imm & 0x1 != 0;
                symbol = Some(read_cstr(d, &mut pos)?);
            }
            0x50 => {} // SET_TYPE_IMM (only pointers exist on arm64)
            0x60 => addend = read_sleb(d, &mut pos)?, // SET_ADDEND_SLEB
            0x70 => {
                // SET_SEGMENT_AND_OFFSET_ULEB
                seg = imm as usize;
                off = read_uleb(d, &mut pos)?;
            }
            0x80 => off = off.wrapping_add(read_uleb(d, &mut pos)?), // ADD_ADDR_ULEB
            0x90 => {
                // DO_BIND
                emit!();
                off = off.wrapping_add(8);
            }
            0xa0 => {
                // DO_BIND_ADD_ADDR_ULEB
                emit!();
                off = off.wrapping_add(read_uleb(d, &mut pos)?).wrapping_add(8);
            }
            0xb0 => {
                // DO_BIND_ADD_ADDR_IMM_SCALED
                emit!();
                off = off.wrapping_add(imm as u64 * 8).wrapping_add(8);
            }
            0xc0 => {
                // DO_BIND_ULEB_TIMES_SKIPPING_ULEB
                let times = read_uleb(d, &mut pos)?;
                let skip = read_uleb(d, &mut pos)?;
                if times > MAX_REPEAT {
                    return Err(format!("bind repeat count {times} is implausible"));
                }
                for _ in 0..times {
                    emit!();
                    off = off.wrapping_add(skip).wrapping_add(8);
                }
            }
            0xd0 => return Err("threaded binds (arm64e / chained) are not supported".into()),
            _ => return Err(format!("unknown bind opcode {op:#x}")),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Top-level parsing

/// Slice of the image holding one dyld_info stream (empty if `size` is 0).
fn stream_slice<'a>(d: &'a [u8], off: u32, size: u32, what: &str) -> Result<&'a [u8], String> {
    if size == 0 {
        return Ok(&[]);
    }
    let (o, s) = (off as usize, size as usize);
    o.checked_add(s)
        .and_then(|e| d.get(o..e))
        .ok_or_else(|| format!("{what} stream extends past end of file"))
}

/// If `file` is a fat binary, find the arm64 slice; returns (offset, length).
fn find_arm64_slice(file: &[u8]) -> Result<(usize, usize), String> {
    if file.len() >= 4 && be_u32(file, 0)? == FAT_MAGIC {
        let n = be_u32(file, 4)? as usize;
        for i in 0..n.min(32) {
            let e = 8 + i * 20;
            if be_u32(file, e)? == CPU_TYPE_ARM64 {
                let (o, s) = (be_u32(file, e + 8)? as usize, be_u32(file, e + 12)? as usize);
                let end = o.checked_add(s).ok_or("fat slice overflows")?;
                if end > file.len() {
                    return Err("fat arm64 slice extends past end of file".into());
                }
                return Ok((o, s));
            }
        }
        return Err("fat binary has no arm64 slice".into());
    }
    Ok((0, file.len()))
}

impl MachO64 {
    pub fn load_from_bytes(file: &[u8], name: String) -> Result<MachO64, String> {
        let (image_offset, image_len) = find_arm64_slice(file)?;
        let d = &file[image_offset..image_offset + image_len];

        if rd_u32(d, 0)? != MH_MAGIC_64 {
            return Err("not a 64-bit little-endian Mach-O".into());
        }
        if rd_u32(d, 4)? != CPU_TYPE_ARM64 {
            return Err("Mach-O is not arm64".into());
        }
        let ncmds = rd_u32(d, 16)? as usize;
        let sizeofcmds = rd_u32(d, 20)? as usize;
        let cmds_end = 32usize
            .checked_add(sizeofcmds)
            .filter(|&e| e <= d.len())
            .ok_or("load commands extend past end of file")?;

        let mut segments: Vec<Segment64> = Vec::new();
        let mut dylibs: Vec<DylibRef> = Vec::new();
        let mut dyld_info: Option<[u32; 10]> = None;
        let mut symtab: Option<[u32; 4]> = None;
        let mut entryoff: Option<u64> = None;
        let mut has_chained_fixups = false;

        let mut off = 32usize;
        for _ in 0..ncmds {
            let cmd = rd_u32(d, off)?;
            let size = rd_u32(d, off + 4)? as usize;
            if size < 8 || off + size > cmds_end {
                return Err(format!("bad load command size {size:#x} at {off:#x}"));
            }
            match cmd {
                LC_SEGMENT_64 => {
                    if size < 72 {
                        return Err("LC_SEGMENT_64 too small".into());
                    }
                    let nsects = rd_u32(d, off + 64)? as usize;
                    if size < 72 + nsects * 80 {
                        return Err("LC_SEGMENT_64 section list truncated".into());
                    }
                    let mut sections = Vec::with_capacity(nsects);
                    for i in 0..nsects {
                        let s = off + 72 + i * 80;
                        sections.push(Section64 {
                            name: fixed_str(&d[s..s + 16]),
                            segment_name: fixed_str(&d[s + 16..s + 32]),
                            addr: rd_u64(d, s + 32)?,
                            size: rd_u64(d, s + 40)?,
                            file_offset: rd_u32(d, s + 48)?,
                            flags: rd_u32(d, s + 64)?,
                        });
                    }
                    let seg = Segment64 {
                        name: fixed_str(&d[off + 8..off + 24]),
                        vmaddr: rd_u64(d, off + 24)?,
                        vmsize: rd_u64(d, off + 32)?,
                        fileoff: rd_u64(d, off + 40)?,
                        filesize: rd_u64(d, off + 48)?,
                        initprot: rd_u32(d, off + 60)?,
                        sections,
                    };
                    let file_end = seg
                        .fileoff
                        .checked_add(seg.filesize)
                        .ok_or("segment file range overflows")?;
                    if file_end > d.len() as u64 {
                        return Err(format!("segment {} extends past end of file", seg.name));
                    }
                    if seg.filesize > seg.vmsize {
                        return Err(format!("segment {} has filesize > vmsize", seg.name));
                    }
                    segments.push(seg);
                }
                LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB | LC_LAZY_LOAD_DYLIB
                | LC_LOAD_UPWARD_DYLIB => {
                    let name_off = rd_u32(d, off + 8)? as usize;
                    if name_off >= size {
                        return Err("dylib name offset out of range".into());
                    }
                    let mut p = off + name_off;
                    dylibs.push(DylibRef {
                        path: read_cstr(d, &mut p)?,
                        weak: cmd == LC_LOAD_WEAK_DYLIB,
                        reexport: cmd == LC_REEXPORT_DYLIB,
                    });
                }
                LC_DYLD_INFO | LC_DYLD_INFO_ONLY => {
                    let mut f = [0u32; 10];
                    for (i, v) in f.iter_mut().enumerate() {
                        *v = rd_u32(d, off + 8 + i * 4)?;
                    }
                    dyld_info = Some(f);
                }
                LC_SYMTAB => {
                    let mut f = [0u32; 4];
                    for (i, v) in f.iter_mut().enumerate() {
                        *v = rd_u32(d, off + 8 + i * 4)?;
                    }
                    symtab = Some(f);
                }
                LC_MAIN => entryoff = Some(rd_u64(d, off + 8)?),
                LC_DYLD_CHAINED_FIXUPS => has_chained_fixups = true,
                _ => {}
            }
            off += size;
        }

        if has_chained_fixups && dyld_info.is_none() {
            return Err("this binary uses chained fixups (iOS 15+ SDK), not supported yet".into());
        }

        let text = segments
            .iter()
            .find(|s| s.name == "__TEXT")
            .ok_or("no __TEXT segment")?;
        let text_base = text.vmaddr;
        let last_segment_end = segments
            .iter()
            .map(|s| s.vmaddr.wrapping_add(s.vmsize))
            .max()
            .unwrap_or(0);

        // dyld_info stream slices: [rebase, bind, weak_bind, lazy_bind, export]
        let (mut rebases, mut binds) = (Vec::new(), Vec::new());
        if let Some(f) = dyld_info {
            rebases = parse_rebases(stream_slice(d, f[0], f[1], "rebase")?, &segments)?;
            binds = parse_binds(stream_slice(d, f[2], f[3], "bind")?, &segments, false)?;
            // Weak binds (f[4], f[5]) are for C++ symbol coalescing between
            // images; with one executable image they can be ignored.
            binds.extend(parse_binds(
                stream_slice(d, f[6], f[7], "lazy bind")?,
                &segments,
                true,
            )?);
        }

        // Static initializers.
        let mut init_funcs = Vec::new();
        for sec in segments.iter().flat_map(|s| s.sections.iter()) {
            match sec.section_type() {
                S_MOD_INIT_FUNC_POINTERS => {
                    for i in 0..sec.size / 8 {
                        init_funcs.push(rd_u64(d, sec.file_offset as usize + i as usize * 8)?);
                    }
                }
                S_INIT_FUNC_OFFSETS => {
                    for i in 0..sec.size / 4 {
                        let o = rd_u32(d, sec.file_offset as usize + i as usize * 4)?;
                        init_funcs.push(text_base.wrapping_add(o as u64));
                    }
                }
                _ => {}
            }
        }

        // Symbol table (defined symbols only; undefined ones are covered by
        // the bind streams).
        let (mut exported_symbols, mut external_symbols) = (HashMap::new(), HashMap::new());
        if let Some([symoff, nsyms, stroff, strsize]) = symtab {
            let strtab = d
                .get(stroff as usize..(stroff as usize).saturating_add(strsize as usize))
                .ok_or("string table extends past end of file")?;
            for i in 0..nsyms as usize {
                let e = symoff as usize + i * 16;
                let strx = rd_u32(d, e)? as usize;
                let n_type = *d.get(e + 4).ok_or("symbol table truncated")?;
                if n_type & N_STAB != 0 || n_type & N_TYPE != N_SECT {
                    continue;
                }
                let mut p = strx;
                let sym_name = read_cstr(strtab, &mut p)?;
                let value = rd_u64(d, e + 8)?;
                if n_type & N_EXT != 0 {
                    external_symbols.insert(sym_name.clone(), value);
                }
                exported_symbols.insert(sym_name, value);
            }
        }

        Ok(MachO64 {
            name,
            image_offset,
            entry_point_pc: entryoff.map(|o| text_base.wrapping_add(o)),
            segments,
            dylibs,
            rebases,
            binds,
            init_funcs,
            text_base,
            last_segment_end,
            exported_symbols,
            external_symbols,
        })
    }

    /// Library a bind refers to, if it names a specific one.
    pub fn dylib_for_ordinal(&self, ordinal: i64) -> Option<&DylibRef> {
        if ordinal >= 1 {
            self.dylibs.get(ordinal as usize - 1)
        } else {
            None
        }
    }

    /// Copy all segments (except `__PAGEZERO`) into guest memory at
    /// `vmaddr + slide`, zero-filling the part beyond the file data (`.bss`).
    /// `file` must be the same bytes that were passed to `load_from_bytes`.
    pub fn map_into<M: GuestMemory64>(
        &self,
        file: &[u8],
        mem: &mut M,
        slide: u64,
    ) -> Result<(), String> {
        for seg in &self.segments {
            if seg.name == "__PAGEZERO" || seg.vmsize == 0 {
                continue;
            }
            let base = seg.vmaddr.wrapping_add(slide);
            let start = self.image_offset + seg.fileoff as usize;
            let data = file
                .get(start..start + seg.filesize as usize)
                .ok_or_else(|| format!("segment {} data out of range", seg.name))?;
            if !mem.write_bytes(base, data) {
                return Err(format!("could not map segment {} at {base:#x}", seg.name));
            }
            let tail = seg.vmsize - seg.filesize;
            if tail > 0 && !mem.zero(base + seg.filesize, tail) {
                return Err(format!("could not zero tail of segment {}", seg.name));
            }
        }
        Ok(())
    }

    /// Add `slide` to every rebased pointer. Call after `map_into`.
    pub fn apply_rebases<M: GuestMemory64>(&self, mem: &mut M, slide: u64) -> Result<(), String> {
        if slide == 0 {
            return Ok(());
        }
        for &addr in &self.rebases {
            let a = addr.wrapping_add(slide);
            let v = mem
                .read_u64(a)
                .ok_or_else(|| format!("rebase target {a:#x} is not mapped"))?;
            if !mem.write_u64(a, v.wrapping_add(slide)) {
                return Err(format!("could not write rebased pointer at {a:#x}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<Segment64> {
        ["__PAGEZERO", "__TEXT", "__DATA"]
            .iter()
            .enumerate()
            .map(|(i, n)| Segment64 {
                name: n.to_string(),
                vmaddr: [0, 0x1_0000_0000, 0x1_0082_4000][i],
                vmsize: 0x1000,
                fileoff: 0,
                filesize: 0,
                initprot: 3,
                sections: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn leb128() {
        let mut p = 0;
        assert_eq!(read_uleb(&[0xe5, 0x8e, 0x26], &mut p).unwrap(), 624485);
        let mut p = 0;
        assert_eq!(read_sleb(&[0x7f], &mut p).unwrap(), -1);
        let mut p = 0;
        assert_eq!(read_sleb(&[0xc0, 0xbb, 0x78], &mut p).unwrap(), -123456);
        let mut p = 0;
        assert!(read_uleb(&[0x80], &mut p).is_err());
    }

    #[test]
    fn rebase_stream() {
        // SET_TYPE_IMM(pointer), SET_SEGMENT_AND_OFFSET(seg 2, 8),
        // DO_REBASE_IMM_TIMES(3), DONE
        let d = [0x11, 0x22, 0x08, 0x53, 0x00];
        let r = parse_rebases(&d, &segs()).unwrap();
        let base = 0x1_0082_4000u64;
        assert_eq!(r, vec![base + 8, base + 16, base + 24]);
    }

    #[test]
    fn bind_stream_wraps_negative_offsets() {
        // Real binaries encode "step back 8" as ULEB 2^64 - 8 after a bind.
        let mut d = vec![0x72, 0x10, 0x40, b'_', b'x', 0, 0x11, 0xa0];
        d.extend_from_slice(&[0xf8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
        d.extend_from_slice(&[0x90, 0x00]);
        let b = parse_binds(&d, &segs(), false).unwrap();
        let want = 0x1_0082_4000u64 + 0x10;
        assert_eq!(b.len(), 2);
        assert_eq!((b[0].addr, b[1].addr), (want, want));
        assert_eq!((b[0].symbol.as_str(), b[0].lib_ordinal), ("_x", 1));
    }

    #[test]
    fn special_ordinals_and_weak_flag() {
        // SET_DYLIB_SPECIAL_IMM(0xe = -2), SET_SYMBOL(weak), segment, DO_BIND
        let d = [0x3e, 0x41, b'a', 0, 0x72, 0x00, 0x90, 0x00];
        let b = parse_binds(&d, &segs(), false).unwrap();
        assert_eq!(b[0].lib_ordinal, -2);
        assert!(b[0].weak_import);
    }

    #[test]
    fn lazy_stream_continues_past_done() {
        let d = [
            0x72, 0x00, 0x11, 0x40, b'a', 0, 0x90, 0x00, //
            0x72, 0x08, 0x11, 0x40, b'b', 0, 0x90, 0x00,
        ];
        let b = parse_binds(&d, &segs(), true).unwrap();
        assert_eq!(b.len(), 2);
        assert!(b.iter().all(|x| x.lazy));
        assert_eq!(parse_binds(&d, &segs(), false).unwrap().len(), 1);
    }

    #[test]
    fn bind_without_symbol_is_an_error() {
        assert!(parse_binds(&[0x72, 0x00, 0x90], &segs(), false).is_err());
    }

    #[test]
    fn rejects_32_bit_and_garbage() {
        assert!(MachO64::load_from_bytes(&[0xce, 0xfa, 0xed, 0xfe, 0, 0, 0, 0], "x".into()).is_err());
        assert!(MachO64::load_from_bytes(&[], "x".into()).is_err());
    }

    struct FakeMem {
        base: u64,
        bytes: Vec<u8>,
    }
    impl GuestMemory64 for FakeMem {
        fn write_bytes(&mut self, a: u64, b: &[u8]) -> bool {
            let o = (a - self.base) as usize;
            match self.bytes.get_mut(o..o + b.len()) {
                Some(s) => {
                    s.copy_from_slice(b);
                    true
                }
                None => false,
            }
        }
        fn zero(&mut self, a: u64, n: u64) -> bool {
            self.write_bytes(a, &vec![0; n as usize])
        }
        fn read_u64(&self, a: u64) -> Option<u64> {
            let o = (a - self.base) as usize;
            Some(u64::from_le_bytes(self.bytes.get(o..o + 8)?.try_into().ok()?))
        }
        fn write_u64(&mut self, a: u64, v: u64) -> bool {
            self.write_bytes(a, &v.to_le_bytes())
        }
    }

    #[test]
    fn rebases_add_slide() {
        let mut mem = FakeMem { base: 0x2_0000_0000, bytes: vec![0; 0x1000] };
        let img = MachO64 {
            name: "t".into(),
            image_offset: 0,
            segments: Vec::new(),
            dylibs: Vec::new(),
            rebases: vec![0x1_0000_0010],
            binds: Vec::new(),
            init_funcs: Vec::new(),
            entry_point_pc: None,
            text_base: 0x1_0000_0000,
            last_segment_end: 0,
            exported_symbols: HashMap::new(),
            external_symbols: HashMap::new(),
        };
        let slide = 0x1_0000_0000;
        mem.write_u64(0x2_0000_0010, 0x1_0000_5000);
        img.apply_rebases(&mut mem, slide).unwrap();
        assert_eq!(mem.read_u64(0x2_0000_0010), Some(0x2_0000_5000));
    }
}

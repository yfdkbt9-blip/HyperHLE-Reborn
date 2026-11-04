/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! AArch64 dynamic-linker stub instruction helpers.
//!
//! These encode the minimal SVC/RET bridge. Symbol resolution and framework
//! dispatch remain separate work because the current dyld dispatcher is A32.

use super::memory::{GuestAddr, GuestMemory, MemoryError};
use std::convert::TryInto;


/// One imported symbol described by an LC_DYLD_CHAINED_FIXUPS import table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainedImport {
    /// Signed Mach-O library ordinal (including special ordinals).
    pub library_ordinal: i16,
    pub weak: bool,
    pub symbol: String,
    pub addend: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainedImportError {
    Truncated,
    UnsupportedFormat,
    UnsupportedSymbolsCompression,
    InvalidNameOffset,
    UnterminatedName,
    InvalidUtf8,
    InvalidReservedBits,
}

/// Decode import records and their symbol names. This does not resolve symbols
/// or patch bind locations; callers must supply a dyld symbol resolver.
pub fn parse_chained_imports(payload: &[u8]) -> Result<Vec<ChainedImport>, ChainedImportError> {
    fn u32_at(data: &[u8], offset: usize) -> Result<u32, ChainedImportError> {
        let end=offset.checked_add(4).ok_or(ChainedImportError::Truncated)?;
        let b=data.get(offset..end).ok_or(ChainedImportError::Truncated)?;
        Ok(u32::from_le_bytes(b.try_into().map_err(|_|ChainedImportError::Truncated)?))
    }
    fn u64_at(data: &[u8], offset: usize) -> Result<u64, ChainedImportError> {
        let end=offset.checked_add(8).ok_or(ChainedImportError::Truncated)?;
        let b=data.get(offset..end).ok_or(ChainedImportError::Truncated)?;
        Ok(u64::from_le_bytes(b.try_into().map_err(|_|ChainedImportError::Truncated)?))
    }
    if payload.len()<28 { return Err(ChainedImportError::Truncated); }
    let imports_offset=usize::try_from(u32_at(payload,8)?).map_err(|_|ChainedImportError::Truncated)?;
    let symbols_offset=usize::try_from(u32_at(payload,12)?).map_err(|_|ChainedImportError::Truncated)?;
    let count=usize::try_from(u32_at(payload,16)?).map_err(|_|ChainedImportError::Truncated)?;
    let format=u32_at(payload,20)?;
    let symbols_format=u32_at(payload,24)?;
    if symbols_format!=0 { return Err(ChainedImportError::UnsupportedSymbolsCompression); }
    let width=match format { 1=>4, 2=>8, 3=>16, _=>return Err(ChainedImportError::UnsupportedFormat) };
    let table_size=count.checked_mul(width).ok_or(ChainedImportError::Truncated)?;
    let table_end=imports_offset.checked_add(table_size).ok_or(ChainedImportError::Truncated)?;
    if imports_offset<28 || table_end>symbols_offset || symbols_offset>payload.len() {
        return Err(ChainedImportError::Truncated);
    }
    let mut imports=Vec::with_capacity(count);
    for index in 0..count {
        let at=imports_offset.checked_add(index.checked_mul(width).ok_or(ChainedImportError::Truncated)?)
            .ok_or(ChainedImportError::Truncated)?;
        let (ordinal,weak,name_offset,addend)=if format==3 {
            let bits=u64_at(payload,at)?;
            if (bits>>17)&0x7FFF != 0 { return Err(ChainedImportError::InvalidReservedBits); }
            let ordinal=(bits&0xFFFF) as u16 as i16;
            let weak=((bits>>16)&1)!=0;
            let name=(bits>>32) as u32;
            let addend=u64_at(payload,at+8)? as i64;
            (ordinal,weak,name,addend)
        } else {
            let bits=u32_at(payload,at)?;
            let ordinal=(bits&0xFF) as u8 as i8 as i16;
            let weak=((bits>>8)&1)!=0;
            let name=bits>>9;
            let addend=if format==2 { u32_at(payload,at+4)? as i32 as i64 } else { 0 };
            (ordinal,weak,name,addend)
        };
        let name_at=symbols_offset.checked_add(usize::try_from(name_offset).map_err(|_|ChainedImportError::InvalidNameOffset)?)
            .ok_or(ChainedImportError::InvalidNameOffset)?;
        let tail=payload.get(name_at..).ok_or(ChainedImportError::InvalidNameOffset)?;
        let nul=tail.iter().position(|b|*b==0).ok_or(ChainedImportError::UnterminatedName)?;
        let symbol=std::str::from_utf8(&tail[..nul]).map_err(|_|ChainedImportError::InvalidUtf8)?.to_owned();
        imports.push(ChainedImport { library_ordinal:ordinal, weak, symbol, addend });
    }
    Ok(imports)
}


/// One symbol decoded from an export trie.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportKind {
    Regular { address: u64 },
    Reexport { library_ordinal: u64, imported_name: String },
    StubAndResolver { stub: u64, resolver: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedSymbol {
    pub name: String,
    pub flags: u64,
    pub kind: ExportKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportTrieError {
    Truncated,
    InvalidUleb128,
    InvalidEdge,
    InvalidUtf8,
    InvalidTerminal,
    UnsupportedFlags,
    ChildOutOfRange,
    Cycle,
    ExcessiveDepth,
}

/// Decode a bounded Mach-O export trie. Child offsets are relative to the
/// beginning of the trie; only recursion-stack cycles are rejected, so valid
/// shared subtrees remain representable.
pub fn parse_export_trie(data: &[u8]) -> Result<Vec<ExportedSymbol>, ExportTrieError> {
    fn uleb(data: &[u8], cursor: &mut usize, end: usize) -> Result<u64, ExportTrieError> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            if *cursor >= end { return Err(ExportTrieError::Truncated); }
            let byte = data[*cursor];
            *cursor += 1;
            let part = u64::from(byte & 0x7f);
            if shift >= 64 || (shift == 63 && part > 1) {
                return Err(ExportTrieError::InvalidUleb128);
            }
            value |= part.checked_shl(shift).ok_or(ExportTrieError::InvalidUleb128)?;
            if byte & 0x80 == 0 { return Ok(value); }
            shift = shift.checked_add(7).ok_or(ExportTrieError::InvalidUleb128)?;
            if shift >= 70 { return Err(ExportTrieError::InvalidUleb128); }
        }
    }
    fn nul_string(data: &[u8], cursor: &mut usize, end: usize) -> Result<String, ExportTrieError> {
        let tail = data.get(*cursor..end).ok_or(ExportTrieError::Truncated)?;
        let len = tail.iter().position(|b| *b == 0).ok_or(ExportTrieError::InvalidEdge)?;
        let value = std::str::from_utf8(&tail[..len]).map_err(|_| ExportTrieError::InvalidUtf8)?.to_owned();
        *cursor = cursor.checked_add(len + 1).ok_or(ExportTrieError::Truncated)?;
        Ok(value)
    }
    fn walk(
        data: &[u8], offset: usize, prefix: &str, active: &mut std::collections::HashSet<usize>,
        output: &mut Vec<ExportedSymbol>, depth: usize,
    ) -> Result<(), ExportTrieError> {
        if depth > 1024 { return Err(ExportTrieError::ExcessiveDepth); }
        if offset >= data.len() { return Err(ExportTrieError::ChildOutOfRange); }
        if !active.insert(offset) { return Err(ExportTrieError::Cycle); }
        let result = (|| {
            let mut cursor = offset;
            let terminal_size = usize::try_from(uleb(data, &mut cursor, data.len())?)
                .map_err(|_| ExportTrieError::InvalidTerminal)?;
            let terminal_end = cursor.checked_add(terminal_size).ok_or(ExportTrieError::InvalidTerminal)?;
            if terminal_end > data.len() { return Err(ExportTrieError::InvalidTerminal); }
            if terminal_size != 0 {
                let mut terminal_cursor = cursor;
                let flags = uleb(data, &mut terminal_cursor, terminal_end)?;
                let kind = if flags & 0x08 != 0 {
                    let ordinal = uleb(data, &mut terminal_cursor, terminal_end)?;
                    let imported_name = nul_string(data, &mut terminal_cursor, terminal_end)?;
                    ExportKind::Reexport { library_ordinal: ordinal, imported_name }
                } else {
                    let address = uleb(data, &mut terminal_cursor, terminal_end)?;
                    if flags & 0x10 != 0 {
                        let resolver = uleb(data, &mut terminal_cursor, terminal_end)?;
                        ExportKind::StubAndResolver { stub: address, resolver }
                    } else {
                        ExportKind::Regular { address }
                    }
                };
                if terminal_cursor != terminal_end { return Err(ExportTrieError::InvalidTerminal); }
                output.push(ExportedSymbol { name: prefix.to_owned(), flags, kind });
            }
            cursor = terminal_end;
            let child_count = *data.get(cursor).ok_or(ExportTrieError::Truncated)? as usize;
            cursor += 1;
            for _ in 0..child_count {
                let edge = nul_string(data, &mut cursor, data.len())?;
                if edge.is_empty() { return Err(ExportTrieError::InvalidEdge); }
                let child = usize::try_from(uleb(data, &mut cursor, data.len())?)
                    .map_err(|_| ExportTrieError::ChildOutOfRange)?;
                let mut name = String::with_capacity(prefix.len() + edge.len());
                name.push_str(prefix);
                name.push_str(&edge);
                walk(data, child, &name, active, output, depth + 1)?;
            }
            Ok(())
        })();
        active.remove(&offset);
        result
    }

    if data.is_empty() { return Ok(Vec::new()); }
    let mut output = Vec::new();
    let mut active = std::collections::HashSet::new();
    walk(data, 0, "", &mut active, &mut output, 0)?;
    Ok(output)
}


/// Loaded image registered with the experimental dyld symbol table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedDylib {
    pub install_name: String,
    exports: std::collections::HashMap<String, RegistryExport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RegistryExport {
    Address { address: GuestAddr, weak_definition: bool },
    Reexport { library_ordinal: u64, imported_name: String },
}

/// Per-image dependency ordinals and globally loaded dylib exports.
/// Ordinals are interpreted in the importing image's load-command order.
#[derive(Clone, Debug, Default)]
pub struct DylibRegistry {
    images: Vec<LoadedDylib>,
    dependencies: std::collections::HashMap<usize, Vec<Option<usize>>>,
    main_executable: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DylibRegistryError {
    InvalidImageId,
    InvalidDependencyId,
    MissingRequiredDependency(String),
    DuplicateInstallName,
    InvalidOrdinal(i16),
    InvalidReexportOrdinal(u64),
    ReexportCycle,
    ReexportDepthExceeded,
    InvalidExportTrie,
    UnsupportedExportKind,
    AddressOverflow,
}

impl DylibRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register_image(&mut self, install_name: impl Into<String>, is_main: bool) -> Result<usize, DylibRegistryError> {
        let install_name = install_name.into();
        if self.images.iter().any(|image| image.install_name == install_name) {
            return Err(DylibRegistryError::DuplicateInstallName);
        }
        let id = self.images.len();
        self.images.push(LoadedDylib { install_name, exports: std::collections::HashMap::new() });
        if is_main { self.main_executable = Some(id); }
        Ok(id)
    }

    pub fn add_dependency(&mut self, importer: usize, dependency: usize) -> Result<(), DylibRegistryError> {
        if importer >= self.images.len() { return Err(DylibRegistryError::InvalidImageId); }
        if dependency >= self.images.len() { return Err(DylibRegistryError::InvalidDependencyId); }
        self.dependencies.entry(importer).or_default().push(Some(dependency));
        Ok(())
    }

    /// Link the importing image's ordered LC_LOAD_* commands against images
    /// already registered in this registry. Weak missing dependencies retain
    /// their ordinal slot; missing required dependencies fail atomically.
    pub fn link_image_dependencies(
        &mut self,
        importer: usize,
        dependencies: &[super::mach_o::DylibDependency],
    ) -> Result<(), DylibRegistryError> {
        if importer >= self.images.len() { return Err(DylibRegistryError::InvalidImageId); }
        let mut linked = Vec::with_capacity(dependencies.len());
        for dependency in dependencies {
            let found = self.images.iter().position(|image| image.install_name == dependency.install_name);
            match (found, dependency.kind) {
                (Some(id), _) => linked.push(Some(id)),
                (None, super::mach_o::DylibDependencyKind::Weak) => linked.push(None),
                (None, _) => return Err(DylibRegistryError::MissingRequiredDependency(dependency.install_name.clone())),
            }
        }
        self.dependencies.insert(importer, linked);
        Ok(())
    }

    pub fn add_export(&mut self, image: usize, name: impl Into<String>, address: GuestAddr) -> Result<(), DylibRegistryError> {
        let image = self.images.get_mut(image).ok_or(DylibRegistryError::InvalidImageId)?;
        image.exports.insert(name.into(), RegistryExport::Address { address, weak_definition: false });
        Ok(())
    }

    pub fn add_weak_export(
        &mut self,
        image: usize,
        name: impl Into<String>,
        address: GuestAddr,
    ) -> Result<(), DylibRegistryError> {
        let image = self.images.get_mut(image).ok_or(DylibRegistryError::InvalidImageId)?;
        image.exports.insert(name.into(), RegistryExport::Address { address, weak_definition: true });
        Ok(())
    }

    pub fn add_reexport(
        &mut self,
        image: usize,
        name: impl Into<String>,
        library_ordinal: u64,
        imported_name: impl Into<String>,
    ) -> Result<(), DylibRegistryError> {
        let image = self.images.get_mut(image).ok_or(DylibRegistryError::InvalidImageId)?;
        image.exports.insert(name.into(), RegistryExport::Reexport {
            library_ordinal,
            imported_name: imported_name.into(),
        });
        Ok(())
    }

    pub fn image(&self, id: usize) -> Option<&LoadedDylib> { self.images.get(id) }

    fn resolve_in(
        &self,
        image_id: usize,
        name: &str,
        active: &mut std::collections::HashSet<(usize, String)>,
        depth: usize,
    ) -> Result<Option<GuestAddr>, DylibRegistryError> {
        if depth > 128 { return Err(DylibRegistryError::ReexportDepthExceeded); }
        let key = (image_id, name.to_owned());
        if !active.insert(key.clone()) { return Err(DylibRegistryError::ReexportCycle); }
        let result = match self.images.get(image_id).and_then(|image| image.exports.get(name)) {
            None => Ok(None),
            Some(RegistryExport::Address { address, .. }) => Ok(Some(*address)),
            Some(RegistryExport::Reexport { library_ordinal, imported_name }) => {
                let dependencies = self.dependencies.get(&image_id).ok_or(
                    DylibRegistryError::InvalidReexportOrdinal(*library_ordinal)
                )?;
                let dependency_id = library_ordinal.checked_sub(1)
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| dependencies.get(index).copied().flatten())
                    .ok_or(DylibRegistryError::InvalidReexportOrdinal(*library_ordinal))?;
                let forwarded_name = if imported_name.is_empty() { name } else { imported_name.as_str() };
                self.resolve_in(dependency_id, forwarded_name, active, depth + 1)
            }
        };
        active.remove(&key);
        result
    }

    fn find_in(&self, image: usize, name: &str) -> Result<Option<GuestAddr>, DylibRegistryError> {
        self.resolve_in(image, name, &mut std::collections::HashSet::new(), 0)
    }

    /// Resolve an import using its signed Mach-O library ordinal.
    /// Ordinal 0 means self; -1 main executable; -2 flat lookup; -3 weak lookup.
    /// Positive ordinals are one-based indexes in the importer's dependency list.
    pub fn resolve(&self, importer: usize, import: &ChainedImport) -> Result<Option<GuestAddr>, DylibRegistryError> {
        if importer >= self.images.len() { return Err(DylibRegistryError::InvalidImageId); }
        let ordinal = import.library_ordinal;
        match ordinal {
            0 => self.find_in(importer, &import.symbol),
            -1 => match self.main_executable {
                Some(id) => self.find_in(id, &import.symbol),
                None => Ok(None),
            },
            -2 => {
                for id in 0..self.images.len() {
                    if let Some(address) = self.find_in(id, &import.symbol)? { return Ok(Some(address)); }
                }
                Ok(None)
            }
            -3 => {
                for image in &self.images {
                    if let Some(RegistryExport::Address { address, weak_definition: true }) = image.exports.get(&import.symbol) {
                        return Ok(Some(*address));
                    }
                }
                Ok(None)
            }
            value if value > 0 => {
                let dependencies = self.dependencies.get(&importer).ok_or(DylibRegistryError::InvalidOrdinal(value))?;
                let dependency = dependencies.get((value - 1) as usize)
                    .ok_or(DylibRegistryError::InvalidOrdinal(value))?;
                match dependency {
                    Some(id) => self.find_in(*id, &import.symbol),
                    None => Ok(None),
                }
            }
            value => Err(DylibRegistryError::InvalidOrdinal(value)),
        }
    }

    /// Register an already mapped Mach-O image and import its export trie
    /// into the shared symbol table. Dependency linking is a separate step so
    /// callers can register an entire dependency graph before resolving binds.
    pub fn register_macho_image(
        &mut self,
        install_name: impl Into<String>,
        image: &super::mach_o::MachOImage,
        is_main: bool,
    ) -> Result<usize, DylibRegistryError> {
        let exports = image.exported_symbols()
            .map_err(|_| DylibRegistryError::InvalidExportTrie)?;
        // Validate and materialize the complete export table before mutating
        // the registry, so malformed exports cannot leave a half-registered image.
        let mut staged = Vec::with_capacity(exports.len());
        for export in exports {
            let entry = match export.kind {
                ExportKind::Regular { address } => {
                    let resolved = match export.flags & 0x03 {
                        0 => image.text_base.checked_add(address)
                            .ok_or(DylibRegistryError::AddressOverflow)?,
                        2 => address,
                        _ => return Err(DylibRegistryError::UnsupportedExportKind),
                    };
                    let weak = export.flags & 0x04 != 0;
                    RegistryExport::Address { address: resolved, weak_definition: weak }
                }
                ExportKind::StubAndResolver { stub, .. } => {
                    let address = image.text_base.checked_add(stub)
                        .ok_or(DylibRegistryError::AddressOverflow)?;
                    RegistryExport::Address { address, weak_definition: export.flags & 0x04 != 0 }
                }
                ExportKind::Reexport { library_ordinal, imported_name } => {
                    RegistryExport::Reexport { library_ordinal, imported_name }
                }
            };
            staged.push((export.name, entry));
        }
        let id = self.register_image(install_name, is_main)?;
        self.images[id].exports.extend(staged);
        Ok(id)
    }

    pub fn image_count(&self) -> usize { self.images.len() }
}

pub const A64_SVC_BASE: u32 = 0xD400_0001;
pub const A64_RET_X30: u32 = 0xD65F_03C0;
pub const A64_BRK_BASE: u32 = 0xD420_0000;

pub fn encode_svc(immediate: u16) -> u32 {
    A64_SVC_BASE | ((immediate as u32) << 5)
}

pub fn encode_brk(immediate: u16) -> u32 {
    A64_BRK_BASE | ((immediate as u32) << 5)
}

/// Write a two-instruction host-call stub: SVC #immediate; RET.
pub fn write_host_call_stub(
    memory: &mut GuestMemory,
    address: GuestAddr,
    immediate: u16,
) -> Result<(), MemoryError> {
    let instructions = [encode_svc(immediate).to_le_bytes(), A64_RET_X30.to_le_bytes()].concat();
    memory.write(address, &instructions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chained_import_with_signed_ordinal_and_name() {
        let mut payload=vec![0u8;38];
        payload[8..12].copy_from_slice(&28u32.to_le_bytes());
        payload[12..16].copy_from_slice(&32u32.to_le_bytes());
        payload[16..20].copy_from_slice(&1u32.to_le_bytes());
        payload[20..24].copy_from_slice(&1u32.to_le_bytes());
        payload[28..32].copy_from_slice(&(0xFFu32 | (1u32<<9)).to_le_bytes());
        payload[32..].copy_from_slice(b"_test\0");
        let imports=parse_chained_imports(&payload).unwrap();
        assert_eq!(imports,vec![ChainedImport{library_ordinal:-1,weak:false,symbol:"_test".into(),addend:0}]);
    }

    #[test]
    fn parses_import_addend_formats() {
        let mut addend32=vec![0u8;42];
        addend32[8..12].copy_from_slice(&28u32.to_le_bytes());
        addend32[12..16].copy_from_slice(&36u32.to_le_bytes());
        addend32[16..20].copy_from_slice(&1u32.to_le_bytes());
        addend32[20..24].copy_from_slice(&2u32.to_le_bytes());
        addend32[28..32].copy_from_slice(&(2u32 | (1u32<<8)).to_le_bytes());
        addend32[32..36].copy_from_slice(&(-7i32).to_le_bytes());
        addend32[36..].copy_from_slice(b"_foo\0");
        let got=parse_chained_imports(&addend32).unwrap();
        assert_eq!(got[0],ChainedImport{library_ordinal:2,weak:true,symbol:"_foo".into(),addend:-7});

        let mut addend64=vec![0u8;45];
        addend64[8..12].copy_from_slice(&28u32.to_le_bytes());
        addend64[12..16].copy_from_slice(&44u32.to_le_bytes());
        addend64[16..20].copy_from_slice(&1u32.to_le_bytes());
        addend64[20..24].copy_from_slice(&3u32.to_le_bytes());
        let import_word=3u64 | (1u64<<16);
        addend64[28..36].copy_from_slice(&import_word.to_le_bytes());
        addend64[36..44].copy_from_slice(&(-0x1234i64).to_le_bytes());
        addend64[44]=0;
        // A zero-length name is valid at the end of the symbol pool.
        let got=parse_chained_imports(&addend64).unwrap();
        assert_eq!(got[0],ChainedImport{library_ordinal:3,weak:true,symbol:String::new(),addend:-0x1234});
    }

    #[test]
    fn rejects_compressed_symbol_pool_without_decompressor() {
        let mut payload=vec![0u8;28];
        payload[24..28].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse_chained_imports(&payload),Err(ChainedImportError::UnsupportedSymbolsCompression));
    }

    #[test]
    fn svc_and_brk_encodings_match_a64_instruction_layout() {
        assert_eq!(encode_svc(0), 0xD400_0001);
        assert_eq!(encode_svc(0x1234), 0xD402_4681);
        assert_eq!(encode_brk(0), 0xD420_0000);
    }

    #[test]
    fn writes_svc_then_ret() {
        let mut memory = GuestMemory::new();
        memory.map_zeroed(0x1_0000_1000, 8).unwrap();
        write_host_call_stub(&mut memory, 0x1_0000_1000, 7).unwrap();
        let mut bytes = [0u8; 8];
        memory.read(0x1_0000_1000, &mut bytes).unwrap();
        assert_eq!(&bytes[..4], &encode_svc(7).to_le_bytes());
        assert_eq!(&bytes[4..], &A64_RET_X30.to_le_bytes());
    }

    #[test]
    fn parses_regular_export_trie_symbol() {
        // Root edge "_foo" points to a terminal node exporting address 0x1234.
        let trie = [0, 1, b'_', b'f', b'o', b'o', 0, 8, 2, 0, 0xB4, 0x24, 0];
        let exports = parse_export_trie(&trie).unwrap();
        assert_eq!(exports, vec![ExportedSymbol {
            name: "_foo".into(), flags: 0, kind: ExportKind::Regular { address: 0x1234 }
        }]);
    }

    #[test]
    fn parses_reexport_trie_symbol() {
        // Terminal payload: flags(REEXPORT), ordinal 2, imported name "_bar".
        let trie = [8, 0x08, 2, b'_', b'b', b'a', b'r', 0, 0];
        let exports = parse_export_trie(&trie).unwrap();
        assert_eq!(exports[0].kind, ExportKind::Reexport {
            library_ordinal: 2, imported_name: "_bar".into()
        });
    }

    #[test]
    fn rejects_export_trie_cycles_and_bad_child_offsets() {
        let cycle = [0, 1, b'a', 0, 0];
        assert_eq!(parse_export_trie(&cycle), Err(ExportTrieError::Cycle));
        let bad_child = [0, 1, b'a', 0, 0x7f];
        assert_eq!(parse_export_trie(&bad_child), Err(ExportTrieError::ChildOutOfRange));
    }

    #[test]
    fn rejects_truncated_export_terminal() {
        let trie = [2, 0, 1];
        assert_eq!(parse_export_trie(&trie), Err(ExportTrieError::InvalidTerminal));
    }

    #[test]
    fn dylib_registry_resolves_self_and_dependency_ordinals() {
        let mut registry = DylibRegistry::new();
        let main = registry.register_image("/App", true).unwrap();
        let framework = registry.register_image("/System/Library/F.framework/F", false).unwrap();
        let helper = registry.register_image("/usr/lib/libHelper.dylib", false).unwrap();
        registry.add_dependency(main, framework).unwrap();
        registry.add_dependency(main, helper).unwrap();
        registry.add_export(main, "_entry", 0x1000).unwrap();
        registry.add_export(framework, "_draw", 0x2000).unwrap();
        registry.add_export(helper, "_util", 0x3000).unwrap();

        let self_import = ChainedImport { library_ordinal: 0, weak: false, symbol: "_entry".into(), addend: 0 };
        let framework_import = ChainedImport { library_ordinal: 1, weak: false, symbol: "_draw".into(), addend: 0 };
        let helper_import = ChainedImport { library_ordinal: 2, weak: false, symbol: "_util".into(), addend: 0 };
        assert_eq!(registry.resolve(main, &self_import).unwrap(), Some(0x1000));
        assert_eq!(registry.resolve(main, &framework_import).unwrap(), Some(0x2000));
        assert_eq!(registry.resolve(main, &helper_import).unwrap(), Some(0x3000));
    }

    #[test]
    fn dylib_registry_supports_main_and_flat_lookup_ordinals() {
        let mut registry = DylibRegistry::new();
        let main = registry.register_image("/App", true).unwrap();
        let library = registry.register_image("/usr/lib/libSystem.B.dylib", false).unwrap();
        registry.add_dependency(main, library).unwrap();
        registry.add_export(main, "_main_symbol", 0x1110).unwrap();
        registry.add_export(library, "_printf", 0x2220).unwrap();
        let main_import = ChainedImport { library_ordinal: -1, weak: false, symbol: "_main_symbol".into(), addend: 0 };
        let flat_import = ChainedImport { library_ordinal: -2, weak: false, symbol: "_printf".into(), addend: 0 };
        assert_eq!(registry.resolve(library, &main_import).unwrap(), Some(0x1110));
        assert_eq!(registry.resolve(main, &flat_import).unwrap(), Some(0x2220));
    }

    #[test]
    fn dylib_registry_rejects_invalid_dependency_ordinals() {
        let mut registry = DylibRegistry::new();
        let app = registry.register_image("/App", true).unwrap();
        let missing = ChainedImport { library_ordinal: 1, weak: false, symbol: "_missing".into(), addend: 0 };
        assert_eq!(registry.resolve(app, &missing), Err(DylibRegistryError::InvalidOrdinal(1)));
    }

    #[test]
    fn dylib_registry_resolves_reexports_through_dependency_ordinals() {
        let mut registry = DylibRegistry::new();
        let app = registry.register_image("/App", true).unwrap();
        let umbrella = registry.register_image("/Framework/Umbrella", false).unwrap();
        let implementation = registry.register_image("/Framework/Implementation", false).unwrap();
        registry.add_dependency(app, umbrella).unwrap();
        registry.add_dependency(umbrella, implementation).unwrap();
        registry.add_reexport(umbrella, "_public_api", 1, "_implementation").unwrap();
        registry.add_export(implementation, "_implementation", 0x1234_5678).unwrap();
        let import = ChainedImport { library_ordinal: 1, weak: false, symbol: "_public_api".into(), addend: 0 };
        assert_eq!(registry.resolve(app, &import).unwrap(), Some(0x1234_5678));
    }

    #[test]
    fn dylib_registry_detects_reexport_cycles() {
        let mut registry = DylibRegistry::new();
        let first = registry.register_image("/A", true).unwrap();
        let second = registry.register_image("/B", false).unwrap();
        registry.add_dependency(first, second).unwrap();
        registry.add_dependency(second, first).unwrap();
        registry.add_reexport(first, "_cycle", 1, "_cycle").unwrap();
        registry.add_reexport(second, "_cycle", 1, "_cycle").unwrap();
        let import = ChainedImport { library_ordinal: 0, weak: false, symbol: "_cycle".into(), addend: 0 };
        assert_eq!(registry.resolve(first, &import), Err(DylibRegistryError::ReexportCycle));
    }

    #[test]
    fn links_dependencies_in_load_command_order_and_keeps_weak_slots() {
        use super::super::mach_o::{DylibDependency, DylibDependencyKind};
        let mut registry = DylibRegistry::new();
        let app = registry.register_image("/App", true).unwrap();
        let second = registry.register_image("/lib/Second.dylib", false).unwrap();
        registry.add_export(second, "_second", 0x2222).unwrap();
        let deps = vec![
            DylibDependency { install_name: "/lib/Optional.dylib".into(), kind: DylibDependencyKind::Weak },
            DylibDependency { install_name: "/lib/Second.dylib".into(), kind: DylibDependencyKind::Required },
        ];
        registry.link_image_dependencies(app, &deps).unwrap();
        let absent = ChainedImport { library_ordinal: 1, weak: true, symbol: "_optional".into(), addend: 0 };
        let present = ChainedImport { library_ordinal: 2, weak: false, symbol: "_second".into(), addend: 0 };
        assert_eq!(registry.resolve(app, &absent).unwrap(), None);
        assert_eq!(registry.resolve(app, &present).unwrap(), Some(0x2222));
    }

    #[test]
    fn required_dependency_linking_fails_without_replacing_existing_order() {
        use super::super::mach_o::{DylibDependency, DylibDependencyKind};
        let mut registry = DylibRegistry::new();
        let app = registry.register_image("/App", true).unwrap();
        let lib = registry.register_image("/lib/Present.dylib", false).unwrap();
        registry.add_dependency(app, lib).unwrap();
        let missing = [DylibDependency { install_name: "/lib/Missing.dylib".into(), kind: DylibDependencyKind::Required }];
        assert_eq!(
            registry.link_image_dependencies(app, &missing),
            Err(DylibRegistryError::MissingRequiredDependency("/lib/Missing.dylib".into()))
        );
        let old = ChainedImport { library_ordinal: 1, weak: false, symbol: "_x".into(), addend: 0 };
        assert_eq!(registry.resolve(app, &old).unwrap(), None);
    }

    #[test]
    fn weak_lookup_only_returns_weak_definitions() {
        let mut registry = DylibRegistry::new();
        let app = registry.register_image("/App", true).unwrap();
        let strong = registry.register_image("/Strong.dylib", false).unwrap();
        let weak = registry.register_image("/Weak.dylib", false).unwrap();
        registry.add_export(strong, "_symbol", 0x1000).unwrap();
        registry.add_weak_export(weak, "_symbol", 0x2000).unwrap();
        let weak_lookup = ChainedImport { library_ordinal: -3, weak: false, symbol: "_symbol".into(), addend: 0 };
        assert_eq!(registry.resolve(app, &weak_lookup).unwrap(), Some(0x2000));
    }

    #[test]
    fn registers_exports_from_a_macho_export_trie() {
        let image = super::super::mach_o::MachOImage {
            segments: vec![],
            entry_point: 0,
            dynamic_libraries: vec![],
            dependencies: vec![],
            chained_fixups: None,
            // Root edge "_foo" -> terminal exporting text-relative address 0x1234.
            exports_trie: Some(vec![0, 1, b'_', b'f', b'o', b'o', 0, 8, 2, 0, 0xB4, 0x24, 0]),
            text_base: 0x1_0000_0000,
            end_address: 0,
        };
        let mut registry = DylibRegistry::new();
        let id = registry.register_macho_image("/App", &image, true).unwrap();
        let import = ChainedImport { library_ordinal: 0, weak: false, symbol: "_foo".into(), addend: 0 };
        assert_eq!(registry.resolve(id, &import).unwrap(), Some(0x1_0000_1234));
    }

}

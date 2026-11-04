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
    pub exports: std::collections::HashMap<String, GuestAddr>,
}

/// Per-image dependency ordinals and globally loaded dylib exports.
/// Ordinals are interpreted in the importing image's load-command order.
#[derive(Clone, Debug, Default)]
pub struct DylibRegistry {
    images: Vec<LoadedDylib>,
    dependencies: std::collections::HashMap<usize, Vec<usize>>,
    main_executable: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DylibRegistryError {
    InvalidImageId,
    InvalidDependencyId,
    DuplicateInstallName,
    InvalidOrdinal(i16),
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
        self.dependencies.entry(importer).or_default().push(dependency);
        Ok(())
    }

    pub fn add_export(&mut self, image: usize, name: impl Into<String>, address: GuestAddr) -> Result<(), DylibRegistryError> {
        let image = self.images.get_mut(image).ok_or(DylibRegistryError::InvalidImageId)?;
        image.exports.insert(name.into(), address);
        Ok(())
    }

    pub fn image(&self, id: usize) -> Option<&LoadedDylib> { self.images.get(id) }

    fn find_in(&self, image: usize, name: &str) -> Option<GuestAddr> {
        self.images.get(image)?.exports.get(name).copied()
    }

    /// Resolve an import using its signed Mach-O library ordinal.
    /// Ordinal 0 means self; -1 main executable; -2 flat lookup; -3 weak lookup.
    /// Positive ordinals are one-based indexes in the importer's dependency list.
    pub fn resolve(&self, importer: usize, import: &ChainedImport) -> Result<Option<GuestAddr>, DylibRegistryError> {
        if importer >= self.images.len() { return Err(DylibRegistryError::InvalidImageId); }
        let ordinal = import.library_ordinal;
        match ordinal {
            0 => Ok(self.find_in(importer, &import.symbol)),
            -1 => Ok(self.main_executable.and_then(|id| self.find_in(id, &import.symbol))),
            -2 | -3 => {
                for (id, image) in self.images.iter().enumerate() {
                    if let Some(address) = image.exports.get(&import.symbol) { return Ok(Some(*address)); }
                }
                Ok(None)
            }
            value if value > 0 => {
                let dependencies = self.dependencies.get(&importer).ok_or(DylibRegistryError::InvalidOrdinal(value))?;
                let dependency = dependencies.get((value - 1) as usize)
                    .ok_or(DylibRegistryError::InvalidOrdinal(value))?;
                Ok(self.find_in(*dependency, &import.symbol))
            }
            value => Err(DylibRegistryError::InvalidOrdinal(value)),
        }
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

}

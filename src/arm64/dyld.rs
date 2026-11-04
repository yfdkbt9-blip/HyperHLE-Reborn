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
}

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

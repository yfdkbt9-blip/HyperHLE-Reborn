/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Initial AAPCS64 call-frame helpers. This is intentionally not wired into
//! the A32 framework dispatcher yet; that dispatcher uses 32-bit guest
//! pointers and must remain unchanged for existing games.

use super::cpu::CpuContext;
use super::memory::{GuestAddr, GuestMemory, MemoryError};

pub const INTEGER_ARGUMENT_REGISTERS: usize = 8;
pub const FLOAT_ARGUMENT_REGISTERS: usize = 8;
pub const FRAME_POINTER_REGISTER: usize = 29;
pub const LINK_REGISTER: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestFunction(pub GuestAddr);

impl GuestFunction {
    pub const fn address(self) -> GuestAddr { self.0 }
    pub const fn is_null(self) -> bool { self.0 == 0 }
}

/// Cursor over the integer argument area of one AAPCS64 call.
pub struct IntegerArguments<'a> {
    context: &'a CpuContext,
    memory: &'a GuestMemory,
    register_index: usize,
    stack_offset: u64,
}

impl<'a> IntegerArguments<'a> {
    pub fn new(context: &'a CpuContext, memory: &'a GuestMemory) -> Self {
        Self { context, memory, register_index: 0, stack_offset: 0 }
    }

    /// Read the next 64-bit integer/pointer argument. X0-X7 are used first;
    /// subsequent arguments are read from the caller-provided stack area.
    pub fn next_u64(&mut self) -> Result<u64, MemoryError> {
        if self.register_index < INTEGER_ARGUMENT_REGISTERS {
            let value = self.context.x[self.register_index];
            self.register_index += 1;
            return Ok(value);
        }
        let address = self.context.sp.checked_add(self.stack_offset)
            .ok_or(MemoryError::AddressOverflow)?;
        let mut bytes = [0u8; 8];
        self.memory.read(address, &mut bytes)?;
        self.stack_offset = self.stack_offset.checked_add(8)
            .ok_or(MemoryError::AddressOverflow)?;
        Ok(u64::from_le_bytes(bytes))
    }
}

pub struct FloatingArguments<'a> {
    context: &'a CpuContext,
    register_index: usize,
}

impl<'a> FloatingArguments<'a> {
    pub fn new(context: &'a CpuContext) -> Self {
        Self { context, register_index: 0 }
    }

    /// Read the next scalar floating-point argument from V0-V7. The low
    /// 64-bit lane is used for both single- and double-precision values.
    pub fn next_f64_bits(&mut self) -> Option<u64> {
        if self.register_index >= FLOAT_ARGUMENT_REGISTERS { return None; }
        let value = self.context.v[self.register_index][0];
        self.register_index += 1;
        Some(value)
    }

    pub fn next_f32(&mut self) -> Option<f32> {
        self.next_f64_bits().map(|bits| f32::from_bits(bits as u32))
    }

    pub fn next_f64(&mut self) -> Option<f64> {
        self.next_f64_bits().map(f64::from_bits)
    }
}

/// AAPCS64 integer return values use X0 and, for a two-word result, X1.
pub fn set_integer_return(context: &mut CpuContext, low: u64, high: Option<u64>) {
    context.x[0] = low;
    if let Some(high) = high { context.x[1] = high; }
}

/// AAPCS64 scalar floating-point return values use V0.
pub fn set_float_return(context: &mut CpuContext, bits: u64) {
    context.v[0][0] = bits;
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::memory::PAGE_SIZE;

    #[test]
    fn integer_arguments_use_x0_to_x7_then_stack() {
        let mut memory = GuestMemory::new();
        memory.map_zeroed(0x1_0000_0000, PAGE_SIZE as u64).unwrap();
        memory.write(0x1_0000_0000, &0xfeed_face_cafe_beefu64.to_le_bytes()).unwrap();
        let mut context = CpuContext::default();
        context.x = [10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        context.sp = 0x1_0000_0000;
        let mut args = IntegerArguments::new(&context, &memory);
        for expected in 10..=17 { assert_eq!(args.next_u64().unwrap(), expected); }
        assert_eq!(args.next_u64().unwrap(), 0xfeed_face_cafe_beef);
    }

    #[test]
    fn floating_arguments_use_independent_v_register_bank() {
        let mut context = CpuContext::default();
        context.x[0] = 123;
        context.v[0][0] = 1.5f64.to_bits();
        context.v[1][0] = 2.5f64.to_bits();
        let mut args = FloatingArguments::new(&context);
        assert_eq!(args.next_f64(), Some(1.5));
        assert_eq!(args.next_f64(), Some(2.5));
    }
}

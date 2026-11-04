/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! AArch64 process-entry stack construction for an LC_MAIN-style entry point.

use super::cpu::CpuContext;
use super::memory::{GuestAddr, GuestMemory, MemoryError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitialStack {
    pub sp: GuestAddr,
    pub argc: u64,
    pub argv: GuestAddr,
    pub envp: GuestAddr,
    pub apple: GuestAddr,
}

fn push_bytes(
    memory: &mut GuestMemory,
    cursor: &mut GuestAddr,
    bytes: &[u8],
) -> Result<GuestAddr, MemoryError> {
    let length = u64::try_from(bytes.len()).map_err(|_| MemoryError::AddressOverflow)?;
    *cursor = cursor.checked_sub(length).ok_or(MemoryError::AddressOverflow)?;
    memory.write(*cursor, bytes)?;
    Ok(*cursor)
}

fn push_string(
    memory: &mut GuestMemory,
    cursor: &mut GuestAddr,
    value: &str,
) -> Result<GuestAddr, MemoryError> {
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    push_bytes(memory, cursor, &bytes)
}

fn push_pointer_array(
    memory: &mut GuestMemory,
    cursor: &mut GuestAddr,
    pointers: &[GuestAddr],
) -> Result<GuestAddr, MemoryError> {
    let bytes_len = (pointers.len() as u64 + 1)
        .checked_mul(8).ok_or(MemoryError::AddressOverflow)?;
    *cursor = cursor.checked_sub(bytes_len).ok_or(MemoryError::AddressOverflow)?;
    let start = *cursor;
    for (index, pointer) in pointers.iter().enumerate() {
        memory.write(start + index as u64 * 8, &pointer.to_le_bytes())?;
    }
    memory.write(start + pointers.len() as u64 * 8, &0u64.to_le_bytes())?;
    Ok(start)
}

/// Map a caller-selected stack reservation and place strings/pointer arrays
/// into it. The returned SP is 16-byte aligned as required by AAPCS64.
pub fn build_initial_stack(
    memory: &mut GuestMemory,
    stack_base: GuestAddr,
    stack_size: u64,
    argv: &[&str],
    envp: &[&str],
    apple: &[&str],
) -> Result<InitialStack, MemoryError> {
    let stack_top = stack_base.checked_add(stack_size).ok_or(MemoryError::AddressOverflow)?;
    memory.map_zeroed(stack_base, stack_size)?;
    let mut cursor = stack_top;

    let mut argv_ptrs = Vec::with_capacity(argv.len());
    for value in argv.iter().rev() {
        argv_ptrs.push(push_string(memory, &mut cursor, value)?);
    }
    argv_ptrs.reverse();

    let mut envp_ptrs = Vec::with_capacity(envp.len());
    for value in envp.iter().rev() {
        envp_ptrs.push(push_string(memory, &mut cursor, value)?);
    }
    envp_ptrs.reverse();

    let mut apple_ptrs = Vec::with_capacity(apple.len());
    for value in apple.iter().rev() {
        apple_ptrs.push(push_string(memory, &mut cursor, value)?);
    }
    apple_ptrs.reverse();

    let apple_ptr = push_pointer_array(memory, &mut cursor, &apple_ptrs)?;
    let envp_ptr = push_pointer_array(memory, &mut cursor, &envp_ptrs)?;
    let argv_ptr = push_pointer_array(memory, &mut cursor, &argv_ptrs)?;

    cursor = cursor.checked_sub(8).ok_or(MemoryError::AddressOverflow)?;
    let argc = argv.len() as u64;
    memory.write(cursor, &argc.to_le_bytes())?;

    let sp = cursor & !0xf;
    if sp < stack_base { return Err(MemoryError::AddressOverflow); }
    Ok(InitialStack { sp, argc, argv: argv_ptr, envp: envp_ptr, apple: apple_ptr })
}

pub fn apply_initial_stack(context: &mut CpuContext, stack: InitialStack) {
    context.sp = stack.sp;
    context.x[0] = stack.argc;
    context.x[1] = stack.argv;
    context.x[2] = stack.envp;
    context.x[3] = stack.apple;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_aligned_stack_and_argument_registers() {
        let mut memory = GuestMemory::new();
        let stack = build_initial_stack(
            &mut memory,
            0x1_0000_0000,
            0x4000,
            &["jester", "--safe"],
            &["LANG=en"],
            &["platform=ios"],
        ).unwrap();
        assert_eq!(stack.sp & 0xf, 0);
        let mut context = CpuContext::default();
        apply_initial_stack(&mut context, stack);
        assert_eq!(context.sp, stack.sp);
        assert_eq!(context.x[0], 2);
        assert_eq!(context.x[1], stack.argv);
        assert_eq!(context.x[2], stack.envp);
        assert_eq!(context.x[3], stack.apple);
    }
}

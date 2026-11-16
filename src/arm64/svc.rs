/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Dispatch table for AArch64 SVC-based host-call stubs.
//!
//! This is the bridge point for future iOS framework HLE exports. It does not
//! claim to implement the iOS kernel syscall ABI; registered handlers are
//! emulator-owned services selected by the SVC immediate.

use super::cpu::CpuContext;
use super::memory::GuestMemory;
use std::collections::HashMap;

pub type SvcHandler = Box<
    dyn FnMut(&mut CpuContext, &mut GuestMemory) -> Result<(), String> + 'static
>;

#[derive(Default)]
pub struct SvcDispatcher {
    handlers: HashMap<u16, SvcHandler>,
}

impl SvcDispatcher {
    pub fn new() -> Self { Self::default() }

    pub fn register<F>(&mut self, immediate: u16, handler: F) -> Result<(), &'static str>
    where
        F: FnMut(&mut CpuContext, &mut GuestMemory) -> Result<(), String> + 'static,
    {
        if self.handlers.contains_key(&immediate) {
            return Err("duplicate A64 SVC immediate");
        }
        self.handlers.insert(immediate, Box::new(handler));
        Ok(())
    }

    /// Dispatch one SVC. Returns false when the immediate is not registered.
    pub fn dispatch(
        &mut self,
        immediate: u16,
        context: &mut CpuContext,
        memory: &mut GuestMemory,
    ) -> Result<bool, String> {
        match self.handlers.get_mut(&immediate) {
            Some(handler) => {
                handler(context, memory)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn contains(&self, immediate: u16) -> bool {
        self.handlers.contains_key(&immediate)
    }
}


/// Darwin arm64 BSD syscall trap convention: SVC #0x80, syscall number in X16,
/// arguments in X0-X8, and the NZCV carry bit marks an errno return.
pub const DARWIN_SYSCALL_SVC: u16 = 0x80;
pub const DARWIN_CARRY_FLAG: u32 = 1 << 29;
pub type DarwinSyscallHandler = Box<
    dyn FnMut([u64; 9], &mut GuestMemory) -> Result<u64, u16> + 'static
>;

#[derive(Default)]
pub struct DarwinSyscallDispatcher {
    handlers: HashMap<u64, DarwinSyscallHandler>,
}

impl DarwinSyscallDispatcher {
    pub fn new() -> Self { Self::default() }

    pub fn register<F>(&mut self, number: u64, handler: F) -> Result<(), &'static str>
    where F: FnMut([u64; 9], &mut GuestMemory) -> Result<u64, u16> + 'static,
    {
        if self.handlers.contains_key(&number) { return Err("duplicate Darwin syscall number"); }
        self.handlers.insert(number, Box::new(handler));
        Ok(())
    }

    /// Returns false for non-Darwin SVCs or unimplemented syscall numbers.
    /// Errno values are returned positively in X0 with NZCV.C set.
    pub fn dispatch(&mut self, immediate: u16, context: &mut CpuContext, memory: &mut GuestMemory) -> Result<bool, String> {
        if immediate != DARWIN_SYSCALL_SVC { return Ok(false); }
        let number = context.x[16];
        let handler = match self.handlers.get_mut(&number) {
            Some(handler) => handler,
            None => return Ok(false),
        };
        let mut args = [0u64; 9];
        args.copy_from_slice(&context.x[..9]);
        match handler(args, memory) {
            Ok(value) => {
                context.x[0] = value;
                context.pstate &= !DARWIN_CARRY_FLAG;
            }
            Err(errno) => {
                context.x[0] = u64::from(errno);
                context.pstate |= DARWIN_CARRY_FLAG;
            }
        }
        Ok(true)
    }

    pub fn contains(&self, number: u64) -> bool { self.handlers.contains_key(&number) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatches_registered_svc_and_updates_registers() {
        let mut dispatcher = SvcDispatcher::new();
        dispatcher.register(17, |context, _| {
            context.x[0] = 42;
            Ok(())
        }).unwrap();
        let mut context = CpuContext::default();
        let mut memory = GuestMemory::new();
        assert!(dispatcher.dispatch(17, &mut context, &mut memory).unwrap());
        assert_eq!(context.x[0], 42);
    }

    #[test]
    fn rejects_duplicate_svc_immediates() {
        let mut dispatcher = SvcDispatcher::new();
        dispatcher.register(1, |_, _| Ok(())).unwrap();
        assert!(dispatcher.register(1, |_, _| Ok(())).is_err());
    }
    #[test]
    fn darwin_syscalls_use_x16_nine_arguments_and_carry_errno() {
        let mut dispatcher = DarwinSyscallDispatcher::new();
        dispatcher.register(0x2000004, |args, _| {
            assert_eq!(args, [1,2,3,4,5,6,7,8,9]);
            Ok(123)
        }).unwrap();
        dispatcher.register(0x2000005, |_args, _| Err(13)).unwrap();
        let mut context = CpuContext::default();
        context.x[..9].copy_from_slice(&[1,2,3,4,5,6,7,8,9]);
        context.x[16] = 0x2000004;
        context.pstate |= DARWIN_CARRY_FLAG;
        let mut memory = GuestMemory::new();
        assert!(dispatcher.dispatch(0x80, &mut context, &mut memory).unwrap());
        assert_eq!(context.x[0], 123);
        assert_eq!(context.pstate & DARWIN_CARRY_FLAG, 0);

        context.x[16] = 0x2000005;
        assert!(dispatcher.dispatch(0x80, &mut context, &mut memory).unwrap());
        assert_eq!(context.x[0], 13);
        assert_ne!(context.pstate & DARWIN_CARRY_FLAG, 0);
        assert!(!dispatcher.dispatch(0, &mut context, &mut memory).unwrap());
    }


}

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
}

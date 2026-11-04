/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Safe Rust-facing shell around the experimental Dynarmic A64 JIT.

use super::memory::{GuestMemory, GuestAddr};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuContext {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub v: [[u64; 2]; 32],
    pub fpcr: u32,
    pub fpsr: u32,
    pub pstate: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum StopReason {
    Normal = 0,
    SupervisorCall = 1,
    Exception = 2,
    MemoryFault = 3,
}

pub struct Cpu {
    raw: *mut std::ffi::c_void,
}

impl Cpu {
    pub fn new(memory: &mut GuestMemory) -> Self {
        let raw = unsafe { touchHLE_A64_new((memory as *mut GuestMemory).cast()) };
        assert!(!raw.is_null(), "Dynarmic A64 failed to create CPU");
        Self { raw }
    }

    pub fn set_context(&mut self, context: &CpuContext) {
        unsafe { touchHLE_A64_set_context(self.raw, context); }
    }

    pub fn get_context(&self) -> CpuContext {
        let mut context = CpuContext::default();
        unsafe { touchHLE_A64_get_context(self.raw, &mut context); }
        context
    }

    /// Run guest instructions up to the supplied cycle budget.
    pub fn run(&mut self, ticks: u64) -> (StopReason, u32, GuestAddr) {
        self.execute(ticks, false)
    }

    /// Execute one guest instruction.
    pub fn step(&mut self) -> (StopReason, u32, GuestAddr) {
        self.execute(1, true)
    }

    fn execute(&mut self, ticks: u64, step: bool) -> (StopReason, u32, GuestAddr) {
        let mut svc = 0;
        let mut exception_pc = 0;
        let raw = unsafe {
            touchHLE_A64_run(self.raw, ticks, step, &mut svc, &mut exception_pc)
        };
        let reason = match raw {
            0 => StopReason::Normal,
            1 => StopReason::SupervisorCall,
            2 => StopReason::Exception,
            3 => StopReason::MemoryFault,
            _ => StopReason::Exception,
        };
        (reason, svc, exception_pc)
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        unsafe { touchHLE_A64_delete(self.raw); }
    }
}

// These callbacks are intentionally narrow: all memory access is routed
// through GuestMemory, and a failed lookup becomes a Dynarmic guest fault.
#[no_mangle]
extern "C" fn touchHLE_a64_read(
    memory: *mut std::ffi::c_void,
    address: GuestAddr,
    output: *mut u8,
    size: usize,
) -> bool {
    if memory.is_null() || (output.is_null() && size != 0) { return false; }
    let memory = unsafe { &*(memory.cast::<GuestMemory>()) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, size) };
    memory.read(address, output).is_ok()
}

#[no_mangle]
extern "C" fn touchHLE_a64_write(
    memory: *mut std::ffi::c_void,
    address: GuestAddr,
    input: *const u8,
    size: usize,
) -> bool {
    if memory.is_null() || (input.is_null() && size != 0) { return false; }
    let memory = unsafe { &mut *(memory.cast::<GuestMemory>()) };
    let input = unsafe { std::slice::from_raw_parts(input, size) };
    memory.write(address, input).is_ok()
}

extern "C" {
    fn touchHLE_A64_new(memory: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn touchHLE_A64_delete(cpu: *mut std::ffi::c_void);
    fn touchHLE_A64_set_context(cpu: *mut std::ffi::c_void, context: *const CpuContext);
    fn touchHLE_A64_get_context(cpu: *mut std::ffi::c_void, context: *mut CpuContext);
    fn touchHLE_A64_run(
        cpu: *mut std::ffi::c_void,
        ticks: u64,
        step: bool,
        svc: *mut u32,
        exception_pc: *mut GuestAddr,
    ) -> u32;
}

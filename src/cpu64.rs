/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! 64-bit (AArch64) guest CPU, a thin safe wrapper around
//! `src/cpu/dynarmic_wrapper/lib64.cpp`.
//!
//! This is the 64-bit counterpart of `cpu.rs` and is independent of it: it does
//! not use the 32-bit `Mem` or `Ptr` types. Guest memory is provided by the
//! caller through the [GuestBus] trait.
#![allow(dead_code)]
#![allow(non_snake_case)]

use std::ffi::c_void;

/// Opaque C++ `Wrapper`.
#[repr(C)]
struct RawCpu {
    _private: [u8; 0],
}

/// Must match `touchHLE64_MemVTable` in lib64.cpp.
#[repr(C)]
struct MemVTable {
    ctx: *mut c_void,
    read: extern "C" fn(*mut c_void, u64, *mut u8, usize) -> bool,
    write: extern "C" fn(*mut c_void, u64, *const u8, usize) -> bool,
}

/// Full register state of one guest thread. Must match `touchHLE64_Context` in
/// lib64.cpp exactly.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Context64 {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub pstate: u32,
    pub fpcr: u32,
    pub fpsr: u32,
    pub pad: u32,
    /// 32 vector registers as (low, high) pairs.
    pub v: [u64; 64],
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
}

impl Default for Context64 {
    fn default() -> Self {
        Context64 {
            x: [0; 31],
            sp: 0,
            pc: 0,
            pstate: 0,
            fpcr: 0,
            fpsr: 0,
            pad: 0,
            v: [0; 64],
            tpidr_el0: 0,
            tpidrro_el0: 0,
        }
    }
}

extern "C" {
    fn touchHLE64_new() -> *mut RawCpu;
    fn touchHLE64_delete(cpu: *mut RawCpu);
    fn touchHLE64_get_reg(cpu: *const RawCpu, index: u32) -> u64;
    fn touchHLE64_set_reg(cpu: *mut RawCpu, index: u32, value: u64);
    fn touchHLE64_get_vec(cpu: *const RawCpu, index: u32, out: *mut u64);
    fn touchHLE64_set_vec(cpu: *mut RawCpu, index: u32, lo: u64, hi: u64);
    fn touchHLE64_pstate(cpu: *const RawCpu) -> u32;
    fn touchHLE64_set_pstate(cpu: *mut RawCpu, value: u32);
    fn touchHLE64_fpcr(cpu: *const RawCpu) -> u32;
    fn touchHLE64_set_fpcr(cpu: *mut RawCpu, value: u32);
    fn touchHLE64_fpsr(cpu: *const RawCpu) -> u32;
    fn touchHLE64_set_fpsr(cpu: *mut RawCpu, value: u32);
    fn touchHLE64_set_tpidr_el0(cpu: *mut RawCpu, value: u64);
    fn touchHLE64_set_tpidrro_el0(cpu: *mut RawCpu, value: u64);
    fn touchHLE64_last_exception_pc(cpu: *const RawCpu) -> u64;
    fn touchHLE64_last_fault_addr(cpu: *const RawCpu) -> u64;
    fn touchHLE64_swap_context(cpu: *mut RawCpu, context: *mut Context64);
    fn touchHLE64_invalidate_cache_range(cpu: *mut RawCpu, start: u64, size: u64);
    fn touchHLE64_run_or_step(
        cpu: *mut RawCpu,
        mem: *const MemVTable,
        ticks: *mut u64,
    ) -> i32;
}

/// Guest memory as seen by the CPU. Return `false` for an unmapped address;
/// the CPU then stops with [StopReason::MemoryAbort].
pub trait GuestBus {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool;
    fn write(&mut self, addr: u64, data: &[u8]) -> bool;
}

extern "C" fn read_trampoline(ctx: *mut c_void, addr: u64, out: *mut u8, size: usize) -> bool {
    // SAFETY: `ctx` is the `&mut &mut dyn GuestBus` set up in `run_or_step`,
    // which outlives the call into C++.
    let bus = unsafe { &mut *(ctx as *mut &mut dyn GuestBus) };
    let buf = unsafe { std::slice::from_raw_parts_mut(out, size) };
    bus.read(addr, buf)
}

extern "C" fn write_trampoline(ctx: *mut c_void, addr: u64, data: *const u8, size: usize) -> bool {
    let bus = unsafe { &mut *(ctx as *mut &mut dyn GuestBus) };
    let buf = unsafe { std::slice::from_raw_parts(data, size) };
    bus.write(addr, buf)
}

/// Why [Cpu64::run_or_step] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ran out of ticks, or finished a single step.
    Finished,
    /// Access to unmapped memory (data access or instruction fetch).
    MemoryAbort { addr: u64 },
    /// Undefined/unallocated instruction at `pc`.
    UndefinedInstruction { pc: u64 },
    /// `brk` instruction at `pc`.
    Breakpoint { pc: u64 },
    /// Guest executed `svc #imm`. This is how HLE functions are called. The PC
    /// already points to the instruction after the `svc`.
    Svc(u32),
}

pub struct Cpu64 {
    raw: *mut RawCpu,
}

// The wrapper owns all its state and is only used through &mut self.
unsafe impl Send for Cpu64 {}

impl Default for Cpu64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Cpu64 {
    fn drop(&mut self) {
        unsafe { touchHLE64_delete(self.raw) }
    }
}

impl Cpu64 {
    pub fn new() -> Cpu64 {
        let raw = unsafe { touchHLE64_new() };
        assert!(!raw.is_null());
        Cpu64 { raw }
    }

    /// X0..X30.
    pub fn reg(&self, index: usize) -> u64 {
        assert!(index < 31);
        unsafe { touchHLE64_get_reg(self.raw, index as u32) }
    }
    pub fn set_reg(&mut self, index: usize, value: u64) {
        assert!(index < 31);
        unsafe { touchHLE64_set_reg(self.raw, index as u32, value) }
    }
    pub fn sp(&self) -> u64 {
        unsafe { touchHLE64_get_reg(self.raw, 31) }
    }
    pub fn set_sp(&mut self, value: u64) {
        unsafe { touchHLE64_set_reg(self.raw, 31, value) }
    }
    pub fn pc(&self) -> u64 {
        unsafe { touchHLE64_get_reg(self.raw, 32) }
    }
    pub fn set_pc(&mut self, value: u64) {
        unsafe { touchHLE64_set_reg(self.raw, 32, value) }
    }
    /// X30.
    pub fn lr(&self) -> u64 {
        self.reg(30)
    }
    pub fn set_lr(&mut self, value: u64) {
        self.set_reg(30, value)
    }

    /// 128-bit vector register V0..V31.
    pub fn vec(&self, index: usize) -> u128 {
        assert!(index < 32);
        let mut out = [0u64; 2];
        unsafe { touchHLE64_get_vec(self.raw, index as u32, out.as_mut_ptr()) };
        (out[0] as u128) | ((out[1] as u128) << 64)
    }
    pub fn set_vec(&mut self, index: usize, value: u128) {
        assert!(index < 32);
        unsafe { touchHLE64_set_vec(self.raw, index as u32, value as u64, (value >> 64) as u64) }
    }
    /// Low 64 bits of Vn viewed as a double (D register).
    pub fn f64(&self, index: usize) -> f64 {
        f64::from_bits(self.vec(index) as u64)
    }
    pub fn set_f64(&mut self, index: usize, value: f64) {
        self.set_vec(index, value.to_bits() as u128)
    }
    /// Low 32 bits of Vn viewed as a float (S register).
    pub fn f32(&self, index: usize) -> f32 {
        f32::from_bits(self.vec(index) as u32)
    }
    pub fn set_f32(&mut self, index: usize, value: f32) {
        self.set_vec(index, value.to_bits() as u128)
    }

    pub fn pstate(&self) -> u32 {
        unsafe { touchHLE64_pstate(self.raw) }
    }
    pub fn set_pstate(&mut self, value: u32) {
        unsafe { touchHLE64_set_pstate(self.raw, value) }
    }
    pub fn fpcr(&self) -> u32 {
        unsafe { touchHLE64_fpcr(self.raw) }
    }
    pub fn set_fpcr(&mut self, value: u32) {
        unsafe { touchHLE64_set_fpcr(self.raw, value) }
    }
    pub fn fpsr(&self) -> u32 {
        unsafe { touchHLE64_fpsr(self.raw) }
    }
    pub fn set_fpsr(&mut self, value: u32) {
        unsafe { touchHLE64_set_fpsr(self.raw, value) }
    }

    /// Thread pointer registers. On iOS, TPIDRRO_EL0 holds the thread's
    /// thread-specific-data array (the 64-bit counterpart of TPIDRURO).
    pub fn set_tpidr_el0(&mut self, value: u64) {
        unsafe { touchHLE64_set_tpidr_el0(self.raw, value) }
    }
    pub fn set_tpidrro_el0(&mut self, value: u64) {
        unsafe { touchHLE64_set_tpidrro_el0(self.raw, value) }
    }

    /// Exchange the whole register state with `context` (thread switch).
    pub fn swap_context(&mut self, context: &mut Context64) {
        unsafe { touchHLE64_swap_context(self.raw, context) }
    }

    /// Must be called after guest code is modified in memory.
    pub fn invalidate_cache_range(&mut self, start: u64, size: u64) {
        unsafe { touchHLE64_invalidate_cache_range(self.raw, start, size) }
    }

    /// Run until `ticks` run out (updated on return) or something happens, or,
    /// if `ticks` is `None`, execute exactly one instruction.
    pub fn run_or_step(&mut self, bus: &mut dyn GuestBus, ticks: Option<&mut u64>) -> StopReason {
        let mut bus_ref: &mut dyn GuestBus = bus;
        let vtable = MemVTable {
            ctx: &mut bus_ref as *mut &mut dyn GuestBus as *mut c_void,
            read: read_trampoline,
            write: write_trampoline,
        };
        let ticks_ptr = match ticks {
            Some(t) => t as *mut u64,
            None => std::ptr::null_mut(),
        };
        let res = unsafe { touchHLE64_run_or_step(self.raw, &vtable, ticks_ptr) };
        match res {
            -1 => StopReason::Finished,
            -2 => StopReason::MemoryAbort {
                addr: unsafe { touchHLE64_last_fault_addr(self.raw) },
            },
            -3 => StopReason::UndefinedInstruction {
                pc: unsafe { touchHLE64_last_exception_pc(self.raw) },
            },
            -4 => StopReason::Breakpoint {
                pc: unsafe { touchHLE64_last_exception_pc(self.raw) },
            },
            n if n >= 0 => StopReason::Svc(n as u32),
            n => panic!("unexpected result {n} from touchHLE64_run_or_step"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlatMem(Vec<u8>);

    impl GuestBus for FlatMem {
        fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
            let a = addr as usize;
            match self.0.get(a..a + out.len()) {
                Some(s) => {
                    out.copy_from_slice(s);
                    true
                }
                None => false,
            }
        }
        fn write(&mut self, addr: u64, data: &[u8]) -> bool {
            let a = addr as usize;
            match self.0.get_mut(a..a + data.len()) {
                Some(s) => {
                    s.copy_from_slice(data);
                    true
                }
                None => false,
            }
        }
    }

    fn put_code(mem: &mut FlatMem, at: usize, code: &[u32]) {
        for (i, insn) in code.iter().enumerate() {
            let o = at + i * 4;
            mem.0[o..o + 4].copy_from_slice(&insn.to_le_bytes());
        }
    }

    /// The smallest possible end-to-end check of the AArch64 CPU: run three real
    /// instructions and an `svc`.
    ///   mov x0, #42
    ///   add x1, x0, #1
    ///   svc #7
    #[test]
    fn runs_tiny_program_and_stops_on_svc() {
        let mut mem = FlatMem(vec![0; 0x10000]);
        put_code(&mut mem, 0x1000, &[0xD280_0540, 0x9100_0401, 0xD400_00E1]);
        let mut cpu = Cpu64::new();
        cpu.set_pc(0x1000);
        let mut ticks = 1000u64;
        let stop = cpu.run_or_step(&mut mem, Some(&mut ticks));
        assert_eq!(stop, StopReason::Svc(7));
        assert_eq!(cpu.reg(0), 42);
        assert_eq!(cpu.reg(1), 43);
        assert_eq!(cpu.pc(), 0x100C);
    }

    /// Loads and stores go through the memory callbacks.
    ///   mov x2, #0x2000
    ///   mov x3, #0x55
    ///   str x3, [x2]
    ///   ldr x4, [x2]
    ///   svc #1
    #[test]
    fn memory_callbacks_work() {
        let mut mem = FlatMem(vec![0; 0x10000]);
        put_code(
            &mut mem,
            0x1000,
            &[0xD284_0002, 0xD280_0AA3, 0xF900_0043, 0xF940_0044, 0xD400_0021],
        );
        let mut cpu = Cpu64::new();
        cpu.set_pc(0x1000);
        let mut ticks = 1000u64;
        assert_eq!(cpu.run_or_step(&mut mem, Some(&mut ticks)), StopReason::Svc(1));
        assert_eq!(cpu.reg(4), 0x55);
        assert_eq!(&mem.0[0x2000..0x2008], &0x55u64.to_le_bytes());
    }

    #[test]
    fn unmapped_fetch_is_a_memory_abort() {
        let mut mem = FlatMem(vec![0; 0x1000]);
        let mut cpu = Cpu64::new();
        cpu.set_pc(0x5000_0000);
        let mut ticks = 1000u64;
        assert!(matches!(
            cpu.run_or_step(&mut mem, Some(&mut ticks)),
            StopReason::MemoryAbort { .. }
        ));
    }

    #[test]
    fn context_swap_exchanges_registers() {
        let mut cpu = Cpu64::new();
        cpu.set_reg(5, 111);
        cpu.set_sp(0x7000);
        let mut other = Context64::default();
        other.x[5] = 222;
        other.sp = 0x9000;
        cpu.swap_context(&mut other);
        assert_eq!((cpu.reg(5), cpu.sp()), (222, 0x9000));
        assert_eq!((other.x[5], other.sp), (111, 0x7000));
    }
}

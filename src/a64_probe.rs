/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Experimental first run of a 64-bit (arm64) app.
//!
//! This is NOT a finished 64-bit port. It is a probe: it loads an arm64
//! executable with [crate::mach_o64], gives it a small flat 64-bit memory, runs
//! it on [crate::cpu64], and answers every imported function with a stub that
//! logs the call. A handful of libc functions (malloc, memcpy, ...) really work
//! so that start-up code can get somewhere. The goal is a log that shows what
//! has to be implemented next, in the order the app needs it.
//!
//! Sequence, like dyld: map the image, apply rebases, point every bind at a
//! stub, run every static initializer, then call `main`.
//!
//! The run is considered a success when the app reaches `UIApplicationMain`.
#![allow(dead_code)]

use crate::bundle::Bundle;
use crate::cpu64::{Cpu64, GuestBus, StopReason};
use crate::mach_o64::{GuestMemory64, MachO64};
use std::collections::{HashMap, HashSet, VecDeque};

// Memory map. The image itself lives at its preferred address (slide 0),
// 0x1_0000_0000 for ordinary iOS executables.
const STACK_BASE: u64 = 0x1_6fc0_0000;
const STACK_SIZE: u64 = 0x40_0000; // 4 MiB
const STACK_TOP: u64 = STACK_BASE + STACK_SIZE;
const HEAP_BASE: u64 = 0x2_0000_0000;
const HEAP_SIZE: u64 = 0x200_0000; // 32 MiB
const STUB_BASE: u64 = 0x1_8000_0000;

/// `svc` immediate used by the return trap. Everything below this is a stub.
const TRAP_SVC: u32 = 0xffff;
/// Instruction budget for one `run` slice, and how many slices a single call
/// into the guest may use before we decide it is stuck in a loop.
const TICKS_PER_SLICE: u64 = 50_000_000;
const MAX_SLICES: u32 = 40;
const MAX_HLE_CALLS: u64 = 3_000_000;

// ---------------------------------------------------------------------------
// Memory

struct Region {
    base: u64,
    data: Vec<u8>,
}

/// Flat 64-bit guest memory made of a few separate regions.
struct Mem64 {
    regions: Vec<Region>,
}

impl Mem64 {
    fn locate(&self, addr: u64, len: usize) -> Option<(usize, usize)> {
        let end = addr.checked_add(len as u64)?;
        for (i, r) in self.regions.iter().enumerate() {
            if addr >= r.base && end <= r.base + r.data.len() as u64 {
                return Some((i, (addr - r.base) as usize));
            }
        }
        None
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<&[u8]> {
        let (i, o) = self.locate(addr, len)?;
        Some(&self.regions[i].data[o..o + len])
    }
    fn write_slice(&mut self, addr: u64, data: &[u8]) -> bool {
        match self.locate(addr, data.len()) {
            Some((i, o)) => {
                self.regions[i].data[o..o + data.len()].copy_from_slice(data);
                true
            }
            None => false,
        }
    }
    /// NUL-terminated bytes (without the NUL), at most `limit` of them.
    fn read_c_bytes(&self, addr: u64, limit: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..limit {
            match self.read_bytes(addr + i as u64, 1)? {
                [0] => return Some(out),
                [b] => out.push(*b),
                _ => unreachable!(),
            }
        }
        Some(out)
    }
}

impl GuestBus for Mem64 {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        match self.read_bytes(addr, out.len()) {
            Some(s) => {
                out.copy_from_slice(s);
                true
            }
            None => false,
        }
    }
    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        self.write_slice(addr, data)
    }
}

impl GuestMemory64 for Mem64 {
    fn write_bytes(&mut self, addr: u64, bytes: &[u8]) -> bool {
        self.write_slice(addr, bytes)
    }
    fn zero(&mut self, addr: u64, len: u64) -> bool {
        match self.locate(addr, len as usize) {
            Some((i, o)) => {
                self.regions[i].data[o..o + len as usize].fill(0);
                true
            }
            None => false,
        }
    }
    fn read_u64(&self, addr: u64) -> Option<u64> {
        Some(u64::from_le_bytes(self.read_bytes(addr, 8)?.try_into().ok()?))
    }
    fn write_u64(&mut self, addr: u64, value: u64) -> bool {
        self.write_slice(addr, &value.to_le_bytes())
    }
}

// ---------------------------------------------------------------------------
// Stubs for imported functions

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Malloc,
    Calloc,
    Realloc,
    Free,
    Memcpy,
    Memset,
    Memcmp,
    Strlen,
    Strcmp,
    Strncmp,
    StackChkFail,
    Abort,
    ReturnZero,
    MachAbsTime,
    MachTimebase,
    UiApplicationMain,
    /// Not implemented: logs the first call and returns 0.
    Unknown,
}

fn classify(name: &str) -> Kind {
    match name {
        "_malloc" | "_valloc" | "__Znwm" | "__Znam" | "__ZnwmRKSt9nothrow_t"
        | "__ZnamRKSt9nothrow_t" => Kind::Malloc,
        "_calloc" => Kind::Calloc,
        "_realloc" => Kind::Realloc,
        "_free" | "__ZdlPv" | "__ZdaPv" | "__ZdlPvm" | "__ZdaPvm" => Kind::Free,
        "_memcpy" | "_memmove" | "___memcpy_chk" | "___memmove_chk" => Kind::Memcpy,
        "_memset" | "___memset_chk" => Kind::Memset,
        "_memcmp" | "_bcmp" => Kind::Memcmp,
        "_strlen" => Kind::Strlen,
        "_strcmp" => Kind::Strcmp,
        "_strncmp" => Kind::Strncmp,
        "___stack_chk_fail" => Kind::StackChkFail,
        "_abort" | "_exit" | "__exit" => Kind::Abort,
        "___cxa_atexit" | "_atexit" => Kind::ReturnZero,
        "_mach_absolute_time" => Kind::MachAbsTime,
        "_mach_timebase_info" => Kind::MachTimebase,
        "_UIApplicationMain" => Kind::UiApplicationMain,
        _ => Kind::Unknown,
    }
}

enum Flow {
    Continue,
    /// Stop the whole run, with a human-readable reason.
    Done(String),
}

/// All "high-level emulation" state: the imported functions and their
/// bookkeeping.
struct Hle {
    names: Vec<String>,
    kinds: Vec<Kind>,
    heap_next: u64,
    alloc_sizes: HashMap<u64, u64>,
    seen: HashSet<u32>,
    counts: HashMap<u32, u64>,
    calls: u64,
    recent: VecDeque<String>,
    clock: u64,
}

impl Hle {
    fn alloc(&mut self, size: u64) -> u64 {
        let size = size.max(1).checked_add(15).map(|s| s & !15);
        let Some(size) = size else { return 0 };
        let end = match self.heap_next.checked_add(size) {
            Some(e) if e <= HEAP_BASE + HEAP_SIZE => e,
            _ => {
                log!("[a64] guest heap exhausted (asked for {size:#x} bytes)");
                return 0;
            }
        };
        let p = self.heap_next;
        self.heap_next = end;
        self.alloc_sizes.insert(p, size);
        p
    }

    fn handle(&mut self, cpu: &mut Cpu64, mem: &mut Mem64, id: u32) -> Flow {
        let Some(&kind) = self.kinds.get(id as usize) else {
            return Flow::Done(format!("svc #{id} does not match any stub"));
        };
        self.calls += 1;
        *self.counts.entry(id).or_insert(0) += 1;
        let lr = cpu.lr();
        let name = &self.names[id as usize];
        if self.recent.len() >= 40 {
            self.recent.pop_front();
        }
        self.recent.push_back(format!("{name}  (called from {lr:#x})"));

        let (x0, x1, x2) = (cpu.reg(0), cpu.reg(1), cpu.reg(2));
        match kind {
            Kind::Malloc => {
                let p = self.alloc(x0);
                cpu.set_reg(0, p);
            }
            Kind::Calloc => {
                let p = self.alloc(x0.saturating_mul(x1));
                cpu.set_reg(0, p); // heap memory is never reused, so it is zero
            }
            Kind::Realloc => {
                let new = self.alloc(x1);
                if x0 != 0 && new != 0 {
                    let old_size = self.alloc_sizes.get(&x0).copied().unwrap_or(0);
                    let n = old_size.min(x1) as usize;
                    if let Some(data) = mem.read_bytes(x0, n).map(|s| s.to_vec()) {
                        mem.write_slice(new, &data);
                    }
                }
                cpu.set_reg(0, new);
            }
            Kind::Free => {}
            Kind::Memcpy => {
                let n = x2 as usize;
                if n > 0x400_0000 {
                    return Flow::Done(format!("memcpy of {n:#x} bytes is implausible"));
                }
                match mem.read_bytes(x1, n).map(|s| s.to_vec()) {
                    Some(data) if mem.write_slice(x0, &data) => cpu.set_reg(0, x0),
                    _ => {
                        return Flow::Done(format!(
                            "memcpy({x0:#x}, {x1:#x}, {n:#x}) touched unmapped memory"
                        ))
                    }
                }
            }
            Kind::Memset => {
                let n = x2 as usize;
                if n > 0x400_0000 || !mem.write_slice(x0, &vec![x1 as u8; n]) {
                    return Flow::Done(format!(
                        "memset({x0:#x}, {x1:#x}, {n:#x}) touched unmapped memory"
                    ));
                }
                cpu.set_reg(0, x0);
            }
            Kind::Memcmp => {
                let n = x2 as usize;
                match (mem.read_bytes(x0, n), mem.read_bytes(x1, n)) {
                    (Some(a), Some(b)) => {
                        let r: i64 = match a.cmp(b) {
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                            std::cmp::Ordering::Greater => 1,
                        };
                        cpu.set_reg(0, r as u64);
                    }
                    _ => {
                        return Flow::Done(format!(
                            "memcmp({x0:#x}, {x1:#x}, {n:#x}) touched unmapped memory"
                        ))
                    }
                }
            }
            Kind::Strlen => match mem.read_c_bytes(x0, 1 << 24) {
                Some(s) => cpu.set_reg(0, s.len() as u64),
                None => return Flow::Done(format!("strlen({x0:#x}) touched unmapped memory")),
            },
            Kind::Strcmp | Kind::Strncmp => {
                let limit = if kind == Kind::Strncmp { x2 as usize } else { 1 << 24 };
                match (mem.read_c_bytes(x0, limit), mem.read_c_bytes(x1, limit)) {
                    (Some(a), Some(b)) => {
                        let r: i64 = match a.cmp(&b) {
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                            std::cmp::Ordering::Greater => 1,
                        };
                        cpu.set_reg(0, r as u64);
                    }
                    _ => {
                        return Flow::Done(format!(
                            "strcmp({x0:#x}, {x1:#x}) touched unmapped memory"
                        ))
                    }
                }
            }
            Kind::StackChkFail => return Flow::Done("the app's stack check failed".into()),
            Kind::Abort => return Flow::Done(format!("the app called {name}")),
            Kind::ReturnZero => cpu.set_reg(0, 0),
            Kind::MachAbsTime => {
                self.clock += 1000;
                cpu.set_reg(0, 1_000_000_000 + self.clock);
            }
            Kind::MachTimebase => {
                // struct mach_timebase_info { u32 numer; u32 denom; } = 1/1
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&1u32.to_le_bytes());
                b[4..].copy_from_slice(&1u32.to_le_bytes());
                mem.write_slice(x0, &b);
                cpu.set_reg(0, 0);
            }
            Kind::UiApplicationMain => {
                return Flow::Done(
                    "SUCCESS: main() ran and reached UIApplicationMain. The 64-bit loader, \
                     CPU and start-up code work."
                        .into(),
                )
            }
            Kind::Unknown => {
                if self.seen.insert(id) {
                    log!("[a64] first call to unimplemented {name} (from {lr:#x}); returning 0");
                }
                cpu.set_reg(0, 0);
            }
        }
        Flow::Continue
    }
}

// ---------------------------------------------------------------------------
// Running

enum CallEnd {
    Returned(u64),
    Stopped(String),
}

struct Machine {
    cpu: Cpu64,
    mem: Mem64,
    hle: Hle,
    trap_addr: u64,
    text_base: u64,
    image_end: u64,
    /// Defined symbols sorted by address, for describing crash addresses.
    syms: Vec<(u64, String)>,
}

impl Machine {
    /// Call guest function `pc` and run until it returns to the trap.
    fn call(&mut self, pc: u64, args: &[u64], what: &str) -> CallEnd {
        for (i, a) in args.iter().enumerate() {
            self.cpu.set_reg(i, *a);
        }
        self.cpu.set_pc(pc);
        self.cpu.set_lr(self.trap_addr);
        self.cpu.set_sp(STACK_TOP - 0x800);
        let mut slices = 0;
        loop {
            let mut ticks = TICKS_PER_SLICE;
            match self.cpu.run_or_step(&mut self.mem, Some(&mut ticks)) {
                StopReason::Svc(n) if n == TRAP_SVC => {
                    return CallEnd::Returned(self.cpu.reg(0));
                }
                StopReason::Svc(n) => {
                    if let Flow::Done(msg) = self.hle.handle(&mut self.cpu, &mut self.mem, n) {
                        return CallEnd::Stopped(msg);
                    }
                    if self.hle.calls > MAX_HLE_CALLS {
                        return CallEnd::Stopped(format!(
                            "{what}: more than {MAX_HLE_CALLS} imported-function calls, probably a loop"
                        ));
                    }
                }
                StopReason::Finished => {
                    slices += 1;
                    if slices >= MAX_SLICES {
                        return CallEnd::Stopped(format!(
                            "{what}: still running after {MAX_SLICES} time slices (stuck in a loop?) at {}",
                            self.describe(self.cpu.pc())
                        ));
                    }
                }
                StopReason::MemoryAbort { addr } => {
                    return CallEnd::Stopped(format!(
                        "{what}: memory abort, address {addr:#x}, at {}",
                        self.describe(self.cpu.pc())
                    ));
                }
                StopReason::UndefinedInstruction { pc } => {
                    let word = self
                        .mem
                        .read_bytes(pc, 4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                    return CallEnd::Stopped(format!(
                        "{what}: undefined instruction {word:08x?} at {}",
                        self.describe(pc)
                    ));
                }
                StopReason::Breakpoint { pc } => {
                    return CallEnd::Stopped(format!(
                        "{what}: brk instruction at {}",
                        self.describe(pc)
                    ));
                }
            }
        }
    }

    /// "image+0x1234 (nearest_symbol+0x10)" for addresses inside the image.
    fn describe(&self, addr: u64) -> String {
        if (STUB_BASE..=self.trap_addr).contains(&addr) {
            return format!("{addr:#x} (import stub area)");
        }
        if !(self.text_base..self.image_end).contains(&addr) {
            return format!("{addr:#x} (outside the app image)");
        }
        let mut s = format!("{addr:#x} (image+{:#x})", addr - self.text_base);
        let i = self.syms.partition_point(|(a, _)| *a <= addr);
        if i > 0 {
            let (a, n) = &self.syms[i - 1];
            if addr - a < 0x10000 {
                s.push_str(&format!(" [{n}+{:#x}]", addr - a));
            }
        }
        s
    }

    fn dump_state(&self, why: &str) {
        log!("[a64] STOPPED: {why}");
        log!("[a64] pc = {}", self.describe(self.cpu.pc()));
        log!("[a64] lr = {}", self.describe(self.cpu.lr()));
        log!("[a64] sp = {:#x}", self.cpu.sp());
        for row in 0..8 {
            let r = |i: usize| self.cpu.reg(i);
            let i = row * 4;
            if i + 3 < 31 {
                log!(
                    "[a64] x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x}",
                    i, r(i), i + 1, r(i + 1), i + 2, r(i + 2), i + 3, r(i + 3)
                );
            } else if i < 31 {
                log!("[a64] x{i}={:#018x} x{}={:#018x} x{}={:#018x}", r(i), i + 1, r(i + 1), i + 2, r(i + 2));
            }
        }
        log!("[a64] last imported-function calls (oldest first):");
        for c in &self.hle.recent {
            log!("[a64]   {c}");
        }
    }

    fn summary(&self) {
        let mut called: Vec<(u32, u64)> = self.hle.counts.iter().map(|(k, v)| (*k, *v)).collect();
        called.sort_by(|a, b| b.1.cmp(&a.1));
        log!(
            "[a64] {} imported-function calls in total, {} different functions",
            self.hle.calls,
            called.len()
        );
        let unimpl: Vec<&(u32, u64)> = called
            .iter()
            .filter(|(id, _)| self.hle.kinds[*id as usize] == Kind::Unknown)
            .collect();
        log!(
            "[a64] {} of them are NOT implemented yet (they just returned 0). Most called first:",
            unimpl.len()
        );
        for (id, n) in unimpl.iter().take(150) {
            log!("[a64]   {:>8} x  {}", n, self.hle.names[*id as usize]);
        }
    }
}

/// Entry point, called from `main` for 64-bit-only apps.
pub fn run(bundle: &Bundle, exe: &[u8]) -> Result<(), String> {
    let exe_path = bundle.executable_path();
    let name = exe_path.file_name().unwrap_or("executable").to_string();
    echo!("[a64] === experimental 64-bit probe: {name} ===");

    let macho = MachO64::load_from_bytes(exe, name.clone())?;
    echo!(
        "[a64] image {:#x}..{:#x}, {} rebases, {} binds, {} initializers, main at {:?}",
        macho.text_base,
        macho.last_segment_end,
        macho.rebases.len(),
        macho.binds.len(),
        macho.init_funcs.len(),
        macho.entry_point_pc
    );
    let main_pc = macho
        .entry_point_pc
        .ok_or("this executable has no LC_MAIN entry point")?;

    // One stub per distinct imported symbol.
    let mut names: Vec<String> = Vec::new();
    let mut index: HashMap<&str, u32> = HashMap::new();
    for b in &macho.binds {
        if !index.contains_key(b.symbol.as_str()) {
            index.insert(b.symbol.as_str(), names.len() as u32);
            names.push(b.symbol.clone());
        }
    }
    if names.len() >= TRAP_SVC as usize {
        return Err("too many imported symbols for the stub table".into());
    }
    let kinds: Vec<Kind> = names.iter().map(|n| classify(n)).collect();
    // Each stub is `svc #i; ret`. After the svc the CPU continues at the `ret`,
    // which returns to the caller. The extra last entry is the return trap.
    let mut stub_code = Vec::with_capacity((names.len() + 1) * 8);
    for i in 0..=names.len() {
        let imm = if i == names.len() { TRAP_SVC } else { i as u32 };
        stub_code.extend_from_slice(&(0xD400_0001u32 | (imm << 5)).to_le_bytes());
        stub_code.extend_from_slice(&0xD65F_03C0u32.to_le_bytes());
    }
    let trap_addr = STUB_BASE + names.len() as u64 * 8;

    let image_size = (macho.last_segment_end - macho.text_base + 0xfff) & !0xfff;
    let mut mem = Mem64 {
        regions: vec![
            Region { base: macho.text_base, data: vec![0; image_size as usize] },
            Region { base: STACK_BASE, data: vec![0; STACK_SIZE as usize] },
            Region { base: HEAP_BASE, data: vec![0; HEAP_SIZE as usize] },
            Region { base: STUB_BASE, data: stub_code },
        ],
    };

    macho.map_into(exe, &mut mem, 0)?;
    macho.apply_rebases(&mut mem, 0)?;
    for b in &macho.binds {
        let target = STUB_BASE + index[b.symbol.as_str()] as u64 * 8;
        if !mem.write_u64(b.addr, target.wrapping_add(b.addend as u64)) {
            return Err(format!("could not write bind for {} at {:#x}", b.symbol, b.addr));
        }
    }
    echo!("[a64] mapped the image and bound {} imports to stubs", names.len());

    let mut syms: Vec<(u64, String)> =
        macho.exported_symbols.iter().map(|(n, a)| (*a, n.clone())).collect();
    syms.sort();

    let mut m = Machine {
        cpu: Cpu64::new(),
        mem,
        hle: Hle {
            names,
            kinds,
            heap_next: HEAP_BASE,
            alloc_sizes: HashMap::new(),
            seen: HashSet::new(),
            counts: HashMap::new(),
            calls: 0,
            recent: VecDeque::new(),
            clock: 0,
        },
        trap_addr,
        text_base: macho.text_base,
        image_end: macho.text_base + image_size,
        syms,
    };

    // Thread-specific data block for TPIDRRO_EL0.
    let tsd = m.hle.alloc(0x1000);
    m.cpu.set_tpidrro_el0(tsd);

    // argc / argv / envp / apple, laid out at the top of the stack like the
    // kernel would.
    let path = format!(
        "/var/containers/Bundle/Application/00000000-0000-0000-0000-000000000000/{}.app/{}",
        name, name
    );
    let mut bytes = path.clone().into_bytes();
    bytes.push(0);
    let path_ptr = STACK_TOP - 0x400;
    let (argv, envp, apple) = (STACK_TOP - 0x200, STACK_TOP - 0x1f0, STACK_TOP - 0x1e0);
    m.mem.write_slice(path_ptr, &bytes);
    m.mem.write_u64(argv, path_ptr);
    m.mem.write_u64(argv + 8, 0);
    m.mem.write_u64(envp, 0);
    m.mem.write_u64(apple, path_ptr);
    m.mem.write_u64(apple + 8, 0);
    let args = [1, argv, envp, apple];

    // Static initializers, then main().
    let mut failure: Option<String> = None;
    for (i, &init) in macho.init_funcs.iter().enumerate() {
        match m.call(init, &args, &format!("initializer #{i}")) {
            CallEnd::Returned(_) => {}
            CallEnd::Stopped(msg) => {
                failure = Some(msg);
                break;
            }
        }
    }
    if failure.is_none() {
        echo!("[a64] all {} initializers returned; calling main", macho.init_funcs.len());
        match m.call(main_pc, &args, "main") {
            CallEnd::Returned(code) => {
                failure = Some(format!("main returned {code} without reaching UIApplicationMain"))
            }
            CallEnd::Stopped(msg) => failure = Some(msg),
        }
    }
    let why = failure.unwrap_or_default();
    if why.starts_with("SUCCESS") {
        echo!("[a64] {why}");
    } else {
        m.dump_state(&why);
    }
    m.summary();
    echo!("[a64] the full log is in touchHLE_log.txt");
    Ok(())
}

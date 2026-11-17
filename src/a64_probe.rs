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
use crate::cpu64::{Context64, Cpu64, GuestBus, StopReason};
use crate::mach_o64::{GuestMemory64, MachO64};
use std::collections::{HashMap, HashSet, VecDeque};

// Memory map. The image itself lives at its preferred address (slide 0),
// 0x1_0000_0000 for ordinary iOS executables.
const STACK_BASE: u64 = 0x1_6fc0_0000;
const STACK_SIZE: u64 = 0x40_0000; // 4 MiB
const STACK_TOP: u64 = STACK_BASE + STACK_SIZE;
const HEAP_BASE: u64 = 0x2_0000_0000;
const HEAP_SIZE: u64 = 0x800_0000; // 128 MiB
const STUB_BASE: u64 = 0x1_8000_0000;

/// `svc` immediate used by the return trap. Everything below this is a stub.
const TRAP_SVC: u32 = 0xffff;
/// Instruction budget for one `run` slice, and how many slices a single call
/// into the guest may use before we decide it is stuck in a loop.
const TICKS_PER_SLICE: u64 = 50_000_000;
const MAX_SLICES: u32 = 40;
const MAX_HLE_CALLS: u64 = 30_000_000;

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
    Strncpy,
    Strcpy,
    Strcat,
    Strchr,
    Strrchr,
    Bzero,
    Time,
    DispatchOnce,
    DispatchOnceF,
    PthreadOnce,
    AllocException,
    CxaThrow,
    PthreadKeyCreate,
    PthreadGetSpecific,
    PthreadSetSpecific,
    ObjcMsgSend,
    ObjcMsgSendSuper2,
    ReturnX0,
    ObjcStoreStrong,
    ObjcGetClass,
    NsClassFromString,
    NsStringFromClass,
    SelRegister,
    ObjectGetClass,
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
        "___cxa_atexit" | "_atexit" | "_dlsym" | "_pthread_mutex_init" | "_pthread_mutex_lock"
        | "_pthread_mutex_unlock" | "_pthread_mutex_destroy" | "_pthread_mutex_trylock"
        | "_pthread_mutexattr_init" | "_pthread_mutexattr_settype"
        | "_pthread_mutexattr_destroy" | "_pthread_cond_init" | "_pthread_cond_destroy"
        | "_pthread_cond_signal" | "_pthread_cond_broadcast" | "_OSSpinLockLock"
        | "_OSSpinLockUnlock" | "_pthread_rwlock_init" | "_pthread_rwlock_rdlock"
        | "_pthread_rwlock_wrlock" | "_pthread_rwlock_unlock" => Kind::ReturnZero,
        "_strncpy" | "___strncpy_chk" => Kind::Strncpy,
        "_strcpy" | "___strcpy_chk" => Kind::Strcpy,
        "_strcat" | "___strcat_chk" => Kind::Strcat,
        "_strchr" => Kind::Strchr,
        "_strrchr" => Kind::Strrchr,
        "_bzero" => Kind::Bzero,
        "_time" => Kind::Time,
        "_dispatch_once" => Kind::DispatchOnce,
        "_dispatch_once_f" => Kind::DispatchOnceF,
        "_pthread_once" => Kind::PthreadOnce,
        "___cxa_allocate_exception" => Kind::AllocException,
        "___cxa_throw" => Kind::CxaThrow,
        "_pthread_key_create" => Kind::PthreadKeyCreate,
        "_pthread_getspecific" => Kind::PthreadGetSpecific,
        "_pthread_setspecific" => Kind::PthreadSetSpecific,
        "_mach_absolute_time" => Kind::MachAbsTime,
        "_mach_timebase_info" => Kind::MachTimebase,
        "_UIApplicationMain" => Kind::UiApplicationMain,
        "_objc_msgSend" => Kind::ObjcMsgSend,
        "_objc_msgSendSuper2" => Kind::ObjcMsgSendSuper2,
        "_objc_retain"
        | "_objc_retainAutoreleasedReturnValue"
        | "_objc_autorelease"
        | "_objc_autoreleaseReturnValue"
        | "_objc_retainAutorelease"
        | "_objc_retainAutoreleaseReturnValue"
        | "_objc_unsafeClaimAutoreleasedReturnValue"
        | "_objc_claimAutoreleasedReturnValue"
        | "_sel_getName" => Kind::ReturnX0,
        "_objc_release" => Kind::Free,
        "_objc_storeStrong" => Kind::ObjcStoreStrong,
        "_objc_autoreleasePoolPush" | "_objc_autoreleasePoolPop" => Kind::ReturnZero,
        "_sel_registerName" | "_sel_getUid" => Kind::SelRegister,
        "_objc_getClass" | "_objc_lookUpClass" | "_objc_getRequiredClass" => Kind::ObjcGetClass,
        "_NSClassFromString" => Kind::NsClassFromString,
        "_NSStringFromClass" => Kind::NsStringFromClass,
        "_object_getClass" => Kind::ObjectGetClass,
        _ => Kind::Unknown,
    }
}

enum Flow {
    Continue,
    /// Stop the current call into the guest, with a human-readable reason.
    Done(String),
    /// Run guest function `pc(args)` to completion (e.g. the block given to
    /// `dispatch_once`), then continue the interrupted call with X0 = `result`.
    CallGuest { pc: u64, args: Vec<u64>, result: u64, retry_at: Option<u64> },
    // `retry_at`: instead of setting X0 = result, jump back to the import stub at
    // this address so the interrupted call is made again (used for +initialize).
}

// ---------------------------------------------------------------------------
// Guest heap and Objective-C runtime
//
// Just enough of the Objective-C runtime to dispatch messages: the app's own
// classes are read from the binary's metadata; system classes (NSObject,
// UIView, ...) are created here as empty placeholders, so that the app can
// subclass them. Messages nobody implements are answered with nil and logged,
// which tells us what to implement next.

struct Heap {
    next: u64,
    sizes: HashMap<u64, u64>,
}

impl Heap {
    fn alloc(&mut self, size: u64) -> u64 {
        let size = size.max(1).checked_add(15).map(|s| s & !15);
        let Some(size) = size else { return 0 };
        let end = match self.next.checked_add(size) {
            Some(e) if e <= HEAP_BASE + HEAP_SIZE => e,
            _ => {
                log!("[a64] guest heap exhausted (asked for {size:#x} bytes)");
                return 0;
            }
        };
        let p = self.next;
        self.next = end;
        self.sizes.insert(p, size);
        p
    }
}

fn rd32(mem: &Mem64, addr: u64) -> Option<u32> {
    Some(u32::from_le_bytes(mem.read_bytes(addr, 4)?.try_into().ok()?))
}

/// (address, size) of a section of the app image, by name.
fn section_range(macho: &MachO64, name: &str) -> Option<(u64, u64)> {
    macho
        .segments
        .iter()
        .flat_map(|s| s.sections.iter())
        .find(|s| s.name == name)
        .map(|s| (s.addr, s.size))
}

/// The pointers stored in a section like `__objc_classlist`.
fn read_ptrs(mem: &Mem64, macho: &MachO64, name: &str) -> Vec<u64> {
    match section_range(macho, name) {
        Some((addr, size)) => (0..size / 8).filter_map(|i| mem.read_u64(addr + i * 8)).collect(),
        None => Vec::new(),
    }
}

/// A method list (`method_list_t`): selector name -> implementation address.
/// Selector pointers are also remembered in `sels`, because inside one image
/// the same selector name always has the same address.
fn parse_method_list(mem: &Mem64, ptr: u64, sels: &mut HashMap<String, u64>) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    if ptr == 0 {
        return out;
    }
    let (Some(entsize_flags), Some(count)) = (rd32(mem, ptr), rd32(mem, ptr + 4)) else {
        return out;
    };
    let entsize = (entsize_flags & 0xfffc) as u64;
    // 24 bytes = absolute pointers (iOS up to 14). Shorter entries are the
    // newer relative format, which this app doesn't use.
    if entsize < 24 || count > 100_000 {
        return out;
    }
    for i in 0..count as u64 {
        let e = ptr + 8 + i * entsize;
        let (Some(name_ptr), Some(imp)) = (mem.read_u64(e), mem.read_u64(e + 16)) else {
            continue;
        };
        if let Some(bytes) = mem.read_c_bytes(name_ptr, 4096) {
            let name = String::from_utf8_lossy(&bytes).into_owned();
            sels.entry(name.clone()).or_insert(name_ptr);
            out.insert(name, imp);
        }
    }
    out
}

/// Text of a constant string object (`@"..."`, a `CFString` literal):
/// { isa, flags, chars pointer, length }.
fn read_cfstring(mem: &Mem64, addr: u64) -> Option<String> {
    if addr == 0 {
        return None;
    }
    let flags = mem.read_u64(addr + 8)?;
    let chars = mem.read_u64(addr + 16)?;
    let len = mem.read_u64(addr + 24)? as usize;
    if len > (1 << 20) {
        return None;
    }
    if flags & 0x10 != 0 {
        // UTF-16
        let raw = mem.read_bytes(chars, len * 2)?;
        let units: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Some(String::from_utf16_lossy(&units))
    } else {
        Some(String::from_utf8_lossy(mem.read_bytes(chars, len)?).into_owned())
    }
}

/// Superclass of a system class that we create as a placeholder.
fn hle_superclass(name: &str) -> Option<&'static str> {
    match name {
        "NSObject" => None,
        "UIApplication" | "UIView" | "UIViewController" => Some("UIResponder"),
        "UIWindow" | "UIControl" | "UILabel" | "UIImageView" | "UIScrollView"
        | "UITableViewCell" | "UIActivityIndicatorView" | "UIAlertView" | "UIWebView" => {
            Some("UIView")
        }
        "UIButton" | "UITextField" | "UISwitch" | "UISlider" => Some("UIControl"),
        "UITextView" | "UITableView" | "UICollectionView" => Some("UIScrollView"),
        "UITableViewController" | "UINavigationController" | "UITabBarController"
        | "UIActivityViewController" => Some("UIViewController"),
        "NSMutableArray" => Some("NSArray"),
        "NSMutableDictionary" => Some("NSDictionary"),
        "NSMutableString" | "NSConstantString" => Some("NSString"),
        "NSMutableData" => Some("NSData"),
        "NSMutableSet" => Some("NSSet"),
        _ => Some("NSObject"),
    }
}

/// Selectors our built-in NSObject answers (for `respondsToSelector:`).
const BUILTIN_SELS: &[&str] = &[
    "init", "self", "retain", "release", "autorelease", "class", "superclass", "isEqual:",
    "hash", "isKindOfClass:", "isMemberOfClass:", "respondsToSelector:", "performSelector:",
    "performSelector:withObject:", "dealloc",
];

struct ClassInfo {
    name: String,
    superclass: u64,
    is_meta: bool,
    /// A placeholder for a system class (as opposed to one from the app).
    is_hle: bool,
    instance_size: u64,
    /// Selector name -> implementation address, for methods the app defines
    /// (including those added by categories).
    methods: HashMap<String, u64>,
    initialized: bool,
}

#[derive(Default)]
struct Objc {
    /// By class_t address; metaclasses are included.
    classes: HashMap<u64, ClassInfo>,
    by_name: HashMap<String, u64>,
    sels: HashMap<String, u64>,
    sel_names: HashMap<u64, String>,
    /// "-[Class selector]" -> how often it was sent without an implementation.
    unhandled: HashMap<String, u64>,
    /// +load methods to run: (class, implementation).
    pending_loads: Vec<(u64, u64)>,
    load_scheduled: HashSet<u64>,
    /// (principal class name, delegate class name) strings given to
    /// UIApplicationMain.
    app_main: Option<(u64, u64)>,
}

impl Objc {
    fn cstr(mem: &Mem64, addr: u64) -> Option<String> {
        mem.read_c_bytes(addr, 4096).map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    fn sel_name(&mut self, mem: &Mem64, sel: u64) -> String {
        if let Some(n) = self.sel_names.get(&sel) {
            return n.clone();
        }
        let n = Self::cstr(mem, sel).unwrap_or_default();
        self.sel_names.insert(sel, n.clone());
        self.sels.entry(n.clone()).or_insert(sel);
        n
    }

    /// Address of the selector called `name`, creating it if the app never
    /// mentions it.
    fn intern_sel(&mut self, mem: &mut Mem64, heap: &mut Heap, name: &str) -> u64 {
        if let Some(a) = self.sels.get(name) {
            return *a;
        }
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        let p = heap.alloc(bytes.len() as u64);
        mem.write_slice(p, &bytes);
        self.sels.insert(name.to_string(), p);
        self.sel_names.insert(p, name.to_string());
        p
    }

    /// The (class, metaclass) addresses of system class `name`, creating the
    /// placeholder objects in guest memory the first time.
    fn system_class(&mut self, mem: &mut Mem64, heap: &mut Heap, name: &str) -> (u64, u64) {
        if let Some(&c) = self.by_name.get(name) {
            return (c, mem.read_u64(c).unwrap_or(0));
        }
        let sup = hle_superclass(name).map(|s| self.system_class(mem, heap, s));
        // class_t: isa, superclass, cache, vtable, data
        let cls = heap.alloc(48);
        let meta = heap.alloc(48);
        mem.write_u64(cls, meta);
        mem.write_u64(cls + 8, sup.map_or(0, |s| s.0));
        mem.write_u64(meta, meta);
        // The root metaclass inherits from the root class, so class methods
        // fall back to instance methods of NSObject, as in the real runtime.
        mem.write_u64(meta + 8, sup.map_or(cls, |s| s.1));
        let info = |is_meta: bool, superclass: u64| ClassInfo {
            name: name.to_string(),
            superclass,
            is_meta,
            is_hle: true,
            instance_size: 0x40,
            methods: HashMap::new(),
            initialized: false,
        };
        self.classes.insert(cls, info(false, sup.map_or(0, |s| s.0)));
        self.classes.insert(meta, info(true, sup.map_or(cls, |s| s.1)));
        self.by_name.insert(name.to_string(), cls);
        (cls, meta)
    }

    /// Register a class (or metaclass) defined in the app image, and the
    /// classes it links to inside the image.
    fn add_app_class(&mut self, mem: &Mem64, c: u64, image: (u64, u64)) {
        if c == 0 || self.classes.contains_key(&c) {
            return;
        }
        let (Some(isa), Some(sup), Some(data)) =
            (mem.read_u64(c), mem.read_u64(c + 8), mem.read_u64(c + 32))
        else {
            return;
        };
        let ro = data & !7;
        let (Some(flags), Some(inst_size), Some(name_ptr), Some(methods_ptr)) = (
            rd32(mem, ro),
            rd32(mem, ro + 8),
            mem.read_u64(ro + 24),
            mem.read_u64(ro + 32),
        ) else {
            return;
        };
        let name = Self::cstr(mem, name_ptr).unwrap_or_default();
        let methods = parse_method_list(mem, methods_ptr, &mut self.sels);
        let is_meta = flags & 1 != 0;
        self.classes.insert(
            c,
            ClassInfo {
                name: name.clone(),
                superclass: sup,
                is_meta,
                is_hle: false,
                instance_size: inst_size as u64,
                methods,
                initialized: false,
            },
        );
        if !is_meta {
            self.by_name.insert(name, c);
        }
        let in_image = |a: u64| a >= image.0 && a < image.1;
        if in_image(isa) {
            self.add_app_class(mem, isa, image);
        }
        if sup != 0 && in_image(sup) {
            self.add_app_class(mem, sup, image);
        }
    }

    /// Queue `cls`'s own +load (superclasses first).
    fn schedule_load(&mut self, mem: &Mem64, cls: u64) {
        if !self.load_scheduled.insert(cls) {
            return;
        }
        let sup = self.classes.get(&cls).map_or(0, |c| c.superclass);
        if sup != 0 && self.classes.get(&sup).map_or(false, |c| !c.is_hle) {
            self.schedule_load(mem, sup);
        }
        if let Some(meta) = mem.read_u64(cls) {
            let imp = self.classes.get(&meta).and_then(|m| m.methods.get("load")).copied();
            if let Some(imp) = imp {
                self.pending_loads.push((cls, imp));
            }
        }
    }

    /// Read all Objective-C metadata of the app image.
    fn load_image(&mut self, mem: &Mem64, macho: &MachO64) {
        let image = (macho.text_base, macho.last_segment_end);
        for p in read_ptrs(mem, macho, "__objc_selrefs") {
            if let Some(n) = Self::cstr(mem, p) {
                self.sels.entry(n).or_insert(p);
            }
        }
        for c in read_ptrs(mem, macho, "__objc_classlist") {
            self.add_app_class(mem, c, image);
        }
        // +load for non-lazy classes; must be decided before categories are
        // merged, because a category may define its own +load as well.
        for c in read_ptrs(mem, macho, "__objc_nlclslist") {
            self.add_app_class(mem, c, image);
            self.schedule_load(mem, c);
        }
        let non_lazy_cats: HashSet<u64> = read_ptrs(mem, macho, "__objc_nlcatlist").into_iter().collect();
        for cat in read_ptrs(mem, macho, "__objc_catlist") {
            // category_t: name, cls, instanceMethods, classMethods, ...
            let (Some(cls), Some(im_ptr), Some(cm_ptr)) =
                (mem.read_u64(cat + 8), mem.read_u64(cat + 16), mem.read_u64(cat + 24))
            else {
                continue;
            };
            let inst = parse_method_list(mem, im_ptr, &mut self.sels);
            let class_m = parse_method_list(mem, cm_ptr, &mut self.sels);
            let load_imp = class_m.get("load").copied();
            let meta = mem.read_u64(cls).unwrap_or(0);
            if let Some(ci) = self.classes.get_mut(&cls) {
                ci.methods.extend(inst);
            }
            if let Some(mi) = self.classes.get_mut(&meta) {
                mi.methods.extend(class_m);
            }
            if non_lazy_cats.contains(&cat) {
                if let Some(imp) = load_imp {
                    self.pending_loads.push((cls, imp));
                }
            }
        }
    }

    /// The class of an object (for a class object: its metaclass).
    fn class_of(&self, mem: &Mem64, obj: u64) -> Option<u64> {
        let isa = mem.read_u64(obj)?;
        let c = isa & 0x0000_000f_ffff_fff8; // no-op for plain isa pointers
        if self.classes.contains_key(&c) {
            Some(c)
        } else {
            None
        }
    }

    /// Method lookup along the superclass chain. Only finds methods the app
    /// implements.
    fn find(&self, start: u64, sel: &str) -> Option<u64> {
        let mut c = start;
        for _ in 0..64 {
            let info = self.classes.get(&c)?;
            if let Some(&imp) = info.methods.get(sel) {
                return Some(imp);
            }
            if info.superclass == 0 || info.superclass == c {
                return None;
            }
            c = info.superclass;
        }
        None
    }

    /// The first not yet initialized class in `recv`'s superclass chain
    /// (superclass first) that has an +initialize, with that method.
    fn pending_initialize(&mut self, mem: &Mem64, recv: u64) -> Option<(u64, u64)> {
        let mut chain = Vec::new();
        let mut c = recv;
        for _ in 0..64 {
            chain.push(c);
            let sup = self.classes.get(&c).map_or(0, |i| i.superclass);
            if sup == 0 || sup == c {
                break;
            }
            c = sup;
        }
        for &c in chain.iter().rev() {
            let Some(info) = self.classes.get_mut(&c) else { continue };
            if info.initialized {
                continue;
            }
            info.initialized = true;
            if info.is_hle {
                continue;
            }
            let meta = mem.read_u64(c)?;
            let imp = self.classes.get(&meta).and_then(|m| m.methods.get("initialize")).copied();
            if let Some(imp) = imp {
                return Some((imp, c));
            }
        }
        None
    }
}

/// All "high-level emulation" state: the imported functions and their
/// bookkeeping.
struct Hle {
    names: Vec<String>,
    kinds: Vec<Kind>,
    heap: Heap,
    objc: Objc,
    seen: HashSet<u32>,
    counts: HashMap<u32, u64>,
    calls: u64,
    recent: VecDeque<String>,
    clock: u64,
    once_done: HashSet<u64>,
    tsd: HashMap<u64, u64>,
    next_key: u64,
}

impl Hle {
    fn alloc(&mut self, size: u64) -> u64 {
        self.heap.alloc(size)
    }

    /// A zeroed instance of `cls` (as `+alloc` would make).
    fn alloc_instance(&mut self, mem: &mut Mem64, cls: u64) -> u64 {
        let size = self.objc.classes.get(&cls).map_or(0x40, |c| c.instance_size).max(16);
        let p = self.heap.alloc(size);
        mem.write_u64(p, cls);
        p
    }

    /// A constant-string object (like an `@"..."` literal) holding `text`.
    fn make_string(&mut self, mem: &mut Mem64, text: &str) -> u64 {
        let (cls, _) = self.objc.system_class(mem, &mut self.heap, "NSConstantString");
        let bytes = text.as_bytes();
        let chars = self.heap.alloc(bytes.len() as u64 + 1);
        mem.write_slice(chars, bytes);
        let obj = self.heap.alloc(32);
        mem.write_u64(obj, cls);
        mem.write_u64(obj + 8, 0x7c8);
        mem.write_u64(obj + 16, chars);
        mem.write_u64(obj + 24, bytes.len() as u64);
        obj
    }

    /// `objc_msgSend`: find the method and jump to it (app code), or run a
    /// built-in, or answer nil. `start` is the class to start searching at
    /// for `super` sends.
    #[allow(clippy::too_many_arguments)]
    fn msg_send(
        &mut self,
        cpu: &mut Cpu64,
        mem: &mut Mem64,
        recv: u64,
        sel_ptr: u64,
        start: Option<u64>,
        stub_addr: u64,
        lr: u64,
    ) -> Flow {
        if recv == 0 {
            // Messages to nil return 0.
            cpu.set_reg(0, 0);
            for i in 0..4 {
                cpu.set_vec(i, 0);
            }
            return Flow::Continue;
        }
        let sel = self.objc.sel_name(mem, sel_ptr);
        let Some(cls) = self.objc.class_of(mem, recv) else {
            let key = format!("message {sel} sent to {recv:#x}, which is not an object");
            let n = self.objc.unhandled.entry(key.clone()).or_insert(0);
            *n += 1;
            if *n == 1 {
                log!("[a64] {key} (from {lr:#x})");
            }
            cpu.set_reg(0, 0);
            return Flow::Continue;
        };
        let recv_is_class = self.objc.classes.get(&recv).map_or(false, |c| !c.is_meta);
        // First message to a class: run +initialize (superclass first), then
        // send this message again.
        if start.is_none() && recv_is_class && !self.objc.classes[&recv].initialized {
            if let Some((imp, c)) = self.objc.pending_initialize(mem, recv) {
                let s = self.objc.intern_sel(mem, &mut self.heap, "initialize");
                return Flow::CallGuest { pc: imp, args: vec![c, s], result: 0, retry_at: Some(stub_addr) };
            }
        }
        if let Some(imp) = self.objc.find(start.unwrap_or(cls), &sel) {
            // Tail call: the method returns straight to our caller.
            cpu.set_pc(imp);
            return Flow::Continue;
        }
        if let Some(flow) = self.nsobject_builtin(cpu, mem, recv, cls, &sel, stub_addr, lr) {
            return flow;
        }
        let (is_meta, cname) = {
            let ci = &self.objc.classes[&cls];
            (ci.is_meta, ci.name.clone())
        };
        let key = format!("{}[{} {}]", if is_meta { '+' } else { '-' }, cname, sel);
        let n = self.objc.unhandled.entry(key.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            log!("[a64] unhandled message {key} (from {lr:#x}); answering nil");
        }
        cpu.set_reg(0, 0);
        for i in 0..4 {
            cpu.set_vec(i, 0);
        }
        Flow::Continue
    }

    /// The few NSObject methods we implement ourselves. `None` = not one of
    /// them.
    #[allow(clippy::too_many_arguments)]
    fn nsobject_builtin(
        &mut self,
        cpu: &mut Cpu64,
        mem: &mut Mem64,
        recv: u64,
        cls: u64,
        sel: &str,
        stub_addr: u64,
        lr: u64,
    ) -> Option<Flow> {
        let recv_is_class = self.objc.classes.get(&recv).map_or(false, |c| !c.is_meta);
        let (x2, x3) = (cpu.reg(2), cpu.reg(3));
        match sel {
            "alloc" | "allocWithZone:" if recv_is_class => {
                let p = self.alloc_instance(mem, recv);
                cpu.set_reg(0, p);
            }
            "new" if recv_is_class => {
                let p = self.alloc_instance(mem, recv);
                cpu.set_reg(0, p);
                let s = self.objc.intern_sel(mem, &mut self.heap, "init");
                cpu.set_reg(1, s);
                return Some(self.msg_send(cpu, mem, p, s, None, stub_addr, lr));
            }
            // These return the receiver, which is already in X0.
            "init" | "self" | "retain" | "autorelease" => {}
            "release" | "dealloc" | "load" | "initialize" => {}
            "class" => {
                let c = if recv_is_class { recv } else { cls };
                cpu.set_reg(0, c);
            }
            "superclass" => {
                let c = if recv_is_class { recv } else { cls };
                let s = self.objc.classes.get(&c).map_or(0, |i| i.superclass);
                cpu.set_reg(0, s);
            }
            "isEqual:" => cpu.set_reg(0, (recv == x2) as u64),
            "hash" => cpu.set_reg(0, recv),
            "isKindOfClass:" | "isMemberOfClass:" => {
                let (mut c, mut res) = (cls, false);
                for _ in 0..64 {
                    if c == x2 {
                        res = true;
                        break;
                    }
                    if sel == "isMemberOfClass:" {
                        break;
                    }
                    let sup = self.objc.classes.get(&c).map_or(0, |i| i.superclass);
                    if sup == 0 || sup == c {
                        break;
                    }
                    c = sup;
                }
                cpu.set_reg(0, res as u64);
            }
            "respondsToSelector:" => {
                let n = self.objc.sel_name(mem, x2);
                let found = self.objc.find(cls, &n).is_some() || BUILTIN_SELS.contains(&n.as_str());
                cpu.set_reg(0, found as u64);
            }
            "performSelector:" => {
                cpu.set_reg(1, x2);
                return Some(self.msg_send(cpu, mem, recv, x2, None, stub_addr, lr));
            }
            "performSelector:withObject:" => {
                cpu.set_reg(1, x2);
                cpu.set_reg(2, x3);
                return Some(self.msg_send(cpu, mem, recv, x2, None, stub_addr, lr));
            }
            _ => return None,
        }
        Some(Flow::Continue)
    }

    fn handle(&mut self, cpu: &mut Cpu64, mem: &mut Mem64, id: u32) -> Flow {
        let Some(&kind) = self.kinds.get(id as usize) else {
            return Flow::Done(format!("svc #{id} does not match any stub"));
        };
        self.calls += 1;
        *self.counts.entry(id).or_insert(0) += 1;
        let lr = cpu.lr();
        let stub_addr = STUB_BASE + id as u64 * 8;
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
                    let old_size = self.heap.sizes.get(&x0).copied().unwrap_or(0);
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
                // argc, argv, principal class name, delegate class name
                self.objc.app_main = Some((x2, cpu.reg(3)));
                return Flow::Done(
                    "SUCCESS: main() ran and reached UIApplicationMain. The 64-bit loader, \
                     CPU and start-up code work."
                        .into(),
                );
            }
            Kind::Strncpy => {
                let n = x2 as usize;
                if n > 0x400_0000 {
                    return Flow::Done(format!("strncpy of {n:#x} bytes is implausible"));
                }
                let Some(src) = mem.read_c_bytes(x1, n) else {
                    return Flow::Done(format!("strncpy source {x1:#x} is unmapped"));
                };
                let mut buf = src;
                buf.resize(n, 0);
                if !mem.write_slice(x0, &buf) {
                    return Flow::Done(format!("strncpy destination {x0:#x} is unmapped"));
                }
                cpu.set_reg(0, x0);
            }
            Kind::Strcpy | Kind::Strcat => {
                let Some(mut src) = mem.read_c_bytes(x1, 1 << 24) else {
                    return Flow::Done(format!("strcpy source {x1:#x} is unmapped"));
                };
                src.push(0);
                let dst = if kind == Kind::Strcat {
                    match mem.read_c_bytes(x0, 1 << 24) {
                        Some(d) => x0 + d.len() as u64,
                        None => return Flow::Done(format!("strcat destination {x0:#x} is unmapped")),
                    }
                } else {
                    x0
                };
                if !mem.write_slice(dst, &src) {
                    return Flow::Done(format!("strcpy destination {dst:#x} is unmapped"));
                }
                cpu.set_reg(0, x0);
            }
            Kind::Strchr | Kind::Strrchr => {
                let Some(s) = mem.read_c_bytes(x0, 1 << 24) else {
                    return Flow::Done(format!("strchr string {x0:#x} is unmapped"));
                };
                let c = x1 as u8;
                let found = if c == 0 {
                    Some(s.len())
                } else if kind == Kind::Strchr {
                    s.iter().position(|&b| b == c)
                } else {
                    s.iter().rposition(|&b| b == c)
                };
                cpu.set_reg(0, found.map_or(0, |i| x0 + i as u64));
            }
            Kind::Bzero => {
                let n = x1 as usize;
                if n > 0x400_0000 || !mem.write_slice(x0, &vec![0u8; n]) {
                    return Flow::Done(format!("bzero({x0:#x}, {n:#x}) touched unmapped memory"));
                }
            }
            Kind::Time => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if x0 != 0 {
                    mem.write_u64(x0, secs);
                }
                cpu.set_reg(0, secs);
            }
            Kind::DispatchOnce => {
                if mem.read_u64(x0) == Some(0) {
                    mem.write_u64(x0, u64::MAX); // DISPATCH_ONCE_DONE
                    // Block layout: isa, flags+reserved, invoke, descriptor.
                    match mem.read_u64(x1.wrapping_add(16)) {
                        Some(invoke) if invoke != 0 => {
                            return Flow::CallGuest { pc: invoke, args: vec![x1], result: 0, retry_at: None }
                        }
                        _ => return Flow::Done(format!("dispatch_once: bad block {x1:#x}")),
                    }
                }
            }
            Kind::DispatchOnceF => {
                if mem.read_u64(x0) == Some(0) {
                    mem.write_u64(x0, u64::MAX);
                    return Flow::CallGuest { pc: x2, args: vec![x1], result: 0, retry_at: None };
                }
            }
            Kind::PthreadOnce => {
                if self.once_done.insert(x0) {
                    return Flow::CallGuest { pc: x1, args: vec![], result: 0, retry_at: None };
                }
                cpu.set_reg(0, 0);
            }
            Kind::AllocException => {
                let p = self.alloc(x0.max(16));
                cpu.set_reg(0, p);
            }
            Kind::CxaThrow => {
                return Flow::Done(format!(
                    "the app threw a C++ exception (type_info {x1:#x}) from {lr:#x}"
                ))
            }
            Kind::PthreadKeyCreate => {
                let key = self.next_key;
                self.next_key += 1;
                mem.write_u64(x0, key);
                cpu.set_reg(0, 0);
            }
            Kind::PthreadGetSpecific => {
                let v = self.tsd.get(&x0).copied().unwrap_or(0);
                cpu.set_reg(0, v);
            }
            Kind::PthreadSetSpecific => {
                self.tsd.insert(x0, x1);
                cpu.set_reg(0, 0);
            }
            Kind::ObjcMsgSend => {
                return self.msg_send(cpu, mem, x0, x1, None, stub_addr, lr);
            }
            Kind::ObjcMsgSendSuper2 => {
                // x0 points to struct objc_super { receiver, current class }.
                let (Some(receiver), Some(current)) = (mem.read_u64(x0), mem.read_u64(x0.wrapping_add(8)))
                else {
                    return Flow::Done(format!("objc_msgSendSuper2: bad objc_super at {x0:#x}"));
                };
                cpu.set_reg(0, receiver);
                let sup = self.objc.classes.get(&current).map(|c| c.superclass);
                return self.msg_send(cpu, mem, receiver, x1, sup, stub_addr, lr);
            }
            Kind::ReturnX0 => {}
            Kind::ObjcStoreStrong => {
                mem.write_u64(x0, x1);
            }
            Kind::SelRegister => {
                let n = Objc::cstr(mem, x0).unwrap_or_default();
                let p = self.objc.intern_sel(mem, &mut self.heap, &n);
                cpu.set_reg(0, p);
            }
            Kind::ObjcGetClass => {
                let n = Objc::cstr(mem, x0).unwrap_or_default();
                let c = self.objc.by_name.get(&n).copied().unwrap_or(0);
                cpu.set_reg(0, c);
            }
            Kind::NsClassFromString => {
                let c = read_cfstring(mem, x0)
                    .and_then(|n| self.objc.by_name.get(&n).copied())
                    .unwrap_or(0);
                cpu.set_reg(0, c);
            }
            Kind::NsStringFromClass => {
                let n = self.objc.classes.get(&x0).map(|c| c.name.clone());
                let s = match n {
                    Some(n) => self.make_string(mem, &n),
                    None => 0,
                };
                cpu.set_reg(0, s);
            }
            Kind::ObjectGetClass => {
                let c = self.objc.class_of(mem, x0).unwrap_or(0);
                cpu.set_reg(0, c);
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
    /// Call guest function `pc` from the top level and run until it returns.
    fn call(&mut self, pc: u64, args: &[u64], what: &str) -> CallEnd {
        self.run_guest(pc, args, STACK_TOP - 0x800, what)
    }

    /// Run guest function `pc` on a stack starting at `sp` until it returns to
    /// the trap. Calls back into the guest from imported functions (like
    /// `dispatch_once`) nest by recursion.
    fn run_guest(&mut self, pc: u64, args: &[u64], sp: u64, what: &str) -> CallEnd {
        for (i, a) in args.iter().enumerate() {
            self.cpu.set_reg(i, *a);
        }
        self.cpu.set_pc(pc);
        self.cpu.set_lr(self.trap_addr);
        self.cpu.set_sp(sp);
        let mut slices = 0;
        loop {
            let mut ticks = TICKS_PER_SLICE;
            match self.cpu.run_or_step(&mut self.mem, Some(&mut ticks)) {
                StopReason::Svc(n) if n == TRAP_SVC => {
                    return CallEnd::Returned(self.cpu.reg(0));
                }
                StopReason::Svc(n) => {
                    match self.hle.handle(&mut self.cpu, &mut self.mem, n) {
                        Flow::Continue => {}
                        Flow::Done(msg) => return CallEnd::Stopped(msg),
                        Flow::CallGuest { pc: f, args: a, result, retry_at } => {
                            // Park the interrupted call's registers, run the
                            // callee on the stack below it, then restore.
                            let mut saved = Context64::default();
                            self.cpu.swap_context(&mut saved);
                            self.cpu.set_tpidrro_el0(saved.tpidrro_el0);
                            self.cpu.set_tpidr_el0(saved.tpidr_el0);
                            let nested_sp = saved.sp.saturating_sub(0x400) & !0xf;
                            let end = self.run_guest(f, &a, nested_sp, what);
                            self.cpu.swap_context(&mut saved);
                            if let CallEnd::Stopped(msg) = end {
                                return CallEnd::Stopped(msg);
                            }
                            match retry_at {
                                Some(at) => self.cpu.set_pc(at),
                                None => self.cpu.set_reg(0, result),
                            }
                        }
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

    /// Send an Objective-C message from Rust. Only methods the app itself
    /// implements are looked up; for anything else the receiver is returned
    /// (which is right for `init`).
    fn send(&mut self, obj: u64, sel: &str, args: &[u64]) -> CallEnd {
        let Some(cls) = self.hle.objc.class_of(&self.mem, obj) else {
            return CallEnd::Stopped(format!("{obj:#x} is not an Objective-C object"));
        };
        let Some(imp) = self.hle.objc.find(cls, sel) else {
            return CallEnd::Returned(obj);
        };
        let sel_ptr = self.hle.objc.intern_sel(&mut self.mem, &mut self.hle.heap, sel);
        let mut a = vec![obj, sel_ptr];
        a.extend_from_slice(args);
        self.call(imp, &a, sel)
    }

    /// What UIApplicationMain would do first: create the app delegate and
    /// tell it the app has launched.
    fn launch_app(&mut self) {
        let Some((principal, delegate)) = self.hle.objc.app_main else {
            return;
        };
        let principal_name = read_cfstring(&self.mem, principal);
        let delegate_name = read_cfstring(&self.mem, delegate);
        echo!("[a64] UIApplicationMain(principal class {principal_name:?}, delegate class {delegate_name:?})");
        let Some(name) = delegate_name else {
            echo!("[a64] no delegate class name was given; stopping here");
            return;
        };
        let Some(&cls) = self.hle.objc.by_name.get(&name) else {
            echo!("[a64] the app has no class called {name}; stopping here");
            return;
        };
        let mut obj = self.hle.alloc_instance(&mut self.mem, cls);
        echo!("[a64] created the app delegate: an instance of {name} at {obj:#x}");
        match self.send(obj, "init", &[]) {
            CallEnd::Returned(r) => {
                if r != 0 {
                    obj = r;
                }
                echo!("[a64] -init done");
            }
            CallEnd::Stopped(msg) => {
                echo!("[a64] -init stopped: {msg}");
                self.brief_state();
            }
        }
        // The UIApplication object and the launch options are nil for now.
        match self.send(obj, "application:didFinishLaunchingWithOptions:", &[0, 0]) {
            CallEnd::Returned(r) => {
                echo!("[a64] application:didFinishLaunchingWithOptions: returned {r:#x}");
            }
            CallEnd::Stopped(msg) => {
                echo!("[a64] application:didFinishLaunchingWithOptions: stopped: {msg}");
                self.brief_state();
            }
        }
    }

    /// Short version of `dump_state`, for failures we then carry on from.
    fn brief_state(&self) {
        log!("[a64]   pc = {}", self.describe(self.cpu.pc()));
        log!("[a64]   lr = {}", self.describe(self.cpu.lr()));
        let skip = self.hle.recent.len().saturating_sub(12);
        for c in self.hle.recent.iter().skip(skip) {
            log!("[a64]   recent: {c}");
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
        let mut un: Vec<(&String, &u64)> = self.hle.objc.unhandled.iter().collect();
        un.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
        log!(
            "[a64] {} different Objective-C messages had no implementation (answered nil). Most sent first:",
            un.len()
        );
        for (k, n) in un.iter().take(250) {
            log!("[a64]   {:>8} x  {}", n, k);
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
    let n_imports = names.len();

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

    let mut hle = Hle {
        names,
        kinds,
        heap: Heap { next: HEAP_BASE, sizes: HashMap::new() },
        objc: Objc::default(),
        seen: HashSet::new(),
        counts: HashMap::new(),
        calls: 0,
        recent: VecDeque::new(),
        clock: 0,
        once_done: HashSet::new(),
        tsd: HashMap::new(),
        next_key: 1,
    };
    // Every import points at its stub, except Objective-C classes, which get
    // placeholder class objects so the app can subclass and message them.
    for b in &macho.binds {
        let sym = b.symbol.as_str();
        let target = if let Some(cname) = sym.strip_prefix("_OBJC_CLASS_$_") {
            hle.objc.system_class(&mut mem, &mut hle.heap, cname).0
        } else if let Some(cname) = sym.strip_prefix("_OBJC_METACLASS_$_") {
            hle.objc.system_class(&mut mem, &mut hle.heap, cname).1
        } else if sym == "___CFConstantStringClassReference" {
            hle.objc.system_class(&mut mem, &mut hle.heap, "NSConstantString").0
        } else {
            STUB_BASE + index[sym] as u64 * 8
        };
        if !mem.write_u64(b.addr, target.wrapping_add(b.addend as u64)) {
            return Err(format!("could not write bind for {} at {:#x}", b.symbol, b.addr));
        }
    }
    echo!("[a64] mapped the image and bound {n_imports} imports");
    hle.objc.load_image(&mem, &macho);
    echo!(
        "[a64] Objective-C: {} app classes, {} system class placeholders, {} selectors, {} +load methods to run",
        hle.objc.classes.values().filter(|c| !c.is_hle && !c.is_meta).count(),
        hle.objc.classes.values().filter(|c| c.is_hle && !c.is_meta).count(),
        hle.objc.sels.len(),
        hle.objc.pending_loads.len()
    );

    let mut syms: Vec<(u64, String)> =
        macho.exported_symbols.iter().map(|(n, a)| (*a, n.clone())).collect();
    syms.sort();

    let mut m = Machine {
        cpu: Cpu64::new(),
        mem,
        hle,
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

    // +load methods run first, like in the real runtime.
    let loads = std::mem::take(&mut m.hle.objc.pending_loads);
    let sel_load = m.hle.objc.intern_sel(&mut m.mem, &mut m.hle.heap, "load");
    for (cls, imp) in loads {
        if let CallEnd::Stopped(msg) = m.call(imp, &[cls, sel_load], "+load") {
            echo!("[a64] +load at {} FAILED: {msg}", m.describe(imp));
            m.brief_state();
        }
    }

    // Static initializers, then main().
    // A failing initializer is logged and skipped, so one problem doesn't hide
    // everything after it.
    let mut failed_inits = 0;
    for (i, &init) in macho.init_funcs.iter().enumerate() {
        if let CallEnd::Stopped(msg) = m.call(init, &args, &format!("initializer #{i}")) {
            failed_inits += 1;
            echo!("[a64] initializer #{i} ({}) FAILED: {msg}", m.describe(init));
            m.brief_state();
        }
    }
    echo!(
        "[a64] initializers done: {} ran, {failed_inits} failed; calling main",
        macho.init_funcs.len() - failed_inits
    );
    let failure = match m.call(main_pc, &args, "main") {
        CallEnd::Returned(code) => {
            Some(format!("main returned {code} without reaching UIApplicationMain"))
        }
        CallEnd::Stopped(msg) => Some(msg),
    };
    let why = failure.unwrap_or_default();
    if why.starts_with("SUCCESS") {
        echo!("[a64] {why}");
        m.launch_app();
    } else {
        m.dump_state(&why);
    }
    m.summary();
    echo!("[a64] the full log is in touchHLE_log.txt");
    Ok(())
}

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
use crate::fs::{Fs, GuestPath, GuestPathBuf};
use crate::gles::{gles2_raw as gl, gles30_raw as gl30};
use crate::options::Options;
use crate::window::{GLContext, GLVersion, Window};
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
const MAX_HLE_CALLS: u64 = 2_000_000;

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
    ClassGetInstanceMethod,
    ClassGetClassMethod,
    MethodGetImplementation,
    MethodGetTypeEncoding,
    MethodSetImplementation,
    MethodExchange,
    ClassAddMethod,
    ClassReplaceMethod,
    ObjcGetProperty,
    ObjcSetProperty,
    GuardAcquire,
    GuardRelease,
    DyldImageCount,
    DyldImageName,
    NsSearchPath,
    NsLog,
    NewHandle,
    ReturnOne,
    BlockObjectAssign,
    BlockCopy,
    DispatchAsync,
    DispatchAfter,
    DispatchSync,
    DispatchTime,
    DispatchGroupAsync,
    DispatchGroupNotify,
    ObjcCopyStruct,
    DyldImageHeader,
    Gl,
    Dlsym,
    ObjcStoreWeak,
    NsTemporaryDirectory,
    Snprintf,
    Strnlen,
    CfStringCreate,
    CfDictionaryCreate,
    Socketpair,
    MathF64,
    MathF32,
    Printf,
    Puts,
    StdStringInit,
    StdStringAssign,
    StdStringAssignChar,
    StdStringCopy,
    StdStringCompare,
    Vsnprintf,
    SecItemNotFound,
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
        "___cxa_atexit" | "_atexit" | "_pthread_mutex_init" | "_pthread_mutex_lock"
        | "_pthread_mutex_unlock" | "_pthread_mutex_destroy" | "_pthread_mutex_trylock"
        | "_pthread_mutexattr_init" | "_pthread_mutexattr_settype"
        | "_pthread_mutexattr_destroy" | "_pthread_cond_init" | "_pthread_cond_destroy"
        | "_pthread_cond_signal" | "_pthread_cond_broadcast" | "_OSSpinLockLock"
        | "_OSSpinLockUnlock" | "_pthread_rwlock_init" | "_pthread_rwlock_rdlock"
        | "_pthread_rwlock_wrlock" | "_pthread_rwlock_unlock" => Kind::ReturnZero,
        "_dlsym" => Kind::Dlsym,
        "_objc_storeWeak" => Kind::ObjcStoreWeak,
        "_NSTemporaryDirectory" => Kind::NsTemporaryDirectory,
        "_snprintf" => Kind::Snprintf,
        "_strnlen" => Kind::Strnlen,
        "_CFStringCreateWithCString" => Kind::CfStringCreate,
        "_CFDictionaryCreate" => Kind::CfDictionaryCreate,
        "_socketpair" => Kind::Socketpair,
        "_pow" | "_sin" | "_cos" | "_tan" | "_atan" | "_atan2" | "_sqrt" | "_exp" | "_log" | "_log10"
        | "_floor" | "_ceil" | "_fmod" | "_asin" | "_acos" => Kind::MathF64,
        "_powf" | "_sinf" | "_cosf" | "_tanf" | "_atanf" | "_atan2f" | "_sqrtf" | "_expf" | "_logf"
        | "_floorf" | "_ceilf" | "_fmodf" | "_asinf" | "_acosf" => Kind::MathF32,
        "_printf" => Kind::Printf,
        "_puts" => Kind::Puts,
        "__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEPKcm" => Kind::StdStringInit,
        "__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6assignEPKc" => Kind::StdStringAssign,
        "__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEaSEc" => Kind::StdStringAssignChar,
        "__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_" => Kind::StdStringCopy,
        "__ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEPKc" => Kind::StdStringCompare,
        "_vsnprintf" => Kind::Vsnprintf,
        "_SecItemCopyMatching" => Kind::SecItemNotFound,
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
        "_class_getInstanceMethod" => Kind::ClassGetInstanceMethod,
        "_class_getClassMethod" => Kind::ClassGetClassMethod,
        "_method_getImplementation" => Kind::MethodGetImplementation,
        "_method_getTypeEncoding" => Kind::MethodGetTypeEncoding,
        "_method_setImplementation" => Kind::MethodSetImplementation,
        "_method_exchangeImplementations" => Kind::MethodExchange,
        "_class_addMethod" => Kind::ClassAddMethod,
        "_class_replaceMethod" => Kind::ClassReplaceMethod,
        "_objc_getProperty" => Kind::ObjcGetProperty,
        "_objc_setProperty_atomic"
        | "_objc_setProperty_nonatomic"
        | "_objc_setProperty_atomic_copy"
        | "_objc_setProperty_nonatomic_copy"
        | "_objc_setProperty" => Kind::ObjcSetProperty,
        "___cxa_guard_acquire" => Kind::GuardAcquire,
        "___cxa_guard_release" => Kind::GuardRelease,
        "__dyld_image_count" => Kind::DyldImageCount,
        "__dyld_get_image_name" => Kind::DyldImageName,
        "_NSSearchPathForDirectoriesInDomains" => Kind::NsSearchPath,
        "_NSLog" => Kind::NsLog,
        "_SCNetworkReachabilityCreateWithAddress"
        | "_SCNetworkReachabilityCreateWithName"
        | "_CFRunLoopGetCurrent"
        | "_CFRunLoopGetMain" => Kind::NewHandle,
        "_SCNetworkReachabilitySetCallback"
        | "_SCNetworkReachabilityScheduleWithRunLoop"
        | "_SCNetworkReachabilityUnscheduleFromRunLoop"
        | "_SCNetworkReachabilitySetDispatchQueue" => Kind::ReturnOne,
        "__Block_object_assign" => Kind::BlockObjectAssign,
        "_CFRetain" => Kind::ReturnX0,
        "_Block_copy" | "_objc_retainBlock" => Kind::BlockCopy,
        "_Block_release" | "_CFRelease" => Kind::Free,
        "_objc_sync_enter" | "_objc_sync_exit" | "___cxa_guard_abort" | "__Block_object_dispose" => Kind::ReturnZero,
        "_dispatch_async" | "_dispatch_barrier_async" => Kind::DispatchAsync,
        "_dispatch_after" => Kind::DispatchAfter,
        "_dispatch_sync" | "_dispatch_barrier_sync" => Kind::DispatchSync,
        "_dispatch_time" => Kind::DispatchTime,
        "_dispatch_group_async" => Kind::DispatchGroupAsync,
        "_dispatch_group_notify" => Kind::DispatchGroupNotify,
        "_dispatch_queue_create"
        | "_dispatch_get_global_queue"
        | "_dispatch_group_create"
        | "_dispatch_semaphore_create" => Kind::NewHandle,
        "_dispatch_semaphore_wait"
        | "_dispatch_semaphore_signal"
        | "_dispatch_group_enter"
        | "_dispatch_group_leave"
        | "_dispatch_group_wait"
        | "_dispatch_release"
        | "_dispatch_retain" => Kind::ReturnZero,
        "_objc_copyStruct" => Kind::ObjcCopyStruct,
        "__dyld_get_image_header" => Kind::DyldImageHeader,
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
        // libc++ constructors/destructors return `this` on arm64; the app only
        // needs them not to crash while their state is not modelled.
        | "__ZNSt3__16localeC1Ev"
        | "__ZNSt3__16localeC1ERKS0_"
        | "__ZNSt3__16localeD1Ev"
        | "__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev"
        | "_sel_getName" => Kind::ReturnX0,
        "_objc_release" => Kind::Free,
        "_objc_storeStrong" => Kind::ObjcStoreStrong,
        "_objc_autoreleasePoolPush" | "_objc_autoreleasePoolPop" => Kind::ReturnZero,
        "_sel_registerName" | "_sel_getUid" => Kind::SelRegister,
        "_objc_getClass" | "_objc_lookUpClass" | "_objc_getRequiredClass" => Kind::ObjcGetClass,
        "_NSClassFromString" => Kind::NsClassFromString,
        "_NSStringFromClass" => Kind::NsStringFromClass,
        "_object_getClass" => Kind::ObjectGetClass,
        _ if name.starts_with("_gl") => Kind::Gl,
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
        "NSUserDefaults" => Some("NSMutableDictionary"),
        "CAEAGLLayer" => Some("CALayer"),
        "GLKView" => Some("UIView"),
        "GLKViewController" | "UIAlertController" => Some("UIViewController"),
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
    /// (owning class, selector) -> `Method` handle, and back.
    method_handles: HashMap<(u64, String), u64>,
    handle_info: HashMap<u64, (u64, String)>,
    /// The app swizzled a system method: (class, selector now holding a null
    /// implementation) -> the selector of the system implementation it stands for.
    sys_alias: HashMap<(u64, String), String>,
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

    /// The class that implements `sel` for `start` and the implementation.
    fn find_owner(&self, start: u64, sel: &str) -> Option<(u64, u64)> {
        let mut c = start;
        for _ in 0..64 {
            let info = self.classes.get(&c)?;
            if let Some(&imp) = info.methods.get(sel) {
                return Some((c, imp));
            }
            if info.superclass == 0 || info.superclass == c {
                return None;
            }
            c = info.superclass;
        }
        None
    }

    /// A `Method` handle (as returned by `class_getInstanceMethod`): a small
    /// guest struct { selector, type encoding, implementation }.
    fn method_handle(&mut self, mem: &mut Mem64, heap: &mut Heap, owner: u64, sel: &str) -> u64 {
        let key = (owner, sel.to_string());
        if let Some(&h) = self.method_handles.get(&key) {
            return h;
        }
        let imp = self.classes.get(&owner).and_then(|c| c.methods.get(sel)).copied().unwrap_or(0);
        let h = heap.alloc(24);
        mem.write_u64(h, self.sels.get(sel).copied().unwrap_or(0));
        mem.write_u64(h + 8, 0);
        mem.write_u64(h + 16, imp);
        self.method_handles.insert(key.clone(), h);
        self.handle_info.insert(h, key);
        h
    }

    /// Set the implementation of the method behind handle `h`; returns the old one.
    fn set_method_imp(&mut self, mem: &mut Mem64, h: u64, imp: u64) -> u64 {
        let Some((owner, sel)) = self.handle_info.get(&h).cloned() else { return 0 };
        let old = self
            .classes
            .get_mut(&owner)
            .and_then(|c| c.methods.insert(sel, imp))
            .unwrap_or(0);
        mem.write_u64(h + 16, imp);
        old
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

    /// The system selector behind a swizzled method whose original implementation
    /// we do not have (see `sys_alias`).
    fn sys_alias_of(&self, start: u64, sel: &str) -> Option<String> {
        let mut c = start;
        for _ in 0..64 {
            if let Some(a) = self.sys_alias.get(&(c, sel.to_string())) {
                return Some(a.clone());
            }
            let info = self.classes.get(&c)?;
            if info.superclass == 0 || info.superclass == c {
                return None;
            }
            c = info.superclass;
        }
        None
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

// ---------------------------------------------------------------------------
// Foundation objects
//
// Objects of Foundation classes (NSString, NSNumber, NSArray, ...) are made
// of a small block of guest memory holding the `isa` pointer, plus a Rust
// value in `Hle::objs`, keyed by the object's address.

#[derive(Clone, Copy, Debug)]
struct Num {
    i: i64,
    f: f64,
    float: bool,
    boolean: bool,
}

impl Num {
    fn int(i: i64) -> Num {
        Num { i, f: 0.0, float: false, boolean: false }
    }
    fn dbl(f: f64) -> Num {
        Num { i: 0, f, float: true, boolean: false }
    }
    fn from_bool(b: bool) -> Num {
        Num { i: b as i64, f: 0.0, float: false, boolean: true }
    }
    fn as_i64(&self) -> i64 {
        if self.float {
            self.f as i64
        } else {
            self.i
        }
    }
    fn as_f64(&self) -> f64 {
        if self.float {
            self.f
        } else {
            self.i as f64
        }
    }
}

#[derive(Clone, Debug)]
enum Obj {
    Str(String),
    Num(Num),
    Array(Vec<u64>),
    /// Key/value pairs in insertion order.
    Dict(Vec<(u64, u64)>),
    Url(String),
    Data(Vec<u8>),
    /// An object of a system class that carries no data (a singleton, a pool).
    Plain,
}

/// Screen size we pretend to have, in points (an iPhone 6s in landscape).
const SCREEN_W: f64 = 667.0;
const SCREEN_H: f64 = 375.0;
const SCREEN_SCALE: f64 = 2.0;
/// `NSNotFound`.
const NS_NOT_FOUND: u64 = i64::MAX as u64;
/// Pretend location of the app's sandbox.
const SANDBOX: &str = "/var/mobile/Containers/Data/Application/00000000-0000-0000-0000-000000000000";

/// State of the OpenGL ES context we made for the app.
struct GlState {
    ctx: GLContext,
    /// The renderbuffer that stands for the screen (from `renderbufferStorage:fromDrawable:`).
    drawable_rb: u32,
    present_fbo: u32,
    size: (i32, i32),
    array_buf: u32,
    element_buf: u32,
    /// Buffers mapped with glMapBufferRange: target -> (staging address, offset, length).
    mapped: HashMap<u32, (u64, i64, i64)>,
    presented: u64,
}

/// Something the main run loop will do later.
#[derive(Clone, Debug)]
enum Action {
    /// `[target selector:args...]`
    Send { target: u64, sel: u64, args: Vec<u64> },
    /// Call a block with these arguments.
    Block { block: u64, args: Vec<u64> },
}

/// A timer-like entry of the run loop (NSTimer, dispatch_after, ...).
#[derive(Clone, Debug)]
struct Scheduled {
    /// Virtual time (ns) at which it is due.
    due: u64,
    /// 0 = once, otherwise the repeat interval (ns).
    repeat: u64,
    action: Action,
    /// Object that can cancel it (the NSTimer), or 0.
    handle: u64,
}

fn fnv_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// All "high-level emulation" state: the imported functions and their
/// bookkeeping.
struct Hle<'a> {
    names: Vec<String>,
    kinds: Vec<Kind>,
    heap: Heap,
    objc: Objc,
    /// Rust-side state of Foundation objects, by object address.
    objs: HashMap<u64, Obj>,
    singletons: HashMap<String, u64>,
    cstr_cache: HashMap<u64, u64>,
    /// Threads the app asked for: (target, selector, argument).
    pending_threads: Vec<(u64, u64, u64)>,
    thread_inits: HashMap<u64, (u64, u64, u64)>,
    bundle_path: String,
    bundle_id: String,
    exe_name: String,
    exe_path_ptr: u64,
    seen_logs: HashSet<String>,
    /// String-valued Info.plist entries.
    info_plist: HashMap<String, String>,
    fs: &'a Fs,
    window: &'a mut Window,
    options: &'a Options,
    gl: Option<GlState>,
    /// The current EAGLContext object (0 = none).
    current_ctx: u64,
    gl_err: u32,
    gl_err_logged: u32,
    gl_strings: HashMap<u32, u64>,
    unpack_align: i32,
    /// Real guest path of the app bundle in the emulated file system.
    bundle_guest: String,
    /// Files the app wrote this session.
    overlay: HashMap<String, Vec<u8>>,
    /// Generic UIKit properties: (object, name) -> [X2, D0, D1, D2, D3].
    props: HashMap<(u64, String), [u64; 5]>,
    visible_windows: Vec<u64>,
    /// Virtual time in nanoseconds.
    vtime: u64,
    scheduled: Vec<Scheduled>,
    /// Display links: (link object, target, selector).
    links: Vec<(u64, u64, u64)>,
    total_calls: u64,
    auto_seen: HashMap<String, u64>,
    data_ptrs: HashMap<u64, u64>,
    image_base: u64,
    seen: HashSet<u32>,
    counts: HashMap<u32, u64>,
    calls: u64,
    recent: VecDeque<String>,
    clock: u64,
    once_done: HashSet<u64>,
    tsd: HashMap<u64, u64>,
    next_key: u64,
}

impl<'a> Hle<'a> {
    fn alloc(&mut self, size: u64) -> u64 {
        self.heap.alloc(size)
    }

    /// A zeroed instance of `cls` (as `+alloc` would make).
    fn alloc_instance(&mut self, mem: &mut Mem64, cls: u64) -> u64 {
        let size = self.objc.classes.get(&cls).map_or(0x40, |c| c.instance_size).max(16);
        let p = self.heap.alloc(size);
        mem.write_u64(p, cls);
        self.init_storage(p, cls);
        p
    }

    /// A constant-string object (like an `@"..."` literal) holding `text`.
    fn make_string(&mut self, mem: &mut Mem64, text: &str) -> u64 {
        self.new_string(mem, text)
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
        // Selector trace for crash reports (shown next to the imported calls).
        {
            let traced = self.class_name(cls);
            if self.recent.len() >= 80 {
                self.recent.pop_front();
            }
            self.recent.push_back(format!("  msg [{traced} {sel}] recv={recv:#x}"));
        }
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
            if imp != 0 {
                // Tail call: the method returns straight to our caller.
                cpu.set_pc(imp);
                return Flow::Continue;
            }
            // A swizzled method whose original is a system method: run our
            // implementation of that one (or answer nil if there is none).
            if let Some(orig) = self.objc.sys_alias_of(start.unwrap_or(cls), &sel) {
                if self.hle_message(cpu, mem, recv, cls, &orig, lr) {
                    return Flow::Continue;
                }
                self.log_once(format!("system original -[{} {orig}] of swizzled {sel} is not implemented", self.class_name(cls)));
            }
            cpu.set_reg(0, 0);
            return Flow::Continue;
        }
        if self.hle_message(cpu, mem, recv, cls, &sel, lr) {
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

    // ----- Foundation object helpers -----

    fn new_obj(&mut self, mem: &mut Mem64, class: &str, o: Obj) -> u64 {
        let (cls, _) = self.objc.system_class(mem, &mut self.heap, class);
        let p = self.heap.alloc(0x40);
        mem.write_u64(p, cls);
        self.objs.insert(p, o);
        p
    }

    fn new_string(&mut self, mem: &mut Mem64, s: &str) -> u64 {
        self.new_obj(mem, "NSString", Obj::Str(s.to_string()))
    }

    fn class_name(&self, c: u64) -> String {
        self.objc.classes.get(&c).map(|i| i.name.clone()).unwrap_or_else(|| "?".to_string())
    }

    /// Text of an NSString (or NSURL path), whether made by us or an `@"..."`
    /// literal from the app.
    fn str_of(&self, mem: &Mem64, obj: u64) -> Option<String> {
        match self.objs.get(&obj) {
            Some(Obj::Str(s)) | Some(Obj::Url(s)) => return Some(s.clone()),
            Some(_) => return None,
            None => {}
        }
        let cls = self.objc.class_of(mem, obj)?;
        if self.objc.classes.get(&cls)?.name == "NSConstantString" {
            read_cfstring(mem, obj)
        } else {
            None
        }
    }

    fn num_of(&self, obj: u64) -> Option<Num> {
        match self.objs.get(&obj) {
            Some(Obj::Num(n)) => Some(*n),
            _ => None,
        }
    }

    fn alloc_cstr(&mut self, mem: &mut Mem64, s: &str) -> u64 {
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        let p = self.heap.alloc(b.len() as u64);
        mem.write_slice(p, &b);
        p
    }

    /// A C string with the text of `obj`, kept alive for the whole run.
    fn utf8_ptr(&mut self, mem: &mut Mem64, obj: u64) -> u64 {
        if let Some(p) = self.cstr_cache.get(&obj) {
            return *p;
        }
        let s = self.str_of(mem, obj).unwrap_or_default();
        let p = self.alloc_cstr(mem, &s);
        self.cstr_cache.insert(obj, p);
        p
    }

    /// What `%@` / `-description` prints for an object.
    fn description(&self, mem: &Mem64, obj: u64) -> String {
        if obj == 0 {
            return "(null)".to_string();
        }
        match self.objs.get(&obj) {
            Some(Obj::Str(s)) | Some(Obj::Url(s)) => return s.clone(),
            Some(Obj::Data(d)) => return format!("<{} bytes>", d.len()),
            Some(Obj::Num(n)) => {
                return if n.float { format!("{}", n.f) } else { n.i.to_string() };
            }
            Some(Obj::Array(v)) => {
                let parts: Vec<String> = v.iter().map(|o| self.description(mem, *o)).collect();
                return format!("({})", parts.join(", "));
            }
            Some(Obj::Dict(d)) => {
                let parts: Vec<String> = d
                    .iter()
                    .map(|(k, v)| format!("{} = {};", self.description(mem, *k), self.description(mem, *v)))
                    .collect();
                return format!("{{{}}}", parts.join(" "));
            }
            _ => {}
        }
        if let Some(s) = self.str_of(mem, obj) {
            return s;
        }
        if let Some(info) = self.objc.classes.get(&obj) {
            return info.name.clone();
        }
        let cname = self
            .objc
            .class_of(mem, obj)
            .map(|c| self.class_name(c))
            .unwrap_or_else(|| "?".to_string());
        format!("<{cname}: {obj:#x}>")
    }

    fn obj_eq(&self, mem: &Mem64, a: u64, b: u64) -> bool {
        if a == b {
            return true;
        }
        if let (Some(x), Some(y)) = (self.num_of(a), self.num_of(b)) {
            return if x.float || y.float { x.as_f64() == y.as_f64() } else { x.i == y.i };
        }
        match (self.str_of(mem, a), self.str_of(mem, b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    fn dict_find(&self, mem: &Mem64, d: u64, key: u64) -> Option<usize> {
        match self.objs.get(&d) {
            Some(Obj::Dict(v)) => v.iter().position(|(k, _)| self.obj_eq(mem, *k, key)),
            _ => None,
        }
    }

    fn dict_get(&self, mem: &Mem64, d: u64, key: u64) -> u64 {
        match (self.dict_find(mem, d, key), self.objs.get(&d)) {
            (Some(i), Some(Obj::Dict(v))) => v[i].1,
            _ => 0,
        }
    }

    fn dict_set(&mut self, mem: &Mem64, d: u64, key: u64, val: u64) {
        let idx = self.dict_find(mem, d, key);
        if let Some(Obj::Dict(v)) = self.objs.get_mut(&d) {
            match idx {
                Some(i) => v[i].1 = val,
                None => v.push((key, val)),
            }
        }
    }

    /// One shared object per `key` (like `+[NSUserDefaults standardUserDefaults]`).
    fn singleton(&mut self, mem: &mut Mem64, key: &str, class: &str, o: Obj) -> u64 {
        if let Some(p) = self.singletons.get(key) {
            return *p;
        }
        let p = self.new_obj(mem, class, o);
        self.singletons.insert(key.to_string(), p);
        p
    }

    fn log_once(&mut self, text: String) {
        if self.seen_logs.insert(text.clone()) {
            log!("[a64] {text}");
        }
    }

    /// Give a freshly allocated instance of a Foundation class its storage.
    fn init_storage(&mut self, obj: u64, cls: u64) {
        let mut c = cls;
        for _ in 0..16 {
            let Some(info) = self.objc.classes.get(&c) else { break };
            let o = match info.name.as_str() {
                "NSString" => Some(Obj::Str(String::new())),
                "NSArray" => Some(Obj::Array(Vec::new())),
                "NSDictionary" => Some(Obj::Dict(Vec::new())),
                "NSNumber" => Some(Obj::Num(Num::int(0))),
                "NSURL" => Some(Obj::Url(String::new())),
                "NSData" => Some(Obj::Data(Vec::new())),
                _ => None,
            };
            if let Some(o) = o {
                self.objs.insert(obj, o);
                return;
            }
            if info.superclass == 0 || info.superclass == c {
                break;
            }
            c = info.superclass;
        }
    }

    /// printf-style formatting as `+[NSString stringWithFormat:]` does it.
    /// Variadic arguments are 8-byte slots on the stack (Apple arm64 ABI).
    fn ns_format(&self, mem: &Mem64, fmt: &str, sp: u64) -> String {
        let chars: Vec<char> = fmt.chars().collect();
        let mut out = String::new();
        let mut argi: u64 = 0;
        let mut next = || -> u64 {
            let v = mem.read_u64(sp.wrapping_add(argi * 8)).unwrap_or(0);
            argi += 1;
            v
        };
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            i += 1;
            if c != '%' {
                out.push(c);
                continue;
            }
            let (mut left, mut zero, mut plus, mut space, mut alt) = (false, false, false, false, false);
            while i < chars.len() && "-0+ #".contains(chars[i]) {
                match chars[i] {
                    '-' => left = true,
                    '0' => zero = true,
                    '+' => plus = true,
                    ' ' => space = true,
                    _ => alt = true,
                }
                i += 1;
            }
            let mut width = 0usize;
            if i < chars.len() && chars[i] == '*' {
                let w = next() as i32;
                if w < 0 {
                    left = true;
                }
                width = (w.unsigned_abs() as usize).min(1000);
                i += 1;
            } else {
                while i < chars.len() && chars[i].is_ascii_digit() {
                    width = (width * 10 + chars[i].to_digit(10).unwrap() as usize).min(1000);
                    i += 1;
                }
            }
            let mut prec: Option<usize> = None;
            if i < chars.len() && chars[i] == '.' {
                i += 1;
                let mut p = 0usize;
                if i < chars.len() && chars[i] == '*' {
                    p = (next() as i32).max(0) as usize;
                    i += 1;
                } else {
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        p = (p * 10 + chars[i].to_digit(10).unwrap() as usize).min(1000);
                        i += 1;
                    }
                }
                prec = Some(p);
            }
            let mut long = 0u32;
            while i < chars.len() && "hlzjtqL".contains(chars[i]) {
                if chars[i] != 'h' {
                    long += 1;
                }
                i += 1;
            }
            if i >= chars.len() {
                break;
            }
            let conv = chars[i];
            i += 1;
            let body: String = match conv {
                '%' => "%".to_string(),
                '@' => {
                    let o = next();
                    self.description(mem, o)
                }
                'd' | 'i' => {
                    let v = next();
                    let n: i64 = if long > 0 { v as i64 } else { v as i32 as i64 };
                    let mut digits = n.unsigned_abs().to_string();
                    if let Some(p) = prec {
                        while digits.len() < p {
                            digits.insert(0, '0');
                        }
                    }
                    let sign = if n < 0 { "-" } else if plus { "+" } else if space { " " } else { "" };
                    format!("{sign}{digits}")
                }
                'u' | 'x' | 'X' | 'o' => {
                    let v = next();
                    let n: u64 = if long > 0 { v } else { v as u32 as u64 };
                    let mut d = match conv {
                        'u' => n.to_string(),
                        'x' => format!("{n:x}"),
                        'X' => format!("{n:X}"),
                        _ => format!("{n:o}"),
                    };
                    if let Some(p) = prec {
                        while d.len() < p {
                            d.insert(0, '0');
                        }
                    }
                    if alt && n != 0 {
                        match conv {
                            'x' => d = format!("0x{d}"),
                            'X' => d = format!("0X{d}"),
                            'o' => d = format!("0{d}"),
                            _ => {}
                        }
                    }
                    d
                }
                'c' => {
                    let v = next();
                    char::from_u32(v as u32).map(|c| c.to_string()).unwrap_or_default()
                }
                's' => {
                    let p = next();
                    let mut s = mem
                        .read_c_bytes(p, 1 << 16)
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_else(|| "(null)".to_string());
                    if let Some(pr) = prec {
                        s = s.chars().take(pr).collect();
                    }
                    s
                }
                'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                    let f = f64::from_bits(next());
                    let sign = if f.is_sign_negative() && !f.is_nan() {
                        "-"
                    } else if plus {
                        "+"
                    } else if space {
                        " "
                    } else {
                        ""
                    };
                    let a = f.abs();
                    let s = match conv {
                        'f' | 'F' => format!("{:.*}", prec.unwrap_or(6), a),
                        'e' => format!("{:.*e}", prec.unwrap_or(6), a),
                        'E' => format!("{:.*E}", prec.unwrap_or(6), a),
                        _ => format!("{a}"),
                    };
                    format!("{sign}{s}")
                }
                'p' => {
                    let v = next();
                    format!("0x{v:x}")
                }
                other => format!("%{other}"),
            };
            let numeric = matches!(conv, 'd' | 'i' | 'u' | 'x' | 'X' | 'o' | 'f' | 'F' | 'e' | 'E' | 'g' | 'G');
            let len = body.chars().count();
            if len >= width {
                out.push_str(&body);
            } else if left {
                out.push_str(&body);
                out.push_str(&" ".repeat(width - len));
            } else if zero && numeric {
                let (sign, digits) = if body.starts_with(|c: char| c == '-' || c == '+' || c == ' ') {
                    body.split_at(1)
                } else {
                    ("", body.as_str())
                };
                out.push_str(sign);
                out.push_str(&"0".repeat(width - len));
                out.push_str(digits);
            } else {
                out.push_str(&" ".repeat(width - len));
                out.push_str(&body);
            }
        }
        out
    }

    /// Foundation / UIKit methods that we implement ourselves, looked up
    /// along the system part of the class chain. `false` = nobody does.
    #[allow(clippy::too_many_arguments)]
    fn hle_message(
        &mut self,
        cpu: &mut Cpu64,
        mem: &mut Mem64,
        recv: u64,
        cls: u64,
        sel: &str,
        lr: u64,
    ) -> bool {
        let mut chain: Vec<(String, bool)> = Vec::new();
        let mut c = cls;
        for _ in 0..32 {
            let Some(info) = self.objc.classes.get(&c) else { break };
            chain.push((info.name.clone(), info.is_meta));
            if info.superclass == 0 || info.superclass == c {
                break;
            }
            c = info.superclass;
        }
        for (name, class_level) in &chain {
            if self.hle_call(cpu, mem, recv, name, *class_level, sel, lr) {
                return true;
            }
        }
        // Plain property accessors of UIKit / Core Animation objects.
        let uikit = chain
            .iter()
            .any(|(n, _)| matches!(n.as_str(), "UIResponder" | "CALayer" | "CADisplayLink" | "NSTimer"));
        if uikit && !chain[0].1 && self.auto_property(cpu, recv, sel) {
            return true;
        }
        false
    }

    /// One class's share of `hle_message`. `class_level` = the message goes
    /// to the class itself (a "+" method).
    #[allow(clippy::too_many_arguments, unreachable_code)]
    fn hle_call(
        &mut self,
        cpu: &mut Cpu64,
        mem: &mut Mem64,
        recv: u64,
        name: &str,
        class_level: bool,
        sel: &str,
        lr: u64,
    ) -> bool {
        let _ = lr;
        let (a2, a3, a4) = (cpu.reg(2), cpu.reg(3), cpu.reg(4));
        let sp = cpu.sp();
        macro_rules! ret {
            ($v:expr) => {{
                let v: u64 = $v;
                cpu.set_reg(0, v);
                return true;
            }};
        }
        match (name, class_level, sel) {
            // ----- NSObject -----
            ("NSObject", false, "description") | ("NSObject", false, "debugDescription") => {
                let d = self.description(mem, recv);
                let o = self.new_string(mem, &d);
                ret!(o)
            }

            // ----- NSString -----
            ("NSString", true, "stringWithUTF8String:") | ("NSString", true, "stringWithCString:encoding:") => {
                let s = Objc::cstr(mem, a2).unwrap_or_default();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", true, "stringWithFormat:") => {
                let fmt = self.str_of(mem, a2).unwrap_or_default();
                let s = self.ns_format(mem, &fmt, sp);
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", true, "string") | ("NSString", true, "stringWithCapacity:") => {
                let o = self.new_string(mem, "");
                ret!(o)
            }
            ("NSString", true, "stringWithString:") => {
                let s = self.str_of(mem, a2).unwrap_or_default();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "initWithString:") => {
                let s = self.str_of(mem, a2).unwrap_or_default();
                self.objs.insert(recv, Obj::Str(s));
                ret!(recv)
            }
            ("NSString", false, "initWithUTF8String:") => {
                let s = Objc::cstr(mem, a2).unwrap_or_default();
                self.objs.insert(recv, Obj::Str(s));
                ret!(recv)
            }
            ("NSString", false, "initWithFormat:") => {
                let fmt = self.str_of(mem, a2).unwrap_or_default();
                let s = self.ns_format(mem, &fmt, sp);
                self.objs.insert(recv, Obj::Str(s));
                ret!(recv)
            }
            ("NSString", false, "initWithCapacity:") => ret!(recv),
            ("NSString", false, "UTF8String")
            | ("NSString", false, "cStringUsingEncoding:")
            | ("NSString", false, "fileSystemRepresentation") => {
                let p = self.utf8_ptr(mem, recv);
                ret!(p)
            }
            ("NSString", false, "length") => {
                let n = self.str_of(mem, recv).map_or(0, |s| s.encode_utf16().count());
                ret!(n as u64)
            }
            ("NSString", false, "isEqualToString:") | ("NSString", false, "isEqual:") => {
                let eq = match (self.str_of(mem, recv), self.str_of(mem, a2)) {
                    (Some(x), Some(y)) => x == y,
                    _ => recv == a2,
                };
                ret!(eq as u64)
            }
            ("NSString", false, "hash") => {
                let h = self.str_of(mem, recv).map_or(recv, |s| fnv_hash(&s));
                ret!(h)
            }
            ("NSString", false, "description")
            | ("NSString", false, "copy")
            | ("NSString", false, "copyWithZone:")
            | ("NSString", false, "stringByStandardizingPath") => ret!(recv),
            ("NSString", false, "mutableCopy") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let o = self.new_obj(mem, "NSMutableString", Obj::Str(s));
                ret!(o)
            }
            ("NSString", false, "stringByAppendingFormat:") => {
                // Variadic arguments are on the stack, like stringWithFormat:.
                let fmt = self.str_of(mem, a2).unwrap_or_default();
                let tail = self.ns_format(mem, &fmt, sp);
                let s = self.str_of(mem, recv).unwrap_or_default() + &tail;
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "stringByAppendingString:") => {
                let s = self.str_of(mem, recv).unwrap_or_default() + &self.str_of(mem, a2).unwrap_or_default();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "stringByAppendingPathComponent:") => {
                let base = self.str_of(mem, recv).unwrap_or_default();
                let part = self.str_of(mem, a2).unwrap_or_default();
                let s = if base.is_empty() {
                    part
                } else if base.ends_with('/') {
                    format!("{base}{part}")
                } else {
                    format!("{base}/{part}")
                };
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "lastPathComponent") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let last = s.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_string();
                let o = self.new_string(mem, &last);
                ret!(o)
            }
            ("NSString", false, "pathExtension") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let last = s.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_string();
                let ext = last.rsplit_once('.').map(|(_, e)| e.to_string()).unwrap_or_default();
                let o = self.new_string(mem, &ext);
                ret!(o)
            }
            ("NSString", false, "stringByDeletingLastPathComponent") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let dir = s.trim_end_matches('/').rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default();
                let o = self.new_string(mem, &dir);
                ret!(o)
            }
            ("NSString", false, "hasPrefix:") => {
                let (s, p) = (self.str_of(mem, recv), self.str_of(mem, a2));
                ret!(matches!((s, p), (Some(s), Some(p)) if s.starts_with(&p)) as u64)
            }
            ("NSString", false, "hasSuffix:") => {
                let (s, p) = (self.str_of(mem, recv), self.str_of(mem, a2));
                ret!(matches!((s, p), (Some(s), Some(p)) if s.ends_with(&p)) as u64)
            }
            ("NSString", false, "containsString:") => {
                let (s, p) = (self.str_of(mem, recv), self.str_of(mem, a2));
                ret!(matches!((s, p), (Some(s), Some(p)) if s.contains(&p)) as u64)
            }
            ("NSString", false, "rangeOfString:") => {
                let hay = self.str_of(mem, recv).unwrap_or_default();
                let needle = self.str_of(mem, a2).unwrap_or_default();
                match hay.find(&needle) {
                    Some(b) => {
                        cpu.set_reg(0, hay[..b].encode_utf16().count() as u64);
                        cpu.set_reg(1, needle.encode_utf16().count() as u64);
                    }
                    None => {
                        cpu.set_reg(0, NS_NOT_FOUND);
                        cpu.set_reg(1, 0);
                    }
                }
                return true;
            }
            ("NSString", false, "substringFromIndex:") => {
                let s: String = self.str_of(mem, recv).unwrap_or_default().chars().skip(a2 as usize).collect();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "substringToIndex:") => {
                let s: String = self.str_of(mem, recv).unwrap_or_default().chars().take(a2 as usize).collect();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSString", false, "componentsSeparatedByString:") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let sep = self.str_of(mem, a2).unwrap_or_default();
                let parts: Vec<String> = if sep.is_empty() {
                    vec![s]
                } else {
                    s.split(sep.as_str()).map(|p| p.to_string()).collect()
                };
                let items: Vec<u64> = parts.iter().map(|p| self.new_string(mem, p)).collect();
                let arr = self.new_obj(mem, "NSArray", Obj::Array(items));
                ret!(arr)
            }
            ("NSString", false, "stringByReplacingOccurrencesOfString:withString:") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let from = self.str_of(mem, a2).unwrap_or_default();
                let to = self.str_of(mem, a3).unwrap_or_default();
                let r = if from.is_empty() { s } else { s.replace(&from, &to) };
                let o = self.new_string(mem, &r);
                ret!(o)
            }
            ("NSString", false, "lowercaseString") | ("NSString", false, "uppercaseString") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let r = if sel == "lowercaseString" { s.to_lowercase() } else { s.to_uppercase() };
                let o = self.new_string(mem, &r);
                ret!(o)
            }
            ("NSString", false, "intValue") | ("NSString", false, "integerValue") | ("NSString", false, "longLongValue") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let n: i64 = s.trim().split(|c: char| !(c.is_ascii_digit() || c == '-')).next().unwrap_or("").parse().unwrap_or(0);
                ret!(n as u64)
            }
            ("NSString", false, "doubleValue") | ("NSString", false, "floatValue") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let f: f64 = s.trim().parse().unwrap_or(0.0);
                if sel == "floatValue" {
                    cpu.set_f32(0, f as f32);
                } else {
                    cpu.set_f64(0, f);
                }
                return true;
            }
            ("NSString", false, "boolValue") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let t = s.trim().to_lowercase();
                ret!((t.starts_with('y') || t.starts_with('t') || t.starts_with('1')) as u64)
            }
            ("NSMutableString", false, "appendString:") => {
                let add = self.str_of(mem, a2).unwrap_or_default();
                if let Some(Obj::Str(s)) = self.objs.get_mut(&recv) {
                    s.push_str(&add);
                }
                self.cstr_cache.remove(&recv);
                return true;
            }
            ("NSMutableString", false, "appendFormat:") => {
                let fmt = self.str_of(mem, a2).unwrap_or_default();
                // a2 is the format; the arguments follow on the stack
                let add = self.ns_format(mem, &fmt, sp);
                if let Some(Obj::Str(s)) = self.objs.get_mut(&recv) {
                    s.push_str(&add);
                }
                self.cstr_cache.remove(&recv);
                return true;
            }
            ("NSMutableString", false, "setString:") => {
                let new = self.str_of(mem, a2).unwrap_or_default();
                self.objs.insert(recv, Obj::Str(new));
                self.cstr_cache.remove(&recv);
                return true;
            }

            // ----- NSNumber -----
            ("NSNumber", true, "numberWithBool:") => {
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::from_bool(a2 as u8 != 0)));
                ret!(o)
            }
            ("NSNumber", true, "numberWithInt:") | ("NSNumber", true, "numberWithChar:") | ("NSNumber", true, "numberWithShort:") => {
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::int(a2 as i32 as i64)));
                ret!(o)
            }
            ("NSNumber", true, "numberWithUnsignedInt:") => {
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::int(a2 as u32 as i64)));
                ret!(o)
            }
            ("NSNumber", true, "numberWithInteger:")
            | ("NSNumber", true, "numberWithLong:")
            | ("NSNumber", true, "numberWithLongLong:")
            | ("NSNumber", true, "numberWithUnsignedInteger:")
            | ("NSNumber", true, "numberWithUnsignedLong:")
            | ("NSNumber", true, "numberWithUnsignedLongLong:") => {
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::int(a2 as i64)));
                ret!(o)
            }
            ("NSNumber", true, "numberWithDouble:") => {
                let f = cpu.f64(0);
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::dbl(f)));
                ret!(o)
            }
            ("NSNumber", true, "numberWithFloat:") => {
                let f = cpu.f32(0) as f64;
                let o = self.new_obj(mem, "NSNumber", Obj::Num(Num::dbl(f)));
                ret!(o)
            }
            ("NSNumber", false, "boolValue") => {
                let n = self.num_of(recv).map_or(0, |n| n.as_i64());
                ret!((n != 0) as u64)
            }
            ("NSNumber", false, "intValue")
            | ("NSNumber", false, "integerValue")
            | ("NSNumber", false, "longValue")
            | ("NSNumber", false, "longLongValue")
            | ("NSNumber", false, "unsignedIntValue")
            | ("NSNumber", false, "unsignedIntegerValue")
            | ("NSNumber", false, "unsignedLongLongValue")
            | ("NSNumber", false, "charValue") => {
                let n = self.num_of(recv).map_or(0, |n| n.as_i64());
                ret!(n as u64)
            }
            ("NSNumber", false, "doubleValue") => {
                let f = self.num_of(recv).map_or(0.0, |n| n.as_f64());
                cpu.set_f64(0, f);
                return true;
            }
            ("NSNumber", false, "floatValue") => {
                let f = self.num_of(recv).map_or(0.0, |n| n.as_f64());
                cpu.set_f32(0, f as f32);
                return true;
            }
            ("NSNumber", false, "isEqualToNumber:") | ("NSNumber", false, "isEqual:") => {
                let eq = self.obj_eq(mem, recv, a2);
                ret!(eq as u64)
            }
            ("NSNumber", false, "hash") => {
                let h = self.num_of(recv).map_or(recv, |n| n.as_i64() as u64);
                ret!(h)
            }
            ("NSNumber", false, "description") | ("NSNumber", false, "stringValue") => {
                let d = self.description(mem, recv);
                let o = self.new_string(mem, &d);
                ret!(o)
            }
            ("NSNumber", false, "copy") | ("NSNumber", false, "copyWithZone:") => ret!(recv),

            // ----- NSArray / NSMutableArray -----
            ("NSArray", true, "array") => {
                let c = self.class_name(recv);
                let o = self.new_obj(mem, &c, Obj::Array(Vec::new()));
                ret!(o)
            }
            ("NSArray", true, "arrayWithObject:") => {
                let c = self.class_name(recv);
                let o = self.new_obj(mem, &c, Obj::Array(vec![a2]));
                ret!(o)
            }
            ("NSArray", true, "arrayWithObjects:") => {
                let c = self.class_name(recv);
                let mut items = Vec::new();
                let mut o = a2;
                let mut slot = 0u64;
                while o != 0 && slot < 4000 {
                    items.push(o);
                    o = mem.read_u64(sp.wrapping_add(slot * 8)).unwrap_or(0);
                    slot += 1;
                }
                let arr = self.new_obj(mem, &c, Obj::Array(items));
                ret!(arr)
            }
            ("NSArray", true, "arrayWithArray:") => {
                let c = self.class_name(recv);
                let items = match self.objs.get(&a2) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let o = self.new_obj(mem, &c, Obj::Array(items));
                ret!(o)
            }
            ("NSArray", false, "countByEnumeratingWithState:objects:count:") => {
                // for-in loops. a2 = NSFastEnumerationState { state, itemsPtr,
                // mutationsPtr, extra[5] }, a3 = caller's buffer, a4 = its length.
                let items = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let start = mem.read_u64(a2).unwrap_or(0) as usize;
                let n = items.len().saturating_sub(start).min(a4 as usize);
                for i in 0..n {
                    mem.write_u64(a3.wrapping_add(i as u64 * 8), items[start + i]);
                }
                mem.write_u64(a2, (start + n) as u64);
                mem.write_u64(a2.wrapping_add(8), a3);
                mem.write_u64(a2.wrapping_add(16), a2.wrapping_add(24));
                ret!(n as u64)
            }
            ("NSDictionary", false, "countByEnumeratingWithState:objects:count:") => {
                // Dictionaries are enumerated by key.
                let keys: Vec<u64> = match self.objs.get(&recv) {
                    Some(Obj::Dict(v)) => v.iter().map(|(k, _)| *k).collect(),
                    _ => Vec::new(),
                };
                let start = mem.read_u64(a2).unwrap_or(0) as usize;
                let n = keys.len().saturating_sub(start).min(a4 as usize);
                for i in 0..n {
                    mem.write_u64(a3.wrapping_add(i as u64 * 8), keys[start + i]);
                }
                mem.write_u64(a2, (start + n) as u64);
                mem.write_u64(a2.wrapping_add(8), a3);
                mem.write_u64(a2.wrapping_add(16), a2.wrapping_add(24));
                ret!(n as u64)
            }
            ("NSArray", false, "initWithObjects:") => {
                // object1, object2, ..., nil (variadic arguments are on the stack)
                let mut items = Vec::new();
                let mut o = a2;
                let mut slot = 0u64;
                while o != 0 && slot < 4000 {
                    items.push(o);
                    o = mem.read_u64(sp.wrapping_add(slot * 8)).unwrap_or(0);
                    slot += 1;
                }
                self.objs.insert(recv, Obj::Array(items));
                ret!(recv)
            }
            ("NSArray", false, "initWithCapacity:") | ("NSArray", false, "init") => ret!(recv),
            ("NSArray", false, "count") => {
                let n = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.len(),
                    _ => 0,
                };
                ret!(n as u64)
            }
            ("NSArray", false, "objectAtIndex:") | ("NSArray", false, "objectAtIndexedSubscript:") => {
                let o = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.get(a2 as usize).copied().unwrap_or(0),
                    _ => 0,
                };
                ret!(o)
            }
            ("NSArray", false, "firstObject") | ("NSArray", false, "lastObject") => {
                let o = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => {
                        let e = if sel == "firstObject" { v.first() } else { v.last() };
                        e.copied().unwrap_or(0)
                    }
                    _ => 0,
                };
                ret!(o)
            }
            ("NSArray", false, "containsObject:") | ("NSArray", false, "indexOfObject:") => {
                let items = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let idx = items.iter().position(|o| self.obj_eq(mem, *o, a2));
                if sel == "containsObject:" {
                    ret!(idx.is_some() as u64)
                }
                ret!(idx.map_or(NS_NOT_FOUND, |i| i as u64))
            }
            ("NSArray", false, "addObject:") => {
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    v.push(a2);
                }
                return true;
            }
            ("NSArray", false, "addObjectsFromArray:") => {
                let more = match self.objs.get(&a2) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    v.extend(more);
                }
                return true;
            }
            ("NSArray", false, "insertObject:atIndex:") => {
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    let at = (a3 as usize).min(v.len());
                    v.insert(at, a2);
                }
                return true;
            }
            ("NSArray", false, "removeAllObjects") => {
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    v.clear();
                }
                return true;
            }
            ("NSArray", false, "removeLastObject") => {
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    v.pop();
                }
                return true;
            }
            ("NSArray", false, "removeObjectAtIndex:") => {
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    if (a2 as usize) < v.len() {
                        v.remove(a2 as usize);
                    }
                }
                return true;
            }
            ("NSArray", false, "removeObject:") => {
                let items = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let keep: Vec<u64> = items.into_iter().filter(|o| !self.obj_eq(mem, *o, a2)).collect();
                if let Some(Obj::Array(v)) = self.objs.get_mut(&recv) {
                    *v = keep;
                }
                return true;
            }
            ("NSArray", false, "copy") | ("NSArray", false, "mutableCopy") | ("NSArray", false, "copyWithZone:") => {
                let items = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let c = if sel == "mutableCopy" { "NSMutableArray" } else { "NSArray" };
                let o = self.new_obj(mem, c, Obj::Array(items));
                ret!(o)
            }
            ("NSArray", false, "componentsJoinedByString:") => {
                let sep = self.str_of(mem, a2).unwrap_or_default();
                let items = match self.objs.get(&recv) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let parts: Vec<String> = items.iter().map(|o| self.description(mem, *o)).collect();
                let o = self.new_string(mem, &parts.join(&sep));
                ret!(o)
            }

            // ----- NSDictionary / NSMutableDictionary / NSUserDefaults -----
            ("NSDictionary", true, "dictionary") => {
                let c = self.class_name(recv);
                let o = self.new_obj(mem, &c, Obj::Dict(Vec::new()));
                ret!(o)
            }
            ("NSDictionary", true, "dictionaryWithObject:forKey:") => {
                let c = self.class_name(recv);
                let d = self.new_obj(mem, &c, Obj::Dict(Vec::new()));
                self.dict_set(mem, d, a3, a2);
                ret!(d)
            }
            ("NSDictionary", true, "dictionaryWithDictionary:") => {
                let c = self.class_name(recv);
                let pairs = match self.objs.get(&a2) {
                    Some(Obj::Dict(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let o = self.new_obj(mem, &c, Obj::Dict(pairs));
                ret!(o)
            }
            ("NSDictionary", true, "dictionaryWithObjectsAndKeys:") => {
                // object1, key1, object2, key2, ..., nil
                let c = self.class_name(recv);
                let mut pairs = Vec::new();
                let (mut obj, mut slot) = (a2, 0u64);
                while obj != 0 && slot < 4000 {
                    let key = mem.read_u64(sp.wrapping_add(slot * 8)).unwrap_or(0);
                    if key == 0 {
                        break;
                    }
                    pairs.push((key, obj));
                    obj = mem.read_u64(sp.wrapping_add((slot + 1) * 8)).unwrap_or(0);
                    slot += 2;
                }
                let d = self.new_obj(mem, &c, Obj::Dict(Vec::new()));
                for (k, v) in pairs {
                    self.dict_set(mem, d, k, v);
                }
                ret!(d)
            }
            ("NSDictionary", true, "dictionaryWithObjects:forKeys:count:") => {
                // x2 = id objects[], x3 = id keys[], x4 = count
                let c = self.class_name(recv);
                let d = self.new_obj(mem, &c, Obj::Dict(Vec::new()));
                for i in 0..a4.min(4096) {
                    let obj = mem.read_u64(a2.wrapping_add(i * 8)).unwrap_or(0);
                    let key = mem.read_u64(a3.wrapping_add(i * 8)).unwrap_or(0);
                    if key != 0 {
                        self.dict_set(mem, d, key, obj);
                    }
                }
                ret!(d)
            }
            ("NSDictionary", true, "dictionaryWithObjects:forKeys:") => {
                // x2 = NSArray of objects, x3 = NSArray of keys
                let c = self.class_name(recv);
                let objs = match self.objs.get(&a2) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let keys = match self.objs.get(&a3) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let d = self.new_obj(mem, &c, Obj::Dict(Vec::new()));
                for (k, o) in keys.iter().zip(objs.iter()) {
                    self.dict_set(mem, d, *k, *o);
                }
                ret!(d)
            }
            ("NSDictionary", false, "initWithObjects:forKeys:") => {
                let objs = match self.objs.get(&a2) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let keys = match self.objs.get(&a3) {
                    Some(Obj::Array(v)) => v.clone(),
                    _ => Vec::new(),
                };
                for (k, o) in keys.iter().zip(objs.iter()) {
                    self.dict_set(mem, recv, *k, *o);
                }
                ret!(recv)
            }
            ("NSDictionary", false, "initWithObjects:forKeys:count:") => {
                for i in 0..a4.min(4096) {
                    let obj = mem.read_u64(a2.wrapping_add(i * 8)).unwrap_or(0);
                    let key = mem.read_u64(a3.wrapping_add(i * 8)).unwrap_or(0);
                    if key != 0 {
                        self.dict_set(mem, recv, key, obj);
                    }
                }
                ret!(recv)
            }
            ("NSDictionary", false, "initWithCapacity:") | ("NSDictionary", false, "init") => ret!(recv),
            ("NSDictionary", false, "count") => {
                let n = match self.objs.get(&recv) {
                    Some(Obj::Dict(v)) => v.len(),
                    _ => 0,
                };
                ret!(n as u64)
            }
            ("NSDictionary", false, "objectForKey:")
            | ("NSDictionary", false, "objectForKeyedSubscript:")
            | ("NSDictionary", false, "valueForKey:") => {
                let v = self.dict_get(mem, recv, a2);
                ret!(v)
            }
            ("NSDictionary", false, "allKeys") | ("NSDictionary", false, "allValues") => {
                let items: Vec<u64> = match self.objs.get(&recv) {
                    Some(Obj::Dict(v)) => v.iter().map(|(k, val)| if sel == "allKeys" { *k } else { *val }).collect(),
                    _ => Vec::new(),
                };
                let o = self.new_obj(mem, "NSArray", Obj::Array(items));
                ret!(o)
            }
            ("NSDictionary", false, "setObject:forKey:") => {
                self.dict_set(mem, recv, a3, a2);
                return true;
            }
            ("NSDictionary", false, "setObject:forKeyedSubscript:") => {
                // dict[key] = object; a nil object removes the key
                if a2 == 0 {
                    if let Some(i) = self.dict_find(mem, recv, a3) {
                        if let Some(Obj::Dict(v)) = self.objs.get_mut(&recv) {
                            v.remove(i);
                        }
                    }
                } else {
                    self.dict_set(mem, recv, a3, a2);
                }
                return true;
            }
            ("NSDictionary", false, "removeObjectForKey:") => {
                if let Some(i) = self.dict_find(mem, recv, a2) {
                    if let Some(Obj::Dict(v)) = self.objs.get_mut(&recv) {
                        v.remove(i);
                    }
                }
                return true;
            }
            ("NSDictionary", false, "removeAllObjects") => {
                if let Some(Obj::Dict(v)) = self.objs.get_mut(&recv) {
                    v.clear();
                }
                return true;
            }
            ("NSDictionary", false, "addEntriesFromDictionary:") => {
                let pairs = match self.objs.get(&a2) {
                    Some(Obj::Dict(v)) => v.clone(),
                    _ => Vec::new(),
                };
                for (k, v) in pairs {
                    self.dict_set(mem, recv, k, v);
                }
                return true;
            }
            ("NSDictionary", false, "copy") | ("NSDictionary", false, "mutableCopy") | ("NSDictionary", false, "copyWithZone:") => {
                let pairs = match self.objs.get(&recv) {
                    Some(Obj::Dict(v)) => v.clone(),
                    _ => Vec::new(),
                };
                let c = if sel == "mutableCopy" { "NSMutableDictionary" } else { "NSDictionary" };
                let o = self.new_obj(mem, c, Obj::Dict(pairs));
                ret!(o)
            }
            ("NSUserDefaults", true, "standardUserDefaults") => {
                let o = self.singleton(mem, "NSUserDefaults", "NSUserDefaults", Obj::Dict(Vec::new()));
                ret!(o)
            }
            ("NSUserDefaults", false, "synchronize") => ret!(1),
            ("NSUserDefaults", false, "boolForKey:") | ("NSUserDefaults", false, "integerForKey:") => {
                let v = self.dict_get(mem, recv, a2);
                let n = self.num_of(v).map_or(0, |n| n.as_i64());
                ret!(if sel == "boolForKey:" { (n != 0) as u64 } else { n as u64 })
            }
            ("NSUserDefaults", false, "doubleForKey:") | ("NSUserDefaults", false, "floatForKey:") => {
                let v = self.dict_get(mem, recv, a2);
                let f = self.num_of(v).map_or(0.0, |n| n.as_f64());
                if sel == "floatForKey:" {
                    cpu.set_f32(0, f as f32);
                } else {
                    cpu.set_f64(0, f);
                }
                return true;
            }
            ("NSUserDefaults", false, "stringForKey:") | ("NSUserDefaults", false, "arrayForKey:") | ("NSUserDefaults", false, "dictionaryForKey:") => {
                let v = self.dict_get(mem, recv, a2);
                ret!(v)
            }
            ("NSUserDefaults", false, "setBool:forKey:") => {
                let n = self.new_obj(mem, "NSNumber", Obj::Num(Num::from_bool(a2 as u8 != 0)));
                self.dict_set(mem, recv, a3, n);
                return true;
            }
            ("NSUserDefaults", false, "setInteger:forKey:") => {
                let n = self.new_obj(mem, "NSNumber", Obj::Num(Num::int(a2 as i64)));
                self.dict_set(mem, recv, a3, n);
                return true;
            }
            ("NSUserDefaults", false, "setDouble:forKey:") => {
                let f = cpu.f64(0);
                let n = self.new_obj(mem, "NSNumber", Obj::Num(Num::dbl(f)));
                self.dict_set(mem, recv, a2, n);
                return true;
            }
            ("NSUserDefaults", false, "setFloat:forKey:") => {
                let f = cpu.f32(0) as f64;
                let n = self.new_obj(mem, "NSNumber", Obj::Num(Num::dbl(f)));
                self.dict_set(mem, recv, a2, n);
                return true;
            }
            ("NSUserDefaults", false, "registerDefaults:") => {
                let pairs = match self.objs.get(&a2) {
                    Some(Obj::Dict(v)) => v.clone(),
                    _ => Vec::new(),
                };
                for (k, v) in pairs {
                    if self.dict_find(mem, recv, k).is_none() {
                        self.dict_set(mem, recv, k, v);
                    }
                }
                return true;
            }

            // ----- NSBundle -----
            ("NSBundle", true, "mainBundle") | ("NSBundle", true, "bundleForClass:") | ("NSBundle", true, "bundleWithPath:") => {
                let o = self.singleton(mem, "NSBundle", "NSBundle", Obj::Plain);
                ret!(o)
            }
            ("NSBundle", false, "bundlePath") | ("NSBundle", false, "resourcePath") => {
                let p = self.bundle_path.clone();
                let o = self.new_string(mem, &p);
                ret!(o)
            }
            ("NSBundle", false, "executablePath") => {
                let p = format!("{}/{}", self.bundle_path, self.exe_name);
                let o = self.new_string(mem, &p);
                ret!(o)
            }
            ("NSBundle", false, "bundleIdentifier") => {
                let p = self.bundle_id.clone();
                let o = self.new_string(mem, &p);
                ret!(o)
            }
            ("NSBundle", false, "pathForResource:ofType:") | ("NSBundle", false, "pathForResource:ofType:inDirectory:") => {
                let Some(file) = self.str_of(mem, a2) else { ret!(0) };
                let ext = self.str_of(mem, a3);
                let dir = if sel.ends_with("inDirectory:") { self.str_of(mem, a4) } else { None };
                let mut p = self.bundle_path.clone();
                if let Some(d) = dir {
                    p = format!("{p}/{d}");
                }
                p = match ext {
                    Some(e) if !e.is_empty() => format!("{p}/{file}.{e}"),
                    _ => format!("{p}/{file}"),
                };
                if !self.file_exists(&p).0 {
                    self.log_once(format!("NSBundle {sel}: no such file {p}"));
                    ret!(0)
                }
                let o = self.new_string(mem, &p);
                ret!(o)
            }
            ("NSBundle", false, "infoDictionary") => {
                let o = self.singleton(mem, "InfoDictionary", "NSDictionary", Obj::Dict(Vec::new()));
                let empty = matches!(self.objs.get(&o), Some(Obj::Dict(v)) if v.is_empty());
                if empty {
                    let entries: Vec<(String, String)> =
                        self.info_plist.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    for (k, v) in entries {
                        let ks = self.new_string(mem, &k);
                        let vs = self.new_string(mem, &v);
                        self.dict_set(mem, o, ks, vs);
                    }
                }
                ret!(o)
            }
            ("NSBundle", false, "objectForInfoDictionaryKey:") => {
                let key = self.str_of(mem, a2).unwrap_or_default();
                let v = match key.as_str() {
                    "CFBundleIdentifier" => self.bundle_id.clone(),
                    "CFBundleName" | "CFBundleDisplayName" | "CFBundleExecutable" => self.exe_name.clone(),
                    _ => self.info_plist.get(&key).cloned().unwrap_or_default(),
                };
                if v.is_empty() {
                    self.log_once(format!("Info.plist has no string value for {key}"));
                    ret!(0)
                }
                let o = self.new_string(mem, &v);
                ret!(o)
            }

            // ----- NSFileManager -----
            ("NSFileManager", true, "defaultManager") => {
                let o = self.singleton(mem, "NSFileManager", "NSFileManager", Obj::Plain);
                ret!(o)
            }
            ("NSFileManager", false, "fileExistsAtPath:") | ("NSFileManager", false, "fileExistsAtPath:isDirectory:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                let (exists, is_dir) = self.file_exists(&path);
                self.log_once(format!("NSFileManager {sel} {path} -> {exists}"));
                if sel.ends_with("isDirectory:") && a3 != 0 {
                    mem.write_slice(a3, &[is_dir as u8]);
                }
                ret!(exists as u64)
            }
            ("NSFileManager", false, "createDirectoryAtPath:withIntermediateDirectories:attributes:error:")
            | ("NSFileManager", false, "removeItemAtPath:error:")
            | ("NSFileManager", false, "copyItemAtPath:toPath:error:")
            | ("NSFileManager", false, "moveItemAtPath:toPath:error:")
            | ("NSFileManager", false, "createFileAtPath:contents:attributes:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                self.log_once(format!("NSFileManager {sel} {path}"));
                ret!(1)
            }
            ("NSFileManager", false, "isReadableFileAtPath:") | ("NSFileManager", false, "isWritableFileAtPath:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                let exists = self.file_exists(&path).0;
                self.log_once(format!("NSFileManager {sel} {path} -> {exists}"));
                ret!(exists as u64)
            }
            ("NSFileManager", false, "contentsOfDirectoryAtPath:error:") => {
                let o = self.new_obj(mem, "NSArray", Obj::Array(Vec::new()));
                ret!(o)
            }
            ("NSFileManager", false, "URLsForDirectory:inDomains:") => {
                let sub = match a2 {
                    5 => "Library",
                    13 => "Library/Caches",
                    14 => "Library/Application Support",
                    _ => "Documents",
                };
                let url = self.new_obj(mem, "NSURL", Obj::Url(format!("{SANDBOX}/{sub}")));
                let o = self.new_obj(mem, "NSArray", Obj::Array(vec![url]));
                ret!(o)
            }

            // ----- NSURL -----
            ("NSURL", true, "fileURLWithPath:") | ("NSURL", true, "fileURLWithPath:isDirectory:") | ("NSURL", true, "URLWithString:") => {
                let s = self.str_of(mem, a2).unwrap_or_default();
                let o = self.new_obj(mem, "NSURL", Obj::Url(s));
                ret!(o)
            }
            ("NSURL", false, "path") | ("NSURL", false, "relativePath") | ("NSURL", false, "description") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSURL", false, "absoluteString") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let s = if s.starts_with('/') { format!("file://{s}") } else { s };
                let o = self.new_string(mem, &s);
                ret!(o)
            }
            ("NSURL", false, "isFileURL") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                ret!(s.starts_with('/') as u64)
            }
            ("NSURL", false, "lastPathComponent") => {
                let s = self.str_of(mem, recv).unwrap_or_default();
                let last = s.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_string();
                let o = self.new_string(mem, &last);
                ret!(o)
            }
            ("NSURL", false, "fileSystemRepresentation") => {
                let p = self.utf8_ptr(mem, recv);
                ret!(p)
            }
            ("NSURL", false, "absoluteURL") | ("NSURL", false, "standardizedURL") | ("NSURL", false, "copy") => ret!(recv),

            // ----- UIDevice / UIScreen -----
            ("UIDevice", true, "currentDevice") => {
                let o = self.singleton(mem, "UIDevice", "UIDevice", Obj::Plain);
                ret!(o)
            }
            ("UIDevice", false, "systemVersion") => {
                let o = self.new_string(mem, "10.3.3");
                ret!(o)
            }
            ("UIDevice", false, "systemName") => {
                let o = self.new_string(mem, "iOS");
                ret!(o)
            }
            ("UIDevice", false, "model") | ("UIDevice", false, "localizedModel") | ("UIDevice", false, "name") => {
                let o = self.new_string(mem, "iPhone");
                ret!(o)
            }
            ("UIDevice", false, "userInterfaceIdiom") => ret!(0),
            ("UIDevice", false, "orientation") => ret!(3),
            ("UIScreen", true, "mainScreen") => {
                let o = self.singleton(mem, "UIScreen", "UIScreen", Obj::Plain);
                ret!(o)
            }
            ("UIScreen", false, "bounds") | ("UIScreen", false, "applicationFrame") => {
                cpu.set_f64(0, 0.0);
                cpu.set_f64(1, 0.0);
                cpu.set_f64(2, SCREEN_W);
                cpu.set_f64(3, SCREEN_H);
                return true;
            }
            ("UIScreen", false, "nativeBounds") => {
                cpu.set_f64(0, 0.0);
                cpu.set_f64(1, 0.0);
                cpu.set_f64(2, SCREEN_W * SCREEN_SCALE);
                cpu.set_f64(3, SCREEN_H * SCREEN_SCALE);
                return true;
            }
            ("UIScreen", false, "scale") | ("UIScreen", false, "nativeScale") => {
                cpu.set_f64(0, SCREEN_SCALE);
                return true;
            }

            // ----- UIKit objects: just enough to be created -----

            // ----- notifications, process, pools, threads -----
            ("NSNotificationCenter", true, "defaultCenter") => {
                let o = self.singleton(mem, "NSNotificationCenter", "NSNotificationCenter", Obj::Plain);
                ret!(o)
            }
            ("NSNotificationCenter", false, s)
                if s.starts_with("addObserver") || s.starts_with("removeObserver") || s.starts_with("postNotification") =>
            {
                if s.starts_with("addObserverForName") {
                    let token = self.new_obj(mem, "NSObject", Obj::Plain);
                    ret!(token)
                }
                return true;
            }
            ("NSProcessInfo", true, "processInfo") => {
                let o = self.singleton(mem, "NSProcessInfo", "NSProcessInfo", Obj::Plain);
                ret!(o)
            }
            ("NSProcessInfo", false, "processName") => {
                let n = self.exe_name.clone();
                let o = self.new_string(mem, &n);
                ret!(o)
            }
            ("NSProcessInfo", false, "arguments") => {
                let o = self.new_obj(mem, "NSArray", Obj::Array(Vec::new()));
                ret!(o)
            }
            ("NSProcessInfo", false, "environment") => {
                let o = self.new_obj(mem, "NSDictionary", Obj::Dict(Vec::new()));
                ret!(o)
            }
            ("NSProcessInfo", false, "systemUptime") => {
                cpu.set_f64(0, 1000.0);
                return true;
            }
            ("NSProcessInfo", false, "processorCount") | ("NSProcessInfo", false, "activeProcessorCount") => ret!(6),
            ("NSProcessInfo", false, "physicalMemory") => ret!(3 * 1024 * 1024 * 1024),
            ("NSAutoreleasePool", false, "drain") => return true,
            ("NSThread", true, "detachNewThreadSelector:toTarget:withObject:") => {
                // selector, target, object
                self.pending_threads.push((a3, a2, a4));
                return true;
            }
            ("NSThread", true, "isMainThread") => ret!(1),
            ("NSThread", true, "currentThread") | ("NSThread", true, "mainThread") => {
                let o = self.singleton(mem, "NSThread", "NSThread", Obj::Plain);
                ret!(o)
            }
            ("NSThread", true, "sleepForTimeInterval:") => return true,
            ("NSThread", false, "initWithTarget:selector:object:") => {
                self.thread_inits.insert(recv, (a2, a3, a4));
                ret!(recv)
            }
            ("NSThread", false, "start") => {
                if let Some(t) = self.thread_inits.get(&recv).copied() {
                    self.pending_threads.push(t);
                }
                return true;
            }
            ("NSThread", false, "isMainThread") => ret!(1),
            ("NSThread", false, "setName:") | ("NSThread", false, "setStackSize:") | ("NSThread", false, "setThreadPriority:") => return true,
            // ----- NSString extras -----
            ("NSString", false, "compare:")
            | ("NSString", false, "compare:options:")
            | ("NSString", false, "caseInsensitiveCompare:")
            | ("NSString", false, "localizedCompare:")
            | ("NSString", false, "localizedCaseInsensitiveCompare:") => {
                let (x, y) = (self.str_of(mem, recv).unwrap_or_default(), self.str_of(mem, a2).unwrap_or_default());
                let ci = sel.contains("aseInsensitive") || (sel == "compare:options:" && a3 & 1 != 0);
                let (x, y) = if ci { (x.to_lowercase(), y.to_lowercase()) } else { (x, y) };
                let r: i64 = match x.cmp(&y) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                ret!(r as u64)
            }
            ("NSString", true, "stringWithContentsOfFile:encoding:error:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                match self.read_file(&path) {
                    Some(d) => {
                        let o = self.new_string(mem, &String::from_utf8_lossy(&d));
                        ret!(o)
                    }
                    None => ret!(0),
                }
            }
            ("NSURL", false, "setResourceValue:forKey:error:") => ret!(1),
            ("NSURL", false, "getResourceValue:forKey:error:") => ret!(0),

            // ----- NSData -----
            ("NSData", true, "data") => {
                let c = self.class_name(recv);
                let o = self.new_obj(mem, &c, Obj::Data(Vec::new()));
                ret!(o)
            }
            ("NSData", true, "dataWithContentsOfFile:")
            | ("NSData", true, "dataWithContentsOfFile:options:error:")
            | ("NSData", true, "dataWithContentsOfURL:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                match self.read_file(&path) {
                    Some(d) => {
                        let c = self.class_name(recv);
                        let o = self.new_obj(mem, &c, Obj::Data(d));
                        ret!(o)
                    }
                    None => {
                        self.log_once(format!("NSData: no such file {path}"));
                        ret!(0)
                    }
                }
            }
            ("NSData", false, "initWithContentsOfFile:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                match self.read_file(&path) {
                    Some(d) => {
                        self.objs.insert(recv, Obj::Data(d));
                        ret!(recv)
                    }
                    None => ret!(0),
                }
            }
            ("NSData", true, "dataWithBytes:length:") => {
                let n = (a3 as usize).min(0x400_0000);
                let d = mem.read_bytes(a2, n).map(|s| s.to_vec()).unwrap_or_default();
                let c = self.class_name(recv);
                let o = self.new_obj(mem, &c, Obj::Data(d));
                ret!(o)
            }
            ("NSData", false, "initWithBytes:length:") => {
                let n = (a3 as usize).min(0x400_0000);
                let d = mem.read_bytes(a2, n).map(|s| s.to_vec()).unwrap_or_default();
                self.objs.insert(recv, Obj::Data(d));
                ret!(recv)
            }
            ("NSData", false, "length") => {
                let n = match self.objs.get(&recv) {
                    Some(Obj::Data(d)) => d.len(),
                    _ => 0,
                };
                ret!(n as u64)
            }
            ("NSData", false, "bytes") => {
                if let Some(p) = self.data_ptrs.get(&recv) {
                    ret!(*p)
                }
                let d = match self.objs.get(&recv) {
                    Some(Obj::Data(d)) => d.clone(),
                    _ => Vec::new(),
                };
                let p = self.heap.alloc(d.len() as u64 + 1);
                mem.write_slice(p, &d);
                self.data_ptrs.insert(recv, p);
                ret!(p)
            }
            ("NSData", false, "getBytes:length:") => {
                if let Some(Obj::Data(d)) = self.objs.get(&recv) {
                    let n = (a3 as usize).min(d.len());
                    let chunk = d[..n].to_vec();
                    mem.write_slice(a2, &chunk);
                }
                return true;
            }
            ("NSData", false, "writeToFile:atomically:") => {
                let path = self.str_of(mem, a2).unwrap_or_default();
                if let Some(Obj::Data(d)) = self.objs.get(&recv) {
                    let d = d.clone();
                    self.overlay.insert(path, d);
                }
                ret!(1)
            }

            // ----- UIKit: windows, controllers, views -----
            ("UIApplication", true, "sharedApplication") => {
                let o = self.singleton(mem, "UIApplication", "UIApplication", Obj::Plain);
                ret!(o)
            }
            ("UIApplication", false, "windows") => {
                let w = self.prop(recv, "keyWindow");
                let items = if w != 0 { vec![w] } else { Vec::new() };
                let o = self.new_obj(mem, "NSArray", Obj::Array(items));
                ret!(o)
            }
            ("UIWindow", false, "screen") => {
                let o = self.singleton(mem, "UIScreen", "UIScreen", Obj::Plain);
                ret!(o)
            }
            ("NSError", true, "errorWithDomain:code:userInfo:") => {
                let o = self.new_obj(mem, "NSError", Obj::Plain);
                ret!(o)
            }
            ("NSUUID", false, "UUIDString") => {
                let o = self.new_string(mem, "E621E1F8-C36C-495A-93FC-0C247A3E6E5F");
                ret!(o)
            }
            ("NSUUID", true, "UUID") => {
                let o = self.new_obj(mem, "NSUUID", Obj::Plain);
                ret!(o)
            }
            ("NSCharacterSet", true, "characterSetWithCharactersInString:") => {
                let o = self.new_obj(mem, "NSCharacterSet", Obj::Plain);
                ret!(o)
            }
            ("UIApplication", false, "applicationState") => ret!(0),
            ("UIApplication", false, "statusBarOrientation") => ret!(3),
            ("UIApplication", false, "canOpenURL:") => ret!(0),
            ("UIWindow", false, "makeKeyAndVisible") => {
                self.visible_windows.push(recv);
                let app = self.singleton(mem, "UIApplication", "UIApplication", Obj::Plain);
                self.set_prop(app, "keyWindow", recv);
                return true;
            }
            ("UIView", false, "initWithFrame:") => {
                let vals = [0, cpu.vec(0) as u64, cpu.vec(1) as u64, cpu.vec(2) as u64, cpu.vec(3) as u64];
                self.props.insert((recv, "frame".to_string()), vals);
                ret!(recv)
            }
            ("UIView", false, "layer") => {
                let mut l = self.prop(recv, "layer");
                if l == 0 {
                    // A view whose class overrides +layerClass wants a
                    // special layer (OpenGL ES drawing).
                    let meta = mem.read_u64(mem.read_u64(recv).unwrap_or(0) & 0x0000_000f_ffff_fff8).unwrap_or(0);
                    let wants_gl = self.objc.find(meta, "layerClass").is_some();
                    l = self.new_obj(mem, if wants_gl { "CAEAGLLayer" } else { "CALayer" }, Obj::Plain);
                    self.set_prop(recv, "layer", l);
                }
                ret!(l)
            }
            ("UIView", false, "addSubview:")
            | ("UIView", false, "removeFromSuperview")
            | ("UIView", false, "setNeedsLayout")
            | ("UIView", false, "setNeedsDisplay")
            | ("UIView", false, "layoutIfNeeded")
            | ("UIView", false, "bringSubviewToFront:")
            | ("UIView", false, "sendSubviewToBack:")
            | ("UIView", false, "insertSubview:atIndex:") => return true,
            ("UIViewController", false, "init") | ("UIViewController", false, "initWithNibName:bundle:") => ret!(recv),
            ("UIViewController", false, "view") => {
                let mut v = self.prop(recv, "view");
                if v == 0 {
                    v = self.default_view(mem);
                    self.set_prop(recv, "view", v);
                }
                ret!(v)
            }
            ("UIViewController", false, "loadView")
            | ("UIViewController", false, "viewDidLoad")
            | ("UIViewController", false, "viewWillAppear:")
            | ("UIViewController", false, "viewDidAppear:")
            | ("UIViewController", false, "viewWillDisappear:")
            | ("UIViewController", false, "viewDidDisappear:")
            | ("UIViewController", false, "viewWillLayoutSubviews")
            | ("UIViewController", false, "viewDidLayoutSubviews")
            | ("UIViewController", false, "didReceiveMemoryWarning") => return true,

            // ----- run loop: timers, display links, delayed calls -----
            ("CADisplayLink", true, "displayLinkWithTarget:selector:") => {
                let l = self.new_obj(mem, "CADisplayLink", Obj::Plain);
                self.set_prop(l, "target", a2);
                self.set_prop(l, "selector", a3);
                ret!(l)
            }
            ("CADisplayLink", false, "addToRunLoop:forMode:") => {
                let (t, s) = (self.prop(recv, "target"), self.prop(recv, "selector"));
                if !self.links.iter().any(|l| l.0 == recv) {
                    self.links.push((recv, t, s));
                }
                return true;
            }
            ("CADisplayLink", false, "invalidate") => {
                self.links.retain(|l| l.0 != recv);
                return true;
            }
            ("CADisplayLink", false, "timestamp") => {
                cpu.set_f64(0, self.vtime as f64 / 1e9);
                return true;
            }
            ("CADisplayLink", false, "duration") => {
                cpu.set_f64(0, 1.0 / 60.0);
                return true;
            }
            ("NSTimer", true, "scheduledTimerWithTimeInterval:target:selector:userInfo:repeats:")
            | ("NSTimer", true, "timerWithTimeInterval:target:selector:userInfo:repeats:") => {
                let interval = cpu.f64(0);
                let repeats = cpu.reg(5) as u8 != 0;
                let t = self.new_obj(mem, "NSTimer", Obj::Plain);
                self.set_prop(t, "target", a2);
                self.set_prop(t, "selector", a3);
                self.set_prop(t, "userInfo", a4);
                self.set_prop(t, "interval", interval.to_bits());
                self.set_prop(t, "repeats", repeats as u64);
                if sel.starts_with("scheduled") {
                    self.schedule_timer(t);
                }
                ret!(t)
            }
            ("NSTimer", true, "scheduledTimerWithTimeInterval:repeats:block:") => {
                let interval = cpu.f64(0);
                let repeats = a2 as u8 != 0;
                let b = self.copy_block(mem, a3);
                let t = self.new_obj(mem, "NSTimer", Obj::Plain);
                let ns = (interval.max(0.0) * 1e9) as u64;
                let repeat = if repeats { ns.max(1_000_000) } else { 0 };
                let due = self.vtime + ns;
                self.schedule(due, repeat, Action::Block { block: b, args: vec![t] }, t);
                ret!(t)
            }
            ("NSTimer", false, "invalidate") => {
                self.scheduled.retain(|s| s.handle != recv);
                return true;
            }
            ("NSRunLoop", true, "currentRunLoop") | ("NSRunLoop", true, "mainRunLoop") => {
                let o = self.singleton(mem, "NSRunLoop", "NSRunLoop", Obj::Plain);
                ret!(o)
            }
            ("NSRunLoop", false, "addTimer:forMode:") => {
                self.schedule_timer(a2);
                return true;
            }
            ("NSRunLoop", false, "run")
            | ("NSRunLoop", false, "runUntilDate:")
            | ("NSRunLoop", false, "runMode:beforeDate:")
            | ("NSRunLoop", false, "addPort:forMode:") => return true,
            ("NSObject", false, "performSelector:withObject:afterDelay:") => {
                let ns = (cpu.f64(0).max(0.0) * 1e9) as u64;
                let due = self.vtime + ns;
                self.schedule(due, 0, Action::Send { target: recv, sel: a2, args: vec![a3] }, 0);
                return true;
            }
            ("NSObject", false, "performSelectorOnMainThread:withObject:waitUntilDone:") => {
                let due = self.vtime;
                self.schedule(due, 0, Action::Send { target: recv, sel: a2, args: vec![a3] }, 0);
                return true;
            }
            // ----- EAGL: OpenGL ES contexts -----
            ("EAGLContext", false, "initWithAPI:") | ("EAGLContext", false, "initWithAPI:sharegroup:") => {
                if a2 != 2 && a2 != 3 {
                    log!("[a64] EAGLContext: OpenGL ES API {a2} is not supported");
                    ret!(0)
                }
                if !self.gl_init() {
                    ret!(0)
                }
                self.set_prop(recv, "api", a2);
                ret!(recv)
            }
            ("EAGLContext", true, "setCurrentContext:") => {
                if a2 != 0 {
                    self.gl_make_current();
                }
                self.current_ctx = a2;
                ret!(1)
            }
            ("EAGLContext", true, "currentContext") => ret!(self.current_ctx),
            ("EAGLContext", false, "renderbufferStorage:fromDrawable:") => {
                let ok = self.gl_renderbuffer_storage();
                ret!(ok as u64)
            }
            ("EAGLContext", false, "presentRenderbuffer:") => {
                self.gl_present();
                ret!(1)
            }
            ("UIView", false, "window") => {
                let app = self.singleton(mem, "UIApplication", "UIApplication", Obj::Plain);
                let w = self.prop(app, "keyWindow");
                ret!(w)
            }
            _ => return false,
        }
        true
    }

    // ----- OpenGL ES bridge -----
    //
    // Guest memory is ordinary host memory, so pointers into it (vertex data,
    // textures, indices) can be handed to the driver directly.

    /// Create the OpenGL ES context for the app (once).
    fn gl_init(&mut self) -> bool {
        if self.gl.is_some() {
            return true;
        }
        let ctx = match self.window.create_gl_context(GLVersion::GLES30) {
            Ok(c) => c,
            Err(e) => {
                log!("[a64] could not create an OpenGL ES 3.0 context: {e}");
                return false;
            }
        };
        unsafe {
            self.window.make_gl_context_current(&ctx);
        }
        {
            let w: &Window = &*self.window;
            gl30::load_with(|s| w.gl_get_proc_address(s));
            gl::load_with(|s| w.gl_get_proc_address(s));
        }
        let vp = self.window.viewport();
        let host_fb = self.window.host_framebuffer();
        echo!("[a64] OpenGL ES context created; window viewport {vp:?}, host framebuffer {host_fb}");
        self.gl = Some(GlState {
            ctx,
            drawable_rb: 0,
            present_fbo: 0,
            size: (0, 0),
            array_buf: 0,
            element_buf: 0,
            mapped: HashMap::new(),
            presented: 0,
        });
        true
    }

    fn gl_make_current(&mut self) {
        if let Some(g) = &self.gl {
            unsafe {
                self.window.make_gl_context_current(&g.ctx);
            }
        }
    }

    /// `-[EAGLContext renderbufferStorage:fromDrawable:]`: give the bound
    /// renderbuffer the size of the screen and remember it as the one to show.
    fn gl_renderbuffer_storage(&mut self) -> bool {
        let w = (SCREEN_W * SCREEN_SCALE) as i32;
        let h = (SCREEN_H * SCREEN_SCALE) as i32;
        let Some(g) = self.gl.as_mut() else { return false };
        unsafe {
            let mut rb: i32 = 0;
            gl::GetIntegerv(gl::RENDERBUFFER_BINDING, &mut rb);
            gl::RenderbufferStorage(gl::RENDERBUFFER, gl30::RGBA8, w, h);
            g.drawable_rb = rb as u32;
            g.size = (w, h);
        }
        true
    }

    /// `-[EAGLContext presentRenderbuffer:]`: copy the drawable to the window
    /// and flip buffers.
    fn gl_present(&mut self) {
        let (vx, vy, vw, vh) = self.window.viewport();
        let host_fb = self.window.host_framebuffer();
        let Some(g) = self.gl.as_mut() else { return };
        if g.drawable_rb == 0 {
            return;
        }
        unsafe {
            let mut prev: i32 = 0;
            gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut prev);
            let scissor = gl::IsEnabled(gl::SCISSOR_TEST);
            gl::Disable(gl::SCISSOR_TEST);
            if g.present_fbo == 0 {
                let mut f: u32 = 0;
                gl::GenFramebuffers(1, &mut f);
                g.present_fbo = f;
            }
            gl::BindFramebuffer(gl::FRAMEBUFFER, g.present_fbo);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, g.drawable_rb);
            gl30::BindFramebuffer(gl30::READ_FRAMEBUFFER, g.present_fbo);
            gl30::BindFramebuffer(gl30::DRAW_FRAMEBUFFER, host_fb);
            gl30::BlitFramebuffer(
                0,
                0,
                g.size.0,
                g.size.1,
                vx as i32,
                vy as i32,
                (vx + vw) as i32,
                (vy + vh) as i32,
                gl30::COLOR_BUFFER_BIT,
                gl30::LINEAR,
            );
            gl::BindFramebuffer(gl::FRAMEBUFFER, prev as u32);
            if scissor != 0 {
                gl::Enable(gl::SCISSOR_TEST);
            }
        }
        g.presented += 1;
        self.window.swap_window();
    }

    /// Host pointer for guest address `addr` (null if it isn't mapped).
    fn host_ptr(mem: &Mem64, addr: u64) -> *const std::ffi::c_void {
        mem.read_bytes(addr, 1).map_or(std::ptr::null(), |s| s.as_ptr() as *const std::ffi::c_void)
    }

    /// Host pointer for `len` bytes at guest address `addr`; null if `addr` is
    /// 0 or the range isn't fully mapped.
    fn host_range(mem: &Mem64, addr: u64, len: usize) -> *const std::ffi::c_void {
        if addr == 0 {
            return std::ptr::null();
        }
        mem.read_bytes(addr, len).map_or(std::ptr::null(), |s| s.as_ptr() as *const std::ffi::c_void)
    }

    /// Size in bytes of a client-side image, as `glTexImage2D` reads it.
    fn tex_bytes(format: u32, ty: u32, w: i32, h: i32, align: i32) -> usize {
        let bpp = match ty {
            0x8033 | 0x8034 | 0x8363 => 2, // 4444, 5551, 565
            _ => match format {
                0x1908 => 4, // RGBA
                0x1907 => 3, // RGB
                0x190A => 2, // LUMINANCE_ALPHA
                0x1909 | 0x1906 => 1,
                _ => 4,
            },
        };
        let align = align.max(1) as usize;
        let row = (w.max(0) as usize * bpp + align - 1) / align * align;
        row * h.max(0) as usize
    }

    /// Run an imported `gl*` function.
    fn gl_call(&mut self, cpu: &mut Cpu64, mem: &mut Mem64, name: &str) {
        if self.gl.is_none() {
            self.log_once(format!("{name} was called before any EAGLContext existed"));
            cpu.set_reg(0, 0);
            return;
        }
        let f = name.strip_prefix("_gl").unwrap_or(name);
        let (a0, a1, a2, a3) = (cpu.reg(0), cpu.reg(1), cpu.reg(2), cpu.reg(3));
        let (a4, a5, a6, a7) = (cpu.reg(4), cpu.reg(5), cpu.reg(6), cpu.reg(7));
        let (f0, f1, f2, f3) = (cpu.f32(0), cpu.f32(1), cpu.f32(2), cpu.f32(3));
        let sp = cpu.sp();
        // The 9th integer argument is the first one on the stack.
        let s0 = mem.read_u64(sp).unwrap_or(0);
        let mut known = true;
        unsafe {
            match f {
                "Clear" => gl::Clear(a0 as u32),
                "ClearColor" => gl::ClearColor(f0, f1, f2, f3),
                "Viewport" => gl::Viewport(a0 as i32, a1 as i32, a2 as i32, a3 as i32),
                "Scissor" => gl::Scissor(a0 as i32, a1 as i32, a2 as i32, a3 as i32),
                "Enable" => gl::Enable(a0 as u32),
                "Disable" => gl::Disable(a0 as u32),
                "IsEnabled" => {
                    let v = gl::IsEnabled(a0 as u32);
                    cpu.set_reg(0, v as u64);
                }
                "DepthMask" => gl::DepthMask(a0 as u8),
                "BlendFunc" => gl::BlendFunc(a0 as u32, a1 as u32),
                "BlendEquation" => gl::BlendEquation(a0 as u32),
                "Hint" => gl::Hint(a0 as u32, a1 as u32),
                "PixelStorei" => {
                    if a0 as u32 == 0x0CF5 {
                        self.unpack_align = a1 as i32;
                    }
                    gl::PixelStorei(a0 as u32, a1 as i32)
                }
                "ActiveTexture" => gl::ActiveTexture(a0 as u32),
                "BindTexture" => gl::BindTexture(a0 as u32, a1 as u32),
                "TexParameteri" => gl::TexParameteri(a0 as u32, a1 as u32, a2 as i32),
                "TexParameterf" => gl::TexParameterf(a0 as u32, a1 as u32, f0),
                "GenerateMipmap" => gl::GenerateMipmap(a0 as u32),
                "GetError" => {
                    let host = gl::GetError();
                    let e = if self.gl_err != 0 { self.gl_err } else { host };
                    self.gl_err = 0;
                    cpu.set_reg(0, e as u64);
                }
                "GenBuffers" | "GenTextures" | "GenFramebuffers" | "GenRenderbuffers" => {
                    let n = (a0 as i32).clamp(0, 1024) as usize;
                    let mut ids = vec![0u32; n];
                    match f {
                        "GenBuffers" => gl::GenBuffers(n as i32, ids.as_mut_ptr()),
                        "GenTextures" => gl::GenTextures(n as i32, ids.as_mut_ptr()),
                        "GenFramebuffers" => gl::GenFramebuffers(n as i32, ids.as_mut_ptr()),
                        _ => gl::GenRenderbuffers(n as i32, ids.as_mut_ptr()),
                    }
                    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
                    mem.write_slice(a1, &bytes);
                }
                "DeleteBuffers" | "DeleteTextures" | "DeleteFramebuffers" | "DeleteRenderbuffers" => {
                    let n = (a0 as i32).clamp(0, 1024) as usize;
                    let ids: Vec<u32> = (0..n as u64).filter_map(|i| rd32(mem, a1 + i * 4)).collect();
                    match f {
                        "DeleteBuffers" => {
                            gl::DeleteBuffers(ids.len() as i32, ids.as_ptr());
                            if let Some(g) = self.gl.as_mut() {
                                if ids.contains(&g.array_buf) {
                                    g.array_buf = 0;
                                }
                                if ids.contains(&g.element_buf) {
                                    g.element_buf = 0;
                                }
                            }
                        }
                        "DeleteTextures" => gl::DeleteTextures(ids.len() as i32, ids.as_ptr()),
                        "DeleteFramebuffers" => gl::DeleteFramebuffers(ids.len() as i32, ids.as_ptr()),
                        _ => gl::DeleteRenderbuffers(ids.len() as i32, ids.as_ptr()),
                    }
                }
                "BindBuffer" => {
                    if let Some(g) = self.gl.as_mut() {
                        match a0 as u32 {
                            0x8892 => g.array_buf = a1 as u32,
                            0x8893 => g.element_buf = a1 as u32,
                            _ => {}
                        }
                    }
                    gl::BindBuffer(a0 as u32, a1 as u32)
                }
                "BindFramebuffer" => gl::BindFramebuffer(a0 as u32, a1 as u32),
                "BindRenderbuffer" => gl::BindRenderbuffer(a0 as u32, a1 as u32),
                "BufferData" => {
                    let data = Self::host_range(mem, a2, a1 as usize);
                    gl::BufferData(a0 as u32, a1 as isize, data, a3 as u32)
                }
                "RenderbufferStorage" => gl::RenderbufferStorage(a0 as u32, a1 as u32, a2 as i32, a3 as i32),
                "RenderbufferStorageMultisampleAPPLE" => {
                    // multisampling isn't offered; plain storage instead
                    gl::RenderbufferStorage(a0 as u32, a2 as u32, a3 as i32, a4 as i32)
                }
                "FramebufferRenderbuffer" => gl::FramebufferRenderbuffer(a0 as u32, a1 as u32, a2 as u32, a3 as u32),
                "FramebufferTexture2D" => gl::FramebufferTexture2D(a0 as u32, a1 as u32, a2 as u32, a3 as u32, a4 as i32),
                "CheckFramebufferStatus" => {
                    let s = gl::CheckFramebufferStatus(a0 as u32);
                    cpu.set_reg(0, s as u64);
                }
                "GetRenderbufferParameteriv" => {
                    let mut v: i32 = 0;
                    gl::GetRenderbufferParameteriv(a0 as u32, a1 as u32, &mut v);
                    mem.write_slice(a2, &v.to_le_bytes());
                }
                "ResolveMultisampleFramebufferAPPLE"
                | "DiscardFramebufferEXT"
                | "LabelObjectEXT"
                | "PushGroupMarkerEXT"
                | "PopGroupMarkerEXT" => {}
                "CreateShader" => {
                    let id = gl::CreateShader(a0 as u32);
                    cpu.set_reg(0, id as u64);
                }
                "CreateProgram" => {
                    let id = gl::CreateProgram();
                    cpu.set_reg(0, id as u64);
                }
                "DeleteShader" => gl::DeleteShader(a0 as u32),
                "DeleteProgram" => gl::DeleteProgram(a0 as u32),
                "AttachShader" => gl::AttachShader(a0 as u32, a1 as u32),
                "UseProgram" => gl::UseProgram(a0 as u32),
                "ShaderSource" => {
                    let count = (a1 as i32).clamp(0, 64) as usize;
                    let mut sources: Vec<std::ffi::CString> = Vec::new();
                    for i in 0..count as u64 {
                        let p = mem.read_u64(a2 + i * 8).unwrap_or(0);
                        let len = if a3 != 0 { rd32(mem, a3 + i * 4).map(|l| l as i32) } else { None };
                        let bytes = match len {
                            Some(l) if l >= 0 => mem.read_bytes(p, l as usize).map(|b| b.to_vec()),
                            _ => mem.read_c_bytes(p, 1 << 20),
                        }
                        .unwrap_or_default();
                        let clean: Vec<u8> = bytes.into_iter().filter(|b| *b != 0).collect();
                        sources.push(std::ffi::CString::new(clean).unwrap_or_default());
                    }
                    let ptrs: Vec<*const std::os::raw::c_char> = sources.iter().map(|c| c.as_ptr()).collect();
                    gl::ShaderSource(a0 as u32, count as i32, ptrs.as_ptr(), std::ptr::null());
                }
                "CompileShader" | "LinkProgram" => {
                    let is_shader = f == "CompileShader";
                    let id = a0 as u32;
                    let mut ok: i32 = 0;
                    if is_shader {
                        gl::CompileShader(id);
                        gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut ok);
                    } else {
                        gl::LinkProgram(id);
                        gl::GetProgramiv(id, gl::LINK_STATUS, &mut ok);
                    }
                    if ok == 0 {
                        let mut buf = vec![0u8; 2048];
                        let mut len: i32 = 0;
                        if is_shader {
                            gl::GetShaderInfoLog(id, 2048, &mut len, buf.as_mut_ptr() as *mut _);
                        } else {
                            gl::GetProgramInfoLog(id, 2048, &mut len, buf.as_mut_ptr() as *mut _);
                        }
                        let text = String::from_utf8_lossy(&buf[..(len.max(0) as usize).min(2048)]).into_owned();
                        log!("[a64] {} {id} failed: {text}", if is_shader { "shader" } else { "program" });
                    }
                }
                "GetShaderiv" | "GetProgramiv" => {
                    let mut v: i32 = 0;
                    if f == "GetShaderiv" {
                        gl::GetShaderiv(a0 as u32, a1 as u32, &mut v);
                    } else {
                        gl::GetProgramiv(a0 as u32, a1 as u32, &mut v);
                    }
                    mem.write_slice(a2, &v.to_le_bytes());
                }
                "GetShaderInfoLog" | "GetProgramInfoLog" => {
                    let cap = (a1 as i32).clamp(1, 1 << 16) as usize;
                    let mut buf = vec![0u8; cap];
                    let mut len: i32 = 0;
                    if f == "GetShaderInfoLog" {
                        gl::GetShaderInfoLog(a0 as u32, cap as i32, &mut len, buf.as_mut_ptr() as *mut _);
                    } else {
                        gl::GetProgramInfoLog(a0 as u32, cap as i32, &mut len, buf.as_mut_ptr() as *mut _);
                    }
                    if a2 != 0 {
                        mem.write_slice(a2, &len.to_le_bytes());
                    }
                    let n = ((len.max(0) as usize) + 1).min(cap);
                    mem.write_slice(a3, &buf[..n]);
                }
                "GetUniformLocation" => {
                    let nm = Objc::cstr(mem, a1).unwrap_or_default();
                    let cs = std::ffi::CString::new(nm).unwrap_or_default();
                    let loc = gl::GetUniformLocation(a0 as u32, cs.as_ptr());
                    cpu.set_reg(0, loc as i64 as u64);
                }
                "BindAttribLocation" => {
                    let nm = Objc::cstr(mem, a2).unwrap_or_default();
                    let cs = std::ffi::CString::new(nm).unwrap_or_default();
                    gl::BindAttribLocation(a0 as u32, a1 as u32, cs.as_ptr());
                }
                "Uniform1i" => gl::Uniform1i(a0 as i32, a1 as i32),
                "Uniform1f" => gl::Uniform1f(a0 as i32, f0),
                "Uniform2f" => gl::Uniform2f(a0 as i32, f0, f1),
                "Uniform3f" => gl::Uniform3f(a0 as i32, f0, f1, f2),
                "Uniform4f" => gl::Uniform4f(a0 as i32, f0, f1, f2, f3),
                "UniformMatrix4fv" => {
                    let n = (a1 as i32).clamp(0, 256) as usize;
                    let p = Self::host_range(mem, a3, n * 64);
                    if !p.is_null() {
                        gl::UniformMatrix4fv(a0 as i32, n as i32, a2 as u8, p as *const f32);
                    }
                }
                "EnableVertexAttribArray" => gl::EnableVertexAttribArray(a0 as u32),
                "DisableVertexAttribArray" => gl::DisableVertexAttribArray(a0 as u32),
                "VertexAttribPointer" => {
                    let buffered = self.gl.as_ref().map_or(false, |g| g.array_buf != 0);
                    let p = if buffered { a5 as usize as *const std::ffi::c_void } else { Self::host_ptr(mem, a5) };
                    gl::VertexAttribPointer(a0 as u32, a1 as i32, a2 as u32, a3 as u8, a4 as i32, p);
                }
                "DrawArrays" => gl::DrawArrays(a0 as u32, a1 as i32, a2 as i32),
                "DrawElements" => {
                    let buffered = self.gl.as_ref().map_or(false, |g| g.element_buf != 0);
                    let p = if buffered { a3 as usize as *const std::ffi::c_void } else { Self::host_ptr(mem, a3) };
                    gl::DrawElements(a0 as u32, a1 as i32, a2 as u32, p);
                }
                "TexImage2D" => {
                    let (w, h) = (a3 as i32, a4 as i32);
                    let n = Self::tex_bytes(a6 as u32, a7 as u32, w, h, self.unpack_align);
                    let p = Self::host_range(mem, s0, n);
                    if s0 != 0 && p.is_null() {
                        self.log_once(format!("glTexImage2D: the pixel data at {s0:#x} ({n} bytes) is not mapped"));
                    }
                    gl::TexImage2D(a0 as u32, a1 as i32, a2 as i32, w, h, a5 as i32, a6 as u32, a7 as u32, p);
                }
                "TexSubImage2D" => {
                    let (w, h) = (a4 as i32, a5 as i32);
                    let n = Self::tex_bytes(a6 as u32, a7 as u32, w, h, self.unpack_align);
                    let p = Self::host_range(mem, s0, n);
                    if !p.is_null() {
                        gl::TexSubImage2D(a0 as u32, a1 as i32, a2 as i32, a3 as i32, w, h, a6 as u32, a7 as u32, p);
                    }
                }
                "CompressedTexImage2D" => {
                    let p = Self::host_range(mem, a7, a6 as usize);
                    gl::CompressedTexImage2D(a0 as u32, a1 as i32, a2 as u32, a3 as i32, a4 as i32, a5 as i32, a6 as i32, p);
                }
                "GetIntegerv" => {
                    let mut v = [0i32; 16];
                    gl::GetIntegerv(a0 as u32, v.as_mut_ptr());
                    let n = match a0 as u32 {
                        0x0BA2 | 0x0C10 | 0x0C22 | 0x0C23 => 4,
                        0x0D3A | 0x846D | 0x846E | 0x0B70 => 2,
                        0x86A3 => {
                            let mut c: i32 = 0;
                            gl::GetIntegerv(0x86A2, &mut c);
                            (c.max(0) as usize).min(16)
                        }
                        _ => 1,
                    };
                    let bytes: Vec<u8> = v[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
                    mem.write_slice(a1, &bytes);
                }
                "GetString" => {
                    let key = a0 as u32;
                    let p = match self.gl_strings.get(&key) {
                        Some(p) => *p,
                        None => {
                            let text = match key {
                                0x1F00 => "Apple Inc.",
                                0x1F01 => "Apple A9 GPU",
                                0x1F02 => "OpenGL ES 2.0 APPLE-12.0.38",
                                0x8B8C => "OpenGL ES GLSL ES 1.00",
                                0x1F03 => {
                                    "GL_EXT_discard_framebuffer GL_OES_rgb8_rgba8 GL_OES_depth_texture \
                                     GL_OES_packed_depth_stencil GL_OES_element_index_uint \
                                     GL_OES_standard_derivatives GL_OES_texture_half_float GL_OES_texture_float"
                                }
                                _ => "",
                            };
                            let p = self.alloc_cstr(mem, text);
                            self.gl_strings.insert(key, p);
                            p
                        }
                    };
                    cpu.set_reg(0, p);
                }
                "MapBufferRangeEXT" => {
                    // Data is staged in guest memory and uploaded on unmap.
                    let len = a2 as i64;
                    let p = self.heap.alloc(len.max(16) as u64);
                    if let Some(g) = self.gl.as_mut() {
                        g.mapped.insert(a0 as u32, (p, a1 as i64, len));
                    }
                    cpu.set_reg(0, p);
                }
                "UnmapBufferOES" => {
                    let staged = self.gl.as_mut().and_then(|g| g.mapped.remove(&(a0 as u32)));
                    if let Some((p, off, len)) = staged {
                        let data = Self::host_range(mem, p, len.max(0) as usize);
                        if !data.is_null() {
                            gl::BufferSubData(a0 as u32, off as isize, len as isize, data);
                        }
                    }
                    cpu.set_reg(0, 1);
                }
                _ => {
                    known = false;
                }
            }
        }
        if !known {
            self.log_once(format!("unimplemented OpenGL function {name}"));
            cpu.set_reg(0, 0);
            return;
        }
        // Remember the first error of a call, so it can be logged once and
        // still be returned by the app's own glGetError.
        if f != "GetError" {
            let e = unsafe { gl::GetError() };
            if e != 0 {
                if self.gl_err == 0 {
                    self.gl_err = e;
                }
                if self.gl_err_logged < 40 {
                    self.gl_err_logged += 1;
                    log!("[a64] OpenGL error {e:#x} after {name}");
                }
            }
        }
    }

    // ----- files -----

    /// Our fake guest-visible path -> a path in the emulated file system.
    fn real_path(&self, p: &str) -> Option<String> {
        if let Some(rest) = p.strip_prefix(self.bundle_path.as_str()) {
            Some(format!("{}/{}", self.bundle_guest.trim_end_matches('/'), rest.trim_start_matches('/')))
        } else if let Some(rest) = p.strip_prefix(SANDBOX) {
            Some(format!(
                "{}/{}",
                self.fs.home_directory().as_str().trim_end_matches('/'),
                rest.trim_start_matches('/')
            ))
        } else {
            None
        }
    }

    /// (exists, is a directory)
    fn file_exists(&self, p: &str) -> (bool, bool) {
        if self.overlay.contains_key(p) {
            return (true, false);
        }
        match self.real_path(p) {
            Some(r) => {
                let gp = GuestPath::new(&r);
                (self.fs.exists(gp), self.fs.is_dir(gp))
            }
            None => (false, false),
        }
    }

    /// File contents; files the app wrote this session shadow the real ones.
    fn read_file(&self, p: &str) -> Option<Vec<u8>> {
        if let Some(d) = self.overlay.get(p) {
            return Some(d.clone());
        }
        let r = self.real_path(p)?;
        self.fs.read(GuestPathBuf::from(r)).ok()
    }

    // ----- generic UIKit properties -----

    fn prop(&self, obj: u64, key: &str) -> u64 {
        self.props.get(&(obj, key.to_string())).map_or(0, |p| p[0])
    }

    fn set_prop(&mut self, obj: u64, key: &str, v: u64) {
        self.props.insert((obj, key.to_string()), [v, 0, 0, 0, 0]);
    }

    /// `setFoo:` stores its argument (X2 and D0-D3, so objects, BOOLs, numbers
    /// and CGRects all survive) under "foo"; `foo` / `isFoo` returns it. This
    /// covers the many plain property accessors of UIKit classes.
    fn auto_property(&mut self, cpu: &mut Cpu64, recv: u64, sel: &str) -> bool {
        let colons = sel.matches(':').count();
        let lower_first = |s: &str| -> String {
            let mut c = s.chars();
            match c.next() {
                Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        };
        let is_setter = colons == 1
            && sel.ends_with(':')
            && sel.starts_with("set")
            && sel.len() > 4
            && sel.chars().nth(3).map_or(false, |c| c.is_uppercase());
        if is_setter {
            let key = lower_first(&sel[3..sel.len() - 1]);
            let vals = [
                cpu.reg(2),
                cpu.vec(0) as u64,
                cpu.vec(1) as u64,
                cpu.vec(2) as u64,
                cpu.vec(3) as u64,
            ];
            self.props.insert((recv, key), vals);
            *self.auto_seen.entry(sel.to_string()).or_insert(0) += 1;
            return true;
        }
        if colons == 0 {
            let key = if sel.starts_with("is") && sel.chars().nth(2).map_or(false, |c| c.is_uppercase()) {
                lower_first(&sel[2..])
            } else {
                sel.to_string()
            };
            if let Some(v) = self.props.get(&(recv, key.clone())).copied() {
                cpu.set_reg(0, v[0]);
                for i in 0..4 {
                    cpu.set_vec(i, v[i + 1] as u128);
                }
                *self.auto_seen.entry(sel.to_string()).or_insert(0) += 1;
                return true;
            }
            if key == "bounds" {
                if let Some(f) = self.props.get(&(recv, "frame".to_string())).copied() {
                    cpu.set_reg(0, 0);
                    cpu.set_vec(0, 0);
                    cpu.set_vec(1, 0);
                    cpu.set_vec(2, f[3] as u128);
                    cpu.set_vec(3, f[4] as u128);
                    return true;
                }
            }
        }
        false
    }

    /// A plain full-screen UIView.
    fn default_view(&mut self, mem: &mut Mem64) -> u64 {
        let (cls, _) = self.objc.system_class(mem, &mut self.heap, "UIView");
        let v = self.alloc_instance(mem, cls);
        self.props.insert(
            (v, "frame".to_string()),
            [0, 0, 0, SCREEN_W.to_bits(), SCREEN_H.to_bits()],
        );
        v
    }

    // ----- blocks and the run loop -----

    /// `Block_copy`: stack blocks are copied to the heap so they outlive
    /// the function that made them. (Helpers for captured objects are not run.)
    fn copy_block(&mut self, mem: &mut Mem64, block: u64) -> u64 {
        if block == 0 {
            return 0;
        }
        const NEEDS_FREE: u32 = 1 << 24;
        const IS_GLOBAL: u32 = 1 << 28;
        let flags = rd32(mem, block.wrapping_add(8)).unwrap_or(0);
        if flags & (NEEDS_FREE | IS_GLOBAL) != 0 {
            return block;
        }
        // Block layout: isa, flags, reserved, invoke, descriptor{reserved, size}.
        let size = mem
            .read_u64(block.wrapping_add(24))
            .and_then(|d| mem.read_u64(d.wrapping_add(8)))
            .unwrap_or(0);
        if size < 32 || size > 0x10_0000 {
            return block;
        }
        let Some(data) = mem.read_bytes(block, size as usize).map(|s| s.to_vec()) else {
            return block;
        };
        let p = self.heap.alloc(size);
        mem.write_slice(p, &data);
        mem.write_slice(p.wrapping_add(8), &(flags | NEEDS_FREE).to_le_bytes());
        p
    }

    fn schedule(&mut self, due: u64, repeat: u64, action: Action, handle: u64) {
        self.scheduled.push(Scheduled { due, repeat, action, handle });
    }

    /// Start an NSTimer that was created with its properties filled in.
    fn schedule_timer(&mut self, t: u64) {
        let interval = f64::from_bits(self.prop(t, "interval"));
        let ns = (interval.max(0.0) * 1e9) as u64;
        let repeat = if self.prop(t, "repeats") != 0 { ns.max(1_000_000) } else { 0 };
        let (target, sel) = (self.prop(t, "target"), self.prop(t, "selector"));
        let due = self.vtime + ns;
        self.schedule(due, repeat, Action::Send { target, sel, args: vec![t] }, t);
    }

    /// Contents of a libc++ `std::string` (Apple arm64 layout: short strings keep
    /// their characters in bytes 0..22 and the length in byte 23; a long string
    /// is { data pointer, length, capacity | 1 << 63 }).
    fn std_string_read(&self, mem: &Mem64, this: u64) -> Vec<u8> {
        let last = mem.read_bytes(this.wrapping_add(23), 1).map_or(0, |b| b[0]);
        if last & 0x80 == 0 {
            let n = (last as usize).min(23);
            mem.read_bytes(this, n).map(|b| b.to_vec()).unwrap_or_default()
        } else {
            let p = mem.read_u64(this).unwrap_or(0);
            let n = mem.read_u64(this.wrapping_add(8)).unwrap_or(0) as usize;
            mem.read_bytes(p, n.min(1 << 24)).map(|b| b.to_vec()).unwrap_or_default()
        }
    }

    /// Store `bytes` into the `std::string` at `this` (the old buffer is leaked).
    fn std_string_write(&mut self, mem: &mut Mem64, this: u64, bytes: &[u8]) {
        let n = bytes.len();
        if n <= 22 {
            let mut buf = [0u8; 24];
            buf[..n].copy_from_slice(bytes);
            buf[23] = n as u8;
            mem.write_slice(this, &buf);
        } else {
            let cap = (n as u64 + 1 + 15) & !15;
            let p = self.alloc(cap);
            mem.write_slice(p, bytes); // heap memory is zeroed, so it ends in NUL
            mem.write_u64(this, p);
            mem.write_u64(this.wrapping_add(8), n as u64);
            mem.write_u64(this.wrapping_add(16), cap | (1u64 << 63));
        }
    }

    fn handle(&mut self, cpu: &mut Cpu64, mem: &mut Mem64, id: u32) -> Flow {
        let Some(&kind) = self.kinds.get(id as usize) else {
            return Flow::Done(format!("svc #{id} does not match any stub"));
        };
        self.calls += 1;
        self.total_calls += 1;
        *self.counts.entry(id).or_insert(0) += 1;
        let lr = cpu.lr();
        let stub_addr = STUB_BASE + id as u64 * 8;
        let name = &self.names[id as usize];
        if self.recent.len() >= 80 {
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
            Kind::ClassGetInstanceMethod | Kind::ClassGetClassMethod => {
                let sel = self.objc.sel_name(mem, x1);
                let start = if kind == Kind::ClassGetClassMethod {
                    mem.read_u64(x0).unwrap_or(0)
                } else {
                    x0
                };
                let h = match self.objc.find_owner(start, &sel) {
                    Some((owner, _)) => self.objc.method_handle(mem, &mut self.heap, owner, &sel),
                    None => 0,
                };
                cpu.set_reg(0, h);
            }
            Kind::MethodGetImplementation => {
                let imp = if x0 == 0 { 0 } else { mem.read_u64(x0.wrapping_add(16)).unwrap_or(0) };
                cpu.set_reg(0, imp);
            }
            Kind::MethodGetTypeEncoding => {
                // An empty C string.
                let p = self.heap.alloc(16);
                cpu.set_reg(0, p);
            }
            Kind::MethodSetImplementation => {
                let old = self.objc.set_method_imp(mem, x0, x1);
                cpu.set_reg(0, old);
            }
            Kind::MethodExchange if x0 == 0 || x1 == 0 => {}
            Kind::MethodExchange => {
                let i1 = mem.read_u64(x0.wrapping_add(16)).unwrap_or(0);
                let i2 = mem.read_u64(x1.wrapping_add(16)).unwrap_or(0);
                self.objc.set_method_imp(mem, x0, i2);
                self.objc.set_method_imp(mem, x1, i1);
            }
            Kind::ClassAddMethod => {
                let sel = self.objc.sel_name(mem, x1);
                let added = match self.objc.classes.get_mut(&x0) {
                    Some(c) if !c.methods.contains_key(&sel) => {
                        c.methods.insert(sel, x2);
                        true
                    }
                    _ => false,
                };
                cpu.set_reg(0, added as u64);
            }
            Kind::ClassReplaceMethod => {
                let sel = self.objc.sel_name(mem, x1);
                let old = self.objc.classes.get_mut(&x0).and_then(|c| c.methods.insert(sel.clone(), x2));
                if x2 == 0 {
                    // Swizzling: the original implementation (x2) is a system
                    // one we do not have. The selector that just received the
                    // old implementation of `sel` is the system selector.
                    if let Some(prev) = old {
                        let orig = self.objc.classes.get(&x0).and_then(|c| {
                            c.methods.iter().find(|(s, i)| **i == prev && **s != sel).map(|(s, _)| s.clone())
                        });
                        if let Some(orig_sel) = orig {
                            self.objc.sys_alias.insert((x0, sel.clone()), orig_sel);
                        }
                    }
                }
                cpu.set_reg(0, old.unwrap_or(0));
            }
            Kind::ObjcGetProperty => {
                // (self, _cmd, offset, atomic)
                let v = mem.read_u64(x0.wrapping_add(x2)).unwrap_or(0);
                cpu.set_reg(0, v);
            }
            Kind::ObjcSetProperty => {
                // (self, _cmd, newValue, offset, ...)
                mem.write_u64(x0.wrapping_add(cpu.reg(3)), x2);
            }
            Kind::GuardAcquire => {
                // 1 = the caller must run the initializer
                let done = mem.read_bytes(x0, 1).map_or(true, |b| b[0] != 0);
                cpu.set_reg(0, if done { 0 } else { 1 });
            }
            Kind::GuardRelease => {
                mem.write_slice(x0, &[1u8]);
            }
            Kind::DyldImageCount => cpu.set_reg(0, 1),
            Kind::DyldImageName => {
                if x0 == 0 {
                    if self.exe_path_ptr == 0 {
                        let path = format!("{}/{}", self.bundle_path, self.exe_name);
                        self.exe_path_ptr = self.alloc_cstr(mem, &path);
                    }
                    cpu.set_reg(0, self.exe_path_ptr);
                } else {
                    cpu.set_reg(0, 0);
                }
            }
            Kind::NsSearchPath => {
                let dir = match x0 {
                    5 => format!("{SANDBOX}/Library"),
                    13 => format!("{SANDBOX}/Library/Caches"),
                    14 => format!("{SANDBOX}/Library/Application Support"),
                    _ => format!("{SANDBOX}/Documents"),
                };
                let s = self.new_string(mem, &dir);
                let arr = self.new_obj(mem, "NSArray", Obj::Array(vec![s]));
                cpu.set_reg(0, arr);
            }
            Kind::NsLog => {
                let fmt = self.str_of(mem, x0).unwrap_or_default();
                let text = self.ns_format(mem, &fmt, cpu.sp());
                log!("[a64] NSLog: {text}");
            }
            Kind::NewHandle => {
                let p = self.heap.alloc(32);
                cpu.set_reg(0, p);
            }
            Kind::ReturnOne => cpu.set_reg(0, 1),
            Kind::BlockObjectAssign => {
                mem.write_u64(x0, x1);
            }
            Kind::BlockCopy => {
                let p = self.copy_block(mem, x0);
                cpu.set_reg(0, p);
            }
            Kind::DispatchAsync => {
                let b = self.copy_block(mem, x1);
                let at = self.vtime;
                self.schedule(at, 0, Action::Block { block: b, args: vec![] }, 0);
            }
            Kind::DispatchAfter => {
                let b = self.copy_block(mem, x2);
                let at = x0.max(self.vtime);
                self.schedule(at, 0, Action::Block { block: b, args: vec![] }, 0);
            }
            Kind::DispatchGroupAsync | Kind::DispatchGroupNotify => {
                let b = self.copy_block(mem, x2);
                let at = self.vtime;
                self.schedule(at, 0, Action::Block { block: b, args: vec![] }, 0);
            }
            Kind::DispatchSync => match mem.read_u64(x1.wrapping_add(16)) {
                Some(invoke) if invoke != 0 => {
                    return Flow::CallGuest { pc: invoke, args: vec![x1], result: 0, retry_at: None }
                }
                _ => return Flow::Done(format!("dispatch_sync: bad block {x1:#x}")),
            },
            Kind::DispatchTime => {
                let base = if x0 == 0 { self.vtime } else { x0 };
                let r = if x0 == u64::MAX { u64::MAX } else { (base as i64).wrapping_add(x1 as i64) as u64 };
                cpu.set_reg(0, r);
            }
            Kind::ObjcCopyStruct => {
                // (dest, src, size, atomic, hasStrong)
                let n = x2 as usize;
                if n <= 0x10_0000 {
                    if let Some(d) = mem.read_bytes(x1, n).map(|s| s.to_vec()) {
                        mem.write_slice(x0, &d);
                    }
                }
            }
            Kind::DyldImageHeader => {
                let h = if x0 == 0 { self.image_base } else { 0 };
                cpu.set_reg(0, h);
            }
            Kind::MathF64 => {
                let (a, b) = (cpu.f64(0), cpu.f64(1));
                let r = match name.as_str() {
                    "_pow" => a.powf(b),
                    "_sin" => a.sin(),
                    "_cos" => a.cos(),
                    "_tan" => a.tan(),
                    "_atan" => a.atan(),
                    "_atan2" => a.atan2(b),
                    "_sqrt" => a.sqrt(),
                    "_exp" => a.exp(),
                    "_log" => a.ln(),
                    "_log10" => a.log10(),
                    "_floor" => a.floor(),
                    "_ceil" => a.ceil(),
                    "_fmod" => a % b,
                    "_asin" => a.asin(),
                    "_acos" => a.acos(),
                    _ => 0.0,
                };
                cpu.set_f64(0, r);
            }
            Kind::MathF32 => {
                let (a, b) = (cpu.f32(0), cpu.f32(1));
                let r = match name.as_str() {
                    "_powf" => a.powf(b),
                    "_sinf" => a.sin(),
                    "_cosf" => a.cos(),
                    "_tanf" => a.tan(),
                    "_atanf" => a.atan(),
                    "_atan2f" => a.atan2(b),
                    "_sqrtf" => a.sqrt(),
                    "_expf" => a.exp(),
                    "_logf" => a.ln(),
                    "_floorf" => a.floor(),
                    "_ceilf" => a.ceil(),
                    "_fmodf" => a % b,
                    "_asinf" => a.asin(),
                    "_acosf" => a.acos(),
                    _ => 0.0,
                };
                cpu.set_f32(0, r);
            }
            Kind::Printf => {
                // The app's own diagnostics (assertions, debug output) show up in our log.
                let fmt = Objc::cstr(mem, x0).unwrap_or_default();
                let text = self.ns_format(mem, &fmt, cpu.sp());
                self.log_once(format!("printf: {}", text.trim_end()));
                cpu.set_reg(0, text.len() as u64);
            }
            Kind::Puts => {
                let text = Objc::cstr(mem, x0).unwrap_or_default();
                self.log_once(format!("puts: {text}"));
                cpu.set_reg(0, 1);
            }
            Kind::StdStringInit => {
                // basic_string::__init(const char *s, size_t n)
                let bytes = mem.read_bytes(x1, x2.min(1 << 24) as usize).map(|b| b.to_vec()).unwrap_or_default();
                self.std_string_write(mem, x0, &bytes);
            }
            Kind::StdStringAssign => {
                let bytes = mem.read_c_bytes(x1, 1 << 24).unwrap_or_default();
                self.std_string_write(mem, x0, &bytes);
                cpu.set_reg(0, x0);
            }
            Kind::StdStringAssignChar => {
                self.std_string_write(mem, x0, &[x1 as u8]);
                cpu.set_reg(0, x0);
            }
            Kind::StdStringCopy => {
                let bytes = self.std_string_read(mem, x1);
                self.std_string_write(mem, x0, &bytes);
                cpu.set_reg(0, x0);
            }
            Kind::StdStringCompare => {
                let a = self.std_string_read(mem, x0);
                let b = mem.read_c_bytes(x1, 1 << 24).unwrap_or_default();
                let r: i64 = match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                cpu.set_reg(0, r as u64);
            }
            Kind::NsTemporaryDirectory => {
                let s = format!("{SANDBOX}/tmp/");
                let o = self.new_string(mem, &s);
                cpu.set_reg(0, o);
            }
            Kind::Snprintf => {
                // snprintf(buf, size, fmt, ...): the variadic arguments are on the stack.
                let fmt = Objc::cstr(mem, x2).unwrap_or_default();
                let text = self.ns_format(mem, &fmt, cpu.sp());
                let bytes = text.as_bytes();
                if x0 != 0 && x1 > 0 {
                    let n = bytes.len().min(x1 as usize - 1);
                    let mut out = bytes[..n].to_vec();
                    out.push(0);
                    mem.write_slice(x0, &out);
                }
                cpu.set_reg(0, bytes.len() as u64);
            }
            Kind::Strnlen => {
                let n = mem.read_c_bytes(x0, x1.min(1 << 24) as usize).map_or(0, |b| b.len());
                cpu.set_reg(0, n as u64);
            }
            Kind::CfStringCreate => {
                // CFStringCreateWithCString(allocator, cstr, encoding)
                let text = Objc::cstr(mem, x1).unwrap_or_default();
                let o = self.new_string(mem, &text);
                cpu.set_reg(0, o);
            }
            Kind::CfDictionaryCreate => {
                // CFDictionaryCreate(allocator, keys[], values[], count, ...)
                let d = self.new_obj(mem, "NSDictionary", Obj::Dict(Vec::new()));
                for i in 0..cpu.reg(3).min(4096) {
                    let key = mem.read_u64(x1.wrapping_add(i * 8)).unwrap_or(0);
                    let val = mem.read_u64(x2.wrapping_add(i * 8)).unwrap_or(0);
                    if key != 0 {
                        self.dict_set(mem, d, key, val);
                    }
                }
                cpu.set_reg(0, d);
            }
            Kind::Socketpair => {
                // socketpair(domain, type, protocol, int sv[2]): two made-up descriptors.
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&3u32.to_le_bytes());
                b[4..].copy_from_slice(&4u32.to_le_bytes());
                mem.write_slice(cpu.reg(3), &b);
                cpu.set_reg(0, 0);
            }
            Kind::ObjcStoreWeak => {
                // id objc_storeWeak(id *location, id obj): no zeroing weak table.
                mem.write_u64(x0, x1);
                cpu.set_reg(0, x1);
            }
            Kind::Vsnprintf => {
                // vsnprintf(buf, size, fmt, va_list): on Apple arm64 a va_list is a
                // pointer to 8-byte argument slots, like our stack arguments.
                let fmt = Objc::cstr(mem, x2).unwrap_or_default();
                let text = self.ns_format(mem, &fmt, cpu.reg(3));
                let bytes = text.as_bytes();
                if x0 != 0 && x1 > 0 {
                    let n = bytes.len().min(x1 as usize - 1);
                    let mut out = bytes[..n].to_vec();
                    out.push(0);
                    mem.write_slice(x0, &out);
                }
                cpu.set_reg(0, bytes.len() as u64);
            }
            Kind::SecItemNotFound => {
                // errSecItemNotFound: there is no keychain.
                cpu.set_reg(0, (-25300i64) as u64);
            }
            Kind::Dlsym => {
                // dlsym(handle, name): give out the stub of an imported symbol
                // of that name. Any other symbol gets its own address (a
                // zeroed heap block for data, a no-op stub for functions), so
                // the app's "symbol not found" checks do not fire and two
                // different names never share one address. Names are logged.
                let sym = Objc::cstr(mem, x1).unwrap_or_default();
                if self.seen_logs.insert("dlsym-code-dump".to_string()) {
                    // The code around the first dlsym call site, to see what
                    // the app does with the result.
                    let start = lr.wrapping_sub(0x80) & !3;
                    if let Some(bytes) = mem.read_bytes(start, 0x120).map(|b| b.to_vec()) {
                        for (i, row) in bytes.chunks(16).enumerate() {
                            let words: Vec<String> = row
                                .chunks(4)
                                .map(|w| format!("{:08x}", u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
                                .collect();
                            log!("[a64] code {:#x}: {}", start + i as u64 * 16, words.join(" "));
                        }
                    }
                }
                let key = format!("dlsym:{sym}");
                let want = format!("_{sym}");
                let stub_of = |i: usize| STUB_BASE + i as u64 * 8;
                let (addr, how) = if let Some(&a) = self.singletons.get(&key) {
                    (a, "again")
                } else if let Some(i) = self.names.iter().position(|n| *n == want) {
                    let a = stub_of(i);
                    self.singletons.insert(key, a);
                    (a, "imported symbol")
                } else if sym.contains("vtable") || sym.starts_with("OBJC_") {
                    let a = self.alloc(64);
                    self.singletons.insert(key, a);
                    (a, "not imported: zeroed data block")
                } else {
                    let n = self.singletons.keys().filter(|k| k.starts_with("dlsym:")).count();
                    let zeros: Vec<usize> = self
                        .kinds
                        .iter()
                        .enumerate()
                        .filter(|(_, k)| **k == Kind::ReturnZero)
                        .map(|(i, _)| i)
                        .collect();
                    let a = if zeros.is_empty() { 0 } else { stub_of(zeros[n % zeros.len()]) };
                    self.singletons.insert(key, a);
                    (a, "not imported: no-op stub")
                };
                if how != "again" {
                    log!("[a64] dlsym({sym:?}) -> {addr:#x} ({how})");
                }
                cpu.set_reg(0, addr);
            }
            Kind::Gl => {
                let n = name.clone();
                self.gl_call(cpu, mem, &n);
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

struct Machine<'a> {
    cpu: Cpu64,
    mem: Mem64,
    hle: Hle<'a>,
    trap_addr: u64,
    text_base: u64,
    image_end: u64,
    /// Defined symbols sorted by address, for describing crash addresses.
    syms: Vec<(u64, String)>,
}

impl<'a> Machine<'a> {
    /// Call guest function `pc` from the top level and run until it returns.
    fn call(&mut self, pc: u64, args: &[u64], what: &str) -> CallEnd {
        // The loop watchdog counts per top-level call. Without this reset one
        // runaway initializer made every later initializer and main() fail.
        self.hle.calls = 0;
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
        let app = self.hle.singleton(&mut self.mem, "UIApplication", "UIApplication", Obj::Plain);
        let mut obj = self.hle.alloc_instance(&mut self.mem, cls);
        self.hle.set_prop(app, "delegate", obj);
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
        // The launch options are nil for now.
        self.hle.calls = 0;
        match self.send(obj, "application:didFinishLaunchingWithOptions:", &[app, 0]) {
            CallEnd::Returned(r) => {
                echo!("[a64] application:didFinishLaunchingWithOptions: returned {r:#x}");
            }
            CallEnd::Stopped(msg) => {
                echo!("[a64] application:didFinishLaunchingWithOptions: stopped: {msg}");
                self.brief_state();
            }
        }
        // Threads the app started (often its game loop). Each runs for a
        // limited time, so an endless loop ends up in the log, not as a hang.
        let threads = std::mem::take(&mut self.hle.pending_threads);
        for (target, sel_ptr, arg) in threads {
            let sel = self.hle.objc.sel_name(&self.mem, sel_ptr);
            let cname = self
                .hle
                .objc
                .class_of(&self.mem, target)
                .map(|c| self.hle.class_name(c))
                .unwrap_or_default();
            echo!("[a64] running a thread the app started: -[{cname} {sel}]");
            self.hle.calls = 0;
            match self.send(target, &sel, &[arg]) {
                CallEnd::Returned(_) => echo!("[a64] the thread function returned"),
                CallEnd::Stopped(msg) => {
                    echo!("[a64] the thread stopped: {msg}");
                    self.brief_state();
                }
            }
        }
        self.show_windows();
        self.run_loop(u32::MAX);
    }

    /// Run one entry of the run loop.
    fn exec_action(&mut self, action: Action) -> Result<(), String> {
        match action {
            Action::Send { target, sel, args } => {
                let name = self.hle.objc.sel_name(&self.mem, sel);
                match self.send(target, &name, &args) {
                    CallEnd::Stopped(msg) => Err(format!("{name}: {msg}")),
                    CallEnd::Returned(_) => Ok(()),
                }
            }
            Action::Block { block, args } => {
                let Some(invoke) = self.mem.read_u64(block.wrapping_add(16)) else {
                    return Err(format!("block {block:#x} is not readable"));
                };
                let mut a = vec![block];
                a.extend(args);
                match self.call(invoke, &a, "block") {
                    CallEnd::Stopped(msg) => Err(format!("block {block:#x}: {msg}")),
                    CallEnd::Returned(_) => Ok(()),
                }
            }
        }
    }

    /// A lifecycle message to a view controller; failures are logged and skipped.
    fn lifecycle(&mut self, vc: u64, sel: &str, args: &[u64]) {
        self.hle.calls = 0;
        if let CallEnd::Stopped(msg) = self.send(vc, sel, args) {
            echo!("[a64] {sel} FAILED: {msg}");
            self.brief_state();
        }
    }

    /// What UIKit does when a window with a root view controller becomes
    /// visible: load its view, then tell it that it appears.
    fn show_windows(&mut self) {
        let windows = std::mem::take(&mut self.hle.visible_windows);
        for w in windows {
            let vc = self.hle.prop(w, "rootViewController");
            echo!("[a64] a window became visible (root view controller {vc:#x})");
            if vc == 0 {
                continue;
            }
            if self.hle.prop(vc, "__loaded") == 0 {
                self.hle.set_prop(vc, "__loaded", 1);
                self.lifecycle(vc, "loadView", &[]);
                if self.hle.prop(vc, "view") == 0 {
                    let v = self.hle.default_view(&mut self.mem);
                    self.hle.set_prop(vc, "view", v);
                }
                self.lifecycle(vc, "viewDidLoad", &[]);
            }
            self.lifecycle(vc, "viewWillAppear:", &[0]);
            self.lifecycle(vc, "viewDidAppear:", &[0]);
        }
    }

    /// A small stand-in for the main run loop: every frame (1/60 s of virtual
    /// time) runs what is due (timers, dispatched blocks) and the display
    /// links. Stops at the first failure.
    fn run_loop(&mut self, frames: u32) {
        echo!("[a64] main run loop: running up to {frames} frames");
        for frame in 0..frames {
            self.hle.vtime += 16_666_667;
            let presented_before = self.hle.gl.as_ref().map_or(0, |g| g.presented);
            self.hle.window.poll_for_events(self.hle.options);
            while let Some(ev) = self.hle.window.pop_event() {
                if matches!(ev, crate::window::Event::Quit | crate::window::Event::CloseWindow) {
                    echo!("[a64] the window was closed after {frame} frames");
                    return;
                }
            }
            let now = self.hle.vtime;
            let mut due = Vec::new();
            let mut later = Vec::new();
            for s in std::mem::take(&mut self.hle.scheduled) {
                if s.due <= now {
                    due.push(s);
                } else {
                    later.push(s);
                }
            }
            self.hle.scheduled = later;
            due.sort_by_key(|s| s.due);
            for s in due {
                if s.repeat > 0 {
                    let mut again = s.clone();
                    again.due = now + s.repeat;
                    self.hle.scheduled.push(again);
                }
                self.hle.calls = 0;
                if let Err(msg) = self.exec_action(s.action) {
                    echo!("[a64] the run loop stopped in frame {frame}: {msg}");
                    self.brief_state();
                    return;
                }
            }
            for (link, target, sel) in self.hle.links.clone() {
                self.hle.calls = 0;
                let action = Action::Send { target, sel, args: vec![link] };
                if let Err(msg) = self.exec_action(action) {
                    echo!("[a64] the run loop stopped in frame {frame} (display link): {msg}");
                    self.brief_state();
                    return;
                }
            }
            if !self.hle.visible_windows.is_empty() {
                self.show_windows();
            }
            // Without a shown frame nothing paces the loop; do it ourselves.
            let presented_after = self.hle.gl.as_ref().map_or(0, |g| g.presented);
            if presented_after == presented_before {
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            if frame == 120 || frame == 1800 {
                self.summary();
            }
        }
        echo!("[a64] the run loop ran all {frames} frames without a failure");
    }

    /// Short version of `dump_state`, for failures we then carry on from.
    /// Hex dump of the instructions around `around` (to decode a crash site).
    fn dump_code(&self, around: u64) {
        let start = around.wrapping_sub(0x40) & !3;
        if let Some(bytes) = self.mem.read_bytes(start, 0x80) {
            for (i, row) in bytes.chunks(16).enumerate() {
                let words: Vec<String> = row
                    .chunks(4)
                    .map(|w| format!("{:08x}", u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
                    .collect();
                log!("[a64]   code {:#x}: {}", start + i as u64 * 16, words.join(" "));
            }
        }
    }

    fn brief_state(&self) {
        log!("[a64]   pc = {}", self.describe(self.cpu.pc()));
        log!("[a64]   lr = {}", self.describe(self.cpu.lr()));
        self.dump_code(self.cpu.lr());
        if self.cpu.pc() != self.cpu.lr() {
            log!("[a64]   code around pc:");
            self.dump_code(self.cpu.pc());
        }
        let skip = self.hle.recent.len().saturating_sub(24);
        for c in self.hle.recent.iter().skip(skip) {
            log!("[a64]   recent: {c}");
        }
    }

    fn summary(&self) {
        let mut called: Vec<(u32, u64)> = self.hle.counts.iter().map(|(k, v)| (*k, *v)).collect();
        called.sort_by(|a, b| b.1.cmp(&a.1));
        log!(
            "[a64] {} imported-function calls in total, {} different functions",
            self.hle.total_calls,
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
        let mut auto: Vec<(&String, &u64)> = self.hle.auto_seen.iter().collect();
        auto.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
        log!("[a64] {} UIKit property accessors were handled generically (stored and returned as set):", auto.len());
        for (k, n) in auto.iter().take(150) {
            log!("[a64]   {:>8} x  {}", n, k);
        }
    }
}

/// Entry point, called from `main` for 64-bit-only apps.
pub fn run(
    bundle: &Bundle,
    fs: &Fs,
    exe: &[u8],
    window: &mut Window,
    options: &Options,
) -> Result<(), String> {
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
    // The app wraps __cxa_throw and finds the real one with dlsym, so the table
    // always has a stub for it (a throw ends the running guest call).
    if !names.iter().any(|n| n.as_str() == "___cxa_throw") {
        names.push("___cxa_throw".to_string());
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
        objs: HashMap::new(),
        singletons: HashMap::new(),
        cstr_cache: HashMap::new(),
        pending_threads: Vec::new(),
        thread_inits: HashMap::new(),
        bundle_path: format!("/var/containers/Bundle/Application/00000000-0000-0000-0000-000000000000/{name}.app"),
        bundle_id: bundle.bundle_identifier().to_string(),
        exe_name: name.clone(),
        exe_path_ptr: 0,
        seen_logs: HashSet::new(),
        info_plist: bundle.info_plist_strings().into_iter().collect(),
        fs,
        window,
        options,
        gl: None,
        current_ctx: 0,
        gl_err: 0,
        gl_err_logged: 0,
        gl_strings: HashMap::new(),
        unpack_align: 4,
        bundle_guest: bundle.bundle_path().as_str().to_string(),
        overlay: HashMap::new(),
        props: HashMap::new(),
        visible_windows: Vec::new(),
        vtime: 0,
        scheduled: Vec::new(),
        links: Vec::new(),
        total_calls: 0,
        auto_seen: HashMap::new(),
        data_ptrs: HashMap::new(),
        image_base: macho.text_base,
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

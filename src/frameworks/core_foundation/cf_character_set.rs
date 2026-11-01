/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! `CFCharacterSet`.
//!
//! `CFCharacterSet` is toll-free bridged with `NSCharacterSet`, and this
//! implementation already has `NSCharacterSet` / `NSMutableCharacterSet`
//! (see `foundation::ns_character_set`), so the functions here are thin
//! wrappers that forward to those classes.
//!
//! Only the Basic Multilingual Plane is tracked by `NSCharacterSet` here, so
//! code points above U+FFFF are never reported as members.
//!
//! References:
//! - <https://developer.apple.com/documentation/corefoundation/cfcharacterset>

use crate::dyld::{export_c_func, FunctionExports};
use crate::frameworks::core_foundation::cf_allocator::CFAllocatorRef;
use crate::frameworks::core_foundation::{CFRange, CFTypeRef};
use crate::frameworks::foundation::{NSRange, NSUInteger};
use crate::objc::{msg, msg_class, retain};
use crate::Environment;

/// Toll-free bridged with `NSCharacterSet`.
pub type CFCharacterSetRef = CFTypeRef;
/// Toll-free bridged with `NSMutableCharacterSet`.
pub type CFMutableCharacterSetRef = CFTypeRef;

/// `UTF32Char`
type UTF32Char = u32;

/// `CFMutableCharacterSetRef CFCharacterSetCreateMutable(CFAllocatorRef alloc)`
///
/// Follows the Create rule: the caller owns the returned (empty) set.
fn CFCharacterSetCreateMutable(
    env: &mut Environment,
    _alloc: CFAllocatorRef,
) -> CFMutableCharacterSetRef {
    msg_class![env; NSMutableCharacterSet new]
}

/// `void CFCharacterSetAddCharactersInRange(CFMutableCharacterSetRef theSet,
///     CFRange theRange)`
fn CFCharacterSetAddCharactersInRange(
    env: &mut Environment,
    set: CFMutableCharacterSetRef,
    range: CFRange,
) {
    if set.is_null() || range.location < 0 || range.length <= 0 {
        return;
    }
    let range = NSRange {
        location: range.location as NSUInteger,
        length: range.length as NSUInteger,
    };
    () = msg![env; set addCharactersInRange:range];
}

/// `CFCharacterSetRef CFCharacterSetCreateCopy(CFAllocatorRef alloc,
///     CFCharacterSetRef theSet)`
///
/// Follows the Create rule: the caller owns the returned set.
fn CFCharacterSetCreateCopy(
    env: &mut Environment,
    _alloc: CFAllocatorRef,
    set: CFCharacterSetRef,
) -> CFCharacterSetRef {
    if set.is_null() {
        return set;
    }
    let copy: CFCharacterSetRef = msg![env; set copy];
    // The `copyWithZone:` implementations in `ns_character_set` return an
    // autoreleased object rather than a +1 one. Retaining here guarantees the
    // caller's eventual `CFRelease` can never over-release it (worst case is a
    // small leak, if that is ever fixed upstream).
    retain(env, copy)
}

/// `Boolean CFCharacterSetIsLongCharacterMember(CFCharacterSetRef theSet,
///     UTF32Char theChar)`
fn CFCharacterSetIsLongCharacterMember(
    env: &mut Environment,
    set: CFCharacterSetRef,
    the_char: UTF32Char,
) -> bool {
    if set.is_null() {
        return false;
    }
    // `NSCharacterSet` only tracks UTF-16 code units from plane 0.
    let Ok(code_unit) = u16::try_from(the_char) else {
        return false;
    };
    msg![env; set characterIsMember:code_unit]
}

pub const FUNCTIONS: FunctionExports = &[
    export_c_func!(CFCharacterSetCreateMutable(_)),
    export_c_func!(CFCharacterSetAddCharactersInRange(_, _)),
    export_c_func!(CFCharacterSetCreateCopy(_, _)),
    export_c_func!(CFCharacterSetIsLongCharacterMember(_, _)),
];

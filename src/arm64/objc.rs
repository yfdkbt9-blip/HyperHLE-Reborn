/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Minimal 64-bit Objective-C runtime metadata used by the A64 HLE layer.
//! This registry does not yet parse __objc_* sections or implement ARC.

use std::collections::HashMap;

pub type ObjcClass = u64;
pub type ObjcSelector = u64;
pub type ObjcImp = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjcError {
    DuplicateClass(String),
    UnknownClass,
    UnknownSelector,
    InheritanceCycle,
    InvalidObjectPointer,
}

#[derive(Clone, Debug)]
pub struct ClassInfo {
    pub name: String,
    pub superclass: Option<ObjcClass>,
    pub instance_size: u32,
    methods: HashMap<ObjcSelector, ObjcImp>,
}

#[derive(Clone, Debug, Default)]
pub struct ObjcRuntime {
    classes: HashMap<ObjcClass, ClassInfo>,
    class_names: HashMap<String, ObjcClass>,
    selectors: HashMap<String, ObjcSelector>,
    selector_names: HashMap<ObjcSelector, String>,
    next_class: ObjcClass,
    next_selector: ObjcSelector,
}

impl ObjcRuntime {
    pub fn new() -> Self {
        Self { next_class: 1, next_selector: 1, ..Self::default() }
    }

    pub fn intern_selector(&mut self, name: &str) -> ObjcSelector {
        if let Some(selector) = self.selectors.get(name) { return *selector; }
        let selector = self.next_selector.max(1);
        self.next_selector = selector.saturating_add(1);
        self.selectors.insert(name.to_owned(), selector);
        self.selector_names.insert(selector, name.to_owned());
        selector
    }

    pub fn selector_name(&self, selector: ObjcSelector) -> Option<&str> {
        self.selector_names.get(&selector).map(String::as_str)
    }

    pub fn register_class(&mut self, name: impl Into<String>, superclass: Option<ObjcClass>, instance_size: u32) -> Result<ObjcClass, ObjcError> {
        let name = name.into();
        if self.class_names.contains_key(&name) { return Err(ObjcError::DuplicateClass(name)); }
        if let Some(parent) = superclass {
            if !self.classes.contains_key(&parent) { return Err(ObjcError::UnknownClass); }
        }
        let id = self.next_class.max(1);
        self.next_class = id.saturating_add(1);
        self.class_names.insert(name.clone(), id);
        self.classes.insert(id, ClassInfo { name, superclass, instance_size, methods: HashMap::new() });
        Ok(id)
    }

    pub fn class_by_name(&self, name: &str) -> Option<ObjcClass> {
        self.class_names.get(name).copied()
    }

    pub fn class(&self, class: ObjcClass) -> Option<&ClassInfo> { self.classes.get(&class) }

    pub fn add_method(&mut self, class: ObjcClass, selector: ObjcSelector, imp: ObjcImp) -> Result<(), ObjcError> {
        if !self.selector_names.contains_key(&selector) { return Err(ObjcError::UnknownSelector); }
        let info = self.classes.get_mut(&class).ok_or(ObjcError::UnknownClass)?;
        info.methods.insert(selector, imp);
        Ok(())
    }

    /// Search the class and then its superclasses, rejecting corrupt cycles.
    pub fn lookup_method(&self, mut class: ObjcClass, selector: ObjcSelector) -> Result<Option<ObjcImp>, ObjcError> {
        if !self.selector_names.contains_key(&selector) { return Err(ObjcError::UnknownSelector); }
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(class) { return Err(ObjcError::InheritanceCycle); }
            let info = self.classes.get(&class).ok_or(ObjcError::UnknownClass)?;
            if let Some(imp) = info.methods.get(&selector) { return Ok(Some(*imp)); }
            match info.superclass {
                Some(parent) => class = parent,
                None => return Ok(None),
            }
        }
    }

    /// Read the isa pointer from the first machine word of an Objective-C object.
    pub fn object_class(&self, object: u64, memory: &super::memory::GuestMemory) -> Result<ObjcClass, ObjcError> {
        if object == 0 { return Err(ObjcError::InvalidObjectPointer); }
        let mut bytes = [0u8; 8];
        memory.read(object, &mut bytes).map_err(|_| ObjcError::InvalidObjectPointer)?;
        let isa = u64::from_le_bytes(bytes);
        if !self.classes.contains_key(&isa) { return Err(ObjcError::UnknownClass); }
        Ok(isa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interns_selectors_and_resolves_inherited_methods() {
        let mut runtime = ObjcRuntime::new();
        let root = runtime.register_class("NSObject", None, 8).unwrap();
        let child = runtime.register_class("GameController", Some(root), 16).unwrap();
        let selector = runtime.intern_selector("description");
        assert_eq!(runtime.intern_selector("description"), selector);
        runtime.add_method(root, selector, 0x1_0000_1234).unwrap();
        assert_eq!(runtime.lookup_method(child, selector).unwrap(), Some(0x1_0000_1234));
        assert_eq!(runtime.selector_name(selector), Some("description"));
    }

    #[test]
    fn rejects_duplicate_classes_and_unknown_superclasses() {
        let mut runtime = ObjcRuntime::new();
        let root = runtime.register_class("NSObject", None, 8).unwrap();
        assert_eq!(runtime.register_class("NSObject", None, 8), Err(ObjcError::DuplicateClass("NSObject".into())));
        assert_eq!(runtime.register_class("Broken", Some(root + 10), 8), Err(ObjcError::UnknownClass));
    }

    #[test]
    fn reads_64_bit_isa_pointer_from_guest_memory() {
        let mut runtime = ObjcRuntime::new();
        let class = runtime.register_class("GameObject", None, 8).unwrap();
        let mut memory = super::super::memory::GuestMemory::new();
        memory.map_zeroed(0x1_0000_0000, 0x1000).unwrap();
        memory.write(0x1_0000_0000, &class.to_le_bytes()).unwrap();
        assert_eq!(runtime.object_class(0x1_0000_0000, &memory).unwrap(), class);
    }
}

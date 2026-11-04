/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! ARM64 guest address-space primitives.
//!
//! Kept deliberately separate from `crate::mem`: the latter models the
//! 32-bit address space and pointer ABI used by the existing A32 runtime.
//! This module starts the A64 path with sparse, demand-allocated pages rather
//! than attempting to reserve a contiguous 2^64-byte guest address space.

use std::collections::BTreeMap;
use std::marker::PhantomData;

/// A guest virtual address in the AArch64 address space.
pub type GuestAddr = u64;

/// Page size used by the initial A64 memory implementation.
pub const PAGE_SIZE: usize = 4096;
const PAGE_MASK: u64 = (PAGE_SIZE as u64) - 1;

/// Typed guest pointer. Its representation is always one 64-bit guest address,
/// never a host pointer.
#[repr(transparent)]
pub struct GuestPtr<T>(GuestAddr, PhantomData<*mut T>);

impl<T> Copy for GuestPtr<T> {}
impl<T> Clone for GuestPtr<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> PartialEq for GuestPtr<T> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl<T> Eq for GuestPtr<T> {}

impl<T> GuestPtr<T> {
    pub const fn null() -> Self { Self(0, PhantomData) }
    pub const fn from_addr(address: GuestAddr) -> Self { Self(address, PhantomData) }
    pub const fn address(self) -> GuestAddr { self.0 }
    pub const fn is_null(self) -> bool { self.0 == 0 }
    pub fn cast<U>(self) -> GuestPtr<U> { GuestPtr::from_addr(self.0) }
    pub fn wrapping_add_bytes(self, bytes: u64) -> Self {
        Self::from_addr(self.0.wrapping_add(bytes))
    }
}

impl<T> std::fmt::Debug for GuestPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() { write!(f, "(null)") } else { write!(f, "{:#018x}", self.0) }
    }
}

/// Errors returned by guest-memory operations. Unmapped memory is not silently
/// allocated on read: the CPU callback layer must turn this into a guest fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryError {
    AddressOverflow,
    Unmapped { address: GuestAddr },
}

/// Sparse A64 guest memory. Pages are allocated explicitly by the loader,
/// stack builder, or allocator; ordinary reads/writes never create mappings.
#[derive(Default)]
pub struct GuestMemory {
    pages: BTreeMap<GuestAddr, Box<[u8; PAGE_SIZE]>>,
}

impl GuestMemory {
    pub fn new() -> Self { Self::default() }

    fn page_base(address: GuestAddr) -> GuestAddr { address & !PAGE_MASK }

    /// Map zero-filled pages covering [address, address + size).
    /// Existing pages are retained and are not cleared.
    pub fn map_zeroed(&mut self, address: GuestAddr, size: u64) -> Result<(), MemoryError> {
        if size == 0 { return Ok(()); }
        let end = address.checked_add(size).ok_or(MemoryError::AddressOverflow)?;
        let mut page = Self::page_base(address);
        let last = Self::page_base(end - 1);
        loop {
            self.pages.entry(page).or_insert_with(|| Box::new([0; PAGE_SIZE]));
            if page == last { break; }
            page = page.checked_add(PAGE_SIZE as u64).ok_or(MemoryError::AddressOverflow)?;
        }
        Ok(())
    }

    /// Remove all pages intersecting [address, address + size).
    pub fn unmap(&mut self, address: GuestAddr, size: u64) -> Result<(), MemoryError> {
        if size == 0 { return Ok(()); }
        let end = address.checked_add(size).ok_or(MemoryError::AddressOverflow)?;
        let mut page = Self::page_base(address);
        let last = Self::page_base(end - 1);
        loop {
            self.pages.remove(&page);
            if page == last { break; }
            page = page.checked_add(PAGE_SIZE as u64).ok_or(MemoryError::AddressOverflow)?;
        }
        Ok(())
    }

    pub fn is_mapped(&self, address: GuestAddr, size: u64) -> bool {
        if size == 0 { return true; }
        let Some(end) = address.checked_add(size) else { return false; };
        let mut page = Self::page_base(address);
        let last = Self::page_base(end - 1);
        loop {
            if !self.pages.contains_key(&page) { return false; }
            if page == last { return true; }
            let Some(next) = page.checked_add(PAGE_SIZE as u64) else { return false; };
            page = next;
        }
    }

    pub fn read(&self, address: GuestAddr, output: &mut [u8]) -> Result<(), MemoryError> {
        let size = u64::try_from(output.len()).map_err(|_| MemoryError::AddressOverflow)?;
        let end = address.checked_add(size).ok_or(MemoryError::AddressOverflow)?;
        let mut current = address;
        let mut written = 0usize;
        while current < end {
            let base = Self::page_base(current);
            let page = self.pages.get(&base).ok_or(MemoryError::Unmapped { address: current })?;
            let offset = (current - base) as usize;
            let count = (PAGE_SIZE - offset).min(output.len() - written);
            output[written..written + count].copy_from_slice(&page[offset..offset + count]);
            written += count;
            current += count as u64;
        }
        Ok(())
    }

    pub fn write(&mut self, address: GuestAddr, input: &[u8]) -> Result<(), MemoryError> {
        let size = u64::try_from(input.len()).map_err(|_| MemoryError::AddressOverflow)?;
        let end = address.checked_add(size).ok_or(MemoryError::AddressOverflow)?;
        if !self.is_mapped(address, size) {
            return Err(MemoryError::Unmapped { address });
        }
        let mut current = address;
        let mut consumed = 0usize;
        while current < end {
            let base = Self::page_base(current);
            let page = self.pages.get_mut(&base).ok_or(MemoryError::Unmapped { address: current })?;
            let offset = (current - base) as usize;
            let count = (PAGE_SIZE - offset).min(input.len() - consumed);
            page[offset..offset + count].copy_from_slice(&input[consumed..consumed + count]);
            consumed += count;
            current += count as u64;
        }
        Ok(())
    }

    pub fn mapped_page_count(&self) -> usize { self.pages.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_addresses_above_four_gibibytes() {
        let address = 0x1_0000_1234;
        let mut memory = GuestMemory::new();
        memory.map_zeroed(address, 16).unwrap();
        memory.write(address, b"arm64 guest").unwrap();
        let mut bytes = [0; 11];
        memory.read(address, &mut bytes).unwrap();
        assert_eq!(&bytes, b"arm64 guest");
    }

    #[test]
    fn reads_and_writes_across_page_boundary() {
        let address = 0x2_0000_0ffb;
        let mut memory = GuestMemory::new();
        memory.map_zeroed(address, 16).unwrap();
        memory.write(address, b"cross-page").unwrap();
        let mut bytes = [0; 10];
        memory.read(address, &mut bytes).unwrap();
        assert_eq!(&bytes, b"cross-page");
    }

    #[test]
    fn unmapped_access_is_an_error() {
        let memory = GuestMemory::new();
        assert_eq!(memory.read(0x1_0000_0000, &mut [0]), Err(MemoryError::Unmapped { address: 0x1_0000_0000 }));
    }

    #[test]
    fn rejects_address_overflow() {
        let mut memory = GuestMemory::new();
        assert_eq!(memory.map_zeroed(u64::MAX - 3, 8), Err(MemoryError::AddressOverflow));
    }
}

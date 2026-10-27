//! Memory utilities — the "memory efficiency" pillar's foundations.
//!
//! * `Arena`: bump allocator for request-lifetime metadata (per-batch scratch
//!   that is allocated once and dropped wholesale). O(1) alloc, no per-object
//!   free, zero fragmentation by construction.
//! * `BytePool`: recycled byte buffers for request/response payloads
//!   (tokenizer I/O, SSE chunks). Amortizes allocation under churn, which is
//!   where most user-space serving allocations actually occur.
//!
//! The KV cache has its own dedicated block pool (kv_cache.rs) because KV
//! pages have fixed size + refcount semantics that deserve first-class types.

use std::collections::VecDeque;

/// Bump arena: allocate slices that all die together on `reset()`.
pub struct Arena {
    buf: Vec<u8>,
    used: usize,
    capacity_high_water: usize,
}

impl Arena {
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: vec![0u8; bytes],
            used: 0,
            capacity_high_water: bytes,
        }
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn used(&self) -> usize {
        self.used
    }

    /// Push `src` into the arena; returns offset into `self.buf`.
    /// Returns None if the remaining capacity is insufficient (caller decides
    /// whether to reset or grow — arenas here are per-batch, so reset is fine).
    pub fn push(&mut self, src: &[u8]) -> Option<usize> {
        let end = self.used.checked_add(src.len())?;
        if end > self.buf.len() {
            return None;
        }
        let off = self.used;
        self.buf[off..end].copy_from_slice(src);
        self.used = end;
        Some(off)
    }

    pub fn slice(&self, off: usize, len: usize) -> Option<&[u8]> {
        let end = off.checked_add(len)?;
        if end > self.used {
            return None;
        }
        Some(&self.buf[off..end])
    }

    pub fn high_water(&self) -> usize {
        self.capacity_high_water
    }
}

/// Recycled byte buffers. `take()` hands out a buffer (contents unspecified),
/// `recycle()` returns it. Internally a simple stack — LIFO maximizes cache
/// warmth. Unbounded by design for v1: payload sizes here are small; a hard
/// cap would add eviction policy for negligible memory win.
pub struct BytePool {
    free: VecDeque<Vec<u8>>,
    outstanding: usize,
    peak_outstanding: usize,
}

impl BytePool {
    pub fn new(preallocate: usize, buf_size: usize) -> Self {
        let mut free = VecDeque::new();
        for _ in 0..preallocate {
            free.push_back(vec![0u8; buf_size]);
        }
        Self {
            free,
            outstanding: 0,
            peak_outstanding: 0,
        }
    }

    pub fn take(&mut self, min_size: usize) -> Vec<u8> {
        self.outstanding += 1;
        self.peak_outstanding = self.peak_outstanding.max(self.outstanding);
        match self.free.pop_back() {
            Some(mut b) => {
                b.resize(min_size.max(16), 0);
                b
            }
            None => vec![0u8; min_size.max(16)],
        }
    }

    pub fn recycle(&mut self, buf: Vec<u8>) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let mut b = buf;
        b.clear();
        b.shrink_to(64 * 1024); // don't keep gigantic one-off buffers forever
        self.free.push_back(b);
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    pub fn peak_outstanding(&self) -> usize {
        self.peak_outstanding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_roundtrip_and_reset() {
        let mut a = Arena::with_capacity(64);
        let off = a.push(b"hello").unwrap();
        assert_eq!(a.slice(off, 5).unwrap(), b"hello");
        assert!(a.push(&[0u8; 100]).is_none()); // exceeds capacity
        a.reset();
        assert_eq!(a.used(), 0);
        let off2 = a.push(b"again").unwrap();
        assert_eq!(a.slice(off2, 5).unwrap(), b"again");
    }

    #[test]
    fn byte_pool_recycles() {
        let mut p = BytePool::new(2, 32);
        let b1 = p.take(32);
        let b2 = p.take(64); // grows beyond pool size
        assert_eq!(b2.len(), 64);
        assert_eq!(p.outstanding(), 2);
        p.recycle(b1);
        p.recycle(b2);
        assert_eq!(p.outstanding(), 0);
        assert_eq!(p.free.len(), 2);
        let b3 = p.take(16);
        assert_eq!(b3.len(), 16.max(16)); // resized on take
        assert_eq!(p.peak_outstanding(), 2);
    }
}

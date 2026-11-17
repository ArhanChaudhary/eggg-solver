//! Pruning-table storage: huge-page-backed buffers, packed tables, a parallel BFS generator,
//! and a simple checksummed file format.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// A zeroed, 2 MiB-aligned buffer of `T` (plain-old-data) backed by transparent huge pages
/// where the kernel allows it.
pub struct HugeBuf<T: Copy> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
}

unsafe impl<T: Copy + Send> Send for HugeBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for HugeBuf<T> {}

impl<T: Copy> HugeBuf<T> {
    pub fn zeroed(len: usize) -> HugeBuf<T> {
        let bytes = (len * std::mem::size_of::<T>())
            .max(1)
            .next_multiple_of(1 << 21);
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            assert!(p != libc::MAP_FAILED, "mmap of {bytes} bytes failed");
            libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
            HugeBuf {
                ptr: p as *mut T,
                len,
                bytes,
            }
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Copy> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.bytes);
        }
    }
}

impl<T: Copy> std::ops::Index<usize> for HugeBuf<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        debug_assert!(i < self.len);
        unsafe { &*self.ptr.add(i) }
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for HugeBuf<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        debug_assert!(i < self.len);
        unsafe { &mut *self.ptr.add(i) }
    }
}

#[inline(always)]
pub fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

/// Distances stored modulo 3 in 2 bits per entry (value 3 = unreachable). The exact distance
/// of a child differs from its parent's by at most one, so it can be decoded from the parent's.
pub struct Mod3Table {
    pub data: HugeBuf<u64>,
    pub n: u64,
}

impl Mod3Table {
    pub fn from_depths(depths: &[u8]) -> Mod3Table {
        let n = depths.len() as u64;
        let mut data = HugeBuf::<u64>::zeroed(depths.len().div_ceil(32));
        let words = data.as_mut_slice();
        crate::par::for_chunks(words.len(), 1 << 14, |range| {
            for w in range {
                let mut x = 0u64;
                for j in 0..32 {
                    let i = w * 32 + j;
                    let v = if i < depths.len() {
                        let d = depths[i];
                        if d == UNSEEN { 3 } else { d % 3 }
                    } else {
                        3
                    };
                    x |= (v as u64) << (2 * j);
                }
                unsafe { *(words.as_ptr() as *mut u64).add(w) = x };
            }
        });
        Mod3Table { data, n }
    }

    #[inline(always)]
    pub fn get(&self, i: u64) -> u8 {
        ((self.data[(i >> 5) as usize] >> ((i & 31) * 2)) & 3) as u8
    }

    #[inline(always)]
    pub fn prefetch(&self, i: u64) {
        prefetch(unsafe { self.data.as_ptr().add((i >> 5) as usize) });
    }

    /// The exact distance of a child whose parent is at exact distance `d`, from its stored
    /// value `v` (d - 1, d or d + 1).
    #[inline(always)]
    pub fn decode(d: u8, v: u8) -> u8 {
        // (v - (d - 1)) mod 3 gives 0, 1, 2 for d - 1, d, d + 1.
        d + ((v + 4 - d % 3) % 3) - 1
    }
}

/// Distances in 4 bits per entry (15 = unreachable).
pub struct NibbleTable {
    pub data: HugeBuf<u8>,
    pub n: u64,
}

impl NibbleTable {
    pub fn from_depths(depths: &[u8]) -> NibbleTable {
        let mut data = HugeBuf::<u8>::zeroed(depths.len().div_ceil(2));
        let bytes = data.as_mut_slice();
        for (i, b) in bytes.iter_mut().enumerate() {
            let lo = depths.get(2 * i).copied().unwrap_or(UNSEEN).min(15);
            let hi = depths.get(2 * i + 1).copied().unwrap_or(UNSEEN).min(15);
            *b = lo | (hi << 4);
        }
        NibbleTable {
            data,
            n: depths.len() as u64,
        }
    }

    #[inline(always)]
    pub fn get(&self, i: u64) -> u8 {
        (self.data[(i >> 1) as usize] >> ((i & 1) * 4)) & 15
    }

    #[inline(always)]
    pub fn prefetch(&self, i: u64) {
        prefetch(unsafe { self.data.as_ptr().add((i >> 1) as usize) });
    }
}

pub const UNSEEN: u8 = 0xff;

/// Breadth-first search over `n` states, from the `goals`, with a neighbour function. Moves
/// must be closed under inverse (so that the backward step is valid). Returns the depth of
/// every state (`UNSEEN` for unreachable states).
pub fn bfs<F>(n: usize, goals: &[u64], threads: usize, neighbours: F) -> Vec<u8>
where
    F: Fn(u64, &mut dyn FnMut(u64)) + Sync,
{
    let depth: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(UNSEEN)).collect();
    for &g in goals {
        depth[g as usize].store(0, Ordering::Relaxed);
    }
    let mut d = 0u8;
    let mut count = goals.len();
    let mut unseen = n - count;
    let mut frontier = count;
    while frontier > 0 && unseen > 0 {
        let forward = frontier < unseen / 4;
        let next = AtomicUsize::new(0);
        let chunk = 1 << 12;
        let new = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut local = 0usize;
                    loop {
                        let start = next.fetch_add(chunk, Ordering::Relaxed);
                        if start >= n {
                            break;
                        }
                        for i in start..(start + chunk).min(n) {
                            let di = depth[i].load(Ordering::Relaxed);
                            if forward {
                                if di == d {
                                    neighbours(i as u64, &mut |j| {
                                        if depth[j as usize]
                                            .compare_exchange(
                                                UNSEEN,
                                                d + 1,
                                                Ordering::Relaxed,
                                                Ordering::Relaxed,
                                            )
                                            .is_ok()
                                        {
                                            local += 1;
                                        }
                                    });
                                }
                            } else if di == UNSEEN {
                                let mut found = false;
                                neighbours(i as u64, &mut |j| {
                                    if !found && depth[j as usize].load(Ordering::Relaxed) == d {
                                        found = true;
                                    }
                                });
                                if found {
                                    depth[i].store(d + 1, Ordering::Relaxed);
                                    local += 1;
                                }
                            }
                        }
                    }
                    new.fetch_add(local, Ordering::Relaxed);
                });
            }
        });
        frontier = new.load(Ordering::Relaxed);
        count += frontier;
        unseen -= frontier;
        d += 1;
        if std::env::var_os("EGGG_DEBUG").is_some() {
            eprintln!("  bfs depth {d}: {frontier} ({count} of {n})");
        }
    }
    depth.into_iter().map(|a| a.into_inner()).collect()
}

/// Checksum of a byte slice (fast, not cryptographic).
pub fn checksum(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let (chunks, rest) = bytes.as_chunks::<8>();
    for c in chunks {
        h = (h ^ u64::from_le_bytes(*c))
            .wrapping_mul(0x100000001b3)
            .rotate_left(23);
    }
    for &b in rest {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

const MAGIC: &[u8; 8] = b"EGGGTAB1";

/// Save raw table bytes with a header: magic, name length + name, byte length, checksum.
pub fn save(path: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&(name.len() as u64).to_le_bytes())?;
        f.write_all(name.as_bytes())?;
        f.write_all(&(bytes.len() as u64).to_le_bytes())?;
        f.write_all(&checksum(bytes).to_le_bytes())?;
        f.write_all(bytes)?;
        f.flush()?;
    }
    std::fs::rename(tmp, path)
}

/// Load a table saved by [`save`] into `out` (which must have exactly the saved length).
pub fn load(path: &Path, name: &str, out: &mut [u8]) -> Result<(), String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    if &hdr[..8] != MAGIC {
        return Err("bad magic".into());
    }
    let nl = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let mut nm = vec![0u8; nl];
    f.read_exact(&mut nm).map_err(|e| e.to_string())?;
    if nm != name.as_bytes() {
        return Err(format!(
            "table name mismatch: {}",
            String::from_utf8_lossy(&nm)
        ));
    }
    let mut lc = [0u8; 16];
    f.read_exact(&mut lc).map_err(|e| e.to_string())?;
    let len = u64::from_le_bytes(lc[..8].try_into().unwrap()) as usize;
    let sum = u64::from_le_bytes(lc[8..].try_into().unwrap());
    if len != out.len() {
        return Err(format!("length mismatch: {len} vs {}", out.len()));
    }
    f.read_exact(out).map_err(|e| e.to_string())?;
    if checksum(out) != sum {
        return Err("checksum mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod3_decode() {
        for d in 1..20u8 {
            for delta in [-1i32, 0, 1] {
                let c = (d as i32 + delta) as u8;
                assert_eq!(Mod3Table::decode(d, c % 3), c);
            }
        }
    }

    #[test]
    fn bfs_on_a_cycle() {
        let n = 100;
        let depths = bfs(n, &[0], 4, |i, f| {
            f((i + 1) % n as u64);
            f((i + n as u64 - 1) % n as u64);
        });
        assert_eq!(depths[50], 50);
        assert_eq!(depths[99], 1);
    }
}

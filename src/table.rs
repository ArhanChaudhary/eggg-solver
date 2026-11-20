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
    /// The buffer as bytes (for loading and saving; `T` is plain data).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.byte_len()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.byte_len()) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Grow the buffer to `len` elements (the new ones zeroed). The mapping grows by half at
    /// least, and moves if it must without copying (Linux mremap); its untouched part takes
    /// no memory.
    pub fn grow(&mut self, len: usize) {
        assert!(len >= self.len);
        let need = (len * std::mem::size_of::<T>()).next_multiple_of(1 << 21);
        if need > self.bytes {
            let bytes = need.max((self.bytes + self.bytes / 2).next_multiple_of(1 << 21));
            unsafe {
                let p = libc::mremap(
                    self.ptr as *mut libc::c_void,
                    self.bytes,
                    bytes,
                    libc::MREMAP_MAYMOVE,
                );
                assert!(p != libc::MAP_FAILED, "mremap to {bytes} bytes failed");
                libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
                self.ptr = p as *mut T;
                self.bytes = bytes;
            }
        }
        self.len = len;
    }

    /// Shrink the buffer to `len` elements, unmapping what is left beyond them.
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        let bytes = (len * std::mem::size_of::<T>())
            .max(1)
            .next_multiple_of(1 << 21);
        if bytes < self.bytes {
            unsafe {
                libc::munmap(
                    (self.ptr as *mut u8).add(bytes) as *mut libc::c_void,
                    self.bytes - bytes,
                );
            }
            self.bytes = bytes;
        }
        self.len = len;
    }

    /// Ask the kernel to back the buffer with huge pages now (MADV_COLLAPSE, Linux 6.1+);
    /// tables are read at random, and 4 KiB pages make every lookup a TLB miss.
    pub fn collapse(&self) {
        const MADV_COLLAPSE: i32 = 25;
        unsafe {
            libc::madvise(self.ptr as *mut libc::c_void, self.bytes, MADV_COLLAPSE);
        }
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

/// Prefetch the cache line of `p` into all cache levels (a hint: `p` need not be valid).
#[inline(always)]
pub fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) p,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// Distances stored modulo 3 in 2 bits per entry (value 3 = unreachable). The exact distance
/// of a child differs from its parent's by at most one, so it can be decoded from the parent's.
pub struct Mod3Table {
    pub data: HugeBuf<u64>,
}

impl Mod3Table {
    pub fn from_depths(depths: &[u8]) -> Mod3Table {
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
        Mod3Table { data }
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
        const T: [[u8; 4]; 3] = {
            let mut t = [[0u8; 4]; 3];
            let mut r = 0;
            while r < 3 {
                let mut v = 0;
                while v < 4 {
                    t[r][v] = ((v + 4 - r) % 3) as u8;
                    v += 1;
                }
                r += 1;
            }
            t
        };
        const MOD3: [u8; 32] = {
            let mut t = [0u8; 32];
            let mut i = 0;
            while i < 32 {
                t[i] = (i % 3) as u8;
                i += 1;
            }
            t
        };
        d + T[MOD3[(d & 31) as usize] as usize][(v & 3) as usize] - 1
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
    let words = bytes.len() / 8 * 8;
    checksum_tail(
        checksum_words(CHECKSUM_INIT, &bytes[..words]),
        &bytes[words..],
    )
}

const CHECKSUM_INIT: u64 = 0xcbf29ce484222325;

/// [`checksum`] state `h` carried over whole 8-byte words (`bytes.len()` a multiple of 8).
fn checksum_words(mut h: u64, bytes: &[u8]) -> u64 {
    let (chunks, rest) = bytes.as_chunks::<8>();
    debug_assert!(rest.is_empty());
    for c in chunks {
        h = (h ^ u64::from_le_bytes(*c))
            .wrapping_mul(0x100000001b3)
            .rotate_left(23);
    }
    h
}

/// The end of [`checksum`]: the last (fewer than 8) bytes.
fn checksum_tail(mut h: u64, rest: &[u8]) -> u64 {
    for &b in rest {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Read `out.len()` bytes of `f` at offset `off` into `out` and return the [`checksum`] state
/// `h` carried over them (a tail of fewer than 8 bytes ends it). The reads are spread over a
/// few threads in chunks; the checksum (one sequential chain) follows them chunk by chunk on
/// this thread, so a load takes about as long as the checksum alone.
fn read_checksummed(
    f: &std::fs::File,
    off: u64,
    out: &mut [u8],
    mut h: u64,
) -> std::io::Result<u64> {
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::AtomicBool;
    const CHUNK: usize = 4 << 20;
    const READERS: usize = 3;
    let n = out.len();
    let chunks = n.div_ceil(CHUNK);
    let ready: Vec<AtomicBool> = (0..chunks).map(|_| AtomicBool::new(false)).collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    // The chunks are disjoint: each is written by one reader, then read here once marked ready.
    let base = out.as_mut_ptr() as usize;
    let chunk = |i: usize| (i * CHUNK, ((i + 1) * CHUNK).min(n));
    std::thread::scope(|s| {
        let mut readers = vec![];
        for _ in 0..READERS.min(chunks) {
            readers.push(s.spawn(|| -> std::io::Result<()> {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= chunks || failed.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let (lo, hi) = chunk(i);
                    // SAFETY: chunk i is only touched by this reader until it is marked ready.
                    let buf =
                        unsafe { std::slice::from_raw_parts_mut((base + lo) as *mut u8, hi - lo) };
                    if let Err(e) = f.read_exact_at(buf, off + lo as u64) {
                        failed.store(true, Ordering::Relaxed);
                        return Err(e);
                    }
                    ready[i].store(true, Ordering::Release);
                }
            }));
        }
        for (i, r) in ready.iter().enumerate() {
            while !r.load(Ordering::Acquire) {
                if failed.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::yield_now();
            }
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let (lo, hi) = chunk(i);
            // SAFETY: chunk i is complete (Acquire above) and no longer written.
            let buf = unsafe { std::slice::from_raw_parts((base + lo) as *const u8, hi - lo) };
            let words = buf.len() / 8 * 8;
            h = checksum_words(h, &buf[..words]);
            if hi == n {
                h = checksum_tail(h, &buf[words..]);
            }
        }
        for r in readers {
            r.join().expect("table reader panicked")?;
        }
        Ok(h)
    })
}

const MAGIC: &[u8; 8] = b"EGGGTAB1";

/// Save raw table bytes with a header: magic, name length + name, byte length, checksum.
pub fn save(path: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    save_parts(path, name, &[bytes])
}

/// [`save`] the concatenation of `parts` (each but the last a multiple of 8 bytes), without
/// concatenating them in memory.
pub fn save_parts(path: &Path, name: &str, parts: &[&[u8]]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // A temporary name of its own, so concurrent saves of the same table (e.g. tests that
    // generate a missing table in parallel) never write or rename each other's file.
    static SAVES: AtomicUsize = AtomicUsize::new(0);
    let tmp = path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        SAVES.fetch_add(1, Ordering::Relaxed)
    ));
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&(name.len() as u64).to_le_bytes())?;
        f.write_all(name.as_bytes())?;
        let mut h = CHECKSUM_INIT;
        for (i, p) in parts.iter().enumerate() {
            let words = p.len() / 8 * 8;
            assert!(
                words == p.len() || i + 1 == parts.len(),
                "part not 8-byte sized"
            );
            h = checksum_tail(checksum_words(h, &p[..words]), &p[words..]);
        }
        let len: usize = parts.iter().map(|p| p.len()).sum();
        f.write_all(&(len as u64).to_le_bytes())?;
        f.write_all(&h.to_le_bytes())?;
        for p in parts {
            f.write_all(p)?;
        }
        f.flush()?;
    }
    std::fs::rename(tmp, path)
}

/// Load a table saved by [`save`] into `out` (which must have exactly the saved length). The
/// whole table is checksummed before this returns. Inside a [`LoadGroup`], a failed load (after
/// which the caller generates the table) first waits for the generation token.
pub fn load(path: &Path, name: &str, out: &mut [u8]) -> Result<(), String> {
    let r = load_checked(path, name, out);
    if r.is_err() {
        LoadGroup::before_generating();
    }
    r
}

fn load_checked(path: &Path, name: &str, out: &mut [u8]) -> Result<(), String> {
    let mut r = Reader::open(path, name)?;
    if r.len() != out.len() {
        return Err(format!("length mismatch: {} vs {}", r.len(), out.len()));
    }
    r.read(out)?;
    r.finish()
}

/// Reads a table saved by [`save_parts`] part by part (each but the last a multiple of 8
/// bytes); [`Reader::finish`] checks the checksum. After a failed load (in a [`LoadGroup`],
/// before the caller generates the table) call [`load_failed`].
pub struct Reader {
    f: std::fs::File,
    off: u64,
    left: usize,
    len: usize,
    sum: u64,
    h: u64,
}

impl Reader {
    /// Open the table `name` at `path` (its header).
    pub fn open(path: &Path, name: &str) -> Result<Reader, String> {
        let (f, off, len, sum) = open_checked(path, name)?;
        Ok(Reader {
            f,
            off,
            left: len,
            len,
            sum,
            h: CHECKSUM_INIT,
        })
    }

    /// The table's length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read the next `out.len()` bytes.
    pub fn read(&mut self, out: &mut [u8]) -> Result<(), String> {
        if out.len() > self.left {
            return Err(format!("table too short: {} bytes", self.len));
        }
        if (self.len - self.left) & 7 != 0 {
            return Err("part after a part not 8-byte sized".into());
        }
        self.h = read_checksummed(&self.f, self.off, out, self.h).map_err(|e| e.to_string())?;
        self.off += out.len() as u64;
        self.left -= out.len();
        Ok(())
    }

    /// Check that the whole table was read and its checksum.
    pub fn finish(self) -> Result<(), String> {
        if self.left != 0 {
            return Err(format!("length mismatch: {} bytes left", self.left));
        }
        if self.h != self.sum {
            return Err("checksum mismatch".into());
        }
        Ok(())
    }
}

/// Note a failed [`Reader`] load (see [`load`]): in a [`LoadGroup`], wait for the generation
/// token before generating.
pub fn load_failed() {
    LoadGroup::before_generating();
}

/// Open a table saved by [`save_parts`]: the file, the data's offset, length and checksum.
fn open_checked(path: &Path, name: &str) -> Result<(std::fs::File, u64, usize, u64), String> {
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
    let off = (8 + 8 + nl + 16) as u64;
    Ok((f, off, len, sum))
}

/// Marks the current thread as one of several loading tables concurrently, until dropped.
/// Loads run in parallel, but the tables that fail to load are generated one at a time (a
/// generation takes all threads and can need several times the table's memory): the first
/// failed [`load`] of a group takes a process-wide token, which the group keeps until it ends.
pub struct LoadGroup(());

static GENERATION: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    /// (in a group, the generation token once taken).
    static GROUP: std::cell::RefCell<(bool, Option<std::sync::MutexGuard<'static, ()>>)> =
        const { std::cell::RefCell::new((false, None)) };
}

impl LoadGroup {
    pub fn enter() -> LoadGroup {
        GROUP.with(|g| g.borrow_mut().0 = true);
        LoadGroup(())
    }

    fn before_generating() {
        GROUP.with(|g| {
            let mut g = g.borrow_mut();
            if g.0 && g.1.is_none() {
                g.1 = Some(GENERATION.lock().unwrap_or_else(|e| e.into_inner()));
            }
        });
    }
}

impl Drop for LoadGroup {
    fn drop(&mut self) {
        GROUP.with(|g| *g.borrow_mut() = (false, None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concurrent saves of the same table (as parallel tests generating a missing table do)
    /// all succeed and leave a valid file.
    #[test]
    fn concurrent_saves_of_one_table() {
        let dir = std::env::temp_dir().join(format!("eggg-save-test-{}", std::process::id()));
        let path = dir.join("t.tab");
        let bytes: Vec<u8> = (0..(1usize << 20)).map(|i| (i * 31 % 253) as u8).collect();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| save(&path, "t", &bytes).unwrap());
            }
        });
        let mut out = vec![0u8; bytes.len()];
        load(&path, "t", &mut out).unwrap();
        assert_eq!(out, bytes);
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "temporary files left"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn chunked_load_matches_checksum_and_rejects_corruption() {
        let dir = std::env::temp_dir().join(format!("eggg-table-test-{}", std::process::id()));
        // Sizes around the chunk size, including a tail of fewer than 8 bytes.
        for n in [
            0usize,
            5,
            4096,
            (4 << 20) - 3,
            (4 << 20) + 13,
            (13 << 20) + 7,
        ] {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 7919 % 251) as u8).collect();
            let path = dir.join(format!("t{n}.tab"));
            save(&path, "t", &bytes).unwrap();
            let mut out = vec![0u8; n];
            load(&path, "t", &mut out).unwrap();
            assert_eq!(out, bytes);
            assert!(load(&path, "u", &mut out).is_err());
            assert!(load(&path, "t", &mut vec![0u8; n + 1]).is_err());
            if n > 0 {
                // One flipped bit anywhere is rejected.
                let mut raw = std::fs::read(&path).unwrap();
                let at = raw.len() - 1 - (n * 3 / 5);
                raw[at] ^= 0x10;
                std::fs::write(&path, &raw).unwrap();
                assert_eq!(load(&path, "t", &mut out), Err("checksum mismatch".into()));
                // So is a truncated file.
                raw.truncate(raw.len() - 1);
                std::fs::write(&path, &raw).unwrap();
                assert!(load(&path, "t", &mut out).is_err());
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A table saved in parts loads as their concatenation, whole or part by part (in other
    /// 8-byte sized parts); reading past its end or not to its end fails.
    #[test]
    fn saved_parts_load_whole_or_in_parts() {
        let dir = std::env::temp_dir().join(format!("eggg-parts-test-{}", std::process::id()));
        let path = dir.join("p.tab");
        let all: Vec<u8> = (0..(5usize << 20) + 29)
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        let (a, b) = (24, (4 << 20) + 40);
        save_parts(&path, "p", &[&all[..a], &all[a..b], &all[b..]]).unwrap();
        let mut out = vec![0u8; all.len()];
        load(&path, "p", &mut out).unwrap();
        assert_eq!(out, all);
        let mut r = Reader::open(&path, "p").unwrap();
        let (x, y) = out.split_at_mut(1 << 20);
        r.read(x).unwrap();
        r.read(y).unwrap();
        r.finish().unwrap();
        assert_eq!(out, all);
        let mut r = Reader::open(&path, "p").unwrap();
        r.read(&mut out[..8]).unwrap();
        assert!(r.finish().is_err());
        let mut r = Reader::open(&path, "p").unwrap();
        assert!(r.read(&mut vec![0u8; all.len() + 1]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn generation_is_one_group_at_a_time() {
        use std::sync::atomic::AtomicUsize;
        let missing = std::path::Path::new("/nonexistent/eggg-table.tab");
        let (active, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    let _g = LoadGroup::enter();
                    for _ in 0..2 {
                        // A failed load, then "generating".
                        assert!(load(missing, "t", &mut [0u8; 8]).is_err());
                        let a = active.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(a, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        active.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(most.into_inner(), 1);
        // Outside a group, failed loads do not wait.
        assert!(load(missing, "t", &mut [0u8; 8]).is_err());
    }

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

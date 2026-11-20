//! User-space instruction and cycle counters of the current thread (perf_event_open): the
//! P-core PMU of Apple M-series chips under Linux if present, else the generic hardware
//! events. `None` where counters cannot be opened.

pub struct Counter(i32);

impl Counter {
    /// `what`: "instructions" or "cycles".
    pub fn new(what: &str) -> Option<Counter> {
        let dir = "/sys/bus/event_source/devices/apple_avalanche_pmu";
        let read = |p: &str| std::fs::read_to_string(format!("{dir}/{p}")).ok();
        let (ty, config) = match (read("type"), read(&format!("events/{what}"))) {
            (Some(t), Some(e)) => (
                t.trim().parse::<u64>().ok()?,
                u64::from_str_radix(e.trim().trim_start_matches("event=0x"), 16).ok()?,
            ),
            _ => (0, if what == "instructions" { 1 } else { 0 }),
        };
        let mut attr = [0u64; 16];
        attr[0] = ty | 128 << 32; // type, size 128
        attr[1] = config;
        attr[5] = 1 | 1 << 5 | 1 << 6; // disabled, exclude_kernel, exclude_hv
        let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, attr.as_ptr(), 0, -1, -1, 0u64) }
            as i32;
        (fd >= 0).then_some(Counter(fd))
    }

    pub fn start(&self) {
        unsafe {
            libc::ioctl(self.0, 0x2403, 0); // reset
            libc::ioctl(self.0, 0x2400, 0); // enable
        }
    }

    pub fn stop(&self) -> u64 {
        let mut v = 0u64;
        unsafe {
            libc::ioctl(self.0, 0x2401, 0);
            libc::read(self.0, &mut v as *mut u64 as *mut libc::c_void, 8);
        }
        v
    }
}

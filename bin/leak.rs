use std::error::Error;
use std::os::unix::io::AsRawFd;
use std::process;
use std::{env, fs, ptr};

use ixy::ixgbe::IxgbeDevice;
use ixy::memory::alloc_contiguous_memory;
use ixy::*;

use simple_logger::SimpleLogger;

// 4 KiB pages (TX queue + packet buffer) used for one batch of packets
const NUM_PAGES: usize = 64;
// batch size
const BATCH_SIZE: usize = NUM_PAGES - 1;
// size of our packets
const PACKET_SIZE: usize = 60;

pub fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;

    let mut args = env::args();
    args.next();

    let pci_addr = match args.next() {
        Some(arg) => arg,
        None => {
            eprintln!("Usage: cargo run <pci bus id>");
            process::exit(1);
        }
    };

    let mut dev = IxgbeDevice::init(&pci_addr, 1, 1, 0)?;

    dev.disable_rx_queue(0);
    dev.enable_loopback();

    #[rustfmt::skip]
    let mut pkt_data = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06,         // dst MAC
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10,         // src MAC
        0x08, 0x00,                                 // ether type: IPv4
        0x45, 0x00,                                 // Version, IHL, TOS
        ((PACKET_SIZE - 14) >> 8) as u8,            // ip len excluding ethernet, high byte
        ((PACKET_SIZE - 14) & 0xFF) as u8,          // ip len excluding ethernet, low byte
        0x00, 0x00, 0x00, 0x00,                     // id, flags, fragmentation
        0x40, 0x11, 0x00, 0x00,                     // TTL (64), protocol (UDP), checksum
        0x0A, 0x00, 0x00, 0x01,                     // src ip (10.0.0.1)
        0x0A, 0x00, 0x00, 0x02,                     // dst ip (10.0.0.2)
        0x00, 0x2A, 0x05, 0x39,                     // src and dst ports (42 -> 1337)
        ((PACKET_SIZE - 20 - 14) >> 8) as u8,       // udp len excluding ip & ethernet, high byte
        ((PACKET_SIZE - 20 - 14) & 0xFF) as u8,     // udp len excluding ip & ethernet, low byte
        0x00, 0x00,                                 // udp checksum, optional
        b'i', b'x', b'y'                            // payload
        // rest of the payload is zero-filled because mempools guarantee empty bufs
    ];

    // VFs: src MAC must be MAC of the device (spoof check of PF)
    pkt_data[6..12].clone_from_slice(&dev.get_mac_addr());

    let memory = alloc_contiguous_memory(NUM_PAGES * 4096)?;

    // pre-fill all packet buffer in the pool with data and return them to the packet pool
    for i in 1..NUM_PAGES {
        let p = unsafe { std::slice::from_raw_parts_mut(memory.0.add(4096 * i), PACKET_SIZE) };

        for (i, data) in pkt_data.iter().enumerate() {
            p[i] = *data;
        }

        let checksum = calc_ipv4_checksum(&p[14..14 + 20]);
        // Calculated checksum is little-endian; checksum field is big-endian
        p[24] = (checksum >> 8) as u8;
        p[25] = (checksum & 0xff) as u8;
    }

    let mut dev_stats = Default::default();

    dev.reset_stats();

    unsafe {
        dev.reinit_tx_queue(0, memory.0, memory.1);

        let buffer_addrs: Vec<usize> = (1..NUM_PAGES).map(|i| memory.1 + 4096 * i).collect();

        dev.set_tx_descriptors(0, &buffer_addrs, PACKET_SIZE);
    }

    let mut logger = CPUCycleLogger::new()?;

    // warm up caches
    for _ in 0..(1 << 7) {
        unsafe { dev.tx_descriptors(0, 0, BATCH_SIZE) };
    }

    for _ in 0..(1 << 14) {
        let cpu_cycles = unsafe { dev.tx_descriptors(0, 0, BATCH_SIZE) };

        logger.log(cpu_cycles as u32);
    }

    dev.read_stats(&mut dev_stats);

    println!(
        "Packets / Batches: {} / {}",
        dev_stats.tx_pkts,
        dev_stats.tx_pkts as usize / BATCH_SIZE
    );

    let (mean, variance, sample_variance) = logger.stats();

    println!("Mean:                      {:.5}", mean);
    println!("Standard deviation:        {:.6}", variance.sqrt());
    println!("Sample standard deviation: {:.6}", sample_variance.sqrt());

    logger.save("log.txt")?;

    Ok(())
}

/// Compute variance and mean in a single pass - update accumulators
#[allow(dead_code)]
#[inline(always)]
fn wellford_update(count: u64, mean: f64, squared_distance: f64, value: u64) -> (u64, f64, f64) {
    let count = count + 1;
    let delta = value as f64 - mean;
    let mean = mean + delta / count as f64;
    let squared_distance = squared_distance + delta * (value as f64 - mean);

    (count, mean, squared_distance)
}

/// Compute variance and mean in a single pass - calculate result
#[allow(dead_code)]
fn wellford_finalize(count: u64, mean: f64, squared_distance: f64) -> (f64, f64, f64) {
    (
        mean,
        squared_distance / count as f64,
        squared_distance / (count - 1) as f64,
    )
}

/// Calculates IPv4 header checksum
fn calc_ipv4_checksum(ipv4_header: &[u8]) -> u16 {
    assert_eq!(ipv4_header.len() % 2, 0);
    let mut checksum = 0;
    for i in 0..ipv4_header.len() / 2 {
        if i == 5 {
            // Assume checksum field is set to 0
            continue;
        }
        checksum += (u32::from(ipv4_header[i * 2]) << 8) + u32::from(ipv4_header[i * 2 + 1]);
        if checksum > 0xffff {
            checksum = (checksum & 0xffff) + 1;
        }
    }
    !(checksum as u16)
}

struct CPUCycleLogger {
    path: String,
    addr: *mut u32,
    index: usize,
}

impl CPUCycleLogger {
    fn new() -> Result<CPUCycleLogger, Box<dyn Error>> {
        let path = "/mnt/huge/ixy-cpu-cycle-logger".to_string();

        if let Ok(f) = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
        {
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    1 << 21,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_HUGETLB,
                    f.as_raw_fd(),
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                Err("failed to memory map huge page - huge pages enabled and free?".into())
            } else {
                Ok(CPUCycleLogger {
                    path,
                    addr: ptr as *mut u32,
                    index: 0,
                })
            }
        } else {
            Err("failed to memory map huge page".into())
        }
    }

    fn log(&mut self, value: u32) {
        unsafe {
            *self.addr.add(self.index) = value;
        }
        self.index += 1;
    }

    fn save(&mut self, file: &str) -> std::io::Result<()> {
        fs::copy(&self.path, file)?;

        Ok(fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)?
            .set_len((self.index * 4) as u64)?)
    }

    fn stats(&self) -> (f64, f64, f64) {
        assert!(self.index > 1, "not enough values");

        let mean = unsafe {
            (0..self.index)
                .map(|x| *self.addr.add(x) as usize)
                .sum::<usize>() as f64
                / self.index as f64
        };

        let numerator = unsafe {
            (0..self.index)
                .map(|x| (*self.addr.add(x) as f64 - mean).powi(2))
                .sum::<f64>()
        };

        let variance = numerator / self.index as f64;
        let sample_variance = numerator / (self.index - 1) as f64;

        (mean, variance, sample_variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_checksum() {
        // Test case from the Wikipedia article "IPv4 header checksum"
        assert_eq!(
            calc_ipv4_checksum(
                b"\x45\x00\x00\x73\x00\x00\x40\x00\x40\x11\xb8\x61\xc0\xa8\x00\x01\xc0\xa8\x00\xc7"
            ),
            0xb861
        );
    }
}

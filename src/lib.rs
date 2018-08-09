#![feature(const_fn)]
#![feature(untagged_unions)]

extern crate libc;
extern crate byteorder;
#[macro_use]
extern crate log;

#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
mod constants;

mod ixgbe;
mod pci;
pub mod memory;

use self::ixgbe::*;
use self::memory::*;
use self::pci::*;

use std::error::Error;
use std::io::ErrorKind;

const MAX_QUEUES: u16 = 64;

/// Used for implementing an ixy device driver like ixgbe or virtio.
pub trait IxyDriver: std::cmp::PartialEq {
    /// Initializes an intel 82599 network card.
    fn init(pci_addr: &str, num_rx_queues: u16, num_tx_queues: u16) -> Result<Self, Box<Error>> where Self: Sized;

    /// Returns the driver's name.
    fn get_driver_name(&self) -> &str;

    /// Returns the network card's pci address.
    fn get_pci_addr(&self) -> &str;

    /// Pushes up to `num_packets` `Packet`s onto `buffer` depending on the amount of
    /// received packets by the network card.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dev = ixy_init("0000:01:00.0", 1, 1).unwrap();
    /// let mut buf: Vec<Packet> = Vec::new();
    ///
    /// dev.rx_batch(0, &mut buf, 32);
    /// ```
    fn rx_batch(&mut self, queue_id: u32, buffer: &mut Vec<Packet>, num_packets: usize) -> usize;

    /// Takes `Packet`s out of `buffer` until `buffer` is empty or the network card's tx
    /// queue is full.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dev = ixy_init("0000:01:00.0", 1, 1).unwrap();
    /// let mut buf: Vec<Packet> = Vec::new();
    ///
    /// assert_eq!(dev.tx_batch(0, &mut buf), 0);
    /// ```
    fn tx_batch(&mut self, queue_id: u32, buffer: &mut Vec<Packet>) -> usize;

    /// Reads the network card's stats registers into `stats`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dev = ixy_init("0000:01:00.0", 1, 1).unwrap();
    /// let mut stats: DeviceStats = Default::default();
    ///
    /// dev.read_stats(&mut stats);
    /// ```
    fn read_stats(&self, stats: &mut DeviceStats);

    /// Resets the network card's stats registers.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dev = ixy_init("0000:01:00.0", 1, 1).unwrap();
    /// dev.reset_stats();
    /// ```
    fn reset_stats(&self);

    /// Returns the network card's link speed.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dev = ixy_init("0000:01:00.0", 1, 1).unwrap();
    ///
    /// println!("Link speed is {} Mbit/s", dev.get_link_speed());
    /// ```
    fn get_link_speed(&self) -> u16;
}

/// Holds network card stats about sent and received packets.
#[derive(Default, Copy, Clone)]
pub struct DeviceStats {
    pub rx_pkts: u64,
    pub tx_pkts: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
}

impl DeviceStats {
    ///  Prints the stats differences between `stats_old` and `self`.
    pub fn print_stats_diff(&self, dev: &impl IxyDriver, stats_old: &DeviceStats, nanos: u64) {
        let pci_addr = dev.get_pci_addr();
        let mbits = diff_mbit(self.rx_bytes, stats_old.rx_bytes, self.rx_pkts, stats_old.rx_pkts, nanos);
        let mpps = diff_mpps(self.rx_pkts, stats_old.rx_pkts, nanos);
        println!("[{}] RX: {:.2} Mbit/s {:.2} Mpps", pci_addr, mbits, mpps);

        let mbits = diff_mbit(self.tx_bytes, stats_old.tx_bytes, self.tx_pkts, stats_old.tx_pkts, nanos);
        let mpps = diff_mpps(self.tx_pkts, stats_old.tx_pkts, nanos);
        println!("[{}] TX: {:.2} Mbit/s {:.2} Mpps", pci_addr, mbits, mpps);
    }
}

fn diff_mbit(bytes_new: u64, bytes_old: u64, pkts_new: u64, pkts_old: u64, nanos: u64) -> f64 {
    (((bytes_new - bytes_old) as f64 / 1_000_000.0 / (nanos as f64 / 1_000_000_000.0)) * f64::from(8)
        + diff_mpps(pkts_new, pkts_old, nanos) * f64::from(20) * f64::from(8))
}

fn diff_mpps(pkts_new: u64, pkts_old: u64, nanos: u64) -> f64 {
    (pkts_new - pkts_old) as f64 / 1_000_000.0 / (nanos as f64 / 1_000_000_000.0)
}

/// Initializes the network card at `pci_addr`.
///
/// `rx_queues` and `tx_queues` specify the amount of queues that will be initialized and used.
pub fn ixy_init(pci_addr: &str, rx_queues: u16, tx_queues: u16) -> Result<impl IxyDriver, Box<Error>> {
    let mut config_file = pci_open_resource(pci_addr, "config")?;

    let vendor_id = read_io16(&mut config_file, 0)?;
    let device_id = read_io16(&mut config_file, 2)?;
    let class_id = read_io32(&mut config_file, 8)? >> 24;

    if class_id != 2 {
        return Err(Box::new(std::io::Error::new(ErrorKind::Other, format!("device {} is not a network card", pci_addr))));
    }

    if vendor_id == 0x1af4 && device_id >= 0x1000 {
        Err(Box::new(std::io::Error::new(ErrorKind::Other, "virtio driver is not implemented yet")))
    } else {
        // let's give it a try with ixgbe
        let driver: IxgbeDevice = IxyDriver::init(pci_addr, rx_queues, tx_queues)?;
        Ok(driver)
    }
}
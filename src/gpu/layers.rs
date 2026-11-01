//! The two layers under the kernel IR:
//!
//!   Hardware   one per GPU generation: IR → machine code, and the command
//!              stream that dispatches it (gen75.rs)
//!   Driver     one per kernel driver: device, memory, submission (i915.rs)
//!
//! Supporting another GPU means another Hardware (and, if its kernel driver
//! differs, another Driver); the ray tracer itself does not change.

use std::rc::Rc;

use super::kir::Program;

/// GPU memory, mapped write-back into the CPU; freed through its driver
pub struct Buffer {
    drv: Rc<dyn Driver>,
    pub handle: u32,
    pub size: u64,
    pub map: *mut u8,
}

impl Buffer {
    /// zeroed and mapped
    pub fn alloc(drv: &Rc<dyn Driver>, size: u64) -> Option<Buffer> {
        let (handle, size, map) = drv.alloc(size)?;
        Some(Buffer { drv: drv.clone(), handle, size, map })
    }
    /// the mapping as dwords
    #[allow(clippy::mut_from_ref)]
    pub fn words(&self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.map as *mut u32, self.size as usize / 4) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.drv.release(self.handle, self.size, self.map);
    }
}

pub trait Driver {
    fn name(&self) -> &'static str;
    /// a zeroed, mapped buffer of at least `size` bytes: (handle, size, mapping)
    fn alloc(&self, size: u64) -> Option<(u32, u64, *mut u8)>;
    fn release(&self, handle: u32, size: u64, map: *mut u8);
    /// make the CPU view coherent; waits for the GPU
    fn sync(&self, b: &Buffer, write: bool);
    /// a GPU address inside `target` (+ delta) is written into `batch` at byte `offset`
    fn reloc(&self, batch: &Buffer, offset: u32, target: &Buffer, delta: u32, write: bool);
    /// queue the batch, return at once
    fn exec(&self, batch: &Buffer, len: u32) -> bool;
    /// was one of our batches reset?
    fn hung(&self) -> bool;
}

/// open the first device whose chip id `accept` takes; returns its path
pub type OpenDriver = fn(accept: &mut dyn FnMut(u32) -> bool) -> Option<(Rc<dyn Driver>, String)>;

pub trait Hardware {
    fn name(&self) -> &'static str;
    fn compile(&mut self, p: &Program) -> bool;
    /// bufs[0] scene, bufs[1..2] images of tiles_x*4 by tiles_y*4 dwords
    fn setup(&mut self, drv: &Rc<dyn Driver>, tiles_x: u32, tiles_y: u32, bufs: &[Buffer; 3]) -> bool;
    /// queue the whole grid, return at once
    fn run(&mut self, uniforms: &[u32; 32]) -> bool;
}

/// does this layer drive the chip? Returns the layer and a description
pub type ProbeHardware = fn(chip_id: u32) -> Option<(Box<dyn Hardware>, String)>;

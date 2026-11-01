//! The Driver for the i915 kernel driver: finds a render node, allocates GEM
//! buffers, relocates and executes a batch, waits, and detects GPU resets. Raw
//! ioctls only. It knows nothing about the hardware's command stream or
//! instruction set, so it serves every Intel generation i915 drives.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::rc::Rc;

use super::layers::{Buffer, Driver};
use crate::sys::*;

struct I915 {
    file: File,
    // the batch is built once and executed many times, so relocations are kept
    // and re-applied on every exec (presumed_offset ~0 forces it)
    rel: RefCell<Vec<DrmI915GemRelocationEntry>>,
    obj: RefCell<Vec<DrmI915GemExecObject2>>,
}

pub fn open(accept: &mut dyn FnMut(u32) -> bool) -> Option<(Rc<dyn Driver>, String)> {
    for i in 128..136 {
        let path = format!("/dev/dri/renderD{i}");
        let Ok(file) = OpenOptions::new().read(true).write(true).open(&path) else { continue };
        let mut id = 0i32;
        let mut gp = DrmI915Getparam { param: I915_PARAM_CHIPSET_ID, value: &mut id };
        if ioctl(file.as_raw_fd(), DRM_IOCTL_I915_GETPARAM, &mut gp) && accept(id as u32) {
            let drv = I915 { file, rel: RefCell::new(Vec::new()), obj: RefCell::new(Vec::new()) };
            return Some((Rc::new(drv), path));
        }
    }
    None
}

impl I915 {
    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
    fn close(&self, handle: u32) {
        let mut c = DrmGemClose { handle, pad: 0 };
        ioctl(self.fd(), DRM_IOCTL_GEM_CLOSE, &mut c);
    }
}

impl Driver for I915 {
    fn name(&self) -> &'static str {
        "i915"
    }

    fn alloc(&self, size: u64) -> Option<(u32, u64, *mut u8)> {
        let mut c = DrmI915GemCreate { size: (size + 4095) & !4095, ..Default::default() };
        if !ioctl(self.fd(), DRM_IOCTL_I915_GEM_CREATE, &mut c) {
            perror("GEM_CREATE");
            return None;
        }
        let mut mo = DrmI915GemMmapOffset { handle: c.handle, flags: I915_MMAP_OFFSET_WB, ..Default::default() };
        if !ioctl(self.fd(), DRM_IOCTL_I915_GEM_MMAP_OFFSET, &mut mo) {
            perror("GEM_MMAP_OFFSET");
            self.close(c.handle);
            return None;
        }
        let Some(map) = mmap(self.fd(), c.size, mo.offset) else {
            perror("mmap bo");
            self.close(c.handle);
            return None;
        };
        Some((c.handle, c.size, map))
    }

    fn release(&self, handle: u32, size: u64, map: *mut u8) {
        munmap(map, size);
        self.close(handle);
    }

    fn sync(&self, b: &Buffer, write: bool) {
        let mut sd = DrmI915GemSetDomain { handle: b.handle, read_domains: I915_GEM_DOMAIN_CPU, write_domain: if write { I915_GEM_DOMAIN_CPU } else { 0 } };
        ioctl(self.fd(), DRM_IOCTL_I915_GEM_SET_DOMAIN, &mut sd);
    }

    fn reloc(&self, batch: &Buffer, offset: u32, target: &Buffer, delta: u32, write: bool) {
        self.rel.borrow_mut().push(DrmI915GemRelocationEntry {
            target_handle: target.handle,
            delta,
            offset: offset as u64,
            presumed_offset: !0,
            read_domains: I915_GEM_DOMAIN_RENDER,
            write_domain: if write { I915_GEM_DOMAIN_RENDER } else { 0 },
        });
        batch.words()[offset as usize / 4] = delta;
    }

    fn exec(&self, batch: &Buffer, len: u32) -> bool {
        let mut rel = self.rel.borrow_mut();
        let mut obj = self.obj.borrow_mut();
        obj.clear();
        for r in rel.iter_mut() {
            // every relocation target once, the batch last
            r.presumed_offset = !0;
            if r.target_handle == batch.handle {
                continue;
            }
            let j = match obj.iter().position(|o| o.handle == r.target_handle) {
                Some(j) => j,
                None => {
                    obj.push(DrmI915GemExecObject2 { handle: r.target_handle, ..Default::default() });
                    obj.len() - 1
                }
            };
            if r.write_domain != 0 {
                obj[j].flags |= EXEC_OBJECT_WRITE;
            }
        }
        obj.push(DrmI915GemExecObject2 { handle: batch.handle, relocation_count: rel.len() as u32, relocs_ptr: rel.as_ptr() as u64, ..Default::default() });
        let mut eb = DrmI915GemExecbuffer2 { buffers_ptr: obj.as_ptr() as u64, buffer_count: obj.len() as u32, batch_len: len, flags: I915_EXEC_RENDER, ..Default::default() };
        if !ioctl(self.fd(), DRM_IOCTL_I915_GEM_EXECBUFFER2, &mut eb) {
            perror("EXECBUFFER2");
            return false;
        }
        true
    }

    fn hung(&self) -> bool {
        let mut rs = DrmI915ResetStats::default();
        ioctl(self.fd(), DRM_IOCTL_I915_GET_RESET_STATS, &mut rs) && (rs.batch_active != 0 || rs.batch_pending != 0)
    }
}

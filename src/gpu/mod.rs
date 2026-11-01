//! The GPU backend, without Mesa or any userspace driver:
//!
//!   rt_kernel.rs   the ray tracer, written once in the kernel IR (kir.rs)
//!        │
//!   gen75.rs       Hardware: IR → machine code, command stream
//!        │
//!   i915.rs        Driver: kernel driver ioctls
//!
//! This module picks a driver that finds a GPU some hardware layer accepts,
//! compiles the kernel, uploads the scene, and fills the uniforms per pass.
//! Passes render asynchronously: submit() queues `samples` per pixel of the
//! camera's view and returns, collect() waits and hands back XRGB8888 pixels
//! plus the primary hit distance per pixel (negative on the teapot, 1e6 for sky).

mod gen75;
mod i915;
mod kir;
mod layers;
mod rt_kernel;

use std::rc::Rc;

use crate::scene::{look, Camera, Scene};
use layers::{Buffer, Driver, Hardware, OpenDriver, ProbeHardware};
use rt_kernel::*;

pub struct Gpu {
    hw: Box<dyn Hardware>,
    drv: Rc<dyn Driver>,
    bufs: [Buffer; 3], // scene, rgb, depth
    w: usize,
    h: usize,
    w4: usize, // image row length on the GPU (whole tiles)
    layout: usize,
    nnodes: u32,
}

impl Gpu {
    /// None if no supported GPU is found
    pub fn open(s: &Scene, w: usize, h: usize) -> Option<Gpu> {
        let drivers: [OpenDriver; 1] = [i915::open];
        let layers: [ProbeHardware; 1] = [gen75::probe];
        let mut found = None;
        for open in drivers {
            let mut hw = None;
            let mut accept = |id: u32| {
                hw = layers.iter().find_map(|probe| probe(id));
                hw.is_some()
            };
            if let Some((drv, path)) = open(&mut accept) {
                let (hw, desc) = hw.unwrap();
                found = Some((drv, path, hw, desc));
                break;
            }
        }
        let Some((drv, path, mut hw, desc)) = found else {
            eprintln!("gpu: no supported GPU found");
            return None;
        };
        eprintln!("gpu: {path} {desc} ({}, {})", drv.name(), hw.name());

        let prog = rt_kernel::build(cfg!(feature = "gpu-debug")); // debug: heat map of the traversal
        if !hw.compile(&prog) {
            return None;
        }

        let (tx, ty) = (w.div_ceil(4), h.div_ceil(4));
        let img = (tx * ty * 16 * 4) as u64;
        let bufs = [Buffer::alloc(&drv, s.blob_bytes() as u64)?, Buffer::alloc(&drv, img)?, Buffer::alloc(&drv, img)?];
        drv.sync(&bufs[BUF_SCENE], true);
        s.write_blob(bufs[BUF_SCENE].words());
        if !hw.setup(&drv, tx as u32, ty as u32, &bufs) {
            return None;
        }
        Some(Gpu { hw, drv, bufs, w, h, w4: tx * 4, layout: s.layout_bytes(), nnodes: s.nodes.len() as u32 })
    }

    /// Callers keep passes short (a few samples) so a batch stays far below the
    /// kernel's hang detection.
    pub fn submit(&mut self, cam: &Camera, samples: u32, seed: u32, sample_base: u32) -> bool {
        let mut u = [0u32; U_COUNT];
        let mut f = |i: usize, v: f32| u[i] = v.to_bits();
        for (v, at) in [(cam.pos, U_CAM), (cam.fwd, U_FWD), (cam.right, U_RIGHT), (cam.up, U_UP), (look::LIGHT_POS, U_LIGHT)] {
            for a in 0..3 {
                f(at + a, v[a]);
            }
        }
        f(U_TAN_X, cam.tan * (self.w as f32 / self.h as f32));
        f(U_TAN_Y, cam.tan);
        f(U_INV_W, 2.0 / self.w as f32);
        f(U_INV_H, 2.0 / self.h as f32);
        f(U_LSIZE, 2.0 * look::LIGHT_RADIUS);
        f(U_INV_SPP, 1.0 / samples as f32);
        f(U_SAMPLE_BASE, sample_base as f32);
        u[U_TRI_BASE] = (8 * self.layout / 16) as u32;
        u[U_NNODES] = self.nnodes;
        for a in 0..3 {
            u[U_STRIDE + a] = ((self.layout << a) / 16) as u32;
        }
        u[U_SPP] = samples;
        u[U_SEED] = seed;
        self.hw.run(&u)
    }

    pub fn collect(&mut self, rgb: &mut [u32], depth: Option<&mut [f32]>) -> bool {
        self.drv.sync(&self.bufs[BUF_RGB], false); // waits for the GPU
        if self.drv.hung() {
            eprintln!("gpu: kernel hung the GPU (reset by the driver), giving up on the GPU path");
            return false;
        }
        let (w, w4) = (self.w, self.w4);
        let src = self.bufs[BUF_RGB].words();
        for (y, row) in rgb.chunks_exact_mut(w).take(self.h).enumerate() {
            row.copy_from_slice(&src[y * w4..][..w]);
        }
        if let Some(depth) = depth {
            self.drv.sync(&self.bufs[BUF_DEPTH], false);
            let src = self.bufs[BUF_DEPTH].words();
            for (y, row) in depth.chunks_exact_mut(w).take(self.h).enumerate() {
                for (d, &s) in row.iter_mut().zip(&src[y * w4..][..w]) {
                    *d = f32::from_bits(s);
                }
            }
        }
        true
    }
}

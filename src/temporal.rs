//! Temporal accumulation of animation frames: each new frame is blended with
//! the previous ones, reprojected through the per-pixel depth (with depth
//! rejection), multithreaded over rows.

use crate::scene::Camera;
use crate::tone::{tone, ToneLuts};
use crate::vec::{dot, V3};

#[derive(Clone, Copy, Default)]
struct View {
    pos: V3,
    fwd: V3,
    right: V3,
    up: V3,
    tx: f32,
    ty: f32,
}

#[derive(Default)]
pub struct Temporal {
    // history: linear RGB + sample count, and view-space depth (1e9 = sky);
    // the frame being blended writes `next`, which then becomes `hist`
    hist: Vec<f32>,
    hist_depth: Vec<f32>,
    next: Vec<f32>,
    next_depth: Vec<f32>,
    sx: Vec<f32>, // pixel-centre ray offsets
    sy: Vec<f32>,
    il: Vec<f32>, // 1 / ray length
    hist_view: View,
    ok: bool,
}

// what every row worker reads
struct Pass<'a> {
    w: usize,
    h: usize,
    depth: &'a [f32],
    view: View,
    hist_view: View,
    ok: bool,
    hist: &'a [f32],
    hist_depth: &'a [f32],
    sx: &'a [f32],
    sy: &'a [f32],
    il: &'a [f32],
    lut: &'static ToneLuts,
}

impl Temporal {
    /// blend `frame` (XRGB8888, w x h, rendered from `cam` with primary hit
    /// distances `depth`, negative on the glossy teapot) with the history, in place
    pub fn blend(&mut self, frame: &mut [u32], w: usize, h: usize, depth: &[f32], cam: &Camera, threads: usize) {
        let n = w * h;
        let view = View { pos: cam.pos, fwd: cam.fwd, right: cam.right, up: cam.up, tx: cam.tan * w as f32 / h as f32, ty: cam.tan };
        if self.hist.is_empty() {
            self.hist = vec![0.0; n * 4];
            self.next = vec![0.0; n * 4];
            self.hist_depth = vec![0.0; n];
            self.next_depth = vec![0.0; n];
            // the ray length only depends on the pixel, not on the camera
            self.sx = (0..w).map(|x| ((x as f32 + 0.5) * 2.0 / w as f32 - 1.0) * view.tx).collect();
            self.sy = (0..h).map(|y| (1.0 - (y as f32 + 0.5) * 2.0 / h as f32) * view.ty).collect();
            self.il = (0..n).map(|i| 1.0 / (1.0 + self.sx[i % w] * self.sx[i % w] + self.sy[i / w] * self.sy[i / w]).sqrt()).collect();
        }
        let pass = Pass {
            w,
            h,
            depth,
            view,
            hist_view: self.hist_view,
            ok: self.ok,
            hist: &self.hist,
            hist_depth: &self.hist_depth,
            sx: &self.sx,
            sy: &self.sy,
            il: &self.il,
            lut: tone(),
        };
        std::thread::scope(|s| {
            let (mut frame, mut nh, mut nd) = (&mut frame[..n], &mut self.next[..], &mut self.next_depth[..]);
            for i in 0..threads {
                let (y0, y1) = (h * i / threads, h * (i + 1) / threads);
                let k = (y1 - y0) * w;
                let (f, rest) = std::mem::take(&mut frame).split_at_mut(k);
                frame = rest;
                let (a, rest) = std::mem::take(&mut nh).split_at_mut(k * 4);
                nh = rest;
                let (b, rest) = std::mem::take(&mut nd).split_at_mut(k);
                nd = rest;
                let pass = &pass;
                s.spawn(move || pass.rows(y0, y1, f, a, b));
            }
        });
        std::mem::swap(&mut self.hist, &mut self.next);
        std::mem::swap(&mut self.hist_depth, &mut self.next_depth);
        self.hist_view = view;
        self.ok = true;
    }
}

impl Pass<'_> {
    // rows y0..y1; frame, nh and nd start at row y0
    fn rows(&self, y0: usize, y1: usize, frame: &mut [u32], nh: &mut [f32], nd: &mut [f32]) {
        let (w, h) = (self.w, self.h);
        let (v, pv) = (self.view, self.hist_view);
        let (ph, pd) = (self.hist, self.hist_depth);
        let dp = v.pos - pv.pos;
        let itx = 0.5 * w as f32 / pv.tx;
        let ity = 0.5 * h as f32 / pv.ty;
        for y in y0..y1 {
            let base = v.fwd + v.up * self.sy[y];
            for x in 0..w {
                let i = y * w + x; // whole-frame index
                let o = (y - y0) * w + x; // index into this worker's rows
                let p = frame[o];
                let lin8 = &self.lut.lin8;
                let c = [lin8[(p >> 16 & 255) as usize], lin8[(p >> 8 & 255) as usize], lin8[(p & 255) as usize]];
                let mut t = self.depth[i];
                let il = self.il[i];
                let (mut hc, mut n, mut wsum) = ([0f32; 3], 0f32, 0f32);
                let glossy = t < 0.0; // the GPU marks teapot pixels with a negative depth
                let sky = t >= 1e5;
                t = t.abs();
                if self.ok {
                    // world point of this pixel relative to the previous camera
                    let s = (if sky { 1e4 } else { t }) * il;
                    let q = dp + (base + v.right * self.sx[x]) * s;
                    let z = dot(q, pv.fwd);
                    if z > 1e-3 {
                        let rz = 1.0 / z;
                        let fx = w as f32 * 0.5 + dot(q, pv.right) * rz * itx - 0.5;
                        let fy = h as f32 * 0.5 - dot(q, pv.up) * rz * ity - 0.5;
                        let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
                        let (ax, ay) = (fx - ix as f32, fy - iy as f32);
                        let tol = 0.02 * z + 0.02;
                        // bilinear, but only over taps that saw the same surface
                        for k in 0..4 {
                            let (tx, ty) = (ix + (k & 1), iy + (k >> 1));
                            if tx < 0 || ty < 0 || tx >= w as i32 || ty >= h as i32 {
                                continue;
                            }
                            let j = ty as usize * w + tx as usize;
                            let hd = pd[j];
                            if if sky { hd < 1e8 } else { (hd - z).abs() > tol } {
                                continue;
                            }
                            let wt = (if k & 1 != 0 { ax } else { 1.0 - ax }) * (if k >> 1 != 0 { ay } else { 1.0 - ay }) + 1e-4;
                            let hp = &ph[j * 4..j * 4 + 4];
                            hc[0] += hp[0] * wt;
                            hc[1] += hp[1] * wt;
                            hc[2] += hp[2] * wt;
                            n += hp[3] * wt;
                            wsum += wt;
                        }
                    }
                }
                let mut alpha = 1.0f32;
                if wsum > 0.05 {
                    let iw = 1.0 / wsum;
                    hc[0] *= iw;
                    hc[1] *= iw;
                    hc[2] *= iw;
                    n *= iw;
                    // running mean, then an exponential average; reflections and highlights on
                    // the teapot move with the view, so it keeps much less history than the floor
                    alpha = (1.0 / (n + 1.0)).max(if glossy { 0.5 } else { 0.2 });
                } else {
                    n = 0.0;
                }
                let out = &mut nh[o * 4..o * 4 + 4];
                for a in 0..3 {
                    out[a] = hc[a] + (c[a] - hc[a]) * alpha;
                }
                out[3] = n + 1.0;
                nd[o] = if sky { 1e9 } else { t * il };
                let oc = [out[0].min(1.0), out[1].min(1.0), out[2].min(1.0)];
                frame[o] = self.lut.encode(&oc, 4095.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the same input as a C++ harness for the previous version; its checksum
    // (strict IEEE build, no -ffast-math / FMA contraction) is 203872f244413a09
    #[test]
    fn matches_cpp_version() {
        const W: usize = 320;
        const H: usize = 180;
        let (mut px, mut dep) = (vec![0u32; W * H], vec![0f32; W * H]);
        let mut t = Temporal::default();
        let (mut hash, mut s) = (1469598103934665603u64, 1u32);
        for f in 0..4 {
            for i in 0..W * H {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                px[i] = s >> 8;
                dep[i] = if i % 7 == 0 { 2e6 } else { (if i % 3 != 0 { 1.0 } else { -1.0 }) * (5.0 + (i % W) as f32 * 0.01) };
            }
            t.blend(&mut px, W, H, &dep, &Camera::orbit(f as f32 * 0.05), 4);
            for &p in &px {
                hash = (hash ^ p as u64).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(format!("{hash:016x}"), "203872f244413a09");
    }
}

//! The tessellated Newell teapot on a checkerboard floor, its binned-SAH BVH,
//! the GPU scene layout, the camera, and the look (light, sky) shared with the
//! GPU kernel.

use std::f32::consts::PI;

use crate::vec::{cross, dot, length, normalize, v3, vmax, vmin, V3};

/* ------------------------------------------------- Newell teapot data --- */
// Classic 10 patch / 127 control point form; rim, body, lid and bottom are
// mirrored into 4 quadrants, handle and spout into 2 → 32 bicubic patches.

#[rustfmt::skip]
const PATCHES: [[usize; 16]; 10] = [
    [102, 103, 104, 105, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],       // rim
    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],     // body
    [24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [96, 96, 96, 96, 97, 98, 99, 100, 101, 101, 101, 101, 0, 1, 2, 3],    // lid
    [0, 1, 2, 3, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
    [118, 118, 118, 118, 126, 125, 124, 123, 122, 120, 119, 121, 40, 39, 38, 37], // bottom
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],     // handle
    [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 28, 65, 66, 67],
    [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],     // spout
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
];

// as written in the original data: decimal → f64 → f32, as C rounds a double literal into a float array
#[rustfmt::skip]
const CPDATA: [[f64; 3]; 127] = [
    [0.2, 0.0, 2.7], [0.2, -0.112, 2.7], [0.112, -0.2, 2.7], [0.0, -0.2, 2.7],
    [1.3375, 0.0, 2.53125], [1.3375, -0.749, 2.53125], [0.749, -1.3375, 2.53125], [0.0, -1.3375, 2.53125],
    [1.4375, 0.0, 2.53125], [1.4375, -0.805, 2.53125], [0.805, -1.4375, 2.53125], [0.0, -1.4375, 2.53125],
    [1.5, 0.0, 2.4], [1.5, -0.84, 2.4], [0.84, -1.5, 2.4], [0.0, -1.5, 2.4],
    [1.75, 0.0, 1.875], [1.75, -0.98, 1.875], [0.98, -1.75, 1.875], [0.0, -1.75, 1.875],
    [2.0, 0.0, 1.35], [2.0, -1.12, 1.35], [1.12, -2.0, 1.35], [0.0, -2.0, 1.35],
    [2.0, 0.0, 0.9], [2.0, -1.12, 0.9], [1.12, -2.0, 0.9], [0.0, -2.0, 0.9],
    [-2.0, 0.0, 0.9],
    [2.0, 0.0, 0.45], [2.0, -1.12, 0.45], [1.12, -2.0, 0.45], [0.0, -2.0, 0.45],
    [1.5, 0.0, 0.225], [1.5, -0.84, 0.225], [0.84, -1.5, 0.225], [0.0, -1.5, 0.225],
    [1.5, 0.0, 0.15], [1.5, -0.84, 0.15], [0.84, -1.5, 0.15], [0.0, -1.5, 0.15],
    [-1.6, 0.0, 2.025], [-1.6, -0.3, 2.025], [-1.5, -0.3, 2.25], [-1.5, 0.0, 2.25],
    [-2.3, 0.0, 2.025], [-2.3, -0.3, 2.025], [-2.5, -0.3, 2.25], [-2.5, 0.0, 2.25],
    [-2.7, 0.0, 2.025], [-2.7, -0.3, 2.025], [-3.0, -0.3, 2.25], [-3.0, 0.0, 2.25],
    [-2.7, 0.0, 1.8], [-2.7, -0.3, 1.8], [-3.0, -0.3, 1.8], [-3.0, 0.0, 1.8],
    [-2.7, 0.0, 1.575], [-2.7, -0.3, 1.575], [-3.0, -0.3, 1.35], [-3.0, 0.0, 1.35],
    [-2.5, 0.0, 1.125], [-2.5, -0.3, 1.125], [-2.65, -0.3, 0.9375], [-2.65, 0.0, 0.9375],
    [-2.0, -0.3, 0.9], [-1.9, -0.3, 0.6], [-1.9, 0.0, 0.6],
    [1.7, 0.0, 1.425], [1.7, -0.66, 1.425], [1.7, -0.66, 0.6], [1.7, 0.0, 0.6],
    [2.6, 0.0, 1.425], [2.6, -0.66, 1.425], [3.1, -0.66, 0.825], [3.1, 0.0, 0.825],
    [2.3, 0.0, 2.1], [2.3, -0.25, 2.1], [2.4, -0.25, 2.025], [2.4, 0.0, 2.025],
    [2.7, 0.0, 2.4], [2.7, -0.25, 2.4], [3.3, -0.25, 2.4], [3.3, 0.0, 2.4],
    [2.8, 0.0, 2.475], [2.8, -0.25, 2.475], [3.525, -0.25, 2.49375], [3.525, 0.0, 2.49375],
    [2.9, 0.0, 2.475], [2.9, -0.15, 2.475], [3.45, -0.15, 2.5125], [3.45, 0.0, 2.5125],
    [2.8, 0.0, 2.4], [2.8, -0.15, 2.4], [3.2, -0.15, 2.4], [3.2, 0.0, 2.4],
    [0.0, 0.0, 3.15], [0.8, 0.0, 3.15], [0.8, -0.45, 3.15], [0.45, -0.8, 3.15], [0.0, -0.8, 3.15],
    [0.0, 0.0, 2.85],
    [1.4, 0.0, 2.4], [1.4, -0.784, 2.4], [0.784, -1.4, 2.4], [0.0, -1.4, 2.4],
    [0.4, 0.0, 2.55], [0.4, -0.224, 2.55], [0.224, -0.4, 2.55], [0.0, -0.4, 2.55],
    [1.3, 0.0, 2.55], [1.3, -0.728, 2.55], [0.728, -1.3, 2.55], [0.0, -1.3, 2.55],
    [1.3, 0.0, 2.4], [1.3, -0.728, 2.4], [0.728, -1.3, 2.4], [0.0, -1.3, 2.4],
    [0.0, 0.0, 0.0],
    [1.5, -0.84, 0.075], [0.84, -1.5, 0.075], [1.5, 0.0, 0.075], [0.0, -1.5, 0.075],
    [1.425, 0.0, 0.0], [1.425, -0.798, 0.0], [0.798, -1.425, 0.0], [0.0, -1.425, 0.0],
];

const TESS: usize = 24; // quads per patch edge

pub struct Tri {
    pub v0: V3,
    pub e1: V3,
    pub e2: V3,
    pub i0: usize,
    pub i1: usize,
    pub i2: usize,
}

/// count == 0: an interior node whose children are at first and first + 1
#[derive(Clone, Copy, Default)]
pub struct Node {
    pub bmin: V3,
    pub bmax: V3,
    pub first: usize,
    pub count: usize,
}

#[derive(Default)]
pub struct Scene {
    pub vpos: Vec<V3>,
    pub vnrm: Vec<V3>,
    pub tris: Vec<Tri>,
    pub nodes: Vec<Node>,
    pub tidx: Vec<usize>, // triangle indices in leaf order
}

fn bern(t: f32) -> ([f32; 4], [f32; 4]) {
    let s = 1.0 - t;
    (
        [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t],
        [-3.0 * s * s, 3.0 * s * s - 6.0 * t * s, 6.0 * t * s - 3.0 * t * t, 3.0 * t * t],
    )
}

/// point and the two tangents of a bicubic patch
fn eval_patch(cp: &[[V3; 4]; 4], u: f32, v: f32) -> (V3, V3, V3) {
    let (bu, du_) = bern(u);
    let (bv, dv_) = bern(v);
    let (mut p, mut du, mut dv) = (V3::default(), V3::default(), V3::default());
    for i in 0..4 {
        for j in 0..4 {
            p += cp[i][j] * (bu[i] * bv[j]);
            du += cp[i][j] * (du_[i] * bv[j]);
            dv += cp[i][j] * (bu[i] * dv_[j]);
        }
    }
    (p, du, dv)
}

fn centroid(t: &Tri) -> V3 {
    t.v0 + (t.e1 + t.e2) * (1.0 / 3.0)
}

impl Scene {
    /// tessellate the teapot and build the SAH BVH
    pub fn build() -> Scene {
        let mut s = Scene::default();
        s.build_mesh();
        s.tidx = (0..s.tris.len()).collect();
        s.nodes.reserve(s.tris.len() * 2);
        s.nodes.push(Node::default());
        s.build_node(0, 0, s.tris.len());
        s
    }

    fn build_mesh(&mut self) {
        const QUAD: [[f32; 2]; 4] = [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]];
        let npatch = 6 * 4 + 4 * 2;
        self.vpos.reserve(npatch * (TESS + 1) * (TESS + 1));
        self.vnrm.reserve(self.vpos.capacity());
        self.tris.reserve(npatch * TESS * TESS * 2);

        for (p, patch) in PATCHES.iter().enumerate() {
            let copies = if p < 6 { 4 } else { 2 };
            for q in &QUAD[..copies] {
                let (sx, sy) = (q[0], q[1]);
                let mut cp = [[V3::default(); 4]; 4];
                for i in 0..16 {
                    let d = CPDATA[patch[i]].map(|c| c as f32);
                    // teapot data is z-up; rotate into y-up
                    cp[i / 4][i % 4] = v3(sx * d[0], d[2], -sy * d[1]);
                }
                let base = self.vpos.len();
                for i in 0..=TESS {
                    for j in 0..=TESS {
                        let (u, v) = (i as f32 / TESS as f32, j as f32 / TESS as f32);
                        let (pt, du, dv) = eval_patch(&cp, u, v);
                        let mut n = cross(du, dv);
                        if length(n) < 1e-6 {
                            // degenerate pole (lid knob, bottom centre)
                            let c = |x: f32| x.clamp(1e-3, 1.0 - 1e-3);
                            let (_, du, dv) = eval_patch(&cp, c(u), c(v));
                            n = cross(du, dv);
                        }
                        self.vpos.push(pt);
                        self.vnrm.push(normalize(n));
                    }
                }
                for i in 0..TESS {
                    for j in 0..TESS {
                        let a = base + i * (TESS + 1) + j;
                        let b = a + 1;
                        let c = a + TESS + 1;
                        let d = c + 1;
                        for k in [[a, c, b], [b, c, d]] {
                            let v0 = self.vpos[k[0]];
                            let t = Tri { v0, e1: self.vpos[k[1]] - v0, e2: self.vpos[k[2]] - v0, i0: k[0], i1: k[1], i2: k[2] };
                            if length(cross(t.e1, t.e2)) > 1e-9 {
                                self.tris.push(t);
                            }
                        }
                    }
                }
            }
        }
    }

    fn build_node(&mut self, ni: usize, first: usize, count: usize) {
        let big = v3(1e30, 1e30, 1e30);
        let (mut bmin, mut bmax) = (big, -big);
        let (mut cmin, mut cmax) = (bmin, bmax);
        for &ti in &self.tidx[first..first + count] {
            let t = &self.tris[ti];
            let (a, b, c) = (t.v0, t.v0 + t.e1, t.v0 + t.e2);
            bmin = vmin(bmin, vmin(a, vmin(b, c)));
            bmax = vmax(bmax, vmax(a, vmax(b, c)));
            let ce = centroid(t);
            cmin = vmin(cmin, ce);
            cmax = vmax(cmax, ce);
        }
        self.nodes[ni] = Node { bmin, bmax, first, count };
        if count <= 4 {
            return;
        }

        // binned surface-area heuristic: pick the axis/plane minimising A_l*N_l + A_r*N_r
        const BINS: usize = 16;
        let bin_of = |c: f32, lo: f32, span: f32| (((c - lo) / span * BINS as f32) as i32).clamp(0, BINS as i32 - 1) as usize;
        let area = |e: V3| e.x * e.y + e.y * e.z + e.z * e.x;
        let mut best = 1e30f32;
        let mut best_split: Option<(usize, usize)> = None; // axis, last bin on the left
        for ax in 0..3 {
            let (lo, span) = (cmin[ax], cmax[ax] - cmin[ax]);
            if span < 1e-6 {
                continue;
            }
            let mut cnt = [0usize; BINS];
            let (mut bmn, mut bmx) = ([big; BINS], [-big; BINS]);
            for &ti in &self.tidx[first..first + count] {
                let t = &self.tris[ti];
                let b = bin_of(centroid(t)[ax], lo, span);
                let (p0, p1, p2) = (t.v0, t.v0 + t.e1, t.v0 + t.e2);
                cnt[b] += 1;
                bmn[b] = vmin(bmn[b], vmin(p0, vmin(p1, p2)));
                bmx[b] = vmax(bmx[b], vmax(p0, vmax(p1, p2)));
            }
            let mut right_cost = [0f32; BINS];
            let (mut rmn, mut rmx, mut rn) = (big, -big, 0);
            for b in (1..BINS).rev() {
                rmn = vmin(rmn, bmn[b]);
                rmx = vmax(rmx, bmx[b]);
                rn += cnt[b];
                right_cost[b] = if rn > 0 { area(rmx - rmn) * rn as f32 } else { 0.0 };
            }
            let (mut lmn, mut lmx, mut ln) = (big, -big, 0);
            for b in 0..BINS - 1 {
                lmn = vmin(lmn, bmn[b]);
                lmx = vmax(lmx, bmx[b]);
                ln += cnt[b];
                if ln == 0 || ln == count {
                    continue;
                }
                let c = area(lmx - lmn) * ln as f32 + right_cost[b + 1];
                if c < best {
                    best = c;
                    best_split = Some((ax, b));
                }
            }
        }

        let mid = match best_split {
            Some((ax, best_bin)) => {
                let (lo, span) = (cmin[ax], cmax[ax] - cmin[ax]);
                let (mut i, mut j) = (first, first + count - 1);
                while i <= j {
                    if bin_of(centroid(&self.tris[self.tidx[i]])[ax], lo, span) <= best_bin {
                        i += 1;
                    } else {
                        self.tidx.swap(i, j);
                        j -= 1;
                    }
                }
                i - first
            }
            None => {
                // all centroids coincide: plain median split
                let e = cmax - cmin;
                let ax = if e.x > e.y && e.x > e.z { 0 } else if e.y > e.z { 1 } else { 2 };
                let tris = &self.tris;
                self.tidx[first..first + count].sort_by(|&a, &b| centroid(&tris[a])[ax].total_cmp(&centroid(&tris[b])[ax]));
                count / 2
            }
        };
        let l = self.nodes.len();
        self.nodes.resize(l + 2, Node::default());
        self.nodes[ni].first = l;
        self.nodes[ni].count = 0;
        self.build_node(l, first, mid);
        self.build_node(l + 1, first + mid, count - mid);
    }

    /* --------------------------------------------- GPU scene layout --- */
    // 8 depth-first copies of the BVH with "skip" links (one per ray-direction
    // octant, nearer child first) so a walk needs no stack, then one 128-byte
    // record per triangle in leaf order.
    //   node (32 B):  bmin.xyz, skip (u32) | bmax.xyz, first_tri << 3 | count (u32, 0 = interior)
    //   triangle:     Havel-Herout planes N,d | N1,d1 | N2,d2 | n0 | n1-n0 | n2-n0 | geometric normal

    pub fn layout_bytes(&self) -> usize {
        self.nodes.len() * 32
    }
    pub fn blob_bytes(&self) -> usize {
        8 * self.layout_bytes() + self.tris.len() * 128
    }

    pub fn write_blob(&self, dst: &mut [u32]) {
        let nodes = &self.nodes;
        let n = nodes.len();
        let layout = self.layout_bytes() / 4;
        let (mut order, mut skip, mut size) = (vec![0usize; n], vec![0usize; n], vec![0usize; n]);
        for oct in 0..8 {
            let mut stack = vec![0usize];
            let mut k = 0;
            while let Some(ni) = stack.pop() {
                order[k] = ni;
                k += 1;
                if nodes[ni].count != 0 {
                    continue;
                }
                let (l, r) = (nodes[ni].first, nodes[ni].first + 1);
                // split axis ≈ largest separation of the child centres
                let d = (nodes[r].bmin + nodes[r].bmax) - (nodes[l].bmin + nodes[l].bmax);
                let (x, y, z) = (d.x.abs(), d.y.abs(), d.z.abs());
                let ax = if x > y && x > z { 0 } else if y > z { 1 } else { 2 };
                let neg_dir = oct >> ax & 1 != 0;
                let near = if (d[ax] > 0.0) != neg_dir { l } else { r }; // child a ray in this octant meets first
                stack.push(if near == l { r } else { l });
                stack.push(near);
            }
            for i in (0..n).rev() {
                // skip(i) = i + size of subtree(i)
                let nd = &nodes[order[i]];
                size[order[i]] = 1 + if nd.count != 0 { 0 } else { size[nd.first] + size[nd.first + 1] };
                skip[i] = i + size[order[i]];
            }
            for i in 0..n {
                let nd = &nodes[order[i]];
                let r = &mut dst[oct * layout + i * 8..][..8];
                r[0] = nd.bmin.x.to_bits();
                r[1] = nd.bmin.y.to_bits();
                r[2] = nd.bmin.z.to_bits();
                r[3] = skip[i] as u32;
                r[4] = nd.bmax.x.to_bits();
                r[5] = nd.bmax.y.to_bits();
                r[6] = nd.bmax.z.to_bits();
                r[7] = if nd.count != 0 { (nd.first << 3 | nd.count) as u32 } else { 0 }; // count <= 4
            }
        }
        for (k, &ti) in self.tidx.iter().enumerate() {
            let tr = &self.tris[ti];
            let r = &mut dst[8 * layout + k * 32..][..32];
            r.fill(0);
            let (n0, n1, n2) = (self.vnrm[tr.i0], self.vnrm[tr.i1], self.vnrm[tr.i2]);
            let nv = cross(tr.e1, tr.e2);
            let ng = normalize(nv);
            let nn = dot(nv, nv);
            let n1v = cross(tr.e2, nv) * (1.0 / nn);
            let n2v = cross(nv, tr.e1) * (1.0 / nn);
            let vals = [nv, n1v, n2v, n0, n1 - n0, n2 - n0, ng];
            let w = [dot(nv, tr.v0), -dot(n1v, tr.v0), -dot(n2v, tr.v0)];
            for (j, v) in vals.iter().enumerate() {
                r[j * 4] = v.x.to_bits();
                r[j * 4 + 1] = v.y.to_bits();
                r[j * 4 + 2] = v.z.to_bits();
                if j < 3 {
                    r[j * 4 + 3] = w[j].to_bits();
                }
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Camera {
    pub pos: V3,
    pub fwd: V3,
    pub right: V3,
    pub up: V3,
    pub tan: f32, // tan(fov_y / 2)
}

impl Camera {
    /// on a circle around the teapot; 0 = the original viewpoint
    pub fn orbit(angle: f32) -> Camera {
        let centre = v3(0.2, 0.0, 0.0);
        let look = v3(0.2, 1.25, 0.0);
        let (radius, height, a0) = (9.44f32, 4.4f32, 0.635f32); // a0 = atan2(5.6, 7.6)
        let pos = v3(centre.x + radius * (a0 + angle).sin(), height, centre.z + radius * (a0 + angle).cos());
        let fwd = normalize(look - pos);
        let right = normalize(cross(fwd, v3(0.0, 1.0, 0.0)));
        Camera { pos, fwd, right, up: cross(right, fwd), tan: (34.0 * PI / 360.0).tan() }
    }
}

pub mod look {
    use crate::vec::{v3, V3};
    pub const LIGHT_POS: V3 = v3(-6.0, 8.0, 0.5);
    pub const LIGHT_RADIUS: f32 = 1.6;
    pub const LIGHT_COL: V3 = v3(1.25, 1.15, 1.0);
    pub const SKY_HORIZON: V3 = v3(0.82, 0.86, 0.93);
    pub const SKY_ZENITH: V3 = v3(0.30, 0.45, 0.78);
}

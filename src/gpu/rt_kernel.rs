//! The teapot ray tracer written once in the kernel IR. It knows nothing about
//! any particular GPU.
//!
//! Each packet shades a 4x4 tile. The 16 rays traverse the BVH together: the
//! BVH is stored depth-first with "skip" links (Scene::write_blob), so the walk
//! needs no stack and every branch is uniform ("descend if any lane hits the
//! box"), with per-lane work done by predication.

// The float constants are kept exactly as first written: they are baked into
// the machine code, and changing their last digits would change the image.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use std::f32::consts::PI;
use std::ops::Index;

use super::kir::*;
use crate::scene::look;
use crate::vec::v3;

// uniform slots (32 dwords)
pub const U_CAM: usize = 0; //         0-2 camera  3 tan*aspect  4-6 forward  7 tan
pub const U_TAN_X: usize = 3;
pub const U_FWD: usize = 4;
pub const U_TAN_Y: usize = 7;
pub const U_RIGHT: usize = 8; //       8-10 right  11 2/W  12-14 up  15 2/H
pub const U_INV_W: usize = 11;
pub const U_UP: usize = 12;
pub const U_INV_H: usize = 15;
pub const U_LIGHT: usize = 16; //      16-18 light centre  19 light size
pub const U_LSIZE: usize = 19;
pub const U_TRI_BASE: usize = 20; //   triangle records, in 16-byte units
pub const U_SPP: usize = 21;
pub const U_SEED: usize = 22;
pub const U_NNODES: usize = 23;
pub const U_INV_SPP: usize = 24;
pub const U_SAMPLE_BASE: usize = 26; // first sample index of this pass (float)
pub const U_STRIDE: usize = 28; //     28-30 node layout stride for x, y, z octant bits (16-byte units)
pub const U_COUNT: usize = 32;

// buffers: 0 = scene (Scene::write_blob), 1 = XRGB8888 image, 2 = primary hit
// distance (negative on the glossy teapot)
pub const BUF_SCENE: usize = 0;
pub const BUF_RGB: usize = 1;
pub const BUF_DEPTH: usize = 2;

/// `samples` (U_SPP) per pixel of each packet's tile; debug = traversal heat map
pub fn build(debug: bool) -> Program {
    kir::build(|| Kernel::new(debug).build())
}

use super::kir;

// blocks: node pair, triangle hit half, shading half
const BK_NODE: usize = 0;
const BK_TRI: usize = 1;
const BK_SHADE: usize = 2;

#[derive(Clone, Copy)]
struct Vec3 {
    x: Operand,
    y: Operand,
    z: Operand,
}
impl Index<usize> for Vec3 {
    type Output = Operand;
    fn index(&self, a: usize) -> &Operand {
        match a {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}
fn vreg3() -> Vec3 {
    let x = vreg();
    let y = vreg();
    Vec3 { x, y, z: vreg() }
}

struct Kernel {
    dbg: bool,
    // ray + hit state used by the traversal (allocation order = register order)
    ro: Vec3, // origin
    rd: Vec3, // direction
    ri: Vec3, // 1 / direction
    rq: Vec3, // origin / direction
    ht: Operand, // t
    hn: Vec3,    // interpolated normal
    hg: Vec3,    // geometric normal
    hm: Operand, // material: 0 miss, 1 teapot, 2 floor
    t: [Operand; 11], // temporaries
    // shading state that survives a traversal
    sp: Vec3,
    sn: Vec3,
    sg: Vec3,
    sd: Vec3,
    sm: Operand,
    acc: Vec3,
    thr: Vec3,
    rng: Operand, // rng state
    dep: Operand, // primary hit distance of the last sample
    // scalars
    node: Operand,
    tri: Operand,
    tri_end: Operand,
    sample: Operand,
    gx: Operand, // tile origin
    gy: Operand,
    r2: Operand,  // R2 index
    tmp: Operand, // scratch
    skip: Operand,
    lbase: Operand,
    cached: Operand,
    naddr: Operand,
    taddr: Operand,
    dbg_nodes: Operand,
    dbg_tris: Operand,
}

// clamp a reciprocal so (b - o) * inv never makes inf - inf
fn clamp_big(v: Operand) {
    v.set(min(v, 1e20));
    v.set(max(v, -1e20));
}

impl Kernel {
    fn new(dbg: bool) -> Kernel {
        // vector registers are numbered in the order of these calls
        let (ro, rd, ri, rq) = (vreg3(), vreg3(), vreg3(), vreg3());
        let ht = vreg();
        let (hn, hg) = (vreg3(), vreg3());
        let hm = vreg();
        let t: [Operand; 11] = std::array::from_fn(|_| vreg());
        let (sp, sn, sg, sd) = (vreg3(), vreg3(), vreg3(), vreg3());
        let sm = vreg();
        let (acc, thr) = (vreg3(), vreg3());
        let rng = vreg().u();
        let dep = vreg();
        Kernel {
            dbg,
            ro,
            rd,
            ri,
            rq,
            ht,
            hn,
            hg,
            hm,
            t,
            sp,
            sn,
            sg,
            sd,
            sm,
            acc,
            thr,
            rng,
            dep,
            node: sreg(0).u(),
            tri: sreg(1).u(),
            tri_end: sreg(2).u(),
            sample: sreg(3).u(),
            gx: sreg(4),
            gy: sreg(5),
            r2: sreg(6),
            tmp: sreg(7),
            skip: sreg(8).u(),
            lbase: sreg(9).u(),
            cached: sreg(10).u(),
            naddr: sreg(11).u(),
            taddr: sreg(12).u(),
            dbg_nodes: sreg(13).u(),
            dbg_tris: sreg(14).u(),
        }
    }

    fn dot3(&self, d: Operand, a: Vec3, b: Vec3, tmp: Operand) {
        d.set(a.x * b.x);
        tmp.set(a.y * b.y);
        d.set(d + tmp);
        tmp.set(a.z * b.z);
        d.set(d + tmp);
    }

    fn normalize3(&self, d: Vec3, a: Vec3, tmp: Operand) {
        self.dot3(tmp, a, a, if d.x.same_reg(&a.x) { self.t[10] } else { d.x });
        tmp.set(rsq(tmp));
        d.x.set(a.x * tmp);
        d.y.set(a.y * tmp);
        d.z.set(a.z * tmp);
    }

    // stateless hash of (src ^ key) → [0,1) in dst; 16-bit multipliers because a
    // 32x16 multiply is a single instruction on most GPUs
    fn hash01(&self, dst: Operand, src: Operand, key: u32, tmp: Operand) {
        let (x, tt) = (dst.u(), tmp.u());
        x.set(src.u() ^ key);
        for _ in 0..2 {
            tt.set(x >> 16u32);
            x.set(x ^ tt);
            x.set(x * 0x7febu32);
            tt.set(x >> 15u32);
            x.set(x ^ tt);
            x.set(x * 0x846bu32);
            tt.set(x >> 16u32);
            x.set(x ^ tt);
        }
        x.set(x >> 9u32);
        x.set(x | 0x3f80_0000u32);
        dst.set(dst + -1.0);
    }

    // per-pixel offset in [0,1): interleaved gradient noise of the pixel position,
    // with the axes weighted (wx, wy) and shifted by k; cheap and well spread
    fn ign(&self, dst: Operand, wx: f32, wy: f32, k: f32, tmp: Operand) {
        dst.set(LANE_X + self.gx);
        tmp.set(LANE_Y + self.gy);
        dst.set(dst * wx);
        tmp.set(tmp * wy);
        dst.set(dst + tmp);
        dst.set(dst + k);
        dst.set(frac(dst));
        dst.set(dst * 52.9829189);
        dst.set(frac(dst));
    }

    // sample s of a 2D R2 low-discrepancy sequence, rotated per pixel; s keeps
    // counting across passes and animation frames, so it also stratifies over time
    fn r2_sample(&self, du: Operand, dv: Operand, key: f32, vtmp: Operand) {
        const R2A: f32 = 0.7548776662;
        const R2B: f32 = 0.5698402910;
        self.ign(du, 0.06711056, 0.00583715, key, vtmp);
        self.ign(dv, 0.00583715, 0.06711056, key + 0.5, vtmp);
        let (r2, tmp) = (self.r2, self.tmp);
        tmp.set(r2 * R2A);
        du.set(du + tmp);
        du.set(frac(du));
        tmp.set(r2 * R2B);
        dv.set(dv + tmp);
        dv.set(frac(dv));
    }

    // uniform random float in [0,1) per lane (xorshift32)
    fn rand01(&self, dst: Operand, vtmp: Operand) {
        let (rng, tt) = (self.rng, vtmp.u());
        tt.set(rng << 13u32);
        rng.set(rng ^ tt);
        tt.set(rng >> 17u32);
        rng.set(rng ^ tt);
        tt.set(rng << 5u32);
        rng.set(rng ^ tt);
        tt.set(rng >> 9u32);
        tt.set(tt | 0x3f80_0000u32);
        dst.set(vtmp + -1.0);
    }

    /* --------------------------------------------- packet traversal ----- */
    // In: ro, rd, ht (= tmax). Out: ht, hn (unnormalised), hg, hm (0 miss, 1 teapot, 2 floor).
    // Shadow mode is any-hit: an occluded lane gets ht = -1, which also fails every
    // later box/triangle test for it, and the walk ends once all 16 lanes are
    // occluded. Lanes may enter with ht = -1 to skip the ray entirely. The floor
    // is ignored (shadow rays go up towards the light).

    fn traverse(&self, shadow: bool) {
        let Kernel { ro, rd, ri, rq, ht, hn, hg, hm, t, node, tri, tri_end, tmp, skip, lbase, cached, naddr, taddr, .. } = *self;
        ri.x.set(inv(rd.x));
        ri.y.set(inv(rd.y));
        ri.z.set(inv(rd.z));
        clamp_big(ri.x);
        clamp_big(ri.y);
        clamp_big(ri.z);
        rq.x.set(ro.x * ri.x);
        rq.y.set(ro.y * ri.y);
        rq.z.set(ro.z * ri.z);
        hm.set(0.0);
        hn.x.set(0.0);
        hn.y.set(1.0);
        hn.z.set(0.0);
        hg.x.set(0.0);
        hg.y.set(1.0);
        hg.z.set(0.0);
        node.set(0u32);

        // pick the node layout for the octant of lane 0's direction
        lbase.set(0u32);
        for a in 0..3 {
            F1.set(lt(lane0(rd[a]), 0.0));
            lbase.at(F1).set(lbase + uni(U_STRIDE + a).u());
        }

        // Nodes are 32 bytes, so one 64-byte load returns node i in dwords 0-7 of
        // the node block and node i+1 in 8-15. Descending to the first child (i+1)
        // then needs no load.
        cached.set(0xffff_fff0u32); // nothing cached yet
        let mut top = here();
        let (mut end_walk, mut to_tri) = (Label::new(), Label::new());
        let mut slot_at = [Label::new(), Label::new()];
        if self.dbg {
            self.dbg_nodes.set(self.dbg_nodes + 1u32);
        }
        F1.set(ge(node, uni(U_NNODES).u()));
        jmp_if(F1, &mut end_walk);
        F1.set(eq(node, cached));
        jmp_if(F1, &mut slot_at[0]);
        tmp.u().set(cached + 1u32);
        F1.set(eq(node, tmp.u()));
        jmp_if(F1, &mut slot_at[1]);
        naddr.set(node << 1u32);
        load(BK_NODE, naddr, lbase, 0);
        cached.set(node);

        // node record: bmin .0-2, skip .3, bmax .4-6, first_tri << 3 | count .7
        for (slot, at) in slot_at.iter_mut().enumerate() {
            let nb = |i: usize| blk(BK_NODE, slot * 8 + i);
            at.bind();
            // slab test, scheduled so neighbouring instructions are independent:
            // in-order GPU cores stall on a dependent op
            for a in 0..3 {
                // t(2a) = near plane, t(2a+1) = far plane
                t[2 * a].set(ri[a] * nb(a));
                t[2 * a + 1].set(ri[a] * nb(4 + a));
            }
            for a in 0..3 {
                t[2 * a].set(t[2 * a] - rq[a]);
                t[2 * a + 1].set(t[2 * a + 1] - rq[a]);
            }
            t[6].set(min(t[0], t[1]));
            t[7].set(min(t[2], t[3]));
            t[8].set(min(t[4], t[5]));
            t[9].set(max(t[0], t[1]));
            t[10].set(max(t[2], t[3]));
            t[0].set(max(t[4], t[5]));
            t[6].set(max(t[6], t[7]));
            t[9].set(min(t[9], t[10]));
            t[8].set(max(t[8], 0.0));
            t[0].set(min(t[0], ht));
            t[6].set(max(t[6], t[8]));
            t[9].set(min(t[9], t[0]));
            F0.set(le(t[6], t[9]));
            let (mut miss, mut inner) = (Label::new(), Label::new());
            jmp_none(F0, &mut miss); // no lane hit the box
            tri_end.set(nb(7).u() & 7u32);
            F1.set(eq(tri_end, 0u32));
            jmp_if(F1, &mut inner);
            tri.set(nb(7).u() >> 3u32);
            tri_end.set(tri_end + tri);
            skip.set(nb(3).u());
            jmp(&mut to_tri);
            inner.bind(); // interior: first child is next
            node.set(node + 1u32);
            jmp(&mut top);
            miss.bind(); // missed: follow the skip link
            node.set(nb(3).u());
            jmp(&mut top);
        }

        // leaf: test each triangle. Its first 64 bytes decide the hit; the
        // shading half is only loaded when a lane hits.
        to_tri.bind();
        let mut tri_loop = here();
        if self.dbg {
            self.dbg_tris.set(self.dbg_tris + 1u32);
        }
        taddr.set(tri << 3u32);
        load(BK_TRI, taddr, uni(U_TRI_BASE).u(), 0);
        {
            // triangle record (Havel-Herout planes):
            //   TRI 0-3 N, d   (plane: N·x = d, N = e1 × e2)
            //   TRI 4-7 N1, d1 (u = N1·x + d1)       TRI 8-11 N2, d2 (v = N2·x + d2)
            //   TRI 12-14 n0   SHADE 0-2 dn1  SHADE 4-6 dn2  SHADE 8-10 geometric normal
            // Everything is kept multiplied by det = N·D so no division is needed
            // to decide a hit; t, u, v are divided out only when some lane hits.
            let tb = |i: usize| blk(BK_TRI, i);
            let sb = |i: usize| blk(BK_SHADE, i);
            // (scheduled in independent groups, see the slab test)
            // det = N·D → t0;  t' = d - N·O → t1
            t[0].set(rd.x * tb(0));
            t[1].set(ro.x * tb(0));
            t[7].set(rd.y * tb(1));
            t[9].set(ro.y * tb(1));
            t[8].set(rd.z * tb(2));
            t[10].set(ro.z * tb(2));
            t[0].set(t[0] + t[7]);
            t[1].set(t[1] + t[9]);
            t[0].set(t[0] + t[8]);
            t[1].set(t[1] + t[10]);
            t[1].set(-t[1] + tb(3));
            // P' = O det + D t' (= hit point * det) → t2 t3 t4
            t[2].set(ro.x * t[0]);
            t[3].set(ro.y * t[0]);
            t[4].set(ro.z * t[0]);
            t[7].set(rd.x * t[1]);
            t[8].set(rd.y * t[1]);
            t[9].set(rd.z * t[1]);
            t[2].set(t[2] + t[7]);
            t[3].set(t[3] + t[8]);
            t[4].set(t[4] + t[9]);
            // u' = N1·P' + d1 det → t5;  v' = N2·P' + d2 det → t6
            t[5].set(t[2] * tb(4));
            t[6].set(t[2] * tb(8));
            t[7].set(t[3] * tb(5));
            t[8].set(t[3] * tb(9));
            t[9].set(t[4] * tb(6));
            t[10].set(t[4] * tb(10));
            t[5].set(t[5] + t[7]);
            t[6].set(t[6] + t[8]);
            t[7].set(t[0] * tb(7));
            t[8].set(t[0] * tb(11));
            t[5].set(t[5] + t[9]);
            t[6].set(t[6] + t[10]);
            t[5].set(t[5] + t[7]);
            t[6].set(t[6] + t[8]);
            // hit iff u, v, 1-u-v, t > 0 and t < T; each scaled by det² (sign-safe):
            //   u'det, v'det, (det-u'-v')det, t'det, (T det - t')det
            t[7].set(t[5] * t[0]);
            t[8].set(t[6] * t[0]);
            t[9].set(t[0] - t[5]);
            t[10].set(t[1] * t[0]);
            t[2].set(ht * t[0]);
            t[9].set(t[9] - t[6]);
            t[2].set(t[2] - t[1]);
            t[7].set(min(t[7], t[8]));
            t[9].set(t[9] * t[0]);
            t[2].set(t[2] * t[0]);
            t[7].set(min(t[7], t[10]));
            t[9].set(min(t[9], t[2]));
            t[8].set(min(t[7], t[9]));
            F0.set(gt(t[8], 0.0));
            let mut no_hit = Label::new();
            jmp_none(F0, &mut no_hit);
            t[0].set(inv(t[0]));
            ht.at(F0).set(t[1] * t[0]);
            if shadow {
                ht.at(F0).set(-1.0);
            } else {
                t[5].set(t[5] * t[0]); // u
                t[6].set(t[6] * t[0]); // v
                load(BK_SHADE, taddr, uni(U_TRI_BASE).u(), 4);
                for a in 0..3 {
                    t[7].set(t[6] * sb(4 + a)); // v * dn2
                    hn[a].at(F0).set(t[5] * sb(a)); // u * dn1
                    hn[a].at(F0).set(hn[a] + tb(12 + a)); // + n0
                    hn[a].at(F0).set(hn[a] + t[7]);
                    hg[a].at(F0).set(sb(8 + a));
                }
            }
            hm.at(F0).set(1.0);
            no_hit.bind();
        }
        tri.set(tri + 1u32);
        F1.set(lt(tri, tri_end));
        jmp_if(F1, &mut tri_loop);
        let mut done = Label::new();
        if shadow {
            F0.set(lt(ht, 0.0));
            jmp_all(F0, &mut done); // every lane occluded
        }
        node.set(skip); // leaf done: follow its skip link
        jmp(&mut top);

        end_walk.bind();
        if shadow {
            done.bind();
            return;
        }
        // floor plane y = 0: t = -oy/dy
        t[0].set(-rq.y);
        t[1].set(t[0] + -1e-4);
        t[2].set(ht - t[0]);
        t[1].set(min(t[1], t[2]));
        F0.set(gt(t[1], 0.0));
        ht.at(F0).set(t[0]);
        hm.at(F0).set(2.0);
        hn.x.at(F0).set(0.0);
        hn.y.at(F0).set(1.0);
        hn.z.at(F0).set(0.0);
        hg.x.at(F0).set(0.0);
        hg.y.at(F0).set(1.0);
        hg.z.at(F0).set(0.0);
    }

    /* --------------------------------------------- shading -------------- */

    // sky(dir) → d, using s as scratch
    fn sky3(&self, d: Vec3, ydir: Operand, s: Operand) {
        let (hor, zen) = (look::SKY_HORIZON, look::SKY_ZENITH);
        s.set(max(ydir, 0.0));
        s.set(sqrt(s));
        for a in 0..3 {
            d[a].set(s * (zen[a] - hor[a]));
            d[a].set(d[a] + hor[a]);
        }
    }

    fn bounce(&self, last: bool, stratified: bool) {
        let Kernel { ro, rd, ht, hn, hg, hm, t, sp, sn, sg, sd, sm, acc, thr, dep, .. } = *self;
        ht.set(1e6);
        self.traverse(false);
        if stratified {
            // primary bounce: keep the depth, negative on the (glossy) teapot
            dep.set(ht);
            F0.set(eq(hm, 1.0));
            dep.at(F0).set(-ht);
        }

        // miss lanes: add sky, kill throughput
        let sk = Vec3 { x: t[1], y: t[2], z: t[3] };
        F0.set(eq(hm, 0.0));
        t[0].set(0.0);
        t[0].at(F0).set(1.0);
        self.sky3(sk, rd.y, t[4]);
        for a in 0..3 {
            t[5].set(thr[a] * t[0]);
            t[5].set(t[5] * sk[a]);
            acc[a].set(acc[a] + t[5]);
        }
        t[0].set(-t[0] + 1.0);
        for a in 0..3 {
            thr[a].set(thr[a] * t[0]);
        }

        // hit point, oriented normals
        for a in 0..3 {
            sp[a].set(rd[a] * ht);
            sp[a].set(sp[a] + ro[a]);
            sd[a].set(rd[a]);
            sg[a].set(hg[a]);
        }
        sm.set(hm);
        self.normalize3(sn, hn, t[0]);
        self.dot3(t[0], sg, sd, t[1]);
        F0.set(gt(t[0], 0.0));
        for a in 0..3 {
            sg[a].at(F0).set(-sg[a]);
        }
        self.dot3(t[0], sn, sg, t[1]);
        F0.set(lt(t[0], 0.0));
        for a in 0..3 {
            sn[a].at(F0).set(-sn[a]);
        }

        // shadow ray towards a jittered point on a square area light
        if stratified {
            self.r2_sample(t[1], t[0], 0.31, t[2]); // swapped dims: decorrelate from pixel jitter
        } else {
            self.rand01(t[0], t[2]);
            self.rand01(t[1], t[2]);
        }
        t[0].set(t[0] + -0.5);
        t[0].set(t[0] * uni(U_LSIZE));
        t[1].set(t[1] + -0.5);
        t[1].set(t[1] * uni(U_LSIZE));
        rd.x.set(t[0] + uni(U_LIGHT));
        rd.x.set(rd.x - sp.x);
        rd.y.set(-sp.y + uni(U_LIGHT + 1));
        rd.z.set(t[1] + uni(U_LIGHT + 2));
        rd.z.set(rd.z - sp.z);
        self.dot3(t[0], rd, rd, t[1]);
        t[1].set(rsq(t[0]));
        for a in 0..3 {
            rd[a].set(rd[a] * t[1]);
        }
        ht.set(t[0] * t[1]); // distance to the light sample
        for a in 0..3 {
            t[2].set(sg[a] * 2e-3);
            ro[a].set(sp[a] + t[2]);
        }
        // no shadow ray for lanes facing away from the light or that hit nothing
        self.dot3(t[0], sg, rd, t[1]);
        F0.set(le(t[0], 0.0));
        ht.at(F0).set(-1.0);
        F0.set(eq(sm, 0.0));
        ht.at(F0).set(-1.0);
        self.traverse(true);

        // local shading: vis → hn.x, n·l → hn.y
        let (vis, ndl) = (hn.x, hn.y);
        F0.set(gt(ht, 0.0));
        vis.set(0.0);
        vis.at(F0).set(1.0);
        self.dot3(ndl, sn, rd, t[0]);
        ndl.set(max(ndl, 0.0));
        // half vector
        let h = Vec3 { x: t[0], y: t[1], z: t[2] };
        for a in 0..3 {
            h[a].set(rd[a] - sd[a]);
        }
        self.normalize3(h, h, t[3]);
        self.dot3(t[3], sn, h, t[4]);
        t[3].set(max(t[3], 0.0));
        // material parameters (flag 0 = teapot)
        F0.set(eq(sm, 1.0));
        t[4].set(60.0);
        t[4].at(F0).set(180.0);
        t[3].set(pow(t[3], t[4]));
        t[5].set(0.25f32 * 68.0 / (8.0 * PI));
        t[5].at(F0).set(0.9f32 * 188.0 / (8.0 * PI));
        t[3].set(t[3] * t[5]); // specular
        t[6].set(0.09);
        t[6].at(F0).set(0.05); // F0
        // albedo: checkerboard floor, red teapot → t8 t9 t10
        t[7].set(floor(sp.x));
        t[8].set(floor(sp.z));
        t[7].set(t[7] + t[8]);
        t[7].set(t[7] * 0.5);
        t[7].set(frac(t[7]));
        t[7].set(t[7] * 2.0);
        let (dark, light, red) = (v3(0.16, 0.17, 0.20), v3(0.78, 0.77, 0.74), v3(0.62, 0.07, 0.05));
        let alb = Vec3 { x: t[8], y: t[9], z: t[10] };
        for a in 0..3 {
            alb[a].set(t[7] * (light[a] - dark[a]));
            alb[a].set(alb[a] + dark[a]);
        }
        for a in 0..3 {
            alb[a].at(F0).set(red[a]);
        }
        // direct = (albedo * n·l + spec) * vis * light colour → t0 t1 t2
        for a in 0..3 {
            t[a].set(alb[a] * ndl);
            t[a].set(t[a] + t[3]);
            t[a].set(t[a] * vis);
            t[a].set(t[a] * look::LIGHT_COL[a]);
        }
        // ambient sky light: albedo * sky(n) * (0.231 + 0.154 n.y)
        let skn = hg;
        self.sky3(skn, sn.y, t[4]);
        t[5].set(sn.y * 0.154);
        t[5].set(t[5] + 0.231);
        for a in 0..3 {
            t[7].set(skn[a] * alb[a]);
            t[7].set(t[7] * t[5]);
            t[a].set(t[a] + t[7]);
        }
        // Fresnel (Schlick) → t6
        self.dot3(t[4], sn, sd, t[5]);
        t[4].set(min(t[4], 0.0));
        t[4].set(t[4] + 1.0); // 1 - cos
        t[5].set(t[4] * t[4]);
        t[5].set(t[5] * t[5]);
        t[5].set(t[5] * t[4]);
        t[7].set(-t[6] + 1.0);
        t[7].set(t[7] * t[5]);
        t[6].set(t[6] + t[7]);
        // distance fog towards the horizon colour
        t[7].set(sp.x - uni(U_CAM));
        t[7].set(t[7] * t[7]);
        t[8].set(sp.y - uni(U_CAM + 1));
        t[8].set(t[8] * t[8]);
        t[7].set(t[7] + t[8]);
        t[8].set(sp.z - uni(U_CAM + 2));
        t[8].set(t[8] * t[8]);
        t[7].set(t[7] + t[8]);
        t[7].set(sqrt(t[7]));
        t[7].set(t[7] * (-0.02f32 * 1.442695));
        F0.set(eq(sm, 1.0));
        t[7].at(F0).set(0.0); // fog only on the floor
        t[7].set(exp2(t[7])); // e = exp(-0.02 d)
        let fog = v3(0.80, 0.84, 0.91);
        // acc += thr * (1 - F) * lerp(fog, local, e);  thr *= F
        t[9].set(-t[6] + 1.0);
        for a in 0..3 {
            t[a].set(t[a] + -fog[a]);
            t[a].set(t[a] * t[7]);
            t[a].set(t[a] + fog[a]);
            t[a].set(t[a] * t[9]);
            t[a].set(t[a] * thr[a]);
            acc[a].set(acc[a] + t[a]);
            thr[a].set(thr[a] * t[6]);
        }

        // reflected ray
        self.dot3(t[0], sd, sn, t[1]);
        t[0].set(t[0] * -2.0);
        for a in 0..3 {
            t[1].set(sn[a] * t[0]);
            rd[a].set(sd[a] + t[1]);
            t[2].set(sg[a] * 2e-3);
            ro[a].set(sp[a] + t[2]);
        }
        if last {
            let sr = Vec3 { x: t[1], y: t[2], z: t[3] };
            self.sky3(sr, rd.y, t[4]);
            for a in 0..3 {
                t[5].set(thr[a] * sr[a]);
                acc[a].set(acc[a] + t[5]);
            }
        }
    }

    fn build(&self) {
        let Kernel { ro, rd, ht, hm, t, acc, thr, rng, dep, sample, gx, gy, r2, dbg_nodes, dbg_tris, .. } = *self;
        // tile origin as float scalars
        gx.set(GROUP_X);
        gy.set(GROUP_Y);
        gx.set(gx * 4.0);
        gy.set(gy * 4.0);

        // per-pixel seed = pixel index (exact in float below 2^24), RNG state = hash of it
        t[0].set(LANE_X + gx);
        t[1].set(LANE_Y + gy);
        t[1].set(t[1] * 4096.0);
        t[0].set(t[0] + t[1]);
        t[0].u().set(t[0]);
        self.hash01(rng.f(), t[0], 0x9e37_79b9, t[1]);
        rng.set(rng ^ uni(U_SEED).u()); // fresh random stream per pass
        rng.set(rng | 1u32); // xorshift state must be non-zero

        for a in 0..3 {
            acc[a].set(0.0);
        }
        sample.set(0u32);
        r2.set(uni(U_SAMPLE_BASE)); // R2 sample index continues across passes

        let mut sample_loop = here();
        // camera ray through a jittered point of the pixel
        self.r2_sample(t[2], t[3], 0.0, t[4]);
        t[0].set(LANE_X + gx);
        t[0].set(t[0] + t[2]);
        t[1].set(LANE_Y + gy);
        t[1].set(t[1] + t[3]);
        t[0].set(t[0] * uni(U_INV_W));
        t[0].set(t[0] + -1.0);
        t[0].set(t[0] * uni(U_TAN_X));
        t[1].set(t[1] * uni(U_INV_H));
        t[1].set(-t[1] + 1.0);
        t[1].set(t[1] * uni(U_TAN_Y));
        for a in 0..3 {
            rd[a].set(t[0] * uni(U_RIGHT + a));
            t[2].set(t[1] * uni(U_UP + a));
            rd[a].set(rd[a] + t[2]);
            rd[a].set(rd[a] + uni(U_FWD + a));
            ro[a].set(uni(U_CAM + a));
            thr[a].set(1.0);
        }
        self.normalize3(rd, rd, t[0]);

        if self.dbg {
            // debug view: red = material, green = nodes visited / 4, blue = triangles tested
            dbg_nodes.set(0u32);
            dbg_tris.set(0u32);
            ht.set(1e6);
            self.traverse(false);
            t[0].set(hm * 100.0);
            t[4].u().set(t[0]);
            dbg_nodes.set(dbg_nodes >> 2u32);
            t[5].u().set(dbg_nodes);
            t[6].u().set(dbg_tris);
        } else {
            self.bounce(false, true);
            self.bounce(true, false);

            sample.set(sample + 1u32);
            r2.set(r2 + 1.0);
            F1.set(lt(sample, uni(U_SPP).u()));
            jmp_if(F1, &mut sample_loop);

            // average, ACES tone curve, gamma, pack to XRGB8888
            for a in 0..3 {
                let c = acc[a];
                c.set(c * uni(U_INV_SPP));
                c.set(c * 0.9);
                t[0].set(c * 2.51);
                t[0].set(t[0] + 0.03);
                t[0].set(t[0] * c);
                t[1].set(c * 2.43);
                t[1].set(t[1] + 0.59);
                t[1].set(t[1] * c);
                t[1].set(t[1] + 0.14);
                t[1].set(inv(t[1]));
                t[0].set(sat(t[0] * t[1]));
                t[2].set(1.0f32 / 2.2);
                t[0].set(pow(t[0], t[2]));
                t[0].set(t[0] * 255.0);
                t[0].set(t[0] + 0.5);
                t[a + 4].u().set(t[0]);
            }
        }
        let px = t[10].u();
        px.set(t[4].u() << 16u32);
        t[5].u().set(t[5].u() << 8u32);
        px.set(px | t[5].u());
        px.set(px | t[6].u());
        store(BUF_RGB, px);
        store(BUF_DEPTH, dep);
        end();
    }
}

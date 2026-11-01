// rt_kernel.cpp — the teapot ray tracer written once in the kernel IR. It knows
// nothing about any particular GPU.
//
// Each packet shades a 4x4 tile. The 16 rays traverse the BVH together: the
// BVH is stored depth-first with "skip" links (Scene::write_blob), so the walk
// needs no stack and every branch is uniform ("descend if any lane hits the
// box"), with per-lane work done by predication.

#include "rt_kernel.h"

#include <array>
#include <cmath>
#include <utility>

#include "../scene.h"

namespace rt {
namespace {

using namespace kir;

enum Block { BK_NODE, BK_TRI, BK_SHADE };  // node pair, triangle hit half, shading half

struct Vec3 {
    Reg x, y, z;
    const Reg &operator[](int a) const { return a == 0 ? x : a == 1 ? y : z; }
    Vec3 &operator=(const Vec3 &) = delete;  // assigning registers emits code; never copy a Vec3 over another
};
Vec3 vreg3() {
    Reg x = vreg(), y = vreg(), z = vreg();
    return {x, y, z};
}
template <size_t... I> std::array<Reg, sizeof...(I)> vregs(std::index_sequence<I...>) {
    return {((void)I, vreg())...};          // braced lists evaluate left to right
}

class Kernel {
public:
    explicit Kernel(bool debug) : dbg(debug) {}
    void build();

private:
    const bool dbg;
    // ray + hit state used by the traversal (allocation order = register order)
    const Vec3 RO = vreg3(), RD = vreg3(), RI = vreg3(), RQ = vreg3();  // origin, dir, 1/dir, origin/dir
    const Reg HT = vreg();                                              // t
    const Vec3 HN = vreg3(), HG = vreg3();                              // interpolated / geometric normal
    const Reg HM = vreg();                                              // material: 0 miss, 1 teapot, 2 floor
    const std::array<Reg, 11> t = vregs(std::make_index_sequence<11>());  // temporaries
    // shading state that survives a traversal
    const Vec3 SP = vreg3(), SN = vreg3(), SG = vreg3(), SD = vreg3();
    const Reg SM = vreg();
    const Vec3 ACC = vreg3(), THR = vreg3();
    const Reg RNG = vreg().u(), DEP = vreg();  // rng state, primary hit distance of the last sample

    // scalars
    const Reg node = S(0).u(), tri = S(1).u(), tri_end = S(2).u(), sample = S(3).u();
    const Reg gx = S(4), gy = S(5), r2 = S(6), tmp = S(7);   // tile origin, R2 index, scratch
    const Reg skip = S(8).u(), lbase = S(9).u(), cached = S(10).u(), naddr = S(11).u(), taddr = S(12).u();
    const Reg dbg_nodes = S(13).u(), dbg_tris = S(14).u();

    void dot3(const Reg &d, const Vec3 &a, const Vec3 &b, const Reg &tmp) const;
    void normalize3(const Vec3 &d, const Vec3 &a, const Reg &tmp) const;
    void hash01(const Reg &dst, const Reg &src, uint32_t key, const Reg &tmp) const;
    void ign(const Reg &dst, float wx, float wy, float k, const Reg &tmp) const;
    void r2_sample(const Reg &du, const Reg &dv, float key, const Reg &tmp) const;
    void rand01(const Reg &dst, const Reg &tmp) const;
    void sky3(const Vec3 &d, const Operand &ydir, const Reg &s) const;
    void traverse(bool shadow) const;
    void bounce(bool last, bool stratified) const;
};

void Kernel::dot3(const Reg &d, const Vec3 &a, const Vec3 &b, const Reg &tmp) const {
    d = a.x * b.x;
    tmp = a.y * b.y; d += tmp;
    tmp = a.z * b.z; d += tmp;
}

void Kernel::normalize3(const Vec3 &d, const Vec3 &a, const Reg &tmp) const {
    dot3(tmp, a, a, d.x.same_reg(a.x) ? t[10] : d.x);
    tmp = rsq(tmp);
    d.x = a.x * tmp; d.y = a.y * tmp; d.z = a.z * tmp;
}

// stateless hash of (src ^ key) → [0,1) in dst; 16-bit multipliers because a
// 32x16 multiply is a single instruction on most GPUs
void Kernel::hash01(const Reg &dst, const Reg &src, uint32_t key, const Reg &tmp) const {
    Reg x = dst.u(), tt = tmp.u();
    x = src.u() ^ u32(key);
    for (int r = 0; r < 2; r++) {
        tt = x >> u32(16); x = x ^ tt;
        x = x * u32(0x7feb);
        tt = x >> u32(15); x = x ^ tt;
        x = x * u32(0x846b);
        tt = x >> u32(16); x = x ^ tt;
    }
    x = x >> u32(9); x = x | u32(0x3f800000);
    dst = dst + -1.0f;
}

// per-pixel offset in [0,1): interleaved gradient noise of the pixel position,
// with the axes weighted (wx, wy) and shifted by k; cheap and well spread
void Kernel::ign(const Reg &dst, float wx, float wy, float k, const Reg &tmp) const {
    dst = lane_x + gx; tmp = lane_y + gy;
    dst = dst * wx; tmp = tmp * wy;
    dst += tmp; dst = dst + k;
    dst = frac(dst); dst = dst * 52.9829189f; dst = frac(dst);
}

// sample s of a 2D R2 low-discrepancy sequence, rotated per pixel; s keeps
// counting across passes and animation frames, so it also stratifies over time
void Kernel::r2_sample(const Reg &du, const Reg &dv, float key, const Reg &tmp_) const {
    const float R2A = 0.7548776662f, R2B = 0.5698402910f;
    ign(du, 0.06711056f, 0.00583715f, key, tmp_); ign(dv, 0.00583715f, 0.06711056f, key + 0.5f, tmp_);
    tmp = r2 * R2A; du += tmp; du = frac(du);
    tmp = r2 * R2B; dv += tmp; dv = frac(dv);
}

// uniform random float in [0,1) per lane (xorshift32)
void Kernel::rand01(const Reg &dst, const Reg &tmp_) const {
    Reg tt = tmp_.u();
    tt = RNG << u32(13); RNG = RNG ^ tt;
    tt = RNG >> u32(17); RNG = RNG ^ tt;
    tt = RNG << u32(5); RNG = RNG ^ tt;
    tt = RNG >> u32(9); tt = tt | u32(0x3f800000);
    dst = tmp_ + -1.0f;
}

// clamp a reciprocal so (b - o) * inv never makes inf - inf
void clamp_big(const Reg &v) { v = min(v, 1e20f); v = max(v, -1e20f); }

/* ------------------------------------------------- packet traversal ----- */
// In: RO, RD, HT (= tmax). Out: HT, HN (unnormalised), HG, HM (0 miss, 1 teapot, 2 floor).
// Shadow mode is any-hit: an occluded lane gets HT = -1, which also fails every
// later box/triangle test for it, and the walk ends once all 16 lanes are
// occluded. Lanes may enter with HT = -1 to skip the ray entirely. The floor
// is ignored (shadow rays go up towards the light).

void Kernel::traverse(bool shadow) const {
    RI.x = inv(RD.x); RI.y = inv(RD.y); RI.z = inv(RD.z);
    clamp_big(RI.x); clamp_big(RI.y); clamp_big(RI.z);
    RQ.x = RO.x * RI.x; RQ.y = RO.y * RI.y; RQ.z = RO.z * RI.z;
    HM = 0.0f;
    HN.x = 0.0f; HN.y = 1.0f; HN.z = 0.0f;
    HG.x = 0.0f; HG.y = 1.0f; HG.z = 0.0f;
    node = u32(0);

    // pick the node layout for the octant of lane 0's direction
    lbase = u32(0);
    for (int a = 0; a < 3; a++) {
        f1 = lane0(RD[a]) < 0.0f;
        lbase[f1] += U(U_STRIDE + a).u();
    }

    // Nodes are 32 bytes, so one 64-byte load returns node i in dwords 0-7 of
    // the node block and node i+1 in 8-15. Descending to the first child (i+1)
    // then needs no load.
    cached = u32(0xfffffff0u);                       // nothing cached yet
    Label top = here(), end_walk, slot_at[2], to_tri;
    if (dbg) dbg_nodes += u32(1);
    f1 = node >= U(U_NNODES).u();
    jmp_if(f1, end_walk);
    f1 = node == cached;
    jmp_if(f1, slot_at[0]);
    tmp.u() = cached + u32(1);
    f1 = node == tmp.u();
    jmp_if(f1, slot_at[1]);
    naddr = node << u32(1);
    load(BK_NODE, naddr, lbase);
    cached = node;

    // node record: bmin .0-2, skip .3, bmax .4-6, first_tri << 3 | count .7
    for (int slot = 0; slot < 2; slot++) {
        auto nb = [&](int i) { return B(BK_NODE, slot * 8 + i); };
        slot_at[slot].bind();
        // slab test, scheduled so neighbouring instructions are independent:
        // in-order GPU cores stall on a dependent op
        for (int a = 0; a < 3; a++) {                // t(2a) = near plane, t(2a+1) = far plane
            t[2 * a] = RI[a] * nb(a);
            t[2 * a + 1] = RI[a] * nb(4 + a);
        }
        for (int a = 0; a < 3; a++) {
            t[2 * a] = t[2 * a] - RQ[a];
            t[2 * a + 1] = t[2 * a + 1] - RQ[a];
        }
        t[6] = min(t[0], t[1]); t[7] = min(t[2], t[3]); t[8] = min(t[4], t[5]);
        t[9] = max(t[0], t[1]); t[10] = max(t[2], t[3]); t[0] = max(t[4], t[5]);
        t[6] = max(t[6], t[7]); t[9] = min(t[9], t[10]);
        t[8] = max(t[8], 0.0f); t[0] = min(t[0], HT);
        t[6] = max(t[6], t[8]); t[9] = min(t[9], t[0]);
        f0 = t[6] <= t[9];
        Label miss, inner;
        jmp_none(f0, miss);                          // no lane hit the box
        tri_end = nb(7).u() & u32(7);
        f1 = tri_end == u32(0);
        jmp_if(f1, inner);
        tri = nb(7).u() >> u32(3);
        tri_end += tri;
        skip = nb(3).u();
        jmp(to_tri);
        inner.bind();                                // interior: first child is next
        node += u32(1);
        jmp(top);
        miss.bind();                                 // missed: follow the skip link
        node = nb(3).u();
        jmp(top);
    }

    // leaf: test each triangle. Its first 64 bytes decide the hit; the
    // shading half is only loaded when a lane hits.
    to_tri.bind();
    Label tri_loop = here();
    if (dbg) dbg_tris += u32(1);
    taddr = tri << u32(3);
    load(BK_TRI, taddr, U(U_TRI_BASE).u());
    {
        // triangle record (Havel-Herout planes):
        //   TRI 0-3 N, d   (plane: N·x = d, N = e1 × e2)
        //   TRI 4-7 N1, d1 (u = N1·x + d1)       TRI 8-11 N2, d2 (v = N2·x + d2)
        //   TRI 12-14 n0   SHADE 0-2 dn1  SHADE 4-6 dn2  SHADE 8-10 geometric normal
        // Everything is kept multiplied by det = N·D so no division is needed
        // to decide a hit; t, u, v are divided out only when some lane hits.
        auto tb = [](int i) { return B(BK_TRI, i); };
        auto sb = [](int i) { return B(BK_SHADE, i); };
        // (scheduled in independent groups, see the slab test)
        // det = N·D → t0;  t' = d - N·O → t1
        t[0] = RD.x * tb(0); t[1] = RO.x * tb(0);
        t[7] = RD.y * tb(1); t[9] = RO.y * tb(1);
        t[8] = RD.z * tb(2); t[10] = RO.z * tb(2);
        t[0] += t[7]; t[1] += t[9];
        t[0] += t[8]; t[1] += t[10];
        t[1] = -t[1] + tb(3);
        // P' = O det + D t' (= hit point * det) → t2 t3 t4
        t[2] = RO.x * t[0]; t[3] = RO.y * t[0]; t[4] = RO.z * t[0];
        t[7] = RD.x * t[1]; t[8] = RD.y * t[1]; t[9] = RD.z * t[1];
        t[2] += t[7]; t[3] += t[8]; t[4] += t[9];
        // u' = N1·P' + d1 det → t5;  v' = N2·P' + d2 det → t6
        t[5] = t[2] * tb(4); t[6] = t[2] * tb(8);
        t[7] = t[3] * tb(5); t[8] = t[3] * tb(9);
        t[9] = t[4] * tb(6); t[10] = t[4] * tb(10);
        t[5] += t[7]; t[6] += t[8];
        t[7] = t[0] * tb(7); t[8] = t[0] * tb(11);
        t[5] += t[9]; t[6] += t[10];
        t[5] += t[7]; t[6] += t[8];
        // hit iff u, v, 1-u-v, t > 0 and t < T; each scaled by det² (sign-safe):
        //   u'det, v'det, (det-u'-v')det, t'det, (T det - t')det
        t[7] = t[5] * t[0]; t[8] = t[6] * t[0];
        t[9] = t[0] - t[5]; t[10] = t[1] * t[0]; t[2] = HT * t[0];
        t[9] = t[9] - t[6]; t[2] = t[2] - t[1];
        t[7] = min(t[7], t[8]);
        t[9] = t[9] * t[0]; t[2] = t[2] * t[0];
        t[7] = min(t[7], t[10]);
        t[9] = min(t[9], t[2]);
        t[8] = min(t[7], t[9]);
        f0 = t[8] > 0.0f;
        Label no_hit;
        jmp_none(f0, no_hit);
        t[0] = inv(t[0]);
        HT[f0] = t[1] * t[0];
        if (shadow) {
            HT[f0] = -1.0f;
        } else {
            t[5] = t[5] * t[0]; t[6] = t[6] * t[0];   // u, v
            load(BK_SHADE, taddr, U(U_TRI_BASE).u(), 4);
            for (int a = 0; a < 3; a++) {
                t[7] = t[6] * sb(4 + a);              // v * dn2
                HN[a][f0] = t[5] * sb(a);             // u * dn1
                HN[a][f0] += tb(12 + a);              // + n0
                HN[a][f0] += t[7];
                HG[a][f0] = sb(8 + a);
            }
        }
        HM[f0] = 1.0f;
        no_hit.bind();
    }
    tri += u32(1);
    f1 = tri < tri_end;
    jmp_if(f1, tri_loop);
    Label done;
    if (shadow) {
        f0 = HT < 0.0f;
        jmp_all(f0, done);                           // every lane occluded
    }
    node = skip;                                     // leaf done: follow its skip link
    jmp(top);

    end_walk.bind();
    if (shadow) { done.bind(); return; }
    // floor plane y = 0: t = -oy/dy
    t[0] = -RQ.y;
    t[1] = t[0] + -1e-4f;
    t[2] = HT - t[0];
    t[1] = min(t[1], t[2]);
    f0 = t[1] > 0.0f;
    HT[f0] = t[0];
    HM[f0] = 2.0f;
    HN.x[f0] = 0.0f; HN.y[f0] = 1.0f; HN.z[f0] = 0.0f;
    HG.x[f0] = 0.0f; HG.y[f0] = 1.0f; HG.z[f0] = 0.0f;
}

/* ------------------------------------------------- shading -------------- */

// sky(dir) → d, using s as scratch
void Kernel::sky3(const Vec3 &d, const Operand &ydir, const Reg &s) const {
    using look::sky_horizon, look::sky_zenith;
    s = max(ydir, 0.0f);
    s = sqrt(s);
    for (int a = 0; a < 3; a++) { d[a] = s * (sky_zenith[a] - sky_horizon[a]); d[a] = d[a] + sky_horizon[a]; }
}

void Kernel::bounce(bool last, bool stratified) const {
    HT = 1e6f;
    traverse(false);
    if (stratified) {                                // primary bounce: keep the depth,
        DEP = HT;                                    // negative on the (glossy) teapot
        f0 = HM == 1.0f;
        DEP[f0] = -HT;
    }

    // miss lanes: add sky, kill throughput
    const Vec3 sk{t[1], t[2], t[3]};
    f0 = HM == 0.0f;
    t[0] = 0.0f; t[0][f0] = 1.0f;
    sky3(sk, RD.y, t[4]);
    for (int a = 0; a < 3; a++) {
        t[5] = THR[a] * t[0]; t[5] = t[5] * sk[a];
        ACC[a] += t[5];
    }
    t[0] = -t[0] + 1.0f;
    for (int a = 0; a < 3; a++) THR[a] = THR[a] * t[0];

    // hit point, oriented normals
    for (int a = 0; a < 3; a++) {
        SP[a] = RD[a] * HT; SP[a] += RO[a];
        SD[a] = RD[a];
        SG[a] = HG[a];
    }
    SM = HM;
    normalize3(SN, HN, t[0]);
    dot3(t[0], SG, SD, t[1]);
    f0 = t[0] > 0.0f;
    for (int a = 0; a < 3; a++) SG[a][f0] = -SG[a];
    dot3(t[0], SN, SG, t[1]);
    f0 = t[0] < 0.0f;
    for (int a = 0; a < 3; a++) SN[a][f0] = -SN[a];

    // shadow ray towards a jittered point on a square area light
    if (stratified) r2_sample(t[1], t[0], 0.31f, t[2]);  // swapped dims: decorrelate from pixel jitter
    else { rand01(t[0], t[2]); rand01(t[1], t[2]); }
    t[0] = t[0] + -0.5f; t[0] = t[0] * U(U_LSIZE);
    t[1] = t[1] + -0.5f; t[1] = t[1] * U(U_LSIZE);
    RD.x = t[0] + U(U_LIGHT); RD.x = RD.x - SP.x;
    RD.y = -SP.y + U(U_LIGHT + 1);
    RD.z = t[1] + U(U_LIGHT + 2); RD.z = RD.z - SP.z;
    dot3(t[0], RD, RD, t[1]);
    t[1] = rsq(t[0]);
    for (int a = 0; a < 3; a++) RD[a] = RD[a] * t[1];
    HT = t[0] * t[1];                                // distance to the light sample
    for (int a = 0; a < 3; a++) { t[2] = SG[a] * 2e-3f; RO[a] = SP[a] + t[2]; }
    // no shadow ray for lanes facing away from the light or that hit nothing
    dot3(t[0], SG, RD, t[1]);
    f0 = t[0] <= 0.0f; HT[f0] = -1.0f;
    f0 = SM == 0.0f; HT[f0] = -1.0f;
    traverse(true);

    // local shading: vis → HN.x, n·l → HN.y
    const Reg &vis = HN.x, &ndl = HN.y;
    f0 = HT > 0.0f;
    vis = 0.0f; vis[f0] = 1.0f;
    dot3(ndl, SN, RD, t[0]);
    ndl = max(ndl, 0.0f);
    // half vector
    const Vec3 h{t[0], t[1], t[2]};
    for (int a = 0; a < 3; a++) h[a] = RD[a] - SD[a];
    normalize3(h, h, t[3]);
    dot3(t[3], SN, h, t[4]);
    t[3] = max(t[3], 0.0f);
    // material parameters (flag 0 = teapot)
    f0 = SM == 1.0f;
    t[4] = 60.0f; t[4][f0] = 180.0f;
    t[3] = pow(t[3], t[4]);
    t[5] = 0.25f * 68 / (8 * (float)M_PI); t[5][f0] = 0.9f * 188 / (8 * (float)M_PI);
    t[3] = t[3] * t[5];                              // specular
    t[6] = 0.09f; t[6][f0] = 0.05f;                  // F0
    // albedo: checkerboard floor, red teapot → t8 t9 t10
    t[7] = floor(SP.x); t[8] = floor(SP.z); t[7] += t[8];
    t[7] = t[7] * 0.5f; t[7] = frac(t[7]); t[7] = t[7] * 2.0f;
    const v3 dark(0.16f, 0.17f, 0.20f), light(0.78f, 0.77f, 0.74f), red(0.62f, 0.07f, 0.05f);
    const Vec3 alb{t[8], t[9], t[10]};
    for (int a = 0; a < 3; a++) { alb[a] = t[7] * (light[a] - dark[a]); alb[a] = alb[a] + dark[a]; }
    for (int a = 0; a < 3; a++) alb[a][f0] = red[a];
    // direct = (albedo * n·l + spec) * vis * light colour → t0 t1 t2
    for (int a = 0; a < 3; a++) {
        t[a] = alb[a] * ndl; t[a] += t[3];
        t[a] = t[a] * vis; t[a] = t[a] * look::light_col[a];
    }
    // ambient sky light: albedo * sky(n) * (0.231 + 0.154 n.y)
    const Vec3 &skn = HG;
    sky3(skn, SN.y, t[4]);
    t[5] = SN.y * 0.154f; t[5] = t[5] + 0.231f;
    for (int a = 0; a < 3; a++) {
        t[7] = skn[a] * alb[a]; t[7] = t[7] * t[5];
        t[a] += t[7];
    }
    // Fresnel (Schlick) → t6
    dot3(t[4], SN, SD, t[5]);
    t[4] = min(t[4], 0.0f);
    t[4] = t[4] + 1.0f;                              // 1 - cos
    t[5] = t[4] * t[4]; t[5] = t[5] * t[5]; t[5] = t[5] * t[4];
    t[7] = -t[6] + 1.0f; t[7] = t[7] * t[5]; t[6] += t[7];
    // distance fog towards the horizon colour
    t[7] = SP.x - U(U_CAM); t[7] = t[7] * t[7];
    t[8] = SP.y - U(U_CAM + 1); t[8] = t[8] * t[8]; t[7] += t[8];
    t[8] = SP.z - U(U_CAM + 2); t[8] = t[8] * t[8]; t[7] += t[8];
    t[7] = sqrt(t[7]);
    t[7] = t[7] * (-0.02f * 1.442695f);
    f0 = SM == 1.0f;
    t[7][f0] = 0.0f;                                 // fog only on the floor
    t[7] = exp2(t[7]);                               // e = exp(-0.02 d)
    const v3 fog(0.80f, 0.84f, 0.91f);
    // acc += thr * (1 - F) * lerp(fog, local, e);  thr *= F
    t[9] = -t[6] + 1.0f;
    for (int a = 0; a < 3; a++) {
        t[a] = t[a] + -fog[a]; t[a] = t[a] * t[7]; t[a] = t[a] + fog[a];
        t[a] = t[a] * t[9]; t[a] = t[a] * THR[a];
        ACC[a] += t[a];
        THR[a] = THR[a] * t[6];
    }

    // reflected ray
    dot3(t[0], SD, SN, t[1]);
    t[0] = t[0] * -2.0f;
    for (int a = 0; a < 3; a++) {
        t[1] = SN[a] * t[0]; RD[a] = SD[a] + t[1];
        t[2] = SG[a] * 2e-3f; RO[a] = SP[a] + t[2];
    }
    if (last) {
        const Vec3 sr{t[1], t[2], t[3]};
        sky3(sr, RD.y, t[4]);
        for (int a = 0; a < 3; a++) { t[5] = THR[a] * sr[a]; ACC[a] += t[5]; }
    }
}

void Kernel::build() {
    // tile origin as float scalars
    gx = group_x;
    gy = group_y;
    gx = gx * 4.0f;
    gy = gy * 4.0f;

    // per-pixel seed = pixel index (exact in float below 2^24), RNG state = hash of it
    t[0] = lane_x + gx;
    t[1] = lane_y + gy;
    t[1] = t[1] * 4096.0f; t[0] += t[1];
    t[0].u() = t[0];
    hash01(RNG.f(), t[0], 0x9e3779b9u, t[1]);
    RNG = RNG ^ U(U_SEED).u();                       // fresh random stream per pass
    RNG = RNG | u32(1);                              // xorshift state must be non-zero

    for (int a = 0; a < 3; a++) ACC[a] = 0.0f;
    sample = u32(0);
    r2 = U(U_SAMPLE_BASE);                           // R2 sample index continues across passes

    Label sample_loop = here();
    // camera ray through a jittered point of the pixel
    r2_sample(t[2], t[3], 0.0f, t[4]);
    t[0] = lane_x + gx; t[0] += t[2];
    t[1] = lane_y + gy; t[1] += t[3];
    t[0] = t[0] * U(U_INV_W); t[0] = t[0] + -1.0f; t[0] = t[0] * U(U_TAN_X);
    t[1] = t[1] * U(U_INV_H); t[1] = -t[1] + 1.0f; t[1] = t[1] * U(U_TAN_Y);
    for (int a = 0; a < 3; a++) {
        RD[a] = t[0] * U(U_RIGHT + a);
        t[2] = t[1] * U(U_UP + a); RD[a] += t[2];
        RD[a] += U(U_FWD + a);
        RO[a] = U(U_CAM + a);
        THR[a] = 1.0f;
    }
    normalize3(RD, RD, t[0]);

    if (dbg) {
        // debug view: red = material, green = nodes visited / 4, blue = triangles tested
        dbg_nodes = u32(0); dbg_tris = u32(0);
        HT = 1e6f;
        traverse(false);
        t[0] = HM * 100.0f; t[4].u() = t[0];
        dbg_nodes = dbg_nodes >> u32(2);
        t[5].u() = dbg_nodes; t[6].u() = dbg_tris;
    } else {
        bounce(false, true);
        bounce(true, false);

        sample += u32(1);
        r2 += 1.0f;
        f1 = sample < U(U_SPP).u();
        jmp_if(f1, sample_loop);

        // average, ACES tone curve, gamma, pack to XRGB8888
        for (int a = 0; a < 3; a++) {
            const Reg &c = ACC[a];
            c = c * U(U_INV_SPP);
            c = c * 0.9f;
            t[0] = c * 2.51f; t[0] = t[0] + 0.03f; t[0] = t[0] * c;
            t[1] = c * 2.43f; t[1] = t[1] + 0.59f; t[1] = t[1] * c; t[1] = t[1] + 0.14f;
            t[1] = inv(t[1]);
            t[0] = sat(t[0] * t[1]);
            t[2] = 1 / 2.2f;
            t[0] = pow(t[0], t[2]);
            t[0] = t[0] * 255.0f; t[0] = t[0] + 0.5f;
            t[a + 4].u() = t[0];
        }
    }
    Reg px = t[10].u();
    px = t[4].u() << u32(16);
    t[5].u() = t[5].u() << u32(8);
    px = px | t[5].u();
    px = px | t[6].u();
    store(BUF_RGB, px);
    store(BUF_DEPTH, DEP);
    end();
}

}  // namespace

void build(kir::Program &p, bool debug) {
    kir::Building b(p);
    Kernel(debug).build();
}

}  // namespace rt

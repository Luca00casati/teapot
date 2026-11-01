// rt_kernel.c — the teapot ray tracer as a kernel IR program (kir.c). This is
// the only GPU ray tracing code; it knows nothing about any particular GPU.
//
// Each packet shades a 4x4 tile. The 16 rays traverse the BVH together: the
// BVH is stored depth-first with "skip" links (write_scene_blob), so the walk
// needs no stack and every branch is uniform ("descend if any lane hits the
// box"), with per-lane work done by predication.
//
// Buffers: 0 = scene (write_scene_blob), 1 = XRGB8888 image, 2 = primary hit
// distance (negative on the glossy teapot).

// uniforms, written by the host per dispatch (gpu.c)
enum {
    U_CAM = 0,        // 0-2 camera  3 tan*aspect  4-6 forward  7 tan
    U_TAN_X = 3, U_FWD = 4, U_TAN_Y = 7,
    U_RIGHT = 8,      // 8-10 right  11 2/W  12-14 up  15 2/H
    U_INV_W = 11, U_UP = 12, U_INV_H = 15,
    U_LIGHT = 16,     // 16-18 light centre  19 light size
    U_LSIZE = 19,
    U_TRI_BASE = 20,  // triangle records, in 16-byte units
    U_SPP = 21, U_SEED = 22, U_NNODES = 23,
    U_INV_SPP = 24,
    U_SAMPLE_BASE = 26,  // first sample index of this pass (float)
    U_STRIDE = 28,    // 28-30 node layout stride for x, y, z octant bits (16-byte units)
    U_COUNT = 32,
};

// scalars
enum {
    S_NODE, S_TRI, S_TEND, S_SAMPLE, S_GX, S_GY, S_R2, S_TMP,
    S_SKIP, S_LBASE, S_CACHED, S_NADDR, S_TADDR, S_DBGN, S_DBGT,
};
enum { BK_NODE, BK_TRI, BK_SHADE };  // blocks: node pair, triangle hit half, shading half

static int rt_dbg;

typedef struct { int x, y, z; } v3r;
static v3r vreg3(void) { v3r v = {vreg(), vreg(), vreg()}; return v; }

// ray + hit state used by the traversal
static v3r RO, RD, RI, RQ;      // origin, dir, 1/dir, origin/dir
static int HT, HM;              // t, material
static v3r HN, HG;              // interpolated / geometric normal
static int T[11];               // temporaries
// shading state that survives a traversal
static v3r SP, SN, SG, SD;
static int SM;
static v3r ACC, THR;
static int RNG, DEP;            // rng state, primary hit distance of the last sample

static void alloc_regs(void) {
    RO = vreg3(); RD = vreg3(); RI = vreg3(); RQ = vreg3();
    HT = vreg(); HN = vreg3(); HG = vreg3(); HM = vreg();
    for (int i = 0; i < 11; i++) T[i] = vreg();
    SP = vreg3(); SN = vreg3(); SG = vreg3(); SD = vreg3(); SM = vreg();
    ACC = vreg3(); THR = vreg3(); RNG = vreg(); DEP = vreg();
}

static kop X(v3r v) { return KV(v.x); }
static kop Y(v3r v) { return KV(v.y); }
static kop Z(v3r v) { return KV(v.z); }
static kop comp(v3r v, int a) { return KV(a == 0 ? v.x : a == 1 ? v.y : v.z); }
static int compi(v3r v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }
static kop t(int i) { return KV(T[i]); }

static void dot3(int d, v3r a, v3r b, int tmp) {
    emul(KV(d), X(a), X(b));
    emul(KV(tmp), Y(a), Y(b)); eadd(KV(d), KV(d), KV(tmp));
    emul(KV(tmp), Z(a), Z(b)); eadd(KV(d), KV(d), KV(tmp));
}
static void normalize3(v3r d, v3r a, int tmp) {
    dot3(tmp, a, a, d.x == a.x ? T[10] : d.x);
    ersq(KV(tmp), KV(tmp));
    emul(X(d), X(a), KV(tmp)); emul(Y(d), Y(a), KV(tmp)); emul(Z(d), Z(a), KV(tmp));
}

// stateless hash of (src ^ key) → [0,1) in dst; 16-bit multipliers because a
// 32x16 multiply is a single instruction on most GPUs
static void hash01(int dst, int src, uint32_t key, int tmp) {
    kop x = KVU(dst), tt = KVU(tmp);
    xor_(x, KVU(src), imu(key));
    for (int r = 0; r < 2; r++) {
        shr(tt, x, imu(16)); xor_(x, x, tt);
        emul(x, x, imu(0x7feb));
        shr(tt, x, imu(15)); xor_(x, x, tt);
        emul(x, x, imu(0x846b));
        shr(tt, x, imu(16)); xor_(x, x, tt);
    }
    shr(x, x, imu(9)); or_(x, x, imu(0x3f800000));
    eadd(KV(dst), KV(dst), imf(-1.0f));
}

// per-pixel offset in [0,1): interleaved gradient noise of the pixel position,
// with the axes weighted (wx, wy) and shifted by k; cheap and well spread
static void ign(int dst, float wx, float wy, float k, int tmp) {
    eadd(KV(dst), LANE_X, KS(S_GX)); eadd(KV(tmp), LANE_Y, KS(S_GY));
    emul(KV(dst), KV(dst), imf(wx)); emul(KV(tmp), KV(tmp), imf(wy));
    eadd(KV(dst), KV(dst), KV(tmp)); eadd(KV(dst), KV(dst), imf(k));
    frc(KV(dst), KV(dst)); emul(KV(dst), KV(dst), imf(52.9829189f)); frc(KV(dst), KV(dst));
}

// sample s of a 2D R2 low-discrepancy sequence, rotated per pixel; s keeps
// counting across passes and animation frames, so it also stratifies over time
static const float R2A = 0.7548776662f, R2B = 0.5698402910f;
static void r2_sample(int du, int dv, float key, int tmp) {
    ign(du, 0.06711056f, 0.00583715f, key, tmp); ign(dv, 0.00583715f, 0.06711056f, key + 0.5f, tmp);
    emul(KS(S_TMP), KS(S_R2), imf(R2A)); eadd(KV(du), KV(du), KS(S_TMP)); frc(KV(du), KV(du));
    emul(KS(S_TMP), KS(S_R2), imf(R2B)); eadd(KV(dv), KV(dv), KS(S_TMP)); frc(KV(dv), KV(dv));
}

// uniform random float in [0,1) per lane (xorshift32)
static void rand01(int dst, int tmp) {
    kop r = KVU(RNG), tt = KVU(tmp);
    shl(tt, r, imu(13)); xor_(r, r, tt);
    shr(tt, r, imu(17)); xor_(r, r, tt);
    shl(tt, r, imu(5)); xor_(r, r, tt);
    shr(tt, r, imu(9)); or_(tt, tt, imu(0x3f800000));
    eadd(KV(dst), KV(tmp), imf(-1.0f));
}

// clamp a reciprocal so (b - o) * inv never makes inf - inf
static void clamp_big(kop v) { vmin_(v, v, imf(1e20f)); vmax_(v, v, imf(-1e20f)); }

/* ------------------------------------------------- packet traversal ----- */
// In: RO, RD, HT (= tmax). Out: HT, HN (unnormalised), HG, HM (0 miss, 1 teapot, 2 floor).
// Shadow mode is any-hit: an occluded lane gets HT = -1, which also fails every
// later box/triangle test for it, and the walk ends once all 16 lanes are
// occluded. Lanes may enter with HT = -1 to skip the ray entirely. The floor
// is ignored (shadow rays go up towards the light).

static void traverse(int shadow) {
    einv(X(RI), X(RD)); einv(Y(RI), Y(RD)); einv(Z(RI), Z(RD));
    clamp_big(X(RI)); clamp_big(Y(RI)); clamp_big(Z(RI));
    emul(X(RQ), X(RO), X(RI)); emul(Y(RQ), Y(RO), Y(RI)); emul(Z(RQ), Z(RO), Z(RI));
    emov(KV(HM), imf(0));
    emov(X(HN), imf(0)); emov(Y(HN), imf(1)); emov(Z(HN), imf(0));
    emov(X(HG), imf(0)); emov(Y(HG), imf(1)); emov(Z(HG), imf(0));
    emov(KSU(S_NODE), imu(0));

    // pick the node layout for the octant of lane 0's direction
    emov(KSU(S_LBASE), imu(0));
    for (int a = 0; a < 3; a++) {
        ecmp(KC_LT, 1, KL0(compi(RD, a)), imf(0));
        P(1); eadd(KSU(S_LBASE), KSU(S_LBASE), KUU(U_STRIDE + a));
    }

    // Nodes are 32 bytes, so one 64-byte load returns node i in dwords 0-7
    // of the node block and node i+1 in 8-15. Descending to the first child (i+1) then
    // needs no load.
    emov(KSU(S_CACHED), imu(0xfffffff0u));               // nothing cached yet
    int top = kn;
    if (rt_dbg) eadd(KSU(S_DBGN), KSU(S_DBGN), imu(1));
    ecmp(KC_GE, 1, KSU(S_NODE), KUU(U_NNODES));
    int j_end = jump(K_JIF, 1);
    ecmp(KC_EQ, 1, KSU(S_NODE), KSU(S_CACHED));
    int j_slot0 = jump(K_JIF, 1);
    eadd(KSU(S_TMP), KSU(S_CACHED), imu(1));
    ecmp(KC_EQ, 1, KSU(S_NODE), KSU(S_TMP));
    int j_slot1 = jump(K_JIF, 1);
    shl(KSU(S_NADDR), KSU(S_NODE), imu(1));
    kload(BK_NODE, KSU(S_NADDR), KSU(S_LBASE), 0);
    emov(KSU(S_CACHED), KSU(S_NODE));

    // node record: bmin .0-2, skip .3, bmax .4-6, first_tri << 3 | count .7
    int to_tri[2], slot_at[2] = {j_slot0, j_slot1};
    for (int slot = 0; slot < 2; slot++) {
        int o = slot * 8;
        patch(slot_at[slot], kn);
        // slab test, scheduled so neighbouring instructions are independent:
        // in-order GPU cores stall on a dependent op
        for (int a = 0; a < 3; a++) {                    // t(2a) = near plane, t(2a+1) = far plane
            emul(t(2 * a), comp(RI, a), KB(BK_NODE, o + a));
            emul(t(2 * a + 1), comp(RI, a), KB(BK_NODE, o + 4 + a));
        }
        for (int a = 0; a < 3; a++) {
            eadd(t(2 * a), t(2 * a), neg(comp(RQ, a)));
            eadd(t(2 * a + 1), t(2 * a + 1), neg(comp(RQ, a)));
        }
        vmin_(t(6), t(0), t(1)); vmin_(t(7), t(2), t(3)); vmin_(t(8), t(4), t(5));
        vmax_(t(9), t(0), t(1)); vmax_(t(10), t(2), t(3)); vmax_(t(0), t(4), t(5));
        vmax_(t(6), t(6), t(7)); vmin_(t(9), t(9), t(10));
        vmax_(t(8), t(8), imf(0)); vmin_(t(0), t(0), KV(HT));
        vmax_(t(6), t(6), t(8)); vmin_(t(9), t(9), t(0));
        ecmp(KC_LE, 0, t(6), t(9));
        int j_miss = jump(K_JNONE, 0);                   // no lane hit the box
        and_(KSU(S_TEND), KBU(BK_NODE, o + 7), imu(7));
        ecmp(KC_EQ, 1, KSU(S_TEND), imu(0));
        int j_inner = jump(K_JIF, 1);
        shr(KSU(S_TRI), KBU(BK_NODE, o + 7), imu(3));
        eadd(KSU(S_TEND), KSU(S_TEND), KSU(S_TRI));
        emov(KSU(S_SKIP), KBU(BK_NODE, o + 3));
        to_tri[slot] = jump(K_JMP, 0);
        patch(j_inner, kn);                              // interior: first child is next
        eadd(KSU(S_NODE), KSU(S_NODE), imu(1));
        jmp_to(top, K_JMP, 0);
        patch(j_miss, kn);                               // missed: follow the skip link
        emov(KSU(S_NODE), KBU(BK_NODE, o + 3));
        jmp_to(top, K_JMP, 0);
    }

    // leaf: test each triangle. Its first 64 bytes decide the hit; the
    // shading half is only loaded when a lane hits.
    patch(to_tri[0], kn); patch(to_tri[1], kn);
    int tri_loop = kn;
    if (rt_dbg) eadd(KSU(S_DBGT), KSU(S_DBGT), imu(1));
    shl(KSU(S_TADDR), KSU(S_TRI), imu(3));
    kload(BK_TRI, KSU(S_TADDR), KUU(U_TRI_BASE), 0);
    {
        // triangle record (Havel-Herout planes):
        //   TRI 0-3 N, d   (plane: N·x = d, N = e1 × e2)
        //   TRI 4-7 N1, d1 (u = N1·x + d1)       TRI 8-11 N2, d2 (v = N2·x + d2)
        //   TRI 12-14 n0   SHADE 0-2 dn1  SHADE 4-6 dn2  SHADE 8-10 geometric normal
        // Everything is kept multiplied by det = N·D so no division is needed
        // to decide a hit; t, u, v are divided out only when some lane hits.
        kop Nx = KB(BK_TRI, 0), Ny = KB(BK_TRI, 1), Nz = KB(BK_TRI, 2), Nd = KB(BK_TRI, 3);
        // (scheduled in independent groups, see the slab test)
        // det = N·D → t0;  t' = d - N·O → t1
        emul(t(0), X(RD), Nx); emul(t(1), X(RO), Nx);
        emul(t(7), Y(RD), Ny); emul(t(9), Y(RO), Ny);
        emul(t(8), Z(RD), Nz); emul(t(10), Z(RO), Nz);
        eadd(t(0), t(0), t(7)); eadd(t(1), t(1), t(9));
        eadd(t(0), t(0), t(8)); eadd(t(1), t(1), t(10));
        eadd(t(1), neg(t(1)), Nd);
        // P' = O det + D t' (= hit point * det) → t2 t3 t4
        emul(t(2), X(RO), t(0)); emul(t(3), Y(RO), t(0)); emul(t(4), Z(RO), t(0));
        emul(t(7), X(RD), t(1)); emul(t(8), Y(RD), t(1)); emul(t(9), Z(RD), t(1));
        eadd(t(2), t(2), t(7)); eadd(t(3), t(3), t(8)); eadd(t(4), t(4), t(9));
        // u' = N1·P' + d1 det → t5;  v' = N2·P' + d2 det → t6
        emul(t(5), t(2), KB(BK_TRI, 4)); emul(t(6), t(2), KB(BK_TRI, 8));
        emul(t(7), t(3), KB(BK_TRI, 5)); emul(t(8), t(3), KB(BK_TRI, 9));
        emul(t(9), t(4), KB(BK_TRI, 6)); emul(t(10), t(4), KB(BK_TRI, 10));
        eadd(t(5), t(5), t(7)); eadd(t(6), t(6), t(8));
        emul(t(7), t(0), KB(BK_TRI, 7)); emul(t(8), t(0), KB(BK_TRI, 11));
        eadd(t(5), t(5), t(9)); eadd(t(6), t(6), t(10));
        eadd(t(5), t(5), t(7)); eadd(t(6), t(6), t(8));
        // hit iff u, v, 1-u-v, t > 0 and t < T; each scaled by det² (sign-safe):
        //   u'det, v'det, (det-u'-v')det, t'det, (T det - t')det
        emul(t(7), t(5), t(0)); emul(t(8), t(6), t(0));
        eadd(t(9), t(0), neg(t(5))); emul(t(10), t(1), t(0)); emul(t(2), KV(HT), t(0));
        eadd(t(9), t(9), neg(t(6))); eadd(t(2), t(2), neg(t(1)));
        vmin_(t(7), t(7), t(8));
        emul(t(9), t(9), t(0)); emul(t(2), t(2), t(0));
        vmin_(t(7), t(7), t(10));
        vmin_(t(9), t(9), t(2));
        vmin_(t(8), t(7), t(9));
        ecmp(KC_GT, 0, t(8), imf(0));
        int j_nohit = jump(K_JNONE, 0);
        einv(t(0), t(0));
        P(0); emul(KV(HT), t(1), t(0));
        if (shadow) {
            P(0); emov(KV(HT), imf(-1));
        } else {
            emul(t(5), t(5), t(0)); emul(t(6), t(6), t(0));   // u, v
            kload(BK_SHADE, KSU(S_TADDR), KUU(U_TRI_BASE), 4);
            for (int a = 0; a < 3; a++) {
                kop hn = comp(HN, a), hg = comp(HG, a);
                emul(t(7), t(6), KB(BK_SHADE, 4 + a));          // v * dn2
                P(0); emul(hn, t(5), KB(BK_SHADE, a));          // u * dn1
                P(0); eadd(hn, hn, KB(BK_TRI, 12 + a));         // + n0
                P(0); eadd(hn, hn, t(7));
                P(0); emov(hg, KB(BK_SHADE, 8 + a));
            }
        }
        P(0); emov(KV(HM), imf(1));
        patch(j_nohit, kn);
    }
    eadd(KSU(S_TRI), KSU(S_TRI), imu(1));
    ecmp(KC_LT, 1, KSU(S_TRI), KSU(S_TEND));
    jmp_to(tri_loop, K_JIF, 1);
    int j_done = -1;
    if (shadow) {
        ecmp(KC_LT, 0, KV(HT), imf(0));
        j_done = jump(K_JALL, 0);                        // every lane occluded
    }
    emov(KSU(S_NODE), KSU(S_SKIP));                        // leaf done: follow its skip link
    jmp_to(top, K_JMP, 0);

    patch(j_end, kn);
    if (shadow) { patch(j_done, kn); return; }
    // floor plane y = 0: t = -oy/dy
    emov(t(0), neg(Y(RQ)));
    eadd(t(1), t(0), imf(-1e-4f));
    eadd(t(2), KV(HT), neg(t(0)));
    vmin_(t(1), t(1), t(2));
    ecmp(KC_GT, 0, t(1), imf(0));
    P(0); emov(KV(HT), t(0));
    P(0); emov(KV(HM), imf(2));
    P(0); emov(X(HN), imf(0)); P(0); emov(Y(HN), imf(1)); P(0); emov(Z(HN), imf(0));
    P(0); emov(X(HG), imf(0)); P(0); emov(Y(HG), imf(1)); P(0); emov(Z(HG), imf(0));
}

/* ------------------------------------------------- shading -------------- */

static const v3 sky_h = {0.82f, 0.86f, 0.93f}, sky_z = {0.30f, 0.45f, 0.78f};

// sky(dir) → d (3 temps), using s as scratch
static void sky3(v3r d, kop ydir, int s) {
    vmax_(KV(s), ydir, imf(0));
    esqrt(KV(s), KV(s));
    emul(X(d), KV(s), imf(sky_z.x - sky_h.x)); eadd(X(d), X(d), imf(sky_h.x));
    emul(Y(d), KV(s), imf(sky_z.y - sky_h.y)); eadd(Y(d), Y(d), imf(sky_h.y));
    emul(Z(d), KV(s), imf(sky_z.z - sky_h.z)); eadd(Z(d), Z(d), imf(sky_h.z));
}

static void bounce(int last, int stratified) {
    emov(KV(HT), imf(1e6f));
    traverse(0);
    if (stratified) {                                  // primary bounce: keep the depth,
        emov(KV(DEP), KV(HT));                           // negative on the (glossy) teapot
        ecmp(KC_EQ, 0, KV(HM), imf(1));
        P(0); emov(KV(DEP), neg(KV(HT)));
    }

    // miss lanes: add sky, kill throughput
    v3r sk = {T[1], T[2], T[3]};
    ecmp(KC_EQ, 0, KV(HM), imf(0));
    emov(t(0), imf(0)); P(0); emov(t(0), imf(1));
    sky3(sk, Y(RD), T[4]);
    for (int a = 0; a < 3; a++) {
        emul(t(5), comp(THR, a), t(0)); emul(t(5), t(5), comp(sk, a));
        eadd(comp(ACC, a), comp(ACC, a), t(5));
    }
    eadd(t(0), neg(t(0)), imf(1));
    for (int a = 0; a < 3; a++) emul(comp(THR, a), comp(THR, a), t(0));

    // hit point, oriented normals
    for (int a = 0; a < 3; a++) {
        emul(comp(SP, a), comp(RD, a), KV(HT)); eadd(comp(SP, a), comp(SP, a), comp(RO, a));
        emov(comp(SD, a), comp(RD, a));
        emov(comp(SG, a), comp(HG, a));
    }
    emov(KV(SM), KV(HM));
    normalize3(SN, HN, T[0]);
    dot3(T[0], SG, SD, T[1]);
    ecmp(KC_GT, 0, t(0), imf(0));
    for (int a = 0; a < 3; a++) { P(0); emov(comp(SG, a), neg(comp(SG, a))); }
    dot3(T[0], SN, SG, T[1]);
    ecmp(KC_LT, 0, t(0), imf(0));
    for (int a = 0; a < 3; a++) { P(0); emov(comp(SN, a), neg(comp(SN, a))); }

    // shadow ray towards a jittered point on a square area light
    if (stratified) r2_sample(T[1], T[0], 0.31f, T[2]); // swapped dims: decorrelate from pixel jitter
    else { rand01(T[0], T[2]); rand01(T[1], T[2]); }
    eadd(t(0), t(0), imf(-0.5f)); emul(t(0), t(0), KU(U_LSIZE));
    eadd(t(1), t(1), imf(-0.5f)); emul(t(1), t(1), KU(U_LSIZE));
    eadd(X(RD), t(0), KU(U_LIGHT)); eadd(X(RD), X(RD), neg(X(SP)));
    eadd(Y(RD), neg(Y(SP)), KU(U_LIGHT + 1));
    eadd(Z(RD), t(1), KU(U_LIGHT + 2)); eadd(Z(RD), Z(RD), neg(Z(SP)));
    dot3(T[0], RD, RD, T[1]);
    ersq(t(1), t(0));
    for (int a = 0; a < 3; a++) emul(comp(RD, a), comp(RD, a), t(1));
    emul(KV(HT), t(0), t(1));                           // distance to light sample
    for (int a = 0; a < 3; a++) { emul(t(2), comp(SG, a), imf(2e-3f)); eadd(comp(RO, a), comp(SP, a), t(2)); }
    // no shadow ray for lanes facing away from the light or that hit nothing
    dot3(T[0], SG, RD, T[1]);
    ecmp(KC_LE, 0, t(0), imf(0)); P(0); emov(KV(HT), imf(-1));
    ecmp(KC_EQ, 0, KV(SM), imf(0)); P(0); emov(KV(HT), imf(-1));
    traverse(1);

    // local shading: vis → HN.x, n·l → HN.y
    kop vis = X(HN), ndl = Y(HN);
    ecmp(KC_GT, 0, KV(HT), imf(0));
    emov(vis, imf(0)); P(0); emov(vis, imf(1));
    dot3(HN.y, SN, RD, T[0]);
    vmax_(ndl, ndl, imf(0));
    // half vector
    v3r h = {T[0], T[1], T[2]};
    for (int a = 0; a < 3; a++) eadd(comp(h, a), comp(RD, a), neg(comp(SD, a)));
    normalize3(h, h, T[3]);
    dot3(T[3], SN, h, T[4]);
    vmax_(t(3), t(3), imf(0));
    // material parameters (flag 0 = teapot)
    ecmp(KC_EQ, 0, KV(SM), imf(1));
    emov(t(4), imf(60)); P(0); emov(t(4), imf(180));
    epow(t(3), t(3), t(4));
    emov(t(5), imf(0.25f * 68 / (8 * (float)M_PI))); P(0); emov(t(5), imf(0.9f * 188 / (8 * (float)M_PI)));
    emul(t(3), t(3), t(5));                            // specular
    emov(t(6), imf(0.09f)); P(0); emov(t(6), imf(0.05f)); // F0
    // albedo: checkerboard floor, red teapot → t8 t9 t10
    efloor(t(7), X(SP)); efloor(t(8), Z(SP)); eadd(t(7), t(7), t(8));
    emul(t(7), t(7), imf(0.5f)); frc(t(7), t(7)); emul(t(7), t(7), imf(2));
    static const v3 dark = {0.16f, 0.17f, 0.20f}, light = {0.78f, 0.77f, 0.74f}, red = {0.62f, 0.07f, 0.05f};
    v3r alb = {T[8], T[9], T[10]};
    emul(t(8), t(7), imf(light.x - dark.x)); eadd(t(8), t(8), imf(dark.x));
    emul(t(9), t(7), imf(light.y - dark.y)); eadd(t(9), t(9), imf(dark.y));
    emul(t(10), t(7), imf(light.z - dark.z)); eadd(t(10), t(10), imf(dark.z));
    P(0); emov(t(8), imf(red.x)); P(0); emov(t(9), imf(red.y)); P(0); emov(t(10), imf(red.z));
    // direct = (albedo * n·l + spec) * vis * light colour → t0 t1 t2
    for (int a = 0; a < 3; a++) {
        emul(t(a), comp(alb, a), ndl); eadd(t(a), t(a), t(3));
        emul(t(a), t(a), vis); emul(t(a), t(a), imf(axis(light_col, a)));
    }
    // ambient sky light: albedo * sky(n) * (0.231 + 0.154 n.y)
    v3r skn = {HG.x, HG.y, HG.z};
    sky3(skn, Y(SN), T[4]);
    emul(t(5), Y(SN), imf(0.154f)); eadd(t(5), t(5), imf(0.231f));
    for (int a = 0; a < 3; a++) {
        emul(t(7), comp(skn, a), comp(alb, a)); emul(t(7), t(7), t(5));
        eadd(t(a), t(a), t(7));
    }
    // Fresnel (Schlick) → t6
    dot3(T[4], SN, SD, T[5]);
    vmin_(t(4), t(4), imf(0));
    eadd(t(4), t(4), imf(1));                          // 1 - cos
    emul(t(5), t(4), t(4)); emul(t(5), t(5), t(5)); emul(t(5), t(5), t(4));
    eadd(t(7), neg(t(6)), imf(1)); emul(t(7), t(7), t(5)); eadd(t(6), t(6), t(7));
    // distance fog towards the horizon colour
    eadd(t(7), X(SP), neg(KU(U_CAM))); emul(t(7), t(7), t(7));
    eadd(t(8), Y(SP), neg(KU(U_CAM + 1))); emul(t(8), t(8), t(8)); eadd(t(7), t(7), t(8));
    eadd(t(8), Z(SP), neg(KU(U_CAM + 2))); emul(t(8), t(8), t(8)); eadd(t(7), t(7), t(8));
    esqrt(t(7), t(7));
    emul(t(7), t(7), imf(-0.02f * 1.442695f));
    ecmp(KC_EQ, 0, KV(SM), imf(1));
    P(0); emov(t(7), imf(0));                         // fog only on the floor
    eexp2(t(7), t(7));                                // e = exp(-0.02 d)
    static const v3 fog = {0.80f, 0.84f, 0.91f};
    // acc += thr * (1 - F) * lerp(fog, local, e);  thr *= F
    eadd(t(9), neg(t(6)), imf(1));
    for (int a = 0; a < 3; a++) {
        eadd(t(a), t(a), imf(-axis(fog, a))); emul(t(a), t(a), t(7)); eadd(t(a), t(a), imf(axis(fog, a)));
        emul(t(a), t(a), t(9)); emul(t(a), t(a), comp(THR, a));
        eadd(comp(ACC, a), comp(ACC, a), t(a));
        emul(comp(THR, a), comp(THR, a), t(6));
    }

    // reflected ray
    dot3(T[0], SD, SN, T[1]);
    emul(t(0), t(0), imf(-2));
    for (int a = 0; a < 3; a++) {
        emul(t(1), comp(SN, a), t(0)); eadd(comp(RD, a), comp(SD, a), t(1));
        emul(t(2), comp(SG, a), imf(2e-3f)); eadd(comp(RO, a), comp(SP, a), t(2));
    }
    if (last) {
        v3r sr = {T[1], T[2], T[3]};
        sky3(sr, Y(RD), T[4]);
        for (int a = 0; a < 3; a++) { emul(t(5), comp(THR, a), comp(sr, a)); eadd(comp(ACC, a), comp(ACC, a), t(5)); }
    }
}

// the whole kernel: U_SPP samples per pixel of the packet's tile
static void rt_build(void) {
    kir_reset();
    alloc_regs();

    // tile origin as float scalars
    emov(KS(S_GX), GROUP_X);
    emov(KS(S_GY), GROUP_Y);
    emul(KS(S_GX), KS(S_GX), imf(4));
    emul(KS(S_GY), KS(S_GY), imf(4));

    // per-pixel seed = pixel index (exact in float below 2^24), RNG state = hash of it
    eadd(t(0), LANE_X, KS(S_GX));
    eadd(t(1), LANE_Y, KS(S_GY));
    emul(t(1), t(1), imf(4096)); eadd(t(0), t(0), t(1));
    emov(KVU(T[0]), t(0));
    hash01(RNG, T[0], 0x9e3779b9u, T[1]);
    xor_(KVU(RNG), KVU(RNG), KUU(U_SEED));                // fresh random stream per pass
    or_(KVU(RNG), KVU(RNG), imu(1));                     // xorshift state must be non-zero

    for (int a = 0; a < 3; a++) emov(comp(ACC, a), imf(0));
    emov(KSU(S_SAMPLE), imu(0));
    emov(KS(S_R2), KU(U_SAMPLE_BASE));                   // R2 sample index continues across passes

    int sample_loop = kn;
    // camera ray through a jittered point of the pixel
    r2_sample(T[2], T[3], 0.0f, T[4]);
    eadd(t(0), LANE_X, KS(S_GX)); eadd(t(0), t(0), t(2));
    eadd(t(1), LANE_Y, KS(S_GY)); eadd(t(1), t(1), t(3));
    emul(t(0), t(0), KU(U_INV_W)); eadd(t(0), t(0), imf(-1)); emul(t(0), t(0), KU(U_TAN_X));
    emul(t(1), t(1), KU(U_INV_H)); eadd(t(1), neg(t(1)), imf(1)); emul(t(1), t(1), KU(U_TAN_Y));
    for (int a = 0; a < 3; a++) {
        emul(comp(RD, a), t(0), KU(U_RIGHT + a));
        emul(t(2), t(1), KU(U_UP + a)); eadd(comp(RD, a), comp(RD, a), t(2));
        eadd(comp(RD, a), comp(RD, a), KU(U_FWD + a));
        emov(comp(RO, a), KU(U_CAM + a));
        emov(comp(THR, a), imf(1));
    }
    normalize3(RD, RD, T[0]);

    if (rt_dbg) {
        // debug view: red = material, green = nodes visited / 4, blue = triangles tested
        emov(KSU(S_DBGN), imu(0)); emov(KSU(S_DBGT), imu(0));
        emov(KV(HT), imf(1e6f));
        traverse(0);
        emul(t(0), KV(HM), imf(100)); emov(KVU(T[4]), t(0));
        shr(KSU(S_DBGN), KSU(S_DBGN), imu(2));
        emov(KVU(T[5]), KSU(S_DBGN)); emov(KVU(T[6]), KSU(S_DBGT));
        goto pack;
    }
    bounce(0, 1);
    bounce(1, 0);

    eadd(KSU(S_SAMPLE), KSU(S_SAMPLE), imu(1));
    eadd(KS(S_R2), KS(S_R2), imf(1));
    ecmp(KC_LT, 1, KSU(S_SAMPLE), KUU(U_SPP));
    jmp_to(sample_loop, K_JIF, 1);

    // average, ACES tone curve, gamma, pack to XRGB8888
    for (int a = 0; a < 3; a++) {
        kop c = comp(ACC, a);
        emul(c, c, KU(U_INV_SPP));
        emul(c, c, imf(0.9f));
        emul(t(0), c, imf(2.51f)); eadd(t(0), t(0), imf(0.03f)); emul(t(0), t(0), c);
        emul(t(1), c, imf(2.43f)); eadd(t(1), t(1), imf(0.59f)); emul(t(1), t(1), c); eadd(t(1), t(1), imf(0.14f));
        einv(t(1), t(1));
        SAT(); emul(t(0), t(0), t(1));
        emov(t(2), imf(1 / 2.2f));
        epow(t(0), t(0), t(2));
        emul(t(0), t(0), imf(255)); eadd(t(0), t(0), imf(0.5f));
        emov(KVU(T[a + 4]), t(0));
    }
pack:;
    kop px = KVU(T[10]);
    shl(px, KVU(T[4]), imu(16));
    shl(KVU(T[5]), KVU(T[5]), imu(8));
    or_(px, px, KVU(T[5]));
    or_(px, px, KVU(T[6]));
    kstore(1, px);
    kstore(2, KV(DEP));
    kend();
}

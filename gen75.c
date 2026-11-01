// gen75.c — the hardware layer for Intel Haswell (Gen7.5): encodes the kernel
// IR (kir.c) into native 128-bit EU instructions and builds the GPGPU command
// stream (PIPELINE_SELECT → STATE_BASE_ADDRESS → MEDIA_VFE_STATE → CURBE →
// interface descriptor → GPGPU_WALKER). Buffers and submission go through a
// gpu_drv (i915.c).
//
// A packet is one SIMD16 hardware thread. Register map:
//   r0 thread payload   r1-r4 uniforms (CURBE)   r5-r6 lane x, r7-r8 lane y (CURBE)
//   r9 scalars 0-7, r17 scalars 8-15   r10, r125 load headers   r11-r16 blocks 0-2
//   r18-r121 vector registers (2 GRFs each)   r122-r124 tile store   r127 end of thread
//
// Hardware findings this relies on (measured on an HD 4600):
//   - jmpi distances are in bytes, relative to the next instruction
//   - a send writes back no more GRFs than its execution size covers, so
//     block loads are SIMD16
//   - an inverted any16h predicate applies after the reduction ("no lane")

enum { RF_ARF = 0, RF_GRF = 1, RF_IMM = 3 };
enum { TY_UD = 0, TY_D = 1, TY_UW = 2, TY_W = 3, TY_F = 7 };
enum {
    OP_MOV = 1, OP_SEL = 2, OP_AND = 5, OP_OR = 6, OP_XOR = 7, OP_SHR = 8, OP_SHL = 9,
    OP_CMP = 16, OP_JMPI = 32, OP_SEND = 49, OP_MATH = 56, OP_ADD = 64, OP_MUL = 65,
    OP_FRC = 67, OP_RNDD = 69,
};
enum { CM_Z = 1, CM_NZ = 2, CM_G = 3, CM_GE = 4, CM_L = 5, CM_LE = 6 };
enum { MF_INV = 1, MF_EXP = 3, MF_SQRT = 4, MF_RSQ = 5, MF_POW = 10 };
enum { PR_NORMAL = 1, PR_ANY16H = 10, PR_ALL16H = 11 };
enum { SFID_RENDER_CACHE = 5, SFID_THREAD_SPAWNER = 7, SFID_CONST_CACHE = 9 };
enum { G_UNI = 1, G_LX = 5, G_LY = 7, G_SC0 = 9, G_HDR = 10, G_BLK = 11, G_SC1 = 17, G_VEC = 18,
       G_VEC_END = 122, G_ST = 122, G_HDR2 = 125, G_EOT = 127 };

typedef struct { uint8_t file, type, nr, sub, vs, w, hs, neg; uint32_t imm; } gop;

static uint32_t gcode[8192][4];
static int gn;

// modifiers for the next instruction only
static struct { int pred, pinv, flag, cmod, sat, es, nomask; } GM;

static gop gvec(int r, int type) { return (gop){RF_GRF, type, r, 0, 8, 8, 1, 0, 0}; }        // SIMD16, r:r+1
static gop gsc(int r, int c, int type) { return (gop){RF_GRF, type, r, c * 4, 0, 1, 0, 0, 0}; }  // <0;1,0>
static gop gimm(uint32_t v, int type) { return (gop){RF_IMM, type, 0, 0, 0, 0, 0, 0, v}; }
static gop gnull(int type) { return (gop){RF_ARF, type, 0, 0, 0, 1, 0, 0, 0}; }

static void setb(uint32_t *in, int hi, int lo, uint32_t v) {
    for (int b = lo; b <= hi; b++, v >>= 1)
        in[b / 32] = (in[b / 32] & ~(1u << (b % 32))) | ((v & 1) << (b % 32));
}
static int enc_vs(int v) { return v == 0 ? 0 : v == 1 ? 1 : v == 2 ? 2 : v == 4 ? 3 : v == 8 ? 4 : v == 16 ? 5 : 6; }
static int enc_w(int w) { return w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : w == 8 ? 3 : 4; }
static int enc_hs(int h) { return h == 0 ? 0 : h == 1 ? 1 : h == 2 ? 2 : 3; }
static int enc_es(int e) { return e == 1 ? 0 : e == 2 ? 1 : e == 4 ? 2 : e == 8 ? 3 : 4; }

static int gemit(int opc, gop d, gop s0, gop s1, int nsrc, int fn) {
    if (gn >= (int)(sizeof gcode / sizeof gcode[0])) { fprintf(stderr, "gen75: kernel too big\n"); exit(1); }
    uint32_t *in = gcode[gn];
    memset(in, 0, 16);
    int es = GM.es ? GM.es : 16;
    setb(in, 6, 0, opc);
    setb(in, 9, 9, GM.nomask);
    setb(in, 19, 16, GM.pred);
    setb(in, 20, 20, GM.pinv);
    setb(in, 23, 21, enc_es(es));
    setb(in, 27, 24, fn ? fn : GM.cmod);
    setb(in, 31, 31, GM.sat);
    setb(in, 90, 90, GM.flag >> 1);   // f0 / f1
    setb(in, 89, 89, GM.flag & 1);    // .0 / .1
    // destination (align1, direct)
    setb(in, 33, 32, d.file);
    setb(in, 36, 34, d.type);
    setb(in, 52, 48, d.sub);
    setb(in, 60, 53, d.nr);
    setb(in, 62, 61, enc_hs(d.hs ? d.hs : 1));
    // src0
    setb(in, 38, 37, s0.file);
    setb(in, 41, 39, s0.type);
    if (s0.file == RF_IMM) {
        in[3] = s0.imm;
        setb(in, 43, 42, RF_ARF);
        setb(in, 46, 44, s0.type);
    } else {
        setb(in, 68, 64, s0.sub);
        setb(in, 76, 69, s0.nr);
        setb(in, 78, 78, s0.neg);
        setb(in, 81, 80, enc_hs(s0.hs));
        setb(in, 84, 82, enc_w(s0.w));
        setb(in, 88, 85, enc_vs(s0.vs));
    }
    if (nsrc == 2) {
        setb(in, 43, 42, s1.file);
        setb(in, 46, 44, s1.type);
        if (s1.file == RF_IMM) {
            in[3] = s1.imm;
        } else {
            setb(in, 100, 96, s1.sub);
            setb(in, 108, 101, s1.nr);
            setb(in, 110, 110, s1.neg);
            setb(in, 113, 112, enc_hs(s1.hs));
            setb(in, 116, 114, enc_w(s1.w));
            setb(in, 120, 117, enc_vs(s1.vs));
        }
    }
    memset(&GM, 0, sizeof GM);
    return gn++;
}

static void gscalar(void) { GM.es = 1; GM.nomask = 1; }
static void gsend(int sfid, gop dst, int src, uint32_t desc, int es) {
    GM.es = es; GM.nomask = 1;
    gemit(OP_SEND, dst, gvec(src, TY_UD), gimm(desc, TY_UD), 2, sfid);
}

/* ------------------------------------------------- IR → EU code --------- */

static int is_vec(kop o) { return o.kind == KO_V || o.kind == KO_LX || o.kind == KO_LY; }

static gop gen_op(kop o) {
    int ty = o.type == KT_F ? TY_F : TY_UD;
    gop g;
    switch (o.kind) {
    case KO_V: g = gvec(G_VEC + 2 * o.idx, ty); break;
    case KO_S: g = gsc(o.idx < 8 ? G_SC0 : G_SC1, o.idx % 8, ty); break;
    case KO_U: g = gsc(G_UNI + o.idx / 8, o.idx % 8, ty); break;
    case KO_B: g = gsc(G_BLK + o.idx / 8, o.idx % 8, ty); break;
    case KO_L0: g = gsc(G_VEC + 2 * o.idx, 0, ty); break;
    case KO_LX: g = gvec(G_LX, ty); break;
    case KO_LY: g = gvec(G_LY, ty); break;
    case KO_GX: g = gsc(0, 1, ty); break;
    case KO_GY: g = gsc(0, 6, ty); break;
    case KO_IMM: return gimm(o.imm, ty);
    default: return gnull(ty);
    }
    g.neg = o.neg;
    return g;
}

static int gen75_compile(void) {
    static const uint8_t alu[] = {
        [K_MOV] = OP_MOV, [K_ADD] = OP_ADD, [K_MUL] = OP_MUL, [K_MIN] = OP_SEL, [K_MAX] = OP_SEL,
        [K_AND] = OP_AND, [K_OR] = OP_OR, [K_XOR] = OP_XOR, [K_SHL] = OP_SHL, [K_SHR] = OP_SHR,
        [K_FLOOR] = OP_RNDD, [K_FRAC] = OP_FRC,
    };
    static const uint8_t mathfn[] = {[K_INV] = MF_INV, [K_SQRT] = MF_SQRT, [K_RSQ] = MF_RSQ,
                                     [K_EXP2] = MF_EXP, [K_POW] = MF_POW};
    static const uint8_t cmod[] = {[KC_EQ] = CM_Z, [KC_NE] = CM_NZ, [KC_GT] = CM_G,
                                   [KC_GE] = CM_GE, [KC_LT] = CM_L, [KC_LE] = CM_LE};
    if (G_VEC + 2 * kir_nv > G_VEC_END) {
        fprintf(stderr, "gen75: kernel needs %d vector registers, only %d fit\n", kir_nv, (G_VEC_END - G_VEC) / 2);
        return -1;
    }
    static int at[KIR_MAX + 1];           // first EU instruction of each IR instruction
    gn = 0;
    memset(&GM, 0, sizeof GM);
    int hdr_toggle = 0;
    // load headers start as a copy of the thread payload; only .2 changes
    GM.es = 8; GM.nomask = 1; gemit(OP_MOV, gvec(G_HDR, TY_UD), gvec(0, TY_UD), gnull(0), 1, 0);
    GM.es = 8; GM.nomask = 1; gemit(OP_MOV, gvec(G_HDR2, TY_UD), gvec(0, TY_UD), gnull(0), 1, 0);

    for (int i = 0; i < kn; i++) {
        const kins *k = &kir[i];
        at[i] = gn;
        int scalar = k->op == K_CMP ? !is_vec(k->a) && !is_vec(k->b) : k->d.kind == KO_S;
        gop d = gen_op(k->d), a = gen_op(k->a), b = gen_op(k->b);
        if (scalar && (is_vec(k->a) || is_vec(k->b))) {
            fprintf(stderr, "gen75: IR %d: vector source in a scalar op\n", i);
            return -1;
        }
#define MODS() do {                                                   \
            if (scalar) gscalar();                                    \
            if (k->pred) { GM.pred = PR_NORMAL; GM.flag = (k->pred - 1) * 2; } \
            GM.sat = k->sat;                                          \
        } while (0)
        switch (k->op) {
        case K_MOV: case K_FLOOR: case K_FRAC:
            MODS(); gemit(alu[k->op], d, a, gnull(0), 1, 0);
            break;
        case K_MUL:
            // 32x16 integer multiply: a uint immediate of 16 bits is encoded as UW
            if (b.file == RF_IMM && b.type == TY_UD) {
                if (b.imm > 0xffff) { fprintf(stderr, "gen75: IR %d: 32-bit integer multiply\n", i); return -1; }
                b.type = TY_UW; b.imm |= b.imm << 16;
            }
            MODS(); gemit(OP_MUL, d, a, b, 2, 0);
            break;
        case K_MIN: case K_MAX:
            MODS(); GM.cmod = k->op == K_MIN ? CM_L : CM_GE; gemit(OP_SEL, d, a, b, 2, 0);
            break;
        case K_ADD: case K_AND: case K_OR: case K_XOR: case K_SHL: case K_SHR:
            MODS(); gemit(alu[k->op], d, a, b, 2, 0);
            break;
        case K_INV: case K_SQRT: case K_RSQ: case K_EXP2:
            MODS(); gemit(OP_MATH, d, a, gnull(TY_F), 2, mathfn[k->op]);
            break;
        case K_POW:
            MODS(); gemit(OP_MATH, d, a, b, 2, MF_POW);
            break;
        case K_CMP:
            MODS(); GM.cmod = cmod[k->cond]; GM.flag = k->flag * 2;
            gemit(OP_CMP, gnull(a.type), a, b, 2, 0);
            break;
        case K_JMP: case K_JIF: case K_JANY: case K_JNONE: case K_JALL: {
            gop ip = {RF_ARF, TY_UD, 0xA0, 0, 4, 1, 0, 0, 0};
            gscalar();
            GM.flag = k->flag * 2;
            GM.pred = k->op == K_JMP ? 0 : k->op == K_JIF ? PR_NORMAL : k->op == K_JALL ? PR_ALL16H : PR_ANY16H;
            GM.pinv = k->op == K_JNONE;
            gemit(OP_JMPI, ip, ip, gimm(0, TY_D), 2, 0);
            break;
        }
        case K_LOAD: {
            // constant-cache OWord block read of 4 owords from binding table entry 0
            int h = hdr_toggle ? G_HDR2 : G_HDR;
            hdr_toggle ^= 1;
            gscalar(); gemit(OP_ADD, gsc(h, 2, TY_UD), a, b, 2, 0);
            if (k->off) { gscalar(); gemit(OP_ADD, gsc(h, 2, TY_UD), gsc(h, 2, TY_UD), gimm(k->off, TY_UD), 2, 0); }
            gsend(SFID_CONST_CACHE, gvec(G_BLK + 2 * k->target, TY_UW), h,
                  (1u << 25) | (2u << 20) | (1u << 19) | (0u << 14) | (3u << 8) | 0, 16);
            break;
        }
        case K_STORE:
            // media block write of the 4x4 tile: 16 bytes wide, 4 rows
            GM.es = 8; GM.nomask = 1; gemit(OP_MOV, gvec(G_ST, TY_UD), gvec(0, TY_UD), gnull(0), 1, 0);
            gscalar(); gemit(OP_SHL, gsc(G_ST, 0, TY_UD), gsc(0, 1, TY_UD), gimm(4, TY_UD), 2, 0);
            gscalar(); gemit(OP_SHL, gsc(G_ST, 1, TY_UD), gsc(0, 6, TY_UD), gimm(2, TY_UD), 2, 0);
            gscalar(); gemit(OP_MOV, gsc(G_ST, 2, TY_UD), gimm((3u << 16) | 15, TY_UD), gnull(0), 1, 0);
            a.type = TY_UD;                                    // raw bits, no conversion
            GM.nomask = 1; gemit(OP_MOV, gvec(G_ST + 1, TY_UD), a, gnull(0), 1, 0);
            gsend(SFID_RENDER_CACHE, gnull(TY_UD), G_ST,
                  (3u << 25) | (0u << 20) | (1u << 19) | (10u << 14) | (uint32_t)k->target, 16);
            break;
        case K_END:
            GM.es = 8; GM.nomask = 1; gemit(OP_MOV, gvec(G_EOT, TY_UD), gvec(0, TY_UD), gnull(0), 1, 0);
            gsend(SFID_THREAD_SPAWNER, gnull(TY_UD), G_EOT, 0x82000010u, 16);
            break;
        }
#undef MODS
    }
    at[kn] = gn;
    for (int i = 0; i < kn; i++) {
        if (kir[i].op < K_JMP || kir[i].op > K_JALL) continue;
        if (kir[i].target < 0) { fprintf(stderr, "gen75: IR %d: unpatched jump\n", i); return -1; }
        int from = at[i];
        gcode[from][3] = (uint32_t)(16 * (at[kir[i].target] - from - 1));
    }
    return 0;
}

/* ------------------------------------------------- command stream ------- */

static struct {
    const gpu_drv *drv;
    int threads, tiles_x, tiles_y;
    gpu_buf batch;
    uint32_t batch_len;
    uint32_t *curbe;
} H;

// 0 if chip_id is a Haswell GPU
static int gen75_probe(uint32_t id, char *desc, size_t n) {
    uint32_t fam = id & 0xff00;
    if (fam != 0x0400 && fam != 0x0a00 && fam != 0x0c00 && fam != 0x0d00) return -1;
    int gt = (id >> 4) & 3;
    H.threads = 70 << gt;
    snprintf(desc, n, "Haswell GT%d [8086:%04x]", gt + 1, id);
    return 0;
}

// bufs: 0 scene (raw buffer), 1 and 2 images of tiles_x*4 by tiles_y*4 dwords
static int gen75_setup(const gpu_drv *drv, int tiles_x, int tiles_y, gpu_buf *bufs) {
    H.drv = drv; H.tiles_x = tiles_x; H.tiles_y = tiles_y;
    gpu_buf *batch = &H.batch;
    if (drv->buf_new(batch, 65536)) return -1;
    drv->buf_sync(batch, 1);
    char *base = batch->map;
    memset(base, 0, batch->size);
    // commands at 0, state from 4096
    enum { OFF_CURBE = 4096, OFF_IDD = 4096 + 512, OFF_BT = 4096 + 1024, OFF_SS = 4096 + 1088,
           OFF_KERNEL = 8192 };
    if (OFF_KERNEL + gn * 16 > (int)batch->size) { fprintf(stderr, "gen75: kernel too large\n"); return -1; }
    memcpy(base + OFF_KERNEL, gcode, gn * 16);

    // CURBE: 4 GRFs of uniforms (written per run), then the lane positions
    H.curbe = (uint32_t *)(base + OFF_CURBE);
    float *lane = (float *)H.curbe + 32;
    for (int i = 0; i < 16; i++) { lane[i] = i % 4; lane[16 + i] = i / 4; }

    // interface descriptor
    uint32_t *idd = (uint32_t *)(base + OFF_IDD);
    idd[0] = OFF_KERNEL;
    idd[1] = 1u << 18;                 // single program flow, IEEE float
    idd[3] = OFF_BT;                   // binding table pointer (32-byte aligned)
    idd[4] = 8u << 16;                 // CURBE read length 8 GRFs
    idd[5] = 1;                        // 1 thread per group

    // binding table: 0 scene (raw buffer), 1 and 2 linear 2D images of dwords
    uint32_t *bt = (uint32_t *)(base + OFF_BT);
    int W = tiles_x * 4, Hh = tiles_y * 4;
    for (int i = 0; i < 3; i++) {
        bt[i] = OFF_SS + 64 * i;
        uint32_t *ss = (uint32_t *)(base + bt[i]);
        if (i == 0) {
            uint32_t sz = (uint32_t)bufs[0].size - 1;
            ss[0] = (4u << 29) | (0x1FFu << 18);                // SURFTYPE_BUFFER, RAW
            ss[2] = ((sz >> 7) & 0x3fff) << 16 | (sz & 0x7f);
            ss[3] = ((sz >> 21) & 0x3ff) << 21;
        } else {
            ss[0] = (1u << 29) | (0x0C0u << 18) | (1u << 8);    // SURFTYPE_2D, B8G8R8A8_UNORM, linear
            ss[2] = ((Hh - 1) << 16) | (W - 1);
            ss[3] = W * 4 - 1;
        }
        ss[7] = (4u << 25) | (5u << 22) | (6u << 19) | (7u << 16);
        drv->reloc(batch, bt[i] + 4, &bufs[i], 0, i != 0);
    }

    uint32_t *c = (uint32_t *)base;
#define OUT(x) (*c++ = (x))
#define RELOC_SELF() do { drv->reloc(batch, (uint32_t)((char *)c - base), batch, 1, 0); c++; } while (0)
    OUT((3u << 29) | (1u << 27) | (1u << 24) | (4u << 16) | 2);   // PIPELINE_SELECT GPGPU
    OUT((3u << 29) | (0u << 27) | (1u << 24) | (1u << 16) | 8);   // STATE_BASE_ADDRESS
    OUT(0);
    RELOC_SELF();                                                 // surface
    RELOC_SELF();                                                 // dynamic
    OUT(0);                                                       // indirect
    RELOC_SELF();                                                 // instruction
    OUT(0); OUT(1); OUT(0); OUT(1);
    OUT((3u << 29) | (2u << 27) | (0u << 24) | (0u << 16) | 6);   // MEDIA_VFE_STATE
    OUT(0);
    OUT(((uint32_t)(H.threads - 1) << 16) | (0 << 8) | (1 << 2));
    OUT(0);
    OUT(8);                                                       // CURBE: 8 x 256 bits
    OUT(0); OUT(0); OUT(0);
    OUT((3u << 29) | (2u << 27) | (0u << 24) | (1u << 16) | 2);   // MEDIA_CURBE_LOAD
    OUT(0); OUT(256); OUT(OFF_CURBE);
    OUT((3u << 29) | (2u << 27) | (0u << 24) | (2u << 16) | 2);   // MEDIA_INTERFACE_DESCRIPTOR_LOAD
    OUT(0); OUT(32); OUT(OFF_IDD);
    OUT((3u << 29) | (2u << 27) | (1u << 24) | (5u << 16) | 9);   // GPGPU_WALKER
    OUT(0);
    OUT(1u << 30);                                                // SIMD16, 1x1x1 threads
    OUT(0); OUT(tiles_x);                                         // thread group X range
    OUT(0); OUT(tiles_y);                                         // thread group Y range
    OUT(0); OUT(1);
    OUT(0xffff); OUT(0xffffffff);
    OUT((0x0Au << 23));                                           // MI_BATCH_BUFFER_END
    OUT(0);
#undef OUT
#undef RELOC_SELF
    H.batch_len = (uint32_t)((char *)c - base);
    return 0;
}

// write the uniforms and queue the whole grid; returns at once
static int gen75_run(const uint32_t *uniforms) {
    H.drv->buf_sync(&H.batch, 1);              // the CURBE lives in the batch: waits for the GPU
    memcpy(H.curbe, uniforms, 32 * 4);
    return H.drv->exec(&H.batch, H.batch_len);
}

static void gen75_fini(void) {
    if (H.drv) H.drv->buf_free(&H.batch);
    H.drv = NULL;
}

static const gpu_hw hw_gen75 = {"gen75", gen75_probe, gen75_compile, gen75_setup, gen75_run, gen75_fini};

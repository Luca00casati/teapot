// kir.c — the kernel IR: a tiny SIMD16 vector instruction list that the ray
// tracer (rt_kernel.c) is written in once, and that each GPU's ISA backend
// (gen75.c, ...) encodes into machine code.
//
// Execution model: one program instance runs per *packet* of 16 lanes, which
// is one 4x4 pixel tile (lane i covers pixel (i % 4, i / 4) of the tile).
// Control flow is uniform for the whole packet; per-lane work is done by
// predication on flag 0. Registers are 32-bit and untyped, operands are typed
// views (float or uint) of them; an op whose destination type differs from
// its source type converts (float <-> uint).
//
//   KV(i)    vector register, one value per lane
//   KS(i)    scalar register, one value for the packet
//   KU(i)    uniform input (32 dwords written by the host per dispatch)
//   KB(k,i)  dword i (0..15) of block k, filled by LOAD (read-only)
//   KL0(i)   lane 0 of vector register i, as a scalar
//   LANE_X / LANE_Y   per-lane pixel position inside the tile (float 0..3)
//   GROUP_X / GROUP_Y tile coordinates of the packet (uint)
//   immediates
//
// An instruction is scalar when its destination is a scalar (or, for CMP,
// when no source is a vector). Memory is reached only through LOAD (64 bytes
// of buffer 0 into a block, address in 16-byte units) and STORE (a packet's
// 16 dwords into the 4x4 tile of image buffer 1 or 2).

enum { KO_NONE, KO_V, KO_S, KO_U, KO_B, KO_L0, KO_LX, KO_LY, KO_GX, KO_GY, KO_IMM };
enum { KT_F, KT_U };

typedef struct { uint8_t kind, type, neg; uint16_t idx; uint32_t imm; } kop;

enum {
    K_MOV, K_ADD, K_MUL, K_MIN, K_MAX, K_AND, K_OR, K_XOR, K_SHL, K_SHR,
    K_FLOOR, K_FRAC, K_INV, K_SQRT, K_RSQ, K_EXP2, K_POW,
    K_CMP,                                  // flag = a <cond> b, per lane (or scalar)
    K_JMP, K_JIF, K_JANY, K_JNONE, K_JALL,  // always / scalar flag / any, no, all lanes of flag
    K_LOAD,                                 // block <- buf0[(a + b) * 16 + off * 16 ..+64]
    K_STORE,                                // image buf `slot` tile <- a
    K_END,
};
enum { KC_EQ = 1, KC_NE, KC_GT, KC_GE, KC_LT, KC_LE };

typedef struct {
    uint8_t op, cond, flag, pred, sat;      // pred: 0 none, 1 + f = only lanes with flag f set
    kop d, a, b;
    int target;                             // jumps: instruction index; LOAD: block; STORE: slot
    uint32_t off;                           // LOAD: extra offset in 16-byte units
} kins;

#define KIR_MAX 4096
static kins kir[KIR_MAX];
static int kn;                              // instruction count = index of the next one
static int kir_nv;                          // vector registers allocated
static struct { int pred, sat; } KM;        // modifiers for the next instruction only

static kop KV(int i) { return (kop){KO_V, KT_F, 0, (uint16_t)i, 0}; }
static kop KVU(int i) { return (kop){KO_V, KT_U, 0, (uint16_t)i, 0}; }
static kop KS(int i) { return (kop){KO_S, KT_F, 0, (uint16_t)i, 0}; }
static kop KSU(int i) { return (kop){KO_S, KT_U, 0, (uint16_t)i, 0}; }
static kop KU(int i) { return (kop){KO_U, KT_F, 0, (uint16_t)i, 0}; }
static kop KUU(int i) { return (kop){KO_U, KT_U, 0, (uint16_t)i, 0}; }
static kop KB(int k, int i) { return (kop){KO_B, KT_F, 0, (uint16_t)(k * 16 + i), 0}; }
static kop KBU(int k, int i) { return (kop){KO_B, KT_U, 0, (uint16_t)(k * 16 + i), 0}; }
static kop KL0(int i) { return (kop){KO_L0, KT_F, 0, (uint16_t)i, 0}; }
static const kop LANE_X = {KO_LX, KT_F, 0, 0, 0}, LANE_Y = {KO_LY, KT_F, 0, 0, 0};
static const kop GROUP_X = {KO_GX, KT_U, 0, 0, 0}, GROUP_Y = {KO_GY, KT_U, 0, 0, 0};
static kop imf(float f) { union { float f; uint32_t u; } x = {f}; return (kop){KO_IMM, KT_F, 0, 0, x.u}; }
static kop imu(uint32_t u) { return (kop){KO_IMM, KT_U, 0, 0, u}; }
static kop neg(kop o) {
    if (o.kind == KO_IMM) o.imm ^= 0x80000000u;
    else o.neg ^= 1;
    return o;
}

static int vreg(void) { return kir_nv++; }

static int kemit(int op, kop d, kop a, kop b) {
    if (kn >= KIR_MAX) { fprintf(stderr, "kir: program too big\n"); exit(1); }
    kir[kn] = (kins){.op = op, .pred = KM.pred, .sat = KM.sat, .d = d, .a = a, .b = b};
    memset(&KM, 0, sizeof KM);
    return kn++;
}
static kop none;
static void emov(kop d, kop s) { kemit(K_MOV, d, s, none); }
static void eadd(kop d, kop a, kop b) { kemit(K_ADD, d, a, b); }
static void emul(kop d, kop a, kop b) { kemit(K_MUL, d, a, b); }
static void vmin_(kop d, kop a, kop b) { kemit(K_MIN, d, a, b); }
static void vmax_(kop d, kop a, kop b) { kemit(K_MAX, d, a, b); }
static void and_(kop d, kop a, kop b) { kemit(K_AND, d, a, b); }
static void or_(kop d, kop a, kop b) { kemit(K_OR, d, a, b); }
static void xor_(kop d, kop a, kop b) { kemit(K_XOR, d, a, b); }
static void shl(kop d, kop a, kop b) { kemit(K_SHL, d, a, b); }
static void shr(kop d, kop a, kop b) { kemit(K_SHR, d, a, b); }
static void efloor(kop d, kop a) { kemit(K_FLOOR, d, a, none); }
static void frc(kop d, kop a) { kemit(K_FRAC, d, a, none); }
static void einv(kop d, kop a) { kemit(K_INV, d, a, none); }
static void esqrt(kop d, kop a) { kemit(K_SQRT, d, a, none); }
static void ersq(kop d, kop a) { kemit(K_RSQ, d, a, none); }
static void eexp2(kop d, kop a) { kemit(K_EXP2, d, a, none); }
static void epow(kop d, kop a, kop b) { kemit(K_POW, d, a, b); }
static void ecmp(int cond, int flag, kop a, kop b) {
    int i = kemit(K_CMP, none, a, b);
    kir[i].cond = cond; kir[i].flag = flag;
}
static void P(int flag) { KM.pred = 1 + flag; }  // predicate the next instruction on flag
static void SAT(void) { KM.sat = 1; }             // clamp the next result to [0,1]

// jumps return their index; patch() points one at a target index (kn = here)
static int jump(int op, int flag) {
    int i = kemit(op, none, none, none);
    kir[i].flag = flag; kir[i].target = -1;
    return i;
}
static void patch(int at, int target) { kir[at].target = target; }
static void jmp_to(int target, int op, int flag) { patch(jump(op, flag), target); }

static void kload(int block, kop a, kop b, uint32_t off) {
    int i = kemit(K_LOAD, none, a, b);
    kir[i].target = block; kir[i].off = off;
}
static void kstore(int slot, kop src) { kir[kemit(K_STORE, none, src, none)].target = slot; }
static void kend(void) { kemit(K_END, none, none, none); }

static void kir_reset(void) { kn = 0; kir_nv = 0; memset(&KM, 0, sizeof KM); }

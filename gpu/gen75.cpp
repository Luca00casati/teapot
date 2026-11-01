// gen75.cpp — the Hardware layer for Intel Haswell (Gen7.5): encodes the
// kernel IR into native 128-bit EU instructions and builds the GPGPU command
// stream (PIPELINE_SELECT → STATE_BASE_ADDRESS → MEDIA_VFE_STATE → CURBE →
// interface descriptor → GPGPU_WALKER). Memory and submission go through a
// Driver (i915.cpp).
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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "layers.h"

namespace gpu {
namespace {

using kir::Kind;

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

struct Gop {
    uint8_t file = RF_ARF, type = TY_F, nr = 0, sub = 0, vs = 0, w = 1, hs = 0, neg = 0;  // default: null
    uint32_t imm = 0;
};
Gop gvec(int r, int type) { return {RF_GRF, (uint8_t)type, (uint8_t)r, 0, 8, 8, 1, 0, 0}; }  // SIMD16, r:r+1
Gop gsc(int r, int c, int type) { return {RF_GRF, (uint8_t)type, (uint8_t)r, (uint8_t)(c * 4), 0, 1, 0, 0, 0}; }
Gop gimm(uint32_t v, int type) { return {RF_IMM, (uint8_t)type, 0, 0, 0, 0, 0, 0, v}; }
Gop gnull(int type = TY_F) { Gop g; g.type = (uint8_t)type; return g; }

void setb(uint32_t *in, int hi, int lo, uint32_t v) {
    for (int b = lo; b <= hi; b++, v >>= 1)
        in[b / 32] = (in[b / 32] & ~(1u << (b % 32))) | ((v & 1) << (b % 32));
}
int enc_vs(int v) { return v == 0 ? 0 : v == 1 ? 1 : v == 2 ? 2 : v == 4 ? 3 : v == 8 ? 4 : v == 16 ? 5 : 6; }
int enc_w(int w) { return w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : w == 8 ? 3 : 4; }
int enc_hs(int h) { return h == 0 ? 0 : h == 1 ? 1 : h == 2 ? 2 : 3; }
int enc_es(int e) { return e == 1 ? 0 : e == 2 ? 1 : e == 4 ? 2 : e == 8 ? 3 : 4; }

// modifiers of one EU instruction
struct Mods { int pred = 0, pinv = 0, flag = 0, cmod = 0, sat = 0, es = 16, nomask = 0; };
const Mods scalar_mods = {0, 0, 0, 0, 0, 1, 1};
Mods header_mods() { Mods m; m.es = 8; m.nomask = 1; return m; }

class Encoder {
public:
    std::vector<std::array<uint32_t, 4>> code;

    int emit(int opc, const Mods &m, Gop d, Gop s0, Gop s1 = gnull(), int nsrc = 2, int fn = 0) {
        std::array<uint32_t, 4> a = {};
        uint32_t *in = a.data();
        setb(in, 6, 0, opc);
        setb(in, 9, 9, m.nomask);
        setb(in, 19, 16, m.pred);
        setb(in, 20, 20, m.pinv);
        setb(in, 23, 21, enc_es(m.es));
        setb(in, 27, 24, fn ? fn : m.cmod);
        setb(in, 31, 31, m.sat);
        setb(in, 90, 90, m.flag >> 1);   // f0 / f1
        setb(in, 89, 89, m.flag & 1);    // .0 / .1
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
        code.push_back(a);
        return (int)code.size() - 1;
    }
    void mov(const Mods &m, Gop d, Gop s) { emit(OP_MOV, m, d, s, gnull(0), 1); }
    void send(int sfid, Gop dst, int src, uint32_t desc) {
        Mods m; m.nomask = 1;
        emit(OP_SEND, m, dst, gvec(src, TY_UD), gimm(desc, TY_UD), 2, sfid);
    }
};

Gop gen_op(const kir::Operand &o) {
    int ty = o.type == kir::Type::F ? TY_F : TY_UD;
    Gop g;
    switch (o.kind) {
    case Kind::V: g = gvec(G_VEC + 2 * o.idx, ty); break;
    case Kind::S: g = gsc(o.idx < 8 ? G_SC0 : G_SC1, o.idx % 8, ty); break;
    case Kind::U: g = gsc(G_UNI + o.idx / 8, o.idx % 8, ty); break;
    case Kind::B: g = gsc(G_BLK + o.idx / 8, o.idx % 8, ty); break;
    case Kind::Lane0: g = gsc(G_VEC + 2 * o.idx, 0, ty); break;
    case Kind::LaneX: g = gvec(G_LX, ty); break;
    case Kind::LaneY: g = gvec(G_LY, ty); break;
    case Kind::GroupX: g = gsc(0, 1, ty); break;
    case Kind::GroupY: g = gsc(0, 6, ty); break;
    case Kind::Imm: return gimm(o.imm, ty);
    default: return gnull(ty);
    }
    g.neg = o.neg;
    return g;
}

class Gen75 final : public Hardware {
public:
    const char *name() const override { return "gen75"; }

    bool probe(uint32_t id, std::string &desc) override {
        uint32_t fam = id & 0xff00;
        if (fam != 0x0400 && fam != 0x0a00 && fam != 0x0c00 && fam != 0x0d00) return false;
        int gt = (id >> 4) & 3;
        threads_ = 70 << gt;
        char s[64];
        snprintf(s, sizeof s, "Haswell GT%d [8086:%04x]", gt + 1, id);
        desc = s;
        return true;
    }

    bool compile(const kir::Program &p) override;
    bool setup(Driver &drv, int tiles_x, int tiles_y, Buffer *bufs) override;

    bool run(const uint32_t *uniforms) override {
        drv_->sync(batch_, true);                // the CURBE lives in the batch: waits for the GPU
        memcpy(curbe_, uniforms, 32 * 4);
        return drv_->exec(batch_, batch_len_);
    }

private:
    Encoder enc_;
    int threads_ = 0;
    Driver *drv_ = nullptr;
    Buffer batch_;
    uint32_t batch_len_ = 0;
    uint32_t *curbe_ = nullptr;
};

/* ------------------------------------------------- IR → EU code --------- */

bool Gen75::compile(const kir::Program &p) {
    using namespace kir;
    auto alu = [](Op op) {
        switch (op) {
        case MOV: return OP_MOV; case ADD: return OP_ADD; case MUL: return OP_MUL;
        case AND: return OP_AND; case OR: return OP_OR; case XOR: return OP_XOR;
        case SHL: return OP_SHL; case SHR: return OP_SHR; case FLOOR: return OP_RNDD;
        default: return OP_FRC;
        }
    };
    auto mathfn = [](Op op) {
        return op == INV ? MF_INV : op == SQRT ? MF_SQRT : op == RSQ ? MF_RSQ : op == EXP2 ? MF_EXP : MF_POW;
    };
    auto cmod = [](Cond c) {
        switch (c) {
        case EQ: return CM_Z; case NE: return CM_NZ; case GT: return CM_G;
        case GE: return CM_GE; case LT: return CM_L; default: return CM_LE;
        }
    };
    if (G_VEC + 2 * p.nvec > G_VEC_END) {
        fprintf(stderr, "gen75: kernel needs %d vector registers, only %d fit\n", p.nvec, (G_VEC_END - G_VEC) / 2);
        return false;
    }
    Encoder &e = enc_;
    e.code.clear();
    std::vector<int> at(p.code.size() + 1);  // first EU instruction of each IR instruction
    bool hdr_toggle = false;
    // load headers start as a copy of the thread payload; only .2 changes
    e.mov(header_mods(), gvec(G_HDR, TY_UD), gvec(0, TY_UD));
    e.mov(header_mods(), gvec(G_HDR2, TY_UD), gvec(0, TY_UD));

    for (size_t i = 0; i < p.code.size(); i++) {
        const Inst &k = p.code[i];
        at[i] = (int)e.code.size();
        bool scalar = k.op == CMP ? !k.a.is_vector() && !k.b.is_vector() : k.d.kind == Kind::S;
        if (scalar && (k.a.is_vector() || k.b.is_vector())) {
            fprintf(stderr, "gen75: IR %zu: vector source in a scalar op\n", i);
            return false;
        }
        Gop d = gen_op(k.d), a = gen_op(k.a), b = gen_op(k.b);
        Mods m = scalar ? scalar_mods : Mods();
        if (k.pred) { m.pred = PR_NORMAL; m.flag = (k.pred - 1) * 2; }
        m.sat = k.sat;
        switch (k.op) {
        case MOV: case FLOOR: case FRAC:
            e.emit(alu(k.op), m, d, a, gnull(0), 1);
            break;
        case MUL:
            // 32x16 integer multiply: a uint immediate of 16 bits is encoded as UW
            if (b.file == RF_IMM && b.type == TY_UD) {
                if (b.imm > 0xffff) { fprintf(stderr, "gen75: IR %zu: 32-bit integer multiply\n", i); return false; }
                b.type = TY_UW; b.imm |= b.imm << 16;
            }
            e.emit(OP_MUL, m, d, a, b);
            break;
        case MIN: case MAX:
            m.cmod = k.op == MIN ? CM_L : CM_GE;
            e.emit(OP_SEL, m, d, a, b);
            break;
        case ADD: case AND: case OR: case XOR: case SHL: case SHR:
            e.emit(alu(k.op), m, d, a, b);
            break;
        case INV: case SQRT: case RSQ: case EXP2:
            e.emit(OP_MATH, m, d, a, gnull(TY_F), 2, mathfn(k.op));
            break;
        case POW:
            e.emit(OP_MATH, m, d, a, b, 2, MF_POW);
            break;
        case CMP:
            m.cmod = cmod(k.cond); m.flag = k.flag * 2;
            e.emit(OP_CMP, m, gnull(a.type), a, b);
            break;
        case JMP: case JIF: case JANY: case JNONE: case JALL: {
            Gop ip = {RF_ARF, TY_UD, 0xA0, 0, 4, 1, 0, 0, 0};
            Mods j = scalar_mods;
            j.flag = k.flag * 2;
            j.pred = k.op == JMP ? 0 : k.op == JIF ? PR_NORMAL : k.op == JALL ? PR_ALL16H : PR_ANY16H;
            j.pinv = k.op == JNONE;
            e.emit(OP_JMPI, j, ip, ip, gimm(0, TY_D));
            break;
        }
        case LOAD: {
            // constant-cache OWord block read of 4 owords from binding table entry 0
            int h = hdr_toggle ? G_HDR2 : G_HDR;
            hdr_toggle = !hdr_toggle;
            e.emit(OP_ADD, scalar_mods, gsc(h, 2, TY_UD), a, b);
            if (k.off) e.emit(OP_ADD, scalar_mods, gsc(h, 2, TY_UD), gsc(h, 2, TY_UD), gimm(k.off, TY_UD));
            e.send(SFID_CONST_CACHE, gvec(G_BLK + 2 * k.target, TY_UW), h,
                   (1u << 25) | (2u << 20) | (1u << 19) | (0u << 14) | (3u << 8) | 0);
            break;
        }
        case STORE: {
            // media block write of the 4x4 tile: 16 bytes wide, 4 rows
            e.mov(header_mods(), gvec(G_ST, TY_UD), gvec(0, TY_UD));
            e.emit(OP_SHL, scalar_mods, gsc(G_ST, 0, TY_UD), gsc(0, 1, TY_UD), gimm(4, TY_UD));
            e.emit(OP_SHL, scalar_mods, gsc(G_ST, 1, TY_UD), gsc(0, 6, TY_UD), gimm(2, TY_UD));
            e.mov(scalar_mods, gsc(G_ST, 2, TY_UD), gimm((3u << 16) | 15, TY_UD));
            a.type = TY_UD;                                      // raw bits, no conversion
            Mods all; all.nomask = 1;
            e.mov(all, gvec(G_ST + 1, TY_UD), a);
            e.send(SFID_RENDER_CACHE, gnull(TY_UD), G_ST,
                   (3u << 25) | (0u << 20) | (1u << 19) | (10u << 14) | (uint32_t)k.target);
            break;
        }
        case END:
            e.mov(header_mods(), gvec(G_EOT, TY_UD), gvec(0, TY_UD));
            e.send(SFID_THREAD_SPAWNER, gnull(TY_UD), G_EOT, 0x82000010u);
            break;
        }
    }
    at[p.code.size()] = (int)e.code.size();
    for (size_t i = 0; i < p.code.size(); i++) {
        const Inst &k = p.code[i];
        if (k.op < JMP || k.op > JALL) continue;
        if (k.target < 0) { fprintf(stderr, "gen75: IR %zu: jump to an unbound label\n", i); return false; }
        int from = at[i];
        e.code[from][3] = (uint32_t)(16 * (at[k.target] - from - 1));
    }
#ifdef TEAPOT_DUMP_KERNEL                  // raw EU binary to this path, for disassemblers / diffs
    if (FILE *f = fopen(TEAPOT_DUMP_KERNEL, "wb")) { fwrite(e.code.data(), 16, e.code.size(), f); fclose(f); }
#endif
    return true;
}

/* ------------------------------------------------- command stream ------- */

bool Gen75::setup(Driver &drv, int tiles_x, int tiles_y, Buffer *bufs) {
    drv_ = &drv;
    if (!drv.alloc(batch_, 65536)) return false;
    drv.sync(batch_, true);
    char *base = (char *)batch_.map;
    memset(base, 0, batch_.size);
    // commands at 0, state from 4096
    enum { OFF_CURBE = 4096, OFF_IDD = 4096 + 512, OFF_BT = 4096 + 1024, OFF_SS = 4096 + 1088, OFF_KERNEL = 8192 };
    size_t kbytes = enc_.code.size() * 16;
    if (OFF_KERNEL + kbytes > batch_.size) { fprintf(stderr, "gen75: kernel too large\n"); return false; }
    memcpy(base + OFF_KERNEL, enc_.code.data(), kbytes);

    // CURBE: 4 GRFs of uniforms (written per run), then the lane positions
    curbe_ = (uint32_t *)(base + OFF_CURBE);
    float *lane = (float *)curbe_ + 32;
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
    int W = tiles_x * 4, H = tiles_y * 4;
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
            ss[2] = ((H - 1) << 16) | (W - 1);
            ss[3] = W * 4 - 1;
        }
        ss[7] = (4u << 25) | (5u << 22) | (6u << 19) | (7u << 16);
        drv.reloc(batch_, bt[i] + 4, bufs[i], 0, i != 0);
    }

    uint32_t *c = (uint32_t *)base;
    auto out = [&](uint32_t x) { *c++ = x; };
    auto reloc_self = [&] { drv.reloc(batch_, (uint32_t)((char *)c - base), batch_, 1, false); c++; };
    out((3u << 29) | (1u << 27) | (1u << 24) | (4u << 16) | 2);   // PIPELINE_SELECT GPGPU
    out((3u << 29) | (0u << 27) | (1u << 24) | (1u << 16) | 8);   // STATE_BASE_ADDRESS
    out(0);
    reloc_self();                                                 // surface
    reloc_self();                                                 // dynamic
    out(0);                                                       // indirect
    reloc_self();                                                 // instruction
    out(0); out(1); out(0); out(1);
    out((3u << 29) | (2u << 27) | (0u << 24) | (0u << 16) | 6);   // MEDIA_VFE_STATE
    out(0);
    out(((uint32_t)(threads_ - 1) << 16) | (0 << 8) | (1 << 2));
    out(0);
    out(8);                                                       // CURBE: 8 x 256 bits
    out(0); out(0); out(0);
    out((3u << 29) | (2u << 27) | (0u << 24) | (1u << 16) | 2);   // MEDIA_CURBE_LOAD
    out(0); out(256); out(OFF_CURBE);
    out((3u << 29) | (2u << 27) | (0u << 24) | (2u << 16) | 2);   // MEDIA_INTERFACE_DESCRIPTOR_LOAD
    out(0); out(32); out(OFF_IDD);
    out((3u << 29) | (2u << 27) | (1u << 24) | (5u << 16) | 9);   // GPGPU_WALKER
    out(0);
    out(1u << 30);                                                // SIMD16, 1x1x1 threads
    out(0); out(tiles_x);                                         // thread group X range
    out(0); out(tiles_y);                                         // thread group Y range
    out(0); out(1);
    out(0xffff); out(0xffffffff);
    out((0x0Au << 23));                                           // MI_BATCH_BUFFER_END
    out(0);
    batch_len_ = (uint32_t)((char *)c - base);
    return true;
}

}  // namespace

std::unique_ptr<Hardware> make_gen75() { return std::make_unique<Gen75>(); }

}  // namespace gpu

// kir.h — the kernel IR: a tiny SIMD16 vector instruction list that the ray
// tracer (rt_kernel.cpp) is written in once, and that each GPU's hardware
// layer (gen75.cpp, ...) encodes into machine code.
//
// Execution model: one program instance runs per *packet* of 16 lanes, which
// is one 4x4 pixel tile (lane i covers pixel (i % 4, i / 4) of the tile).
// Control flow is uniform for the whole packet; per-lane work is done by
// predication on a flag. Registers are 32-bit and untyped; operands are typed
// views (float or uint) of them, and an op whose destination type differs
// from its source type converts (float <-> uint).
//
//   V(i)     vector register, one value per lane
//   S(i)     scalar register, one value for the packet
//   U(i)     uniform input (32 dwords written by the host per dispatch)
//   B(k, i)  dword i (0..15) of block k, filled by load() (read-only)
//   lane0(v) lane 0 of vector register v, as a scalar
//   lane_x / lane_y    per-lane pixel position inside the tile (float 0..3)
//   group_x / group_y  tile coordinates of the packet (uint)
//   float immediates (implicit) and u32(x) integer immediates
//
// An instruction is scalar when its destination is a scalar (or, for a
// compare, when no source is a vector). Memory is reached only through load()
// (64 bytes of buffer 0 into a block, address in 16-byte units) and store()
// (a packet's 16 dwords into the 4x4 tile of image buffer 1 or 2).
//
// The builder syntax maps one statement to exactly one instruction, so the
// kernel author controls scheduling (GPU cores issue in order):
//
//   t0 = a * b;          t0 += c;          t0 = sat(t0 * t1);
//   f0 = t6 <= t9;       ht[f0] = -1.0f;   // predicated on flag 0
//
// Compound expressions such as `a * b + c` deliberately do not compile.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace kir {

enum class Kind : uint8_t { None, V, S, U, B, Lane0, LaneX, LaneY, GroupX, GroupY, Imm };
enum class Type : uint8_t { F, U };

enum Op : uint8_t {
    MOV, ADD, MUL, MIN, MAX, AND, OR, XOR, SHL, SHR,
    FLOOR, FRAC, INV, SQRT, RSQ, EXP2, POW,
    CMP,                                    // flag = a <cond> b, per lane (or scalar)
    JMP, JIF, JANY, JNONE, JALL,            // always / scalar flag / any, no, all lanes of flag
    LOAD,                                   // block <- buf0[(a + b + off) * 16 .. +64]
    STORE,                                  // image buffer `slot`, this packet's tile <- a
    END,
};
enum Cond : uint8_t { EQ = 1, NE, GT, GE, LT, LE };

struct Operand {
    Kind kind = Kind::None;
    Type type = Type::F;
    bool neg = false;
    uint16_t idx = 0;
    uint32_t imm = 0;

    constexpr Operand() = default;
    constexpr Operand(Kind k, Type t, int i) : kind(k), type(t), idx((uint16_t)i) {}
    Operand(float f) : kind(Kind::Imm) { __builtin_memcpy(&imm, &f, 4); }  // float immediate

    Operand u() const { Operand o = *this; o.type = Type::U; return o; }   // uint view
    Operand f() const { Operand o = *this; o.type = Type::F; return o; }   // float view
    Operand operator-() const {
        Operand o = *this;
        if (kind == Kind::Imm) o.imm ^= 0x80000000u;
        else o.neg = !o.neg;
        return o;
    }
    bool is_vector() const { return kind == Kind::V || kind == Kind::LaneX || kind == Kind::LaneY; }
    bool same_reg(const Operand &o) const { return kind == o.kind && idx == o.idx; }
};

inline Operand u32(uint32_t v) { Operand o(Kind::Imm, Type::U, 0); o.imm = v; return o; }
const Operand lane_x{Kind::LaneX, Type::F, 0}, lane_y{Kind::LaneY, Type::F, 0};
const Operand group_x{Kind::GroupX, Type::U, 0}, group_y{Kind::GroupY, Type::U, 0};
inline Operand U(int i) { return {Kind::U, Type::F, i}; }
inline Operand B(int block, int i) { return {Kind::B, Type::F, block * 16 + i}; }
inline Operand lane0(const Operand &v) { return {Kind::Lane0, v.type, v.idx}; }

struct Inst {
    explicit Inst(Op o) : op(o) {}
    Op op;
    Cond cond = EQ;
    uint8_t flag = 0, pred = 0;             // pred: 0 none, 1 + f = only lanes with flag f set
    bool sat = false;
    Operand d, a, b;
    int target = -1;                        // jumps: instruction index; LOAD: block; STORE: slot
    uint32_t off = 0;                       // LOAD: extra offset in 16-byte units
};

struct Program {
    std::vector<Inst> code;
    int nvec = 0;                           // vector registers allocated
    int here() const { return (int)code.size(); }
    int emit(const Inst &in) {
        code.push_back(in);
        return here() - 1;
    }
};

// the program being built; set with Building (defined in rt_kernel.cpp)
extern Program *cur;
struct Building {
    explicit Building(Program &p) { p = Program(); cur = &p; }
    ~Building() { cur = nullptr; }
};

// one pending instruction: what an operator returns, emitted on assignment
struct Expr {
    Op op;
    Operand a, b;
    bool sat;
    Expr(Op op, Operand a, Operand b) : op(op), a(a), b(b), sat(false) {}
};

inline Expr operator+(Operand a, Operand b) { return {ADD, a, b}; }
inline Expr operator-(Operand a, Operand b) { return {ADD, a, -b}; }
inline Expr operator*(Operand a, Operand b) { return {MUL, a, b}; }
inline Expr operator&(Operand a, Operand b) { return {AND, a, b}; }
inline Expr operator|(Operand a, Operand b) { return {OR, a, b}; }
inline Expr operator^(Operand a, Operand b) { return {XOR, a, b}; }
inline Expr operator<<(Operand a, Operand b) { return {SHL, a, b}; }
inline Expr operator>>(Operand a, Operand b) { return {SHR, a, b}; }
inline Expr min(Operand a, Operand b) { return {MIN, a, b}; }
inline Expr max(Operand a, Operand b) { return {MAX, a, b}; }
inline Expr pow(Operand a, Operand b) { return {POW, a, b}; }
inline Expr floor(Operand a) { return {FLOOR, a, {}}; }
inline Expr frac(Operand a) { return {FRAC, a, {}}; }
inline Expr inv(Operand a) { return {INV, a, {}}; }
inline Expr sqrt(Operand a) { return {SQRT, a, {}}; }
inline Expr rsq(Operand a) { return {RSQ, a, {}}; }
inline Expr exp2(Operand a) { return {EXP2, a, {}}; }
inline Expr sat(Expr e) { e.sat = true; return e; }  // clamp the result to [0, 1]

struct Cmp { Cond c; Operand a, b; };
inline Cmp operator==(Operand a, Operand b) { return {EQ, a, b}; }
inline Cmp operator!=(Operand a, Operand b) { return {NE, a, b}; }
inline Cmp operator>(Operand a, Operand b) { return {GT, a, b}; }
inline Cmp operator>=(Operand a, Operand b) { return {GE, a, b}; }
inline Cmp operator<(Operand a, Operand b) { return {LT, a, b}; }
inline Cmp operator<=(Operand a, Operand b) { return {LE, a, b}; }

struct Flag {
    uint8_t n;
    void operator=(Cmp c) const {
        Inst in{CMP};
        in.cond = c.c; in.flag = n; in.a = c.a; in.b = c.b;
        cur->emit(in);
    }
};
constexpr Flag f0{0}, f1{1};

inline void emit(Operand d, Expr e, int pred = 0) {
    Inst in{e.op};
    in.d = d; in.a = e.a; in.b = e.b; in.sat = e.sat; in.pred = (uint8_t)pred;
    cur->emit(in);
}

// a register write restricted to the lanes (or the scalar) whose flag is set
struct Masked {
    Operand r;
    Flag f;
    void operator=(Expr e) const { emit(r, e, 1 + f.n); }
    void operator=(Operand o) const { emit(r, {MOV, o, {}}, 1 + f.n); }
    void operator=(float v) const { *this = Operand(v); }
    void operator+=(Operand o) const { emit(r, {ADD, r, o}, 1 + f.n); }
};

// a writable register; assigning to it emits an instruction
struct Reg : Operand {
    using Operand::Operand;
    Reg(const Operand &o) : Operand(o) {}
    Reg(const Reg &) = default;
    const Reg &operator=(Expr e) const { emit(*this, e); return *this; }
    const Reg &operator=(Operand o) const { emit(*this, {MOV, o, {}}); return *this; }
    const Reg &operator=(const Reg &o) const { return *this = (const Operand &)o; }
    const Reg &operator=(float v) const { return *this = Operand(v); }
    const Reg &operator+=(Operand o) const { return *this = *this + o; }
    const Reg &operator-=(Operand o) const { return *this = *this - o; }
    const Reg &operator*=(Operand o) const { return *this = *this * o; }
    Masked operator[](Flag f) const { return {*this, f}; }
    Reg u() const { return Operand::u(); }
    Reg f() const { return Operand::f(); }
};

inline Reg V(int i) { return Operand(Kind::V, Type::F, i); }
inline Reg S(int i) { return Operand(Kind::S, Type::F, i); }
inline Reg vreg() { return V(cur->nvec++); }

// jump targets; a label may be used before it is bound
struct Label {
    int pos = -1;
    std::vector<int> fixups;
    void bind() {
        pos = cur->here();
        for (int i : fixups) cur->code[i].target = pos;
        fixups.clear();
    }
};
inline Label here() { Label l; l.pos = cur->here(); return l; }

inline void jump(Op op, Flag f, Label &l) {
    Inst in{op};
    in.flag = f.n; in.target = l.pos;
    int i = cur->emit(in);
    if (l.pos < 0) l.fixups.push_back(i);
}
inline void jmp(Label &l) { jump(JMP, f0, l); }
inline void jmp_if(Flag f, Label &l) { jump(JIF, f, l); }     // scalar flag set
inline void jmp_any(Flag f, Label &l) { jump(JANY, f, l); }   // some lane has the flag
inline void jmp_none(Flag f, Label &l) { jump(JNONE, f, l); } // no lane has it
inline void jmp_all(Flag f, Label &l) { jump(JALL, f, l); }   // every lane has it

inline void load(int block, Operand a, Operand b, uint32_t off = 0) {
    Inst in{LOAD};
    in.a = a; in.b = b; in.target = block; in.off = off;
    cur->emit(in);
}
inline void store(int slot, Operand src) {
    Inst in{STORE};
    in.a = src; in.target = slot;
    cur->emit(in);
}
inline void end() { cur->emit(Inst{END}); }

}  // namespace kir

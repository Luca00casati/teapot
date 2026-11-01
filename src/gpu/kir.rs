//! The kernel IR: a tiny SIMD16 vector instruction list that the ray tracer
//! (rt_kernel.rs) is written in once, and that each GPU's hardware layer
//! (gen75.rs, ...) encodes into machine code.
//!
//! Execution model: one program instance runs per *packet* of 16 lanes, which
//! is one 4x4 pixel tile (lane i covers pixel (i % 4, i / 4) of the tile).
//! Control flow is uniform for the whole packet; per-lane work is done by
//! predication on a flag. Registers are 32-bit and untyped; operands are typed
//! views (float or uint) of them, and an op whose destination type differs
//! from its source type converts (float <-> uint).
//!
//!   vreg()     a new vector register, one value per lane
//!   sreg(i)    scalar register i, one value for the packet
//!   uni(i)     uniform input (32 dwords written by the host per dispatch)
//!   blk(k, i)  dword i (0..15) of block k, filled by load() (read-only)
//!   lane0(v)   lane 0 of vector register v, as a scalar
//!   LANE_X / LANE_Y    per-lane pixel position inside the tile (float 0..3)
//!   GROUP_X / GROUP_Y  tile coordinates of the packet (uint)
//!   f32 values are float immediates, u32 values integer immediates
//!
//! An instruction is scalar when its destination is a scalar (or, for a
//! compare, when no source is a vector). Memory is reached only through load()
//! (64 bytes of buffer 0 into a block, address in 16-byte units) and store()
//! (a packet's 16 dwords into the 4x4 tile of image buffer 1 or 2).
//!
//! The builder maps one statement to exactly one instruction, so the kernel
//! author controls scheduling (GPU cores issue in order):
//!
//!   t0.set(a * b);       t0.set(t0 + c);       t0.set(sat(t0 * t1));
//!   F0.set(le(t6, t9));  ht.at(F0).set(-1.0);  // predicated on flag 0
//!
//! Operators build an `Expr`, which is not an operand, so a compound
//! expression such as `a * b + c` does not compile.

use std::cell::RefCell;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Neg, Shl, Shr, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    None,
    V,
    S,
    U,
    B,
    Lane0,
    LaneX,
    LaneY,
    GroupX,
    GroupY,
    Imm,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Type {
    F,
    U,
}

// the whole IR, also the parts the ray tracer does not use (yet)
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Mov,
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Floor,
    Frac,
    Inv,
    Sqrt,
    Rsq,
    Exp2,
    Pow,
    Cmp,   // flag = a <cond> b, per lane (or scalar)
    Jmp,   // always
    Jif,   // scalar flag set
    Jany,  // some lane has the flag
    Jnone, // no lane has it
    Jall,  // every lane has it
    Load,  // block <- buf0[(a + b + off) * 16 .. +64]
    Store, // image buffer `slot`, this packet's tile <- a
    End,
}

impl Op {
    pub fn is_jump(self) -> bool {
        matches!(self, Op::Jmp | Op::Jif | Op::Jany | Op::Jnone | Op::Jall)
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cond {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

#[derive(Clone, Copy, Debug)]
pub struct Operand {
    pub kind: Kind,
    pub ty: Type,
    pub neg: bool,
    pub idx: u16,
    pub imm: u32,
}

impl Operand {
    pub const NONE: Operand = Operand::new(Kind::None, Type::F, 0);

    const fn new(kind: Kind, ty: Type, idx: u16) -> Operand {
        Operand { kind, ty, neg: false, idx, imm: 0 }
    }
    /// uint view
    pub fn u(self) -> Operand {
        Operand { ty: Type::U, ..self }
    }
    /// float view
    pub fn f(self) -> Operand {
        Operand { ty: Type::F, ..self }
    }
    pub fn is_vector(&self) -> bool {
        matches!(self.kind, Kind::V | Kind::LaneX | Kind::LaneY)
    }
    pub fn same_reg(&self, o: &Operand) -> bool {
        self.kind == o.kind && self.idx == o.idx
    }

    /// emit `self = e`
    pub fn set(self, e: impl Into<Expr>) {
        emit_expr(self, e.into(), 0);
    }
    /// the lanes (or the scalar) of this register whose flag f is set
    pub fn at(self, f: Flag) -> Masked {
        Masked { r: self, f }
    }
}

impl From<f32> for Operand {
    fn from(f: f32) -> Operand {
        Operand { imm: f.to_bits(), ..Operand::new(Kind::Imm, Type::F, 0) }
    }
}
impl From<u32> for Operand {
    fn from(v: u32) -> Operand {
        Operand { imm: v, ..Operand::new(Kind::Imm, Type::U, 0) }
    }
}
impl Neg for Operand {
    type Output = Operand;
    fn neg(self) -> Operand {
        let mut o = self;
        if o.kind == Kind::Imm {
            o.imm ^= 0x8000_0000;
        } else {
            o.neg = !o.neg;
        }
        o
    }
}

pub const LANE_X: Operand = Operand::new(Kind::LaneX, Type::F, 0);
pub const LANE_Y: Operand = Operand::new(Kind::LaneY, Type::F, 0);
pub const GROUP_X: Operand = Operand::new(Kind::GroupX, Type::U, 0);
pub const GROUP_Y: Operand = Operand::new(Kind::GroupY, Type::U, 0);
pub fn uni(i: usize) -> Operand {
    Operand::new(Kind::U, Type::F, i as u16)
}
pub fn blk(block: usize, i: usize) -> Operand {
    Operand::new(Kind::B, Type::F, (block * 16 + i) as u16)
}
pub fn lane0(v: Operand) -> Operand {
    Operand::new(Kind::Lane0, v.ty, v.idx)
}
pub fn sreg(i: u16) -> Operand {
    Operand::new(Kind::S, Type::F, i)
}
pub fn vreg() -> Operand {
    with(|p| {
        p.nvec += 1;
        Operand::new(Kind::V, Type::F, p.nvec as u16 - 1)
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub op: Op,
    pub cond: Cond,
    pub flag: u8,
    pub pred: u8, // 0 none, 1 + f = only lanes with flag f set
    pub sat: bool,
    pub d: Operand,
    pub a: Operand,
    pub b: Operand,
    pub target: i32, // jumps: instruction index; Load: block; Store: slot
    pub off: u32,    // Load: extra offset in 16-byte units
}

impl Inst {
    fn new(op: Op) -> Inst {
        Inst { op, cond: Cond::Eq, flag: 0, pred: 0, sat: false, d: Operand::NONE, a: Operand::NONE, b: Operand::NONE, target: -1, off: 0 }
    }
}

#[derive(Default)]
pub struct Program {
    pub code: Vec<Inst>,
    pub nvec: usize, // vector registers allocated
}

// the program being built (see build)
thread_local!(static CUR: RefCell<Program> = RefCell::new(Program::default()));

fn with<R>(f: impl FnOnce(&mut Program) -> R) -> R {
    CUR.with(|c| f(&mut c.borrow_mut()))
}
fn emit(i: Inst) -> usize {
    with(|p| {
        p.code.push(i);
        p.code.len() - 1
    })
}
fn here_pos() -> usize {
    with(|p| p.code.len())
}

/// run `f`, which emits a kernel, and return the program it built
pub fn build(f: impl FnOnce()) -> Program {
    with(|p| *p = Program::default());
    f();
    with(std::mem::take)
}

/// one pending instruction: what an operator returns, emitted by `set`
#[derive(Clone, Copy)]
pub struct Expr {
    op: Op,
    a: Operand,
    b: Operand,
    sat: bool,
}

fn ex(op: Op, a: Operand, b: Operand) -> Expr {
    Expr { op, a, b, sat: false }
}
impl From<Operand> for Expr {
    fn from(o: Operand) -> Expr {
        ex(Op::Mov, o, Operand::NONE)
    }
}
impl From<f32> for Expr {
    fn from(v: f32) -> Expr {
        Operand::from(v).into()
    }
}
impl From<u32> for Expr {
    fn from(v: u32) -> Expr {
        Operand::from(v).into()
    }
}

macro_rules! binop {
    ($tr:ident, $m:ident, $op:expr) => {
        impl<T: Into<Operand>> $tr<T> for Operand {
            type Output = Expr;
            fn $m(self, b: T) -> Expr {
                ex($op, self, b.into())
            }
        }
    };
}
binop!(Add, add, Op::Add);
binop!(Mul, mul, Op::Mul);
binop!(BitAnd, bitand, Op::And);
binop!(BitOr, bitor, Op::Or);
binop!(BitXor, bitxor, Op::Xor);
binop!(Shl, shl, Op::Shl);
binop!(Shr, shr, Op::Shr);
impl<T: Into<Operand>> Sub<T> for Operand {
    type Output = Expr;
    fn sub(self, b: T) -> Expr {
        ex(Op::Add, self, -b.into())
    }
}

pub fn min(a: impl Into<Operand>, b: impl Into<Operand>) -> Expr {
    ex(Op::Min, a.into(), b.into())
}
pub fn max(a: impl Into<Operand>, b: impl Into<Operand>) -> Expr {
    ex(Op::Max, a.into(), b.into())
}
pub fn pow(a: impl Into<Operand>, b: impl Into<Operand>) -> Expr {
    ex(Op::Pow, a.into(), b.into())
}
pub fn floor(a: Operand) -> Expr {
    ex(Op::Floor, a, Operand::NONE)
}
pub fn frac(a: Operand) -> Expr {
    ex(Op::Frac, a, Operand::NONE)
}
pub fn inv(a: Operand) -> Expr {
    ex(Op::Inv, a, Operand::NONE)
}
pub fn sqrt(a: Operand) -> Expr {
    ex(Op::Sqrt, a, Operand::NONE)
}
pub fn rsq(a: Operand) -> Expr {
    ex(Op::Rsq, a, Operand::NONE)
}
pub fn exp2(a: Operand) -> Expr {
    ex(Op::Exp2, a, Operand::NONE)
}
/// clamp the result to [0, 1]
pub fn sat(e: Expr) -> Expr {
    Expr { sat: true, ..e }
}

pub struct Cmp {
    c: Cond,
    a: Operand,
    b: Operand,
}
macro_rules! cmpop {
    ($f:ident, $c:expr) => {
        #[allow(dead_code)]
        pub fn $f(a: impl Into<Operand>, b: impl Into<Operand>) -> Cmp {
            Cmp { c: $c, a: a.into(), b: b.into() }
        }
    };
}
cmpop!(eq, Cond::Eq);
cmpop!(ne, Cond::Ne);
cmpop!(gt, Cond::Gt);
cmpop!(ge, Cond::Ge);
cmpop!(lt, Cond::Lt);
cmpop!(le, Cond::Le);

#[derive(Clone, Copy)]
pub struct Flag(u8);
pub const F0: Flag = Flag(0);
pub const F1: Flag = Flag(1);

impl Flag {
    /// emit `flag = c`
    pub fn set(self, c: Cmp) {
        let mut i = Inst::new(Op::Cmp);
        i.cond = c.c;
        i.flag = self.0;
        i.a = c.a;
        i.b = c.b;
        emit(i);
    }
}

fn emit_expr(d: Operand, e: Expr, pred: u8) {
    let mut i = Inst::new(e.op);
    i.d = d;
    i.a = e.a;
    i.b = e.b;
    i.sat = e.sat;
    i.pred = pred;
    emit(i);
}

/// a register write restricted to the lanes (or the scalar) whose flag is set
pub struct Masked {
    r: Operand,
    f: Flag,
}
impl Masked {
    pub fn set(self, e: impl Into<Expr>) {
        emit_expr(self.r, e.into(), 1 + self.f.0);
    }
}

/// a jump target; it may be used before it is bound
#[derive(Default)]
pub struct Label {
    pos: Option<usize>,
    fixups: Vec<usize>,
}

impl Label {
    pub fn new() -> Label {
        Label::default()
    }
    pub fn bind(&mut self) {
        let pos = here_pos();
        self.pos = Some(pos);
        with(|p| {
            for &i in &self.fixups {
                p.code[i].target = pos as i32;
            }
        });
        self.fixups.clear();
    }
}
/// a label bound to the next instruction
pub fn here() -> Label {
    Label { pos: Some(here_pos()), fixups: Vec::new() }
}

fn jump(op: Op, f: Flag, l: &mut Label) {
    let mut i = Inst::new(op);
    i.flag = f.0;
    i.target = l.pos.map_or(-1, |p| p as i32);
    let at = emit(i);
    if l.pos.is_none() {
        l.fixups.push(at);
    }
}
pub fn jmp(l: &mut Label) {
    jump(Op::Jmp, F0, l);
}
/// scalar flag set
pub fn jmp_if(f: Flag, l: &mut Label) {
    jump(Op::Jif, f, l);
}
/// some lane has the flag
#[allow(dead_code)]
pub fn jmp_any(f: Flag, l: &mut Label) {
    jump(Op::Jany, f, l);
}
/// no lane has it
pub fn jmp_none(f: Flag, l: &mut Label) {
    jump(Op::Jnone, f, l);
}
/// every lane has it
pub fn jmp_all(f: Flag, l: &mut Label) {
    jump(Op::Jall, f, l);
}

pub fn load(block: usize, a: Operand, b: Operand, off: u32) {
    let mut i = Inst::new(Op::Load);
    i.a = a;
    i.b = b;
    i.target = block as i32;
    i.off = off;
    emit(i);
}
pub fn store(slot: usize, src: Operand) {
    let mut i = Inst::new(Op::Store);
    i.a = src;
    i.target = slot as i32;
    emit(i);
}
pub fn end() {
    emit(Inst::new(Op::End));
}

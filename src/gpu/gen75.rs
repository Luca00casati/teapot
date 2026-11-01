//! The Hardware layer for Intel Haswell (Gen7.5): encodes the kernel IR into
//! native 128-bit EU instructions and builds the GPGPU command stream
//! (PIPELINE_SELECT → STATE_BASE_ADDRESS → MEDIA_VFE_STATE → CURBE →
//! interface descriptor → GPGPU_WALKER). Memory and submission go through a
//! Driver (i915.rs).
//!
//! A packet is one SIMD16 hardware thread. Register map:
//!   r0 thread payload   r1-r4 uniforms (CURBE)   r5-r6 lane x, r7-r8 lane y (CURBE)
//!   r9 scalars 0-7, r17 scalars 8-15   r10, r125 load headers   r11-r16 blocks 0-2
//!   r18-r121 vector registers (2 GRFs each)   r122-r124 tile store   r127 end of thread
//!
//! Hardware findings this relies on (measured on an HD 4600):
//!   - jmpi distances are in bytes, relative to the next instruction
//!   - a send writes back no more GRFs than its execution size covers, so
//!     block loads are SIMD16
//!   - an inverted any16h predicate applies after the reduction ("no lane")

use std::rc::Rc;

use super::kir::{Cond, Kind, Op, Operand, Program, Type};
use super::layers::{Buffer, Driver, Hardware};

const RF_ARF: u8 = 0;
const RF_GRF: u8 = 1;
const RF_IMM: u8 = 3;
const TY_UD: u8 = 0;
const TY_D: u8 = 1;
const TY_UW: u8 = 2;
const TY_F: u8 = 7;
const OP_MOV: u32 = 1;
const OP_SEL: u32 = 2;
const OP_AND: u32 = 5;
const OP_OR: u32 = 6;
const OP_XOR: u32 = 7;
const OP_SHR: u32 = 8;
const OP_SHL: u32 = 9;
const OP_CMP: u32 = 16;
const OP_JMPI: u32 = 32;
const OP_SEND: u32 = 49;
const OP_MATH: u32 = 56;
const OP_ADD: u32 = 64;
const OP_MUL: u32 = 65;
const OP_FRC: u32 = 67;
const OP_RNDD: u32 = 69;
const CM_Z: u32 = 1;
const CM_NZ: u32 = 2;
const CM_G: u32 = 3;
const CM_GE: u32 = 4;
const CM_L: u32 = 5;
const CM_LE: u32 = 6;
const MF_INV: u32 = 1;
const MF_EXP: u32 = 3;
const MF_SQRT: u32 = 4;
const MF_RSQ: u32 = 5;
const MF_POW: u32 = 10;
const PR_NORMAL: u32 = 1;
const PR_ANY16H: u32 = 10;
const PR_ALL16H: u32 = 11;
const SFID_RENDER_CACHE: u32 = 5;
const SFID_THREAD_SPAWNER: u32 = 7;
const SFID_CONST_CACHE: u32 = 9;
const G_UNI: u8 = 1;
const G_LX: u8 = 5;
const G_LY: u8 = 7;
const G_SC0: u8 = 9;
const G_HDR: u8 = 10;
const G_BLK: u8 = 11;
const G_SC1: u8 = 17;
const G_VEC: usize = 18;
const G_VEC_END: usize = 122;
const G_ST: u8 = 122;
const G_HDR2: u8 = 125;
const G_EOT: u8 = 127;

/// one EU operand; the default is the null register
#[derive(Clone, Copy)]
struct Gop {
    file: u8,
    ty: u8,
    nr: u8,
    sub: u8,
    vs: u8,
    w: u8,
    hs: u8,
    neg: u8,
    imm: u32,
}

fn gnull(ty: u8) -> Gop {
    Gop { file: RF_ARF, ty, nr: 0, sub: 0, vs: 0, w: 1, hs: 0, neg: 0, imm: 0 }
}
/// SIMD16 vector in r:r+1
fn gvec(r: u8, ty: u8) -> Gop {
    Gop { file: RF_GRF, nr: r, vs: 8, w: 8, hs: 1, ..gnull(ty) }
}
/// dword c of r, as a scalar
fn gsc(r: u8, c: u8, ty: u8) -> Gop {
    Gop { file: RF_GRF, nr: r, sub: c * 4, vs: 0, w: 1, hs: 0, ..gnull(ty) }
}
fn gimm(v: u32, ty: u8) -> Gop {
    Gop { file: RF_IMM, w: 0, imm: v, ..gnull(ty) }
}

fn setb(inst: &mut [u32; 4], hi: u32, lo: u32, mut v: u32) {
    for b in lo..=hi {
        let (w, s) = ((b / 32) as usize, b % 32);
        inst[w] = (inst[w] & !(1 << s)) | ((v & 1) << s);
        v >>= 1;
    }
}
fn enc_vs(v: u8) -> u32 {
    match v { 0 => 0, 1 => 1, 2 => 2, 4 => 3, 8 => 4, 16 => 5, _ => 6 }
}
fn enc_w(w: u8) -> u32 {
    match w { 1 => 0, 2 => 1, 4 => 2, 8 => 3, _ => 4 }
}
fn enc_hs(h: u8) -> u32 {
    match h { 0 => 0, 1 => 1, 2 => 2, _ => 3 }
}
fn enc_es(e: u32) -> u32 {
    match e { 1 => 0, 2 => 1, 4 => 2, 8 => 3, _ => 4 }
}

/// modifiers of one EU instruction
#[derive(Clone, Copy)]
struct Mods {
    pred: u32,
    pinv: u32,
    flag: u32,
    cmod: u32,
    sat: u32,
    es: u32,
    nomask: u32,
}
const MODS: Mods = Mods { pred: 0, pinv: 0, flag: 0, cmod: 0, sat: 0, es: 16, nomask: 0 };
const SCALAR_MODS: Mods = Mods { es: 1, nomask: 1, ..MODS };
const HEADER_MODS: Mods = Mods { es: 8, nomask: 1, ..MODS };

#[derive(Default)]
struct Encoder {
    code: Vec<[u32; 4]>,
}

impl Encoder {
    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, opc: u32, m: Mods, d: Gop, s0: Gop, s1: Gop, nsrc: u32, func: u32) {
        let mut a = [0u32; 4];
        let i = &mut a;
        setb(i, 6, 0, opc);
        setb(i, 9, 9, m.nomask);
        setb(i, 19, 16, m.pred);
        setb(i, 20, 20, m.pinv);
        setb(i, 23, 21, enc_es(m.es));
        setb(i, 27, 24, if func != 0 { func } else { m.cmod });
        setb(i, 31, 31, m.sat);
        setb(i, 90, 90, m.flag >> 1); // f0 / f1
        setb(i, 89, 89, m.flag & 1); //  .0 / .1
        // destination (align1, direct)
        setb(i, 33, 32, d.file as u32);
        setb(i, 36, 34, d.ty as u32);
        setb(i, 52, 48, d.sub as u32);
        setb(i, 60, 53, d.nr as u32);
        setb(i, 62, 61, enc_hs(if d.hs != 0 { d.hs } else { 1 }));
        // src0
        setb(i, 38, 37, s0.file as u32);
        setb(i, 41, 39, s0.ty as u32);
        if s0.file == RF_IMM {
            i[3] = s0.imm;
            setb(i, 43, 42, RF_ARF as u32);
            setb(i, 46, 44, s0.ty as u32);
        } else {
            setb(i, 68, 64, s0.sub as u32);
            setb(i, 76, 69, s0.nr as u32);
            setb(i, 78, 78, s0.neg as u32);
            setb(i, 81, 80, enc_hs(s0.hs));
            setb(i, 84, 82, enc_w(s0.w));
            setb(i, 88, 85, enc_vs(s0.vs));
        }
        if nsrc == 2 {
            setb(i, 43, 42, s1.file as u32);
            setb(i, 46, 44, s1.ty as u32);
            if s1.file == RF_IMM {
                i[3] = s1.imm;
            } else {
                setb(i, 100, 96, s1.sub as u32);
                setb(i, 108, 101, s1.nr as u32);
                setb(i, 110, 110, s1.neg as u32);
                setb(i, 113, 112, enc_hs(s1.hs));
                setb(i, 116, 114, enc_w(s1.w));
                setb(i, 120, 117, enc_vs(s1.vs));
            }
        }
        self.code.push(a);
    }
    fn op1(&mut self, opc: u32, m: Mods, d: Gop, s: Gop) {
        self.emit(opc, m, d, s, gnull(TY_UD), 1, 0);
    }
    fn op2(&mut self, opc: u32, m: Mods, d: Gop, a: Gop, b: Gop) {
        self.emit(opc, m, d, a, b, 2, 0);
    }
    fn mov(&mut self, m: Mods, d: Gop, s: Gop) {
        self.op1(OP_MOV, m, d, s);
    }
    fn send(&mut self, sfid: u32, dst: Gop, src: u8, desc: u32) {
        self.emit(OP_SEND, Mods { nomask: 1, ..MODS }, dst, gvec(src, TY_UD), gimm(desc, TY_UD), 2, sfid);
    }
}

fn gen_op(o: &Operand) -> Gop {
    let ty = if o.ty == Type::F { TY_F } else { TY_UD };
    let i = o.idx as usize;
    let g = match o.kind {
        Kind::V => gvec((G_VEC + 2 * i) as u8, ty),
        Kind::S => gsc(if i < 8 { G_SC0 } else { G_SC1 }, (i % 8) as u8, ty),
        Kind::U => gsc(G_UNI + (i / 8) as u8, (i % 8) as u8, ty),
        Kind::B => gsc(G_BLK + (i / 8) as u8, (i % 8) as u8, ty),
        Kind::Lane0 => gsc((G_VEC + 2 * i) as u8, 0, ty),
        Kind::LaneX => gvec(G_LX, ty),
        Kind::LaneY => gvec(G_LY, ty),
        Kind::GroupX => gsc(0, 1, ty),
        Kind::GroupY => gsc(0, 6, ty),
        Kind::Imm => return gimm(o.imm, ty),
        Kind::None => return gnull(ty),
    };
    Gop { neg: o.neg as u8, ..g }
}

struct Gen75 {
    enc: Encoder,
    threads: u32,
    drv: Option<Rc<dyn Driver>>,
    batch: Option<Buffer>,
    batch_len: u32,
}

pub fn probe(id: u32) -> Option<(Box<dyn Hardware>, String)> {
    let fam = id & 0xff00;
    if fam != 0x0400 && fam != 0x0a00 && fam != 0x0c00 && fam != 0x0d00 {
        return None;
    }
    let gt = (id >> 4) & 3;
    let hw = Gen75 { enc: Encoder::default(), threads: 70 << gt, drv: None, batch: None, batch_len: 0 };
    Some((Box::new(hw), format!("Haswell GT{} [8086:{:04x}]", gt + 1, id)))
}

// batch layout: commands at 0, state from 4096
const OFF_CURBE: usize = 4096;
const OFF_IDD: usize = 4096 + 512;
const OFF_BT: usize = 4096 + 1024;
const OFF_SS: usize = 4096 + 1088;
const OFF_KERNEL: usize = 8192;

impl Hardware for Gen75 {
    fn name(&self) -> &'static str {
        "gen75"
    }

    /* --------------------------------------------- IR → EU code --------- */

    fn compile(&mut self, p: &Program) -> bool {
        let alu = |op: Op| match op {
            Op::Mov => OP_MOV,
            Op::Add => OP_ADD,
            Op::Mul => OP_MUL,
            Op::And => OP_AND,
            Op::Or => OP_OR,
            Op::Xor => OP_XOR,
            Op::Shl => OP_SHL,
            Op::Shr => OP_SHR,
            Op::Floor => OP_RNDD,
            _ => OP_FRC,
        };
        let mathfn = |op: Op| match op {
            Op::Inv => MF_INV,
            Op::Sqrt => MF_SQRT,
            Op::Rsq => MF_RSQ,
            Op::Exp2 => MF_EXP,
            _ => MF_POW,
        };
        let cmod = |c: Cond| match c {
            Cond::Eq => CM_Z,
            Cond::Ne => CM_NZ,
            Cond::Gt => CM_G,
            Cond::Ge => CM_GE,
            Cond::Lt => CM_L,
            Cond::Le => CM_LE,
        };
        if G_VEC + 2 * p.nvec > G_VEC_END {
            eprintln!("gen75: kernel needs {} vector registers, only {} fit", p.nvec, (G_VEC_END - G_VEC) / 2);
            return false;
        }
        let e = &mut self.enc;
        e.code.clear();
        let mut at = vec![0usize; p.code.len() + 1]; // first EU instruction of each IR instruction
        let mut hdr_toggle = false;
        // load headers start as a copy of the thread payload; only .2 changes
        e.mov(HEADER_MODS, gvec(G_HDR, TY_UD), gvec(0, TY_UD));
        e.mov(HEADER_MODS, gvec(G_HDR2, TY_UD), gvec(0, TY_UD));

        for (i, k) in p.code.iter().enumerate() {
            at[i] = e.code.len();
            let scalar = if k.op == Op::Cmp { !k.a.is_vector() && !k.b.is_vector() } else { k.d.kind == Kind::S };
            if scalar && (k.a.is_vector() || k.b.is_vector()) {
                eprintln!("gen75: IR {i}: vector source in a scalar op");
                return false;
            }
            let (d, a, mut b) = (gen_op(&k.d), gen_op(&k.a), gen_op(&k.b));
            let mut m = if scalar { SCALAR_MODS } else { MODS };
            if k.pred != 0 {
                m.pred = PR_NORMAL;
                m.flag = (k.pred as u32 - 1) * 2;
            }
            m.sat = k.sat as u32;
            match k.op {
                Op::Mov | Op::Floor | Op::Frac => e.op1(alu(k.op), m, d, a),
                Op::Mul => {
                    // 32x16 integer multiply: a uint immediate of 16 bits is encoded as UW
                    if b.file == RF_IMM && b.ty == TY_UD {
                        if b.imm > 0xffff {
                            eprintln!("gen75: IR {i}: 32-bit integer multiply");
                            return false;
                        }
                        b.ty = TY_UW;
                        b.imm |= b.imm << 16;
                    }
                    e.op2(OP_MUL, m, d, a, b);
                }
                Op::Min | Op::Max => {
                    m.cmod = if k.op == Op::Min { CM_L } else { CM_GE };
                    e.op2(OP_SEL, m, d, a, b);
                }
                Op::Add | Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr => e.op2(alu(k.op), m, d, a, b),
                Op::Inv | Op::Sqrt | Op::Rsq | Op::Exp2 => e.emit(OP_MATH, m, d, a, gnull(TY_F), 2, mathfn(k.op)),
                Op::Pow => e.emit(OP_MATH, m, d, a, b, 2, MF_POW),
                Op::Cmp => {
                    m.cmod = cmod(k.cond);
                    m.flag = k.flag as u32 * 2;
                    e.op2(OP_CMP, m, gnull(a.ty), a, b);
                }
                Op::Jmp | Op::Jif | Op::Jany | Op::Jnone | Op::Jall => {
                    let ip = Gop { file: RF_ARF, ty: TY_UD, nr: 0xA0, sub: 0, vs: 4, w: 1, hs: 0, neg: 0, imm: 0 };
                    let pred = match k.op {
                        Op::Jmp => 0,
                        Op::Jif => PR_NORMAL,
                        Op::Jall => PR_ALL16H,
                        _ => PR_ANY16H,
                    };
                    let j = Mods { flag: k.flag as u32 * 2, pred, pinv: (k.op == Op::Jnone) as u32, ..SCALAR_MODS };
                    e.op2(OP_JMPI, j, ip, ip, gimm(0, TY_D));
                }
                Op::Load => {
                    // constant-cache OWord block read of 4 owords from binding table entry 0
                    let h = if hdr_toggle { G_HDR2 } else { G_HDR };
                    hdr_toggle = !hdr_toggle;
                    e.op2(OP_ADD, SCALAR_MODS, gsc(h, 2, TY_UD), a, b);
                    if k.off != 0 {
                        e.op2(OP_ADD, SCALAR_MODS, gsc(h, 2, TY_UD), gsc(h, 2, TY_UD), gimm(k.off, TY_UD));
                    }
                    e.send(SFID_CONST_CACHE, gvec(G_BLK + 2 * k.target as u8, TY_UW), h, (1 << 25) | (2 << 20) | (1 << 19) | (3 << 8));
                }
                Op::Store => {
                    // media block write of the 4x4 tile: 16 bytes wide, 4 rows
                    e.mov(HEADER_MODS, gvec(G_ST, TY_UD), gvec(0, TY_UD));
                    e.op2(OP_SHL, SCALAR_MODS, gsc(G_ST, 0, TY_UD), gsc(0, 1, TY_UD), gimm(4, TY_UD));
                    e.op2(OP_SHL, SCALAR_MODS, gsc(G_ST, 1, TY_UD), gsc(0, 6, TY_UD), gimm(2, TY_UD));
                    e.mov(SCALAR_MODS, gsc(G_ST, 2, TY_UD), gimm((3 << 16) | 15, TY_UD));
                    let raw = Gop { ty: TY_UD, ..a }; // raw bits, no conversion
                    e.mov(Mods { nomask: 1, ..MODS }, gvec(G_ST + 1, TY_UD), raw);
                    e.send(SFID_RENDER_CACHE, gnull(TY_UD), G_ST, (3 << 25) | (1 << 19) | (10 << 14) | k.target as u32);
                }
                Op::End => {
                    e.mov(HEADER_MODS, gvec(G_EOT, TY_UD), gvec(0, TY_UD));
                    e.send(SFID_THREAD_SPAWNER, gnull(TY_UD), G_EOT, 0x8200_0010);
                }
            }
        }
        at[p.code.len()] = e.code.len();
        for (i, k) in p.code.iter().enumerate() {
            if !k.op.is_jump() {
                continue;
            }
            if k.target < 0 {
                eprintln!("gen75: IR {i}: jump to an unbound label");
                return false;
            }
            let from = at[i];
            e.code[from][3] = (16 * (at[k.target as usize] as i64 - from as i64 - 1)) as u32;
        }
        // raw EU binary to this path, for disassemblers / diffs (TEAPOT_DUMP_KERNEL=path cargo build)
        if let Some(path) = option_env!("TEAPOT_DUMP_KERNEL") {
            let bytes: Vec<u8> = e.code.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
            if let Err(err) = std::fs::write(path, bytes) {
                eprintln!("{path}: {err}");
            }
        }
        true
    }

    /* --------------------------------------------- command stream ------- */

    fn setup(&mut self, drv: &Rc<dyn Driver>, tiles_x: u32, tiles_y: u32, bufs: &[Buffer; 3]) -> bool {
        let Some(batch) = Buffer::alloc(drv, 65536) else { return false };
        drv.sync(&batch, true);
        let mem = batch.words();
        mem.fill(0);
        let kwords = self.enc.code.len() * 4;
        if (OFF_KERNEL / 4 + kwords) * 4 > batch.size as usize {
            eprintln!("gen75: kernel too large");
            return false;
        }
        for (dst, src) in mem[OFF_KERNEL / 4..].chunks_mut(4).zip(&self.enc.code) {
            dst.copy_from_slice(src);
        }

        // CURBE: 4 GRFs of uniforms (written per run), then the lane positions
        let lane = &mut mem[OFF_CURBE / 4 + 32..][..32];
        for i in 0..16 {
            lane[i] = ((i % 4) as f32).to_bits();
            lane[16 + i] = ((i / 4) as f32).to_bits();
        }

        // interface descriptor
        let idd = &mut mem[OFF_IDD / 4..][..8];
        idd[0] = OFF_KERNEL as u32;
        idd[1] = 1 << 18; // single program flow, IEEE float
        idd[3] = OFF_BT as u32; // binding table pointer (32-byte aligned)
        idd[4] = 8 << 16; // CURBE read length 8 GRFs
        idd[5] = 1; // 1 thread per group

        // binding table: 0 scene (raw buffer), 1 and 2 linear 2D images of dwords
        let (w, h) = (tiles_x * 4, tiles_y * 4);
        for (i, buf) in bufs.iter().enumerate() {
            let ss_off = (OFF_SS + 64 * i) as u32;
            mem[OFF_BT / 4 + i] = ss_off;
            let ss = &mut mem[ss_off as usize / 4..][..8];
            if i == 0 {
                let sz = buf.size as u32 - 1;
                ss[0] = (4 << 29) | (0x1FF << 18); // SURFTYPE_BUFFER, RAW
                ss[2] = ((sz >> 7) & 0x3fff) << 16 | (sz & 0x7f);
                ss[3] = ((sz >> 21) & 0x3ff) << 21;
            } else {
                ss[0] = (1 << 29) | (0x0C0 << 18) | (1 << 8); // SURFTYPE_2D, B8G8R8A8_UNORM, linear
                ss[2] = ((h - 1) << 16) | (w - 1);
                ss[3] = w * 4 - 1;
            }
            ss[7] = (4 << 25) | (5 << 22) | (6 << 19) | (7 << 16);
            drv.reloc(&batch, ss_off + 4, buf, 0, i != 0);
        }

        let mut c: Vec<u32> = Vec::new();
        let mut self_relocs = Vec::new(); // dwords holding the batch's own address
        let mut reloc_self = |c: &mut Vec<u32>| {
            self_relocs.push(c.len());
            c.push(1);
        };
        c.push((3 << 29) | (1 << 27) | (1 << 24) | (4 << 16) | 2); // PIPELINE_SELECT GPGPU
        c.push((3 << 29) | (1 << 24) | (1 << 16) | 8); // STATE_BASE_ADDRESS
        c.push(0);
        reloc_self(&mut c); // surface
        reloc_self(&mut c); // dynamic
        c.push(0); // indirect
        reloc_self(&mut c); // instruction
        c.extend([0, 1, 0, 1]);
        c.push((3 << 29) | (2 << 27) | 6); // MEDIA_VFE_STATE
        c.push(0);
        c.push(((self.threads - 1) << 16) | (1 << 2));
        c.push(0);
        c.push(8); // CURBE: 8 x 256 bits
        c.extend([0, 0, 0]);
        c.push((3 << 29) | (2 << 27) | (1 << 16) | 2); // MEDIA_CURBE_LOAD
        c.extend([0, 256, OFF_CURBE as u32]);
        c.push((3 << 29) | (2 << 27) | (2 << 16) | 2); // MEDIA_INTERFACE_DESCRIPTOR_LOAD
        c.extend([0, 32, OFF_IDD as u32]);
        c.push((3 << 29) | (2 << 27) | (1 << 24) | (5 << 16) | 9); // GPGPU_WALKER
        c.push(0);
        c.push(1 << 30); // SIMD16, 1x1x1 threads
        c.extend([0, tiles_x]); // thread group X range
        c.extend([0, tiles_y]); // thread group Y range
        c.extend([0, 1]);
        c.extend([0xffff, 0xffff_ffff]);
        c.push(0x0A << 23); // MI_BATCH_BUFFER_END
        c.push(0);
        mem[..c.len()].copy_from_slice(&c);
        for off in self_relocs {
            drv.reloc(&batch, off as u32 * 4, &batch, 1, false);
        }
        self.batch_len = c.len() as u32 * 4;
        self.drv = Some(drv.clone());
        self.batch = Some(batch);
        true
    }

    fn run(&mut self, uniforms: &[u32; 32]) -> bool {
        let (Some(drv), Some(batch)) = (&self.drv, &self.batch) else { return false };
        drv.sync(batch, true); // the CURBE lives in the batch: waits for the GPU
        batch.words()[OFF_CURBE / 4..][..32].copy_from_slice(uniforms);
        drv.exec(batch, self.batch_len)
    }
}

; kir.lsp — the kernel IR: a tiny SIMD16 vector instruction list that the ray
; tracer (rt-kernel.lsp) is written in once, and that each GPU's hardware
; layer (gen75.lsp, ...) encodes into machine code.
;
; Execution model: one program instance runs per *packet* of 16 lanes, which
; is one 4x4 pixel tile (lane i covers pixel (i % 4, i / 4) of the tile).
; Control flow is uniform for the whole packet; per-lane work is done by
; predication on a flag. Registers are 32-bit and untyped; operands are typed
; views (float or uint) of them, and an op whose destination type differs
; from its source type converts (float <-> uint).
;
;   (vreg)        a new vector register, one value per lane
;   (sreg i)      scalar register i, one value for the packet
;   (uni i)       uniform input (32 dwords written by the host per dispatch)
;   (blk k i)     dword i (0..15) of block k, filled by load (read-only)
;   (lane0 v)     lane 0 of vector register v, as a scalar
;   lane-x lane-y     per-lane pixel position inside the tile (float 0..3)
;   group-x group-y   tile coordinates of the packet (uint)
;   (as-u o) (as-f o) the uint / float view of a register
;   f32 numbers are float immediates, integers uint immediates
;
; An instruction is scalar when its destination is a scalar (or, for a
; compare, when no source is a vector). Memory is reached only through load
; (64 bytes of buffer 0 into a block, address in 16-byte units) and store
; (a packet's 16 dwords into the 4x4 tile of image buffer 1 or 2).
;
; One form is exactly one instruction, so the kernel author controls
; scheduling (GPU cores issue in order):
;
;   (<- t0 (* a b))     (<- t0 (+ t0 c))     (<- t0 (sat (* t0 t1)))
;   (<- f0 (<= t6 t9))  (<- (at ht f0) -1.0) ; predicated on flag 0
;
; Operators build an `expr`, which is not an operand, so a compound
; expression such as (+ (* a b) c) does not compile.

(c-include "stdlib.h")
(import "prelude" :all)

; operand kinds
(const K-NONE 0) (const K-V 1) (const K-S 2) (const K-U 3) (const K-B 4) (const K-LANE0 5)
(const K-LANEX 6) (const K-LANEY 7) (const K-GROUPX 8) (const K-GROUPY 9) (const K-IMM 10)
; operand types
(const TY-F 0) (const TY-U 1)
; ops
(const OP-MOV 0) (const OP-ADD 1) (const OP-MUL 2) (const OP-MIN 3) (const OP-MAX 4)
(const OP-AND 5) (const OP-OR 6) (const OP-XOR 7) (const OP-SHL 8) (const OP-SHR 9)
(const OP-FLOOR 10) (const OP-FRAC 11) (const OP-INV 12) (const OP-SQRT 13) (const OP-RSQ 14)
(const OP-EXP2 15) (const OP-POW 16)
(const OP-CMP 17)                        ; flag = a <cond> b, per lane (or scalar)
(const OP-JMP 18)                        ; always
(const OP-JIF 19)                        ; scalar flag set
(const OP-JANY 20)                       ; some lane has the flag
(const OP-JNONE 21)                      ; no lane has it
(const OP-JALL 22)                       ; every lane has it
(const OP-LOAD 23)                       ; block <- buf0[(a + b + off) * 16 .. +64]
(const OP-STORE 24)                      ; image buffer `slot`, this packet's tile <- a
(const OP-END 25)
; compare conditions
(const C-EQ 0) (const C-NE 1) (const C-GT 2) (const C-GE 3) (const C-LT 4) (const C-LE 5)

(struct operand (kind u8) (ty u8) (neg bool) (idx u16) (imm u32))

(def none operand (make operand))
(def lane-x operand (make operand :kind K-LANEX :ty TY-F))
(def lane-y operand (make operand :kind K-LANEY :ty TY-F))
(def group-x operand (make operand :kind K-GROUPX :ty TY-U))
(def group-y operand (make operand :kind K-GROUPY :ty TY-U))

(defn- opnd ((kind u8) (ty u8) (idx u16)) operand (make operand :kind kind :ty ty :idx idx))
(defn uni ((i i32)) operand (opnd K-U TY-F (cast u16 i)))
(defn blk ((block i32) (i i32)) operand (opnd K-B TY-F (cast u16 (+ (* block 16) i))))
(defn lane0 ((v operand)) operand (opnd K-LANE0 v.ty v.idx))
(defn sreg ((i i32)) operand (opnd K-S TY-F (cast u16 i)))
(defn as-u ((o operand)) operand (let ((r o)) (set r.ty TY-U) r))
(defn as-f ((o operand)) operand (let ((r o)) (set r.ty TY-F) r))
(defn fimm ((v f32)) operand (make operand :kind K-IMM :ty TY-F :imm (bitcast u32 v)))
(defn uimm ((v u32)) operand (make operand :kind K-IMM :ty TY-U :imm v))
(defn is-vector ((o operand)) bool (or (= o.kind K-V) (= o.kind K-LANEX) (= o.kind K-LANEY)))
(defn same-reg ((a operand) (b operand)) bool (and (= a.kind b.kind) (= a.idx b.idx)))

(defop - ((o operand)) operand
  (let ((r o))
    (if (= r.kind K-IMM) (set r.imm (^ r.imm 0x80000000)) (set r.neg (not r.neg)))
    r))

;; ------------------------------------------------------------ program ---

(struct inst
  (op u8) (cond u8) (flag u8)
  (pred u8)                              ; 0 none, 1 + f = only lanes with flag f set
  (sat bool) (d operand) (a operand) (b operand)
  (target i32)                           ; jumps: instruction index; LOAD: block; STORE: slot
  (off u32))                             ; LOAD: extra offset in 16-byte units

(struct program (code (ptr inst)) (n u32) (cap u32) (nvec u32))

(def prog program)                       ; the program being built

(defn- new-inst ((op u8)) inst (make inst :op op :target -1))

(defn- emit ((i inst)) u32
  (when (= prog.n prog.cap)
    (set prog.cap (if (= prog.cap 0) 1024 (* prog.cap 2)))
    (set prog.code (cast (ptr inst) (realloc prog.code (* (sizeof inst) (cast u64 prog.cap))))))
  (set (aref prog.code prog.n) i)
  (+= prog.n 1)
  (- prog.n 1))

(defn vreg () operand
  (+= prog.nvec 1)
  (opnd K-V TY-F (cast u16 (- prog.nvec 1))))

;; -------------------------------------------------------- expressions ---

; one pending instruction: what an operator returns, emitted by <-
(struct expr (op u8) (a operand) (b operand) (sat bool))

(defn- ex ((op u8) (a operand) (b operand)) expr (make expr :op op :a a :b b))
(defn- mov ((a operand)) expr (ex OP-MOV a none))

; an operand from an operand or a number (f32 → float immediate, integer → uint)
(defmacro- arg (x)
  (let ((t (type-of x)))
    (cond ((eq t 'operand) x)
          ((eq (type-kind t) 'float) `(fimm ,x))
          (else `(uimm ,x)))))

; the binary operators, with an operand or a number on the right
(defmacro- kir-binop (op code)
  `(do (defop ,op ((a operand) (b operand)) expr (ex ,code a b))
       (defop ,op ((a operand) (b f32)) expr (ex ,code a (fimm b)))
       (defop ,op ((a operand) (b u32)) expr (ex ,code a (uimm b)))))
(kir-binop + OP-ADD)
(kir-binop * OP-MUL)
(kir-binop & OP-AND)
(kir-binop | OP-OR)
(kir-binop ^ OP-XOR)
(kir-binop << OP-SHL)
(kir-binop >> OP-SHR)
(defop - ((a operand) (b operand)) expr (ex OP-ADD a (- b)))
(defop - ((a operand) (b f32)) expr (ex OP-ADD a (- (fimm b))))

(defmacro min (a b) `(ex OP-MIN (arg ,a) (arg ,b)))
(defmacro max (a b) `(ex OP-MAX (arg ,a) (arg ,b)))
(defmacro pow (a b) `(ex OP-POW (arg ,a) (arg ,b)))
(defn floor ((a operand)) expr (ex OP-FLOOR a none))
(defn frac ((a operand)) expr (ex OP-FRAC a none))
(defn inv ((a operand)) expr (ex OP-INV a none))
(defn sqrt ((a operand)) expr (ex OP-SQRT a none))
(defn rsq ((a operand)) expr (ex OP-RSQ a none))
(defn exp2 ((a operand)) expr (ex OP-EXP2 a none))
(defn sat ((e expr)) expr (let ((r e)) (set r.sat true) r))   ; clamp the result to [0, 1]

;; ----------------------------------------------------- compares, flags ---

(struct cmp (cond u8) (a operand) (b operand))
(struct flag (n u8))
(def f0 flag (make flag :n 0))
(def f1 flag (make flag :n 1))

(defmacro- kir-cmpop (op c)
  `(do (defop ,op ((a operand) (b operand)) cmp (make cmp :cond ,c :a a :b b))
       (defop ,op ((a operand) (b f32)) cmp (make cmp :cond ,c :a a :b (fimm b)))
       (defop ,op ((a operand) (b u32)) cmp (make cmp :cond ,c :a a :b (uimm b)))))
(kir-cmpop = C-EQ)
(kir-cmpop != C-NE)
(kir-cmpop > C-GT)
(kir-cmpop >= C-GE)
(kir-cmpop < C-LT)
(kir-cmpop <= C-LE)

;; ------------------------------------------------------------ emitting ---

(defn- emit-op ((d operand) (e expr) (pred u8)) void
  (let ((i (new-inst e.op)))
    (set i.d d)
    (set i.a e.a)
    (set i.b e.b)
    (set i.sat e.sat)
    (set i.pred pred)
    (emit i)))

(defn- emit-cmp ((fl flag) (c cmp)) void
  (let ((i (new-inst OP-CMP)))
    (set i.cond c.cond)
    (set i.flag fl.n)
    (set i.a c.a)
    (set i.b c.b)
    (emit i)))

; (<- dst e): one instruction. dst is a register, (at reg flag) for a write
; predicated on the flag, or a flag when e is a compare. e is an operator
; form, an operand, or a number.
(defmacro <- (d e)
  (let ((t (type-of e))
        (masked (and (list? d) (eq (first d) 'at)))
        (dst (if masked (nth d 1) d))
        (pred (if masked `(+ 1 (. ,(nth d 2) n)) 0)))
    (cond ((eq t 'cmp) `(emit-cmp ,d ,e))
          ((eq t 'expr) `(emit-op ,dst ,e ,pred))
          (else `(emit-op ,dst (mov (arg ,e)) ,pred)))))

;; -------------------------------------------------------------- labels ---

; a jump target; it may be used before it is bound
(struct label (pos i32) (fix (ptr u32)) (nfix u32))

(defn new-label () label (make label :pos -1))
(defn here () label (make label :pos (cast i32 prog.n)))   ; bound to the next instruction

(defn- bind-label ((l (ptr label))) void
  (set l.pos (cast i32 prog.n))
  (for (i u32 0 l.nfix) (set (. (aref prog.code (aref l.fix i)) target) l.pos))
  (set l.nfix 0))

(defn- kjump ((op u8) (fl flag) (l (ptr label))) void
  (let ((i (new-inst op)))
    (set i.flag fl.n)
    (set i.target l.pos)
    (let ((at (emit i)))
      (when (< l.pos 0)
        (set l.fix (cast (ptr u32) (realloc l.fix (* 4 (cast u64 (+ l.nfix 1))))))
        (set (aref l.fix l.nfix) at)
        (+= l.nfix 1)))))

(defmacro bind! (l) `(bind-label (addr ,l)))
(defmacro jmp (l) `(kjump OP-JMP f0 (addr ,l)))
(defmacro jmp-if (fl l) `(kjump OP-JIF ,fl (addr ,l)))       ; scalar flag set
(defmacro jmp-any (fl l) `(kjump OP-JANY ,fl (addr ,l)))     ; some lane has the flag
(defmacro jmp-none (fl l) `(kjump OP-JNONE ,fl (addr ,l)))   ; no lane has it
(defmacro jmp-all (fl l) `(kjump OP-JALL ,fl (addr ,l)))     ; every lane has it

;; -------------------------------------------------------------- memory ---

(defn load ((block i32) (a operand) (b operand) (off u32)) void
  (let ((i (new-inst OP-LOAD)))
    (set i.a a)
    (set i.b b)
    (set i.target block)
    (set i.off off)
    (emit i)))

(defn store ((slot i32) (src operand)) void
  (let ((i (new-inst OP-STORE)))
    (set i.a src)
    (set i.target slot)
    (emit i)))

(defn end () void (emit (new-inst OP-END)))

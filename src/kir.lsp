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

(import "libc")
(import "prelude")
(using prelude)

;; the IR's vocabulary: enums, so an opcode can't be mixed up with a kind
(enum okind u8 NONE V S U B LANE0 LANEX LANEY GROUPX GROUPY IMM)   ; operand kinds
(enum otype u8 F U)                                                ; operand types: float, uint
(enum opcode u8
  MOV ADD MUL MIN MAX AND OR XOR SHL SHR FLOOR FRAC INV SQRT RSQ EXP2 POW
  CMP                                    ; flag = a <cond> b, per lane (or scalar)
  JMP                                    ; always
  JIF                                    ; scalar flag set
  JANY                                   ; some lane has the flag
  JNONE                                  ; no lane has it
  JALL                                   ; every lane has it
  LOAD                                   ; block <- buf0[(a + b + off) * 16 .. +64]
  STORE                                  ; image buffer `slot`, this packet's tile <- a
  END)
(enum cmpc u8 EQ NE GT GE LT LE)         ; compare conditions

(struct operand (kind okind) (ty otype) (neg bool) (idx u16) (imm u32))

(def none operand (make operand))
(def lane-x operand (make operand :kind okind.LANEX :ty otype.F))
(def lane-y operand (make operand :kind okind.LANEY :ty otype.F))
(def group-x operand (make operand :kind okind.GROUPX :ty otype.U))
(def group-y operand (make operand :kind okind.GROUPY :ty otype.U))

(defn- opnd ((kind okind) (ty otype) (idx u16)) operand (make operand :kind kind :ty ty :idx idx))
(defn uni ((i i32)) operand (opnd okind.U otype.F (cast u16 i)))
(defn blk ((block i32) (i i32)) operand (opnd okind.B otype.F (cast u16 (+ (* block 16) i))))
(defn lane0 ((v operand)) operand (opnd okind.LANE0 v.ty v.idx))
(defn sreg ((i i32)) operand (opnd okind.S otype.F (cast u16 i)))
(defn as-u ((o operand)) operand (let ((r o)) (set r.ty otype.U) r))
(defn as-f ((o operand)) operand (let ((r o)) (set r.ty otype.F) r))
(defn fimm ((v f32)) operand (make operand :kind okind.IMM :ty otype.F :imm (bitcast u32 v)))
(defn uimm ((v u32)) operand (make operand :kind okind.IMM :ty otype.U :imm v))
(defn is-vector ((o operand)) bool (or (= o.kind okind.V) (= o.kind okind.LANEX) (= o.kind okind.LANEY)))
(defn same-reg ((a operand) (b operand)) bool (and (= a.kind b.kind) (= a.idx b.idx)))

(defop - ((o operand)) operand
  (let ((r o))
    (if (= r.kind okind.IMM) (set r.imm (^ r.imm 0x80000000)) (set r.neg (not r.neg)))
    r))

;; ------------------------------------------------------------ program ---

(struct inst
  (op opcode) (cond cmpc) (flag u8)
  (pred u8)                              ; 0 none, 1 + f = only lanes with flag f set
  (sat bool) (d operand) (a operand) (b operand)
  (target i32)                           ; jumps: instruction index; LOAD: block; STORE: slot
  (off u32))                             ; LOAD: extra offset in 16-byte units

(struct program (code (ptr inst)) (n u32) (cap u32) (nvec u32))

(def prog program)                       ; the program being built

(defn- new-inst ((op opcode)) inst (make inst :op op :target -1))

(defn- emit ((i inst)) u32
  (when (= prog.n prog.cap)
    (set prog.cap (if (= prog.cap 0) 1024 (* prog.cap 2)))
    (set prog.code (cast (ptr inst) (libc::realloc prog.code (* (sizeof inst) (cast u64 prog.cap))))))
  (set (aref prog.code prog.n) i)
  (+= prog.n 1)
  (- prog.n 1))

(defn vreg () operand
  (+= prog.nvec 1)
  (opnd okind.V otype.F (cast u16 (- prog.nvec 1))))

;; -------------------------------------------------------- expressions ---

; one pending instruction: what an operator returns, emitted by <-
(struct expr (op opcode) (a operand) (b operand) (sat bool))

(defn- ex ((op opcode) (a operand) (b operand)) expr (make expr :op op :a a :b b))
(defn- mov ((a operand)) expr (ex opcode.MOV a none))

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
(kir-binop + opcode.ADD)
(kir-binop * opcode.MUL)
(kir-binop & opcode.AND)
(kir-binop | opcode.OR)
(kir-binop ^ opcode.XOR)
(kir-binop << opcode.SHL)
(kir-binop >> opcode.SHR)
(defop - ((a operand) (b operand)) expr (ex opcode.ADD a (- b)))
(defop - ((a operand) (b f32)) expr (ex opcode.ADD a (- (fimm b))))

(defmacro min (a b) `(ex opcode.MIN (arg ,a) (arg ,b)))
(defmacro max (a b) `(ex opcode.MAX (arg ,a) (arg ,b)))
(defmacro pow (a b) `(ex opcode.POW (arg ,a) (arg ,b)))
(defn floor ((a operand)) expr (ex opcode.FLOOR a none))
(defn frac ((a operand)) expr (ex opcode.FRAC a none))
(defn inv ((a operand)) expr (ex opcode.INV a none))
(defn sqrt ((a operand)) expr (ex opcode.SQRT a none))
(defn rsq ((a operand)) expr (ex opcode.RSQ a none))
(defn exp2 ((a operand)) expr (ex opcode.EXP2 a none))
(defn sat ((e expr)) expr (let ((r e)) (set r.sat true) r))   ; clamp the result to [0, 1]

;; ----------------------------------------------------- compares, flags ---

(struct cmp (cond cmpc) (a operand) (b operand))
(struct flag (n u8))
(def f0 flag (make flag :n 0))
(def f1 flag (make flag :n 1))

(defmacro- kir-cmpop (op c)
  `(do (defop ,op ((a operand) (b operand)) cmp (make cmp :cond ,c :a a :b b))
       (defop ,op ((a operand) (b f32)) cmp (make cmp :cond ,c :a a :b (fimm b)))
       (defop ,op ((a operand) (b u32)) cmp (make cmp :cond ,c :a a :b (uimm b)))))
(kir-cmpop = cmpc.EQ)
(kir-cmpop != cmpc.NE)
(kir-cmpop > cmpc.GT)
(kir-cmpop >= cmpc.GE)
(kir-cmpop < cmpc.LT)
(kir-cmpop <= cmpc.LE)

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
  (let ((i (new-inst opcode.CMP)))
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

(defn- kjump ((op opcode) (fl flag) (l (ptr label))) void
  (let ((i (new-inst op)))
    (set i.flag fl.n)
    (set i.target l.pos)
    (let ((at (emit i)))
      (when (< l.pos 0)
        (set l.fix (cast (ptr u32) (libc::realloc l.fix (* 4 (cast u64 (+ l.nfix 1))))))
        (set (aref l.fix l.nfix) at)
        (+= l.nfix 1)))))

(defmacro bind! (l) `(bind-label (addr ,l)))
(defmacro jmp (l) `(kjump opcode.JMP f0 (addr ,l)))
(defmacro jmp-if (fl l) `(kjump opcode.JIF ,fl (addr ,l)))       ; scalar flag set
(defmacro jmp-any (fl l) `(kjump opcode.JANY ,fl (addr ,l)))     ; some lane has the flag
(defmacro jmp-none (fl l) `(kjump opcode.JNONE ,fl (addr ,l)))   ; no lane has it
(defmacro jmp-all (fl l) `(kjump opcode.JALL ,fl (addr ,l)))     ; every lane has it

;; -------------------------------------------------------------- memory ---

(defn load ((block i32) (a operand) (b operand) (off u32)) void
  (let ((i (new-inst opcode.LOAD)))
    (set i.a a)
    (set i.b b)
    (set i.target block)
    (set i.off off)
    (emit i)))

(defn store ((slot i32) (src operand)) void
  (let ((i (new-inst opcode.STORE)))
    (set i.a src)
    (set i.target slot)
    (emit i)))

(defn end () void (emit (new-inst opcode.END)))

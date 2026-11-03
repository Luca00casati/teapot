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

(struct operand [kind okind] [ty otype] [neg bool] [idx u16] [imm u32])

(def none operand {operand})
(def lane-x operand {operand :kind okind.LANEX :ty otype.F})
(def lane-y operand {operand :kind okind.LANEY :ty otype.F})
(def group-x operand {operand :kind okind.GROUPX :ty otype.U})
(def group-y operand {operand :kind okind.GROUPY :ty otype.U})

(private
  (defn opnd [kind okind] [ty otype] [idx u16] operand {operand :kind kind :ty ty :idx idx}))
(defn uni [i i32] operand (opnd okind.U otype.F (cast u16 i)))
(defn blk [block i32] [i i32] operand (opnd okind.B otype.F (cast u16 (+ (* block 16) i))))
(defn lane0 [v operand] operand (opnd okind.LANE0 v.ty v.idx))
(defn sreg [i i32] operand (opnd okind.S otype.F (cast u16 i)))
(defn as-u [o operand] operand (let [r o] (set r.ty otype.U) r))
(defn as-f [o operand] operand (let [r o] (set r.ty otype.F) r))
(defn fimm [v f32] operand {operand :kind okind.IMM :ty otype.F :imm (bitcast u32 v)})
(defn uimm [v u32] operand {operand :kind okind.IMM :ty otype.U :imm v})
(defn is-vector [o operand] bool (or (= o.kind okind.V) (= o.kind okind.LANEX) (= o.kind okind.LANEY)))
(defn same-reg [a operand] [b operand] bool (and (= a.kind b.kind) (= a.idx b.idx)))

(defop - [o operand] operand
  (let [r o]
    (if (= r.kind okind.IMM) (set r.imm (^ r.imm 0x80000000)) (set r.neg (not r.neg)))
    r))

;; ------------------------------------------------------------ program ---

(struct inst
  [op opcode] [cond cmpc] [flag u8]
  [pred u8]                              ; 0 none, 1 + f = only lanes with flag f set
  [sat bool] [d operand] [a operand] [b operand]
  [target i32]                           ; jumps: instruction index; LOAD: block; STORE: slot
  [off u32])                             ; LOAD: extra offset in 16-byte units

(struct program [code (ptr inst)] [n u32] [cap u32] [nvec u32])

(def prog program)                       ; the program being built

(private
  (defn new-inst [op opcode] inst {inst :op op :target -1})

  (defn emit [i inst] u32 :noinline
    (when (= prog.n prog.cap)
      (set prog.cap (if (= prog.cap 0) 1024 (* prog.cap 2)))
      (set prog.code (cast (ptr inst) (libc::realloc prog.code (* (sizeof inst) (cast u64 prog.cap))))))
    (set prog.code[prog.n] i)
    (+= prog.n 1)
    (- prog.n 1)))

(defn vreg [] operand
  (+= prog.nvec 1)
  (opnd okind.V otype.F (cast u16 (- prog.nvec 1))))

;; -------------------------------------------------------- expressions ---

; one pending instruction: what an operator returns, emitted by <-
(struct expr [op opcode] [a operand] [b operand] [sat bool])

(private
  (defn ex [op opcode] [a operand] [b operand] expr {expr :op op :a a :b b})
  (defn mov [a operand] expr (ex opcode.MOV a none))

  ; the binary operators, with an operand or a number (f32: a float immediate,
  ; an integer: a uint one) on the right
  (defmacro def-binop! (op code)
    `(do (defop ,op [a operand] [b operand] expr (ex ,code a b))
         (defop ,op [a operand] [b f32] expr (ex ,code a (fimm b)))
         (defop ,op [a operand] [b u32] expr (ex ,code a (uimm b))))))
(def-binop! + opcode.ADD)
(def-binop! * opcode.MUL)
(def-binop! & opcode.AND)
(def-binop! | opcode.OR)
(def-binop! ^ opcode.XOR)
(def-binop! << opcode.SHL)
(def-binop! >> opcode.SHR)
(defop - [a operand] [b operand] expr (ex opcode.ADD a (- b)))
(defop - [a operand] [b f32] expr (ex opcode.ADD a (- (fimm b))))

(def-binop! min opcode.MIN)
(def-binop! max opcode.MAX)
(def-binop! pow opcode.POW)
(defn floor [a operand] expr (ex opcode.FLOOR a none))
(defn frac [a operand] expr (ex opcode.FRAC a none))
(defn inv [a operand] expr (ex opcode.INV a none))
(defn sqrt [a operand] expr (ex opcode.SQRT a none))
(defn rsq [a operand] expr (ex opcode.RSQ a none))
(defn exp2 [a operand] expr (ex opcode.EXP2 a none))
(defn sat [e expr] expr (let [r e] (set r.sat true) r))   ; clamp the result to [0, 1]

;; ----------------------------------------------------- compares, flags ---

(struct cmp [cond cmpc] [a operand] [b operand])
(struct flag [n u8])
(def f0 flag {flag :n 0})
(def f1 flag {flag :n 1})

(private
  (defmacro def-cmpop! (op c)
    `(do (defop ,op [a operand] [b operand] cmp {cmp :cond ,c :a a :b b})
         (defop ,op [a operand] [b f32] cmp {cmp :cond ,c :a a :b (fimm b)})
         (defop ,op [a operand] [b u32] cmp {cmp :cond ,c :a a :b (uimm b)}))))
(def-cmpop! = cmpc.EQ)
(def-cmpop! != cmpc.NE)
(def-cmpop! > cmpc.GT)
(def-cmpop! >= cmpc.GE)
(def-cmpop! < cmpc.LT)
(def-cmpop! <= cmpc.LE)

;; ------------------------------------------------------------ emitting ---
; (the emitters are :noinline: the kernel is built once, at startup, and
; inlining them into every (<- ...) only makes the builder huge to compile)

(private
  (defn emit-op [d operand] [e expr] [pred u8] void :noinline
    (let [i (new-inst e.op)]
      (set i.d d)
      (set i.a e.a)
      (set i.b e.b)
      (set i.sat e.sat)
      (set i.pred pred)
      (emit i)))

  (defn emit-mov [d operand] [a operand] [pred u8] void :noinline
    (let [i (new-inst opcode.MOV)]
      (set i.d d)
      (set i.a a)
      (set i.pred pred)
      (emit i)))

  (defn emit-cmp [fl flag] [c cmp] void :noinline
    (let [i (new-inst opcode.CMP)]
      (set i.cond c.cond)
      (set i.flag fl.n)
      (set i.a c.a)
      (set i.b c.b)
      (emit i))))

; (<- dst e): one instruction. dst is a register, (at reg flag) for a write
; predicated on the flag, or a flag when e is a compare. e is an operator's
; expr, an operand, or a number.
(struct dest [d operand] [pred u8])     ; 0 every lane, 1 + f the lanes with flag f
(defn at [d operand] [fl flag] dest {dest :d d :pred (+ 1 fl.n)})

(defop <- [d operand] [e expr] void (emit-op d e 0))
(defop <- [d operand] [a operand] void (emit-mov d a 0))
(defop <- [d operand] [x f32] void (emit-mov d (fimm x) 0))
(defop <- [d operand] [x u32] void (emit-mov d (uimm x) 0))
(defop <- [d dest] [e expr] void (emit-op d.d e d.pred))
(defop <- [d dest] [a operand] void (emit-mov d.d a d.pred))
(defop <- [d dest] [x f32] void (emit-mov d.d (fimm x) d.pred))
(defop <- [d dest] [x u32] void (emit-mov d.d (uimm x) d.pred))
(defop <- [fl flag] [c cmp] void (emit-cmp fl c))

;; -------------------------------------------------------------- labels ---

; a jump target; it may be used before it is bound
(struct label [pos i32] [fix (ptr u32)] [nfix u32])

(private
  (defn make-label [pos i32] (ptr label)       ; (kept for the whole build: there are few)
    (let [l (cast (ptr label) (libc::calloc 1 (sizeof label)))]
      (set l.pos pos)
      l)))
(defn new-label [] (ptr label) (make-label -1))
(defn here [] (ptr label) (make-label (cast i32 prog.n)))   ; bound to the next instruction

(defn bind [l (ptr label)] void
  (set l.pos (cast i32 prog.n))
  (for [i u32 0 l.nfix] (set (. prog.code[l.fix[i]] target) l.pos))
  (set l.nfix 0))

(private
  (defn kjump [op opcode] [fl flag] [l (ptr label)] void :noinline
    (let [i (new-inst op)]
      (set i.flag fl.n)
      (set i.target l.pos)
      (let [at (emit i)]
        (when (< l.pos 0)
          (set l.fix (cast (ptr u32) (libc::realloc l.fix (* 4 (cast u64 (+ l.nfix 1))))))
          (set l.fix[l.nfix] at)
          (+= l.nfix 1))))))

(defn jmp [l (ptr label)] void (kjump opcode.JMP f0 l))
(defn jmp-if [fl flag] [l (ptr label)] void (kjump opcode.JIF fl l))       ; scalar flag set
(defn jmp-any [fl flag] [l (ptr label)] void (kjump opcode.JANY fl l))     ; some lane has the flag
(defn jmp-none [fl flag] [l (ptr label)] void (kjump opcode.JNONE fl l))   ; no lane has it
(defn jmp-all [fl flag] [l (ptr label)] void (kjump opcode.JALL fl l))     ; every lane has it

;; -------------------------------------------------------------- memory ---

(defn load [block i32] [a operand] [b operand] [off u32] void :noinline
  (let [i (new-inst opcode.LOAD)]
    (set i.a a)
    (set i.b b)
    (set i.target block)
    (set i.off off)
    (emit i)))

(defn store [slot i32] [src operand] void :noinline
  (let [i (new-inst opcode.STORE)]
    (set i.a src)
    (set i.target slot)
    (emit i)))

(defn end [] void (emit (new-inst opcode.END)))

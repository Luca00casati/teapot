; gen75.lsp — the hardware layer for Intel Haswell (Gen7.5): encodes the
; kernel IR into native 128-bit EU instructions and builds the GPGPU command
; stream (PIPELINE_SELECT → STATE_BASE_ADDRESS → MEDIA_VFE_STATE → CURBE →
; interface descriptor → GPGPU_WALKER). Memory and submission go through a
; driver (i915.lsp).
;
; A packet is one SIMD16 hardware thread. Register map:
;   r0 thread payload   r1-r4 uniforms (CURBE)   r5-r6 lane x, r7-r8 lane y (CURBE)
;   r9 scalars 0-7, r17 scalars 8-15   r10, r125 load headers   r11-r16 blocks 0-2
;   r18-r121 vector registers (2 GRFs each)   r122-r124 tile store   r127 end of thread
;
; Hardware findings this relies on (measured on an HD 4600):
;   - jmpi distances are in bytes, relative to the next instruction
;   - a send writes back no more GRFs than its execution size covers, so
;     block loads are SIMD16
;   - an inverted any16h predicate applies after the reduction ("no lane")

(c-include "stdio.h" "stdlib.h" "string.h")
(import "prelude" :all)
(import "kir" :all)
(import "layers" :all)

(const RF-ARF 0) (const RF-GRF 1) (const RF-IMM 3)
(const TY-UD 0) (const TY-D 1) (const TY-UW 2) (const TY-W 3) (const TY-F 7)
(const EU-MOV 1) (const EU-SEL 2) (const EU-AND 5) (const EU-OR 6) (const EU-XOR 7) (const EU-SHR 8)
(const EU-SHL 9) (const EU-CMP 16) (const EU-JMPI 32) (const EU-SEND 49) (const EU-MATH 56)
(const EU-ADD 64) (const EU-MUL 65) (const EU-FRC 67) (const EU-RNDD 69)
(const CM-Z 1) (const CM-NZ 2) (const CM-G 3) (const CM-GE 4) (const CM-L 5) (const CM-LE 6)
(const MF-INV 1) (const MF-EXP 3) (const MF-SQRT 4) (const MF-RSQ 5) (const MF-POW 10)
(const PR-NORMAL 1) (const PR-ANY16H 10) (const PR-ALL16H 11)
(const SFID-RENDER-CACHE 5) (const SFID-THREAD-SPAWNER 7) (const SFID-CONST-CACHE 9)
(const G-UNI 1) (const G-LX 5) (const G-LY 7) (const G-SC0 9) (const G-HDR 10) (const G-BLK 11)
(const G-SC1 17) (const G-VEC 18) (const G-VEC-END 122) (const G-ST 122) (const G-HDR2 125) (const G-EOT 127)

; one EU operand; (gnull ty) is the null register
(struct- gop (file u32) (ty u32) (nr u32) (sub u32) (vs u32) (w u32) (hs u32) (neg u32) (imm u32))

(defn- gnull ((ty u32)) gop (make gop :file RF-ARF :ty ty :w 1))
(defn- gvec ((r u32) (ty u32)) gop (make gop :file RF-GRF :ty ty :nr r :vs 8 :w 8 :hs 1))   ; SIMD16 in r:r+1
(defn- gsc ((r u32) (c u32) (ty u32)) gop (make gop :file RF-GRF :ty ty :nr r :sub (* c 4) :w 1)) ; dword c of r
(defn- gimm ((v u32) (ty u32)) gop (make gop :file RF-IMM :ty ty :imm v))

; bits lo..hi of the 128-bit instruction in[0..3] = v
(defn- setb ((in (ptr u32)) (hi u32) (lo u32) (v u32)) void
  (let ((x v))
    (for (b lo (+ hi 1))
      (let ((w (/ b 32)) (s (% b 32)))
        (set (aref in w) (| (& (aref in w) (~ (<< 1 s))) (<< (& x 1) s)))
        (set x (>> x 1))))))

(defn- enc-vs ((v u32)) u32 (cond ((= v 0) 0) ((= v 1) 1) ((= v 2) 2) ((= v 4) 3) ((= v 8) 4) ((= v 16) 5) (else 6)))
(defn- enc-w ((w u32)) u32 (cond ((= w 1) 0) ((= w 2) 1) ((= w 4) 2) ((= w 8) 3) (else 4)))
(defn- enc-hs ((h u32)) u32 (cond ((= h 0) 0) ((= h 1) 1) ((= h 2) 2) (else 3)))
(defn- enc-es ((e u32)) u32 (cond ((= e 1) 0) ((= e 2) 1) ((= e 4) 2) ((= e 8) 3) (else 4)))

; modifiers of one EU instruction
(struct- mods (pred u32) (pinv u32) (flag u32) (cmod u32) (sat u32) (es u32) (nomask u32))
(def- MODS mods (make mods :es 16))
(def- SCALAR-MODS mods (make mods :es 1 :nomask 1))
(def- HEADER-MODS mods (make mods :es 8 :nomask 1))

; the machine code: 4 dwords per instruction
(struct- encoder (code (ptr u32)) (n u32) (cap u32))
(def enc encoder)

(defn- eu-emit ((opc u32) (m mods) (d gop) (s0 gop) (s1 gop) (nsrc u32) (func u32)) void
  (when (= enc.n enc.cap)
    (set enc.cap (if (= enc.cap 0) 4096 (* enc.cap 2)))
    (set enc.code (cast (ptr u32) (realloc enc.code (* 16 (cast u64 enc.cap))))))
  (let ((i (+ enc.code (* enc.n 4))))
    (memset i 0 16)
    (setb i 6 0 opc)
    (setb i 9 9 m.nomask)
    (setb i 19 16 m.pred)
    (setb i 20 20 m.pinv)
    (setb i 23 21 (enc-es m.es))
    (setb i 27 24 (if (!= func 0) func m.cmod))
    (setb i 31 31 m.sat)
    (setb i 90 90 (>> m.flag 1))        ; f0 / f1
    (setb i 89 89 (& m.flag 1))         ; .0 / .1
    ; destination (align1, direct)
    (setb i 33 32 d.file)
    (setb i 36 34 d.ty)
    (setb i 52 48 d.sub)
    (setb i 60 53 d.nr)
    (setb i 62 61 (enc-hs (if (!= d.hs 0) d.hs 1)))
    ; src0
    (setb i 38 37 s0.file)
    (setb i 41 39 s0.ty)
    (if (= s0.file RF-IMM)
        (do (set (aref i 3) s0.imm)
            (setb i 43 42 RF-ARF)
            (setb i 46 44 s0.ty))
        (do (setb i 68 64 s0.sub)
            (setb i 76 69 s0.nr)
            (setb i 78 78 s0.neg)
            (setb i 81 80 (enc-hs s0.hs))
            (setb i 84 82 (enc-w s0.w))
            (setb i 88 85 (enc-vs s0.vs))))
    (when (= nsrc 2)
      (setb i 43 42 s1.file)
      (setb i 46 44 s1.ty)
      (if (= s1.file RF-IMM)
          (set (aref i 3) s1.imm)
          (do (setb i 100 96 s1.sub)
              (setb i 108 101 s1.nr)
              (setb i 110 110 s1.neg)
              (setb i 113 112 (enc-hs s1.hs))
              (setb i 116 114 (enc-w s1.w))
              (setb i 120 117 (enc-vs s1.vs)))))
    (+= enc.n 1)))

(defn- op1 ((opc u32) (m mods) (d gop) (s gop)) void (eu-emit opc m d s (gnull TY-UD) 1 0))
(defn- op2 ((opc u32) (m mods) (d gop) (a gop) (b gop)) void (eu-emit opc m d a b 2 0))
(defn- eu-mov ((m mods) (d gop) (s gop)) void (op1 EU-MOV m d s))
(defn- eu-send ((sfid u32) (dst gop) (src u32) (desc u32)) void
  (let ((m MODS)) (set m.nomask 1) (eu-emit EU-SEND m dst (gvec src TY-UD) (gimm desc TY-UD) 2 sfid)))

(defn- gen-op ((o operand)) gop
  (let ((ty u32 (if (= o.ty kir::TY-F) TY-F TY-UD)) (i (cast u32 o.idx)) (g (gnull ty)))
    (cond ((= o.kind K-V) (set g (gvec (+ G-VEC (* 2 i)) ty)))
          ((= o.kind K-S) (set g (gsc (if (< i 8) G-SC0 G-SC1) (% i 8) ty)))
          ((= o.kind K-U) (set g (gsc (+ G-UNI (/ i 8)) (% i 8) ty)))
          ((= o.kind K-B) (set g (gsc (+ G-BLK (/ i 8)) (% i 8) ty)))
          ((= o.kind K-LANE0) (set g (gsc (+ G-VEC (* 2 i)) 0 ty)))
          ((= o.kind K-LANEX) (set g (gvec G-LX ty)))
          ((= o.kind K-LANEY) (set g (gvec G-LY ty)))
          ((= o.kind K-GROUPX) (set g (gsc 0 1 ty)))
          ((= o.kind K-GROUPY) (set g (gsc 0 6 ty)))
          ((= o.kind K-IMM) (return (gimm o.imm ty)))
          (else (return g)))
    (set g.neg (cast u32 o.neg))
    g))

(defn- alu-op ((op u8)) u32
  (cond ((= op OP-MOV) EU-MOV) ((= op OP-ADD) EU-ADD) ((= op OP-MUL) EU-MUL) ((= op OP-AND) EU-AND)
        ((= op OP-OR) EU-OR) ((= op OP-XOR) EU-XOR) ((= op OP-SHL) EU-SHL) ((= op OP-SHR) EU-SHR)
        ((= op OP-FLOOR) EU-RNDD) (else EU-FRC)))
(defn- math-fn ((op u8)) u32
  (cond ((= op OP-INV) MF-INV) ((= op OP-SQRT) MF-SQRT) ((= op OP-RSQ) MF-RSQ) ((= op OP-EXP2) MF-EXP) (else MF-POW)))
(defn- cond-mod ((c u8)) u32
  (cond ((= c C-EQ) CM-Z) ((= c C-NE) CM-NZ) ((= c C-GT) CM-G) ((= c C-GE) CM-GE) ((= c C-LT) CM-L) (else CM-LE)))
(defn- is-jump ((op u8)) bool (and (>= op OP-JMP) (<= op OP-JALL)))

;; ------------------------------------------------------- IR → EU code ---

(defn compile ((p (ptr program))) bool
  (when (> (+ G-VEC (* 2 p.nvec)) G-VEC-END)
    (fprintf stderr "gen75: kernel needs %u vector registers, only %d fit\n" p.nvec (/ (- G-VEC-END G-VEC) 2))
    (return false))
  (set enc.n 0)
  (let ((at (cast (ptr u32) (malloc (* 4 (+ (cast u64 p.n) 1)))))  ; first EU instruction of each IR instruction
        (hdr-toggle false))
    ; load headers start as a copy of the thread payload; only .2 changes
    (eu-mov HEADER-MODS (gvec G-HDR TY-UD) (gvec 0 TY-UD))
    (eu-mov HEADER-MODS (gvec G-HDR2 TY-UD) (gvec 0 TY-UD))
    (for (i u32 0 p.n)
      (let ((k (addr (aref p.code i)))
            (scalar (if (= k.op OP-CMP) (and (not (is-vector k.a)) (not (is-vector k.b))) (= k.d.kind K-S)))
            (d (gen-op k.d)) (a (gen-op k.a)) (b (gen-op k.b))
            (m (if scalar SCALAR-MODS MODS)))
        (set (aref at i) enc.n)
        (when (and scalar (or (is-vector k.a) (is-vector k.b)))
          (fprintf stderr "gen75: IR %u: vector source in a scalar op\n" i)
          (return false))
        (when (!= k.pred 0)
          (set m.pred PR-NORMAL)
          (set m.flag (* (- (cast u32 k.pred) 1) 2)))
        (set m.sat (cast u32 k.sat))
        (cond
          ((or (= k.op OP-MOV) (= k.op OP-FLOOR) (= k.op OP-FRAC)) (op1 (alu-op k.op) m d a))
          ((= k.op OP-MUL)
           ; 32x16 integer multiply: a uint immediate of 16 bits is encoded as UW
           (when (and (= b.file RF-IMM) (= b.ty TY-UD))
             (when (> b.imm 0xffff)
               (fprintf stderr "gen75: IR %u: 32-bit integer multiply\n" i)
               (return false))
             (set b.ty TY-UW)
             (set b.imm (| b.imm (<< b.imm 16))))
           (op2 EU-MUL m d a b))
          ((or (= k.op OP-MIN) (= k.op OP-MAX))
           (set m.cmod (if (= k.op OP-MIN) CM-L CM-GE))
           (op2 EU-SEL m d a b))
          ((or (= k.op OP-ADD) (= k.op OP-AND) (= k.op OP-OR) (= k.op OP-XOR) (= k.op OP-SHL) (= k.op OP-SHR))
           (op2 (alu-op k.op) m d a b))
          ((or (= k.op OP-INV) (= k.op OP-SQRT) (= k.op OP-RSQ) (= k.op OP-EXP2))
           (eu-emit EU-MATH m d a (gnull TY-F) 2 (math-fn k.op)))
          ((= k.op OP-POW) (eu-emit EU-MATH m d a b 2 MF-POW))
          ((= k.op OP-CMP)
           (set m.cmod (cond-mod k.cond))
           (set m.flag (* (cast u32 k.flag) 2))
           (op2 EU-CMP m (gnull a.ty) a b))
          ((is-jump k.op)
           (let ((ip (make gop :file RF-ARF :ty TY-UD :nr 0xA0 :vs 4 :w 1)) (j SCALAR-MODS))
             (set j.flag (* (cast u32 k.flag) 2))
             (set j.pred (cond ((= k.op OP-JMP) 0) ((= k.op OP-JIF) PR-NORMAL) ((= k.op OP-JALL) PR-ALL16H) (else PR-ANY16H)))
             (set j.pinv (cast u32 (= k.op OP-JNONE)))
             (op2 EU-JMPI j ip ip (gimm 0 TY-D))))
          ((= k.op OP-LOAD)
           ; constant-cache OWord block read of 4 owords from binding table entry 0
           (let ((h u32 (if hdr-toggle G-HDR2 G-HDR)))
             (set hdr-toggle (not hdr-toggle))
             (op2 EU-ADD SCALAR-MODS (gsc h 2 TY-UD) a b)
             (when (!= k.off 0)
               (op2 EU-ADD SCALAR-MODS (gsc h 2 TY-UD) (gsc h 2 TY-UD) (gimm k.off TY-UD)))
             (eu-send SFID-CONST-CACHE (gvec (+ G-BLK (* 2 (cast u32 k.target))) TY-UW) h
                      (| (<< 1 25) (<< 2 20) (<< 1 19) (<< 3 8)))))
          ((= k.op OP-STORE)
           ; media block write of the 4x4 tile: 16 bytes wide, 4 rows
           (eu-mov HEADER-MODS (gvec G-ST TY-UD) (gvec 0 TY-UD))
           (op2 EU-SHL SCALAR-MODS (gsc G-ST 0 TY-UD) (gsc 0 1 TY-UD) (gimm 4 TY-UD))
           (op2 EU-SHL SCALAR-MODS (gsc G-ST 1 TY-UD) (gsc 0 6 TY-UD) (gimm 2 TY-UD))
           (eu-mov SCALAR-MODS (gsc G-ST 2 TY-UD) (gimm (| (<< 3 16) 15) TY-UD))
           (let ((raw a) (all MODS))
             (set raw.ty TY-UD)                 ; raw bits, no conversion
             (set all.nomask 1)
             (eu-mov all (gvec (+ G-ST 1) TY-UD) raw))
           (eu-send SFID-RENDER-CACHE (gnull TY-UD) G-ST (| (<< 3 25) (<< 1 19) (<< 10 14) (cast u32 k.target))))
          (else                                 ; END
           (eu-mov HEADER-MODS (gvec G-EOT TY-UD) (gvec 0 TY-UD))
           (eu-send SFID-THREAD-SPAWNER (gnull TY-UD) G-EOT 0x82000010)))))
    (set (aref at p.n) enc.n)
    (for (i u32 0 p.n)
      (let ((k (addr (aref p.code i))))
        (when (is-jump k.op)
          (when (< k.target 0)
            (fprintf stderr "gen75: IR %u: jump to an unbound label\n" i)
            (return false))
          (let ((from (aref at i)))
            (set (aref enc.code (+ (* from 4) 3)) (* 16 (- (- (aref at k.target) from) 1)))))))
    (free at))
  ; raw EU binary to this path, for disassemblers / diffs (lisc -DTEAPOT_DUMP_KERNEL=k.bin)
  (if-defined TEAPOT_DUMP_KERNEL
    (let ((f (fopen TEAPOT_DUMP_KERNEL "wb")))
      (if (= f null)
          (perror TEAPOT_DUMP_KERNEL)
          (do (fwrite enc.code 16 (cast u64 enc.n) f) (fclose f)))))
  true)

;; ------------------------------------------------------ command stream ---

(struct- gen75 (threads u32) (drv (ptr driver)) (batch buffer) (batch-len u32))
(defn- state-of ((h (ptr hardware))) (ptr gen75) (cast (ptr gen75) h.state))

; batch layout: commands at 0, state from 4096
(const OFF-CURBE 4096) (const OFF-IDD (+ 4096 512)) (const OFF-BT (+ 4096 1024))
(const OFF-SS (+ 4096 1088)) (const OFF-KERNEL 8192)

; the command stream being written: dwords, and where the batch's own address goes
(struct- cmdbuf (mem (ptr u32)) (n u32) (self (array u32 3)) (nself u32))
(defn- put ((cb (ptr cmdbuf)) (x u32)) void
  (set (aref cb.mem cb.n) x)
  (+= cb.n 1))
(defmacro- out (cb &rest xs) `(do ,@(map (lambda (x) `(put ,cb ,x)) xs)))
(defn- reloc-self ((cb (ptr cmdbuf))) void   ; relocated to the batch itself in setup
  (set (aref cb.self cb.nself) cb.n)
  (+= cb.nself 1)
  (put cb 1))

(defn- hw-compile ((h (ptr hardware)) (p (ptr program))) bool (compile p))

(defn- setup ((h (ptr hardware)) (drv (ptr driver)) (tiles-x u32) (tiles-y u32) (bufs (ptr buffer))) bool
  (let ((g (state-of h)) (batch (addr g.batch)))
    (unless (drv.alloc drv 65536 batch) (return false))
    (drv.sync drv batch true)
    (memset batch.map 0 batch.size)
    (let ((mem (cast (ptr u32) batch.map)))
      (when (> (+ OFF-KERNEL (* 16 (cast u64 enc.n))) batch.size)
        (fprintf stderr "gen75: kernel too large\n")
        (return false))
      (memcpy (+ batch.map OFF-KERNEL) enc.code (* 16 (cast u64 enc.n)))

      ; CURBE: 4 GRFs of uniforms (written per run), then the lane positions
      (for (i u32 0 16)
        (set (aref mem (+ (/ OFF-CURBE 4) 32 i)) (bitcast u32 (cast f32 (% i 4))))
        (set (aref mem (+ (/ OFF-CURBE 4) 48 i)) (bitcast u32 (cast f32 (/ i 4)))))

      ; interface descriptor
      (let ((idd (+ mem (/ OFF-IDD 4))))
        (set (aref idd 0) OFF-KERNEL)
        (set (aref idd 1) (<< 1 18))     ; single program flow, IEEE float
        (set (aref idd 3) OFF-BT)        ; binding table pointer (32-byte aligned)
        (set (aref idd 4) (<< 8 16))     ; CURBE read length 8 GRFs
        (set (aref idd 5) 1))            ; 1 thread per group

      ; binding table: 0 scene (raw buffer), 1 and 2 linear 2D images of dwords
      (let ((w (* tiles-x 4)) (hh (* tiles-y 4)))
        (for (i u32 0 3)
          (let ((ss-off (+ OFF-SS (* 64 i))) (ss (+ mem (/ ss-off 4))))
            (set (aref mem (+ (/ OFF-BT 4) i)) ss-off)
            (if (= i 0)
                (let ((sz (- (cast u32 (. (aref bufs 0) size)) 1)))
                  (set (aref ss 0) (| (<< 4 29) (<< 0x1FF 18)))           ; SURFTYPE_BUFFER, RAW
                  (set (aref ss 2) (| (<< (& (>> sz 7) 0x3fff) 16) (& sz 0x7f)))
                  (set (aref ss 3) (<< (& (>> sz 21) 0x3ff) 21)))
                (do
                  (set (aref ss 0) (| (<< 1 29) (<< 0x0C0 18) (<< 1 8)))  ; SURFTYPE_2D, B8G8R8A8_UNORM, linear
                  (set (aref ss 2) (| (<< (- hh 1) 16) (- w 1)))
                  (set (aref ss 3) (- (* w 4) 1))))
            (set (aref ss 7) (| (<< 4 25) (<< 5 22) (<< 6 19) (<< 7 16)))
            (drv.reloc drv batch (+ ss-off 4) (+ bufs i) 0 (!= i 0)))))

      (let ((cb (make cmdbuf :mem mem)) (k (addr cb)))
        (out k (| (<< 3 29) (<< 1 27) (<< 1 24) (<< 4 16) 2))    ; PIPELINE_SELECT GPGPU
        (out k (| (<< 3 29) (<< 1 24) (<< 1 16) 8))              ; STATE_BASE_ADDRESS
        (out k 0)
        (reloc-self k)                                           ; surface
        (reloc-self k)                                           ; dynamic
        (out k 0)                                                ; indirect
        (reloc-self k)                                           ; instruction
        (out k 0 1 0 1)
        (out k (| (<< 3 29) (<< 2 27) 6))                        ; MEDIA_VFE_STATE
        (out k 0 (| (<< (- g.threads 1) 16) (<< 1 2)) 0)
        (out k 8)                                                ; CURBE: 8 x 256 bits
        (out k 0 0 0)
        (out k (| (<< 3 29) (<< 2 27) (<< 1 16) 2))              ; MEDIA_CURBE_LOAD
        (out k 0 256 OFF-CURBE)
        (out k (| (<< 3 29) (<< 2 27) (<< 2 16) 2))              ; MEDIA_INTERFACE_DESCRIPTOR_LOAD
        (out k 0 32 OFF-IDD)
        (out k (| (<< 3 29) (<< 2 27) (<< 1 24) (<< 5 16) 9))    ; GPGPU_WALKER
        (out k 0)
        (out k (<< 1 30))                                        ; SIMD16, 1x1x1 threads
        (out k 0 tiles-x)                                        ; thread group X range
        (out k 0 tiles-y)                                        ; thread group Y range
        (out k 0 1)
        (out k 0xffff 0xffffffff)
        (out k (<< 0x0A 23))                                     ; MI_BATCH_BUFFER_END
        (out k 0)
        (for (i u32 0 cb.nself) (drv.reloc drv batch (* (aref cb.self i) 4) batch 1 false))
        (set g.batch-len (* cb.n 4))))
    (set g.drv drv)
    true))

(defn- run ((h (ptr hardware)) (uniforms (ptr u32))) bool
  (let ((g (state-of h)) (batch (addr g.batch)))
    (g.drv.sync g.drv batch true)        ; the CURBE lives in the batch: waits for the GPU
    (memcpy (+ batch.map OFF-CURBE) uniforms (* 32 4))
    (g.drv.exec g.drv batch g.batch-len)))

; does this layer drive the chip? A description goes to desc[64]
(defn probe ((id u32) (desc (ptr i8))) (ptr hardware)
  (let ((fam (& id 0xff00)))
    (unless (or (= fam 0x0400) (= fam 0x0a00) (= fam 0x0c00) (= fam 0x0d00)) (return null))
    (let ((gt (& (>> id 4) 3))
          (g (cast (ptr gen75) (calloc 1 (sizeof gen75))))
          (h (cast (ptr hardware) (calloc 1 (sizeof hardware)))))
      (set g.threads (<< 70 gt))
      (snprintf desc 64 "Haswell GT%u [8086:%04x]" (+ gt 1) id)
      (set (deref h) (make hardware :name "gen75" :state g :compile hw-compile :setup setup :run run))
      h)))

; rt-kernel.lsp — the teapot ray tracer written once in the kernel IR. It
; knows nothing about any particular GPU.
;
; Each packet shades a 4x4 tile. The 16 rays traverse the BVH together: the
; BVH is stored depth-first with "skip" links (write-blob), so the walk needs
; no stack and every branch is uniform ("descend if any lane hits the box"),
; with per-lane work done by predication.

(import "prelude")
(import "vec")
(import "kir")
(import "scene")
(using prelude vec kir)

; uniform slots (32 dwords)
(const U-CAM 0)                          ; 0-2 camera  3 tan*aspect  4-6 forward  7 tan
(const U-TAN-X 3) (const U-FWD 4) (const U-TAN-Y 7)
(const U-RIGHT 8)                        ; 8-10 right  11 2/W  12-14 up  15 2/H
(const U-INV-W 11) (const U-UP 12) (const U-INV-H 15)
(const U-LIGHT 16)                       ; 16-18 light centre  19 light size
(const U-LSIZE 19)
(const U-TRI-BASE 20)                    ; triangle records, in 16-byte units
(const U-SPP 21) (const U-SEED 22) (const U-NNODES 23) (const U-INV-SPP 24)
(const U-SAMPLE-BASE 26)                 ; first sample index of this pass (float)
(const U-STRIDE 28)                      ; 28-30 node layout stride for x, y, z octant bits (16-byte units)
(const U-COUNT 32)

; buffers: 0 = scene (write-blob), 1 = XRGB8888 image, 2 = primary hit
; distance (negative on the glossy teapot)
(const BUF-SCENE 0) (const BUF-RGB 1) (const BUF-DEPTH 2)

; blocks: node pair, triangle hit half, shading half
(const BK-NODE 0) (const BK-TRI 1) (const BK-SHADE 2)

(struct reg3 (x operand) (y operand) (z operand))
(defn- vreg3 () reg3
  (let ((x (vreg)) (y (vreg)) (z (vreg)))
    (make reg3 :x x :y y :z z)))
(defn- c3 ((v reg3) (a i32)) operand (if (= a 0) v.x (if (= a 1) v.y v.z)))
(defn- r3 ((x operand) (y operand) (z operand)) reg3 (make reg3 :x x :y y :z z))

(def- kdebug bool false)
; ray + hit state used by the traversal (allocation order = register order)
(def ro reg3) (def rd reg3)              ; origin, direction
(def ri reg3) (def rq reg3)              ; 1 / direction, origin / direction
(def ht operand)                         ; t
(def hn reg3) (def hg reg3)              ; interpolated / geometric normal
(def hm operand)                         ; material: 0 miss, 1 teapot, 2 floor
; temporaries t0..t10, also as an array for computed indices: (tn i)
(defmacro- def-temps (n) `(do ,@(map (lambda (i) `(def ,(sym c"t" i) operand)) (range n)) (def temps (array operand ,n))))
(def-temps 11)
(defn- tn ((i i32)) operand (aref temps i))
; shading state that survives a traversal
(def sp reg3) (def sn reg3) (def sg reg3) (def sd reg3)
(def sm operand)
(def acc reg3) (def thr reg3)
(def rng operand) (def dep operand)      ; rng state, primary hit distance of the last sample
; scalars
(def node operand) (def tri operand) (def tri-end operand) (def sample operand)
(def gx operand) (def gy operand)        ; tile origin
(def r2 operand) (def tmp operand)       ; R2 index, scratch
(def skip operand) (def lbase operand) (def cached operand) (def naddr operand) (def taddr operand)
(def dbg-nodes operand) (def dbg-tris operand)

(defmacro- alloc-temps (n)
  `(do ,@(map (lambda (i) `(do (set ,(sym c"t" i) (vreg)) (set (aref temps ,i) ,(sym c"t" i)))) (range n))))

(defn- alloc-registers () void
  ; vector registers are numbered in the order of these calls
  (set ro (vreg3)) (set rd (vreg3)) (set ri (vreg3)) (set rq (vreg3))
  (set ht (vreg))
  (set hn (vreg3)) (set hg (vreg3))
  (set hm (vreg))
  (alloc-temps 11)
  (set sp (vreg3)) (set sn (vreg3)) (set sg (vreg3)) (set sd (vreg3))
  (set sm (vreg))
  (set acc (vreg3)) (set thr (vreg3))
  (set rng (as-u (vreg)))
  (set dep (vreg))
  (set node (as-u (sreg 0))) (set tri (as-u (sreg 1))) (set tri-end (as-u (sreg 2))) (set sample (as-u (sreg 3)))
  (set gx (sreg 4)) (set gy (sreg 5)) (set r2 (sreg 6)) (set tmp (sreg 7))
  (set skip (as-u (sreg 8))) (set lbase (as-u (sreg 9))) (set cached (as-u (sreg 10)))
  (set naddr (as-u (sreg 11))) (set taddr (as-u (sreg 12)))
  (set dbg-nodes (as-u (sreg 13))) (set dbg-tris (as-u (sreg 14))))

(defn- dot3 ((d operand) (a reg3) (b reg3) (tm operand)) void
  (<- d (* a.x b.x))
  (<- tm (* a.y b.y)) (<- d (+ d tm))
  (<- tm (* a.z b.z)) (<- d (+ d tm)))

(defn- normalize3 ((d reg3) (a reg3) (tm operand)) void
  (dot3 tm a a (if (same-reg d.x a.x) t10 d.x))
  (<- tm (rsq tm))
  (<- d.x (* a.x tm)) (<- d.y (* a.y tm)) (<- d.z (* a.z tm)))

; stateless hash of (src ^ key) → [0,1) in dst; 16-bit multipliers because a
; 32x16 multiply is a single instruction on most GPUs
(defn- hash01 ((dst operand) (src operand) (key u32) (tm operand)) void
  (let ((x (as-u dst)) (tt (as-u tm)))
    (<- x (^ (as-u src) key))
    (for (r 0 2)
      (<- tt (>> x 16)) (<- x (^ x tt))
      (<- x (* x 0x7feb))
      (<- tt (>> x 15)) (<- x (^ x tt))
      (<- x (* x 0x846b))
      (<- tt (>> x 16)) (<- x (^ x tt)))
    (<- x (>> x 9)) (<- x (| x 0x3f800000))
    (<- dst (+ dst -1.0))))

; per-pixel offset in [0,1): interleaved gradient noise of the pixel position,
; with the axes weighted (wx, wy) and shifted by k; cheap and well spread
(defn- ign ((dst operand) (wx f32) (wy f32) (k f32) (tm operand)) void
  (<- dst (+ lane-x gx)) (<- tm (+ lane-y gy))
  (<- dst (* dst wx)) (<- tm (* tm wy))
  (<- dst (+ dst tm)) (<- dst (+ dst k))
  (<- dst (frac dst)) (<- dst (* dst 52.9829189)) (<- dst (frac dst)))

; sample s of a 2D R2 low-discrepancy sequence, rotated per pixel; s keeps
; counting across passes and animation frames, so it also stratifies over time
(defn- r2-sample ((du operand) (dv operand) (key f32) (vtmp operand)) void
  (ign du 0.06711056 0.00583715 key vtmp)
  (ign dv 0.00583715 0.06711056 (+ key 0.5) vtmp)
  (<- tmp (* r2 0.7548776662)) (<- du (+ du tmp)) (<- du (frac du))
  (<- tmp (* r2 0.5698402910)) (<- dv (+ dv tmp)) (<- dv (frac dv)))

; uniform random float in [0,1) per lane (xorshift32)
(defn- rand01 ((dst operand) (vtmp operand)) void
  (let ((tt (as-u vtmp)))
    (<- tt (<< rng 13)) (<- rng (^ rng tt))
    (<- tt (>> rng 17)) (<- rng (^ rng tt))
    (<- tt (<< rng 5)) (<- rng (^ rng tt))
    (<- tt (>> rng 9)) (<- tt (| tt 0x3f800000))
    (<- dst (+ vtmp -1.0))))

; clamp a reciprocal so (b - o) * inv never makes inf - inf
(defn- clamp-big ((v operand)) void
  (<- v (min v 1e20))
  (<- v (max v -1e20)))

;; ---------------------------------------------------- packet traversal ---
; In: ro, rd, ht (= tmax). Out: ht, hn (unnormalised), hg, hm (0 miss, 1 teapot, 2 floor).
; Shadow mode is any-hit: an occluded lane gets ht = -1, which also fails every
; later box/triangle test for it, and the walk ends once all 16 lanes are
; occluded. Lanes may enter with ht = -1 to skip the ray entirely. The floor
; is ignored (shadow rays go up towards the light).

(defn- traverse ((shadow bool)) void
  (<- ri.x (inv rd.x)) (<- ri.y (inv rd.y)) (<- ri.z (inv rd.z))
  (clamp-big ri.x) (clamp-big ri.y) (clamp-big ri.z)
  (<- rq.x (* ro.x ri.x)) (<- rq.y (* ro.y ri.y)) (<- rq.z (* ro.z ri.z))
  (<- hm 0.0)
  (<- hn.x 0.0) (<- hn.y 1.0) (<- hn.z 0.0)
  (<- hg.x 0.0) (<- hg.y 1.0) (<- hg.z 0.0)
  (<- node 0)

  ; pick the node layout for the octant of lane 0's direction
  (<- lbase 0)
  (for (a 0 3)
    (<- f1 (< (lane0 (c3 rd a)) 0.0))
    (<- (at lbase f1) (+ lbase (as-u (uni (+ U-STRIDE a))))))

  ; Nodes are 32 bytes, so one 64-byte load returns node i in dwords 0-7 of
  ; the node block and node i+1 in 8-15. Descending to the first child (i+1)
  ; then needs no load.
  (<- cached 0xfffffff0)                ; nothing cached yet
  (let ((top (here)) (end-walk (new-label)) (to-tri (new-label))
        (slot-at (make (array label 2) (new-label) (new-label))))
    (when kdebug (<- dbg-nodes (+ dbg-nodes 1)))
    (<- f1 (>= node (as-u (uni U-NNODES))))
    (jmp-if f1 end-walk)
    (<- f1 (= node cached))
    (jmp-if f1 (aref slot-at 0))
    (<- (as-u tmp) (+ cached 1))
    (<- f1 (= node (as-u tmp)))
    (jmp-if f1 (aref slot-at 1))
    (<- naddr (<< node 1))
    (load BK-NODE naddr lbase 0)
    (<- cached node)

    ; node record: bmin .0-2, skip .3, bmax .4-6, first_tri << 3 | count .7
    (for (slot 0 2)
      (let ((nb (* slot 8)) (miss (new-label)) (inner (new-label)))
        (bind! (aref slot-at slot))
        ; slab test, scheduled so neighbouring instructions are independent:
        ; in-order GPU cores stall on a dependent op
        (for (a 0 3)                     ; t(2a) = near plane, t(2a+1) = far plane
          (<- (tn (* 2 a)) (* (c3 ri a) (blk BK-NODE (+ nb a))))
          (<- (tn (+ (* 2 a) 1)) (* (c3 ri a) (blk BK-NODE (+ nb 4 a)))))
        (for (a 0 3)
          (<- (tn (* 2 a)) (- (tn (* 2 a)) (c3 rq a)))
          (<- (tn (+ (* 2 a) 1)) (- (tn (+ (* 2 a) 1)) (c3 rq a))))
        (<- t6 (min t0 t1)) (<- t7 (min t2 t3)) (<- t8 (min t4 t5))
        (<- t9 (max t0 t1)) (<- t10 (max t2 t3)) (<- t0 (max t4 t5))
        (<- t6 (max t6 t7)) (<- t9 (min t9 t10))
        (<- t8 (max t8 0.0)) (<- t0 (min t0 ht))
        (<- t6 (max t6 t8)) (<- t9 (min t9 t0))
        (<- f0 (<= t6 t9))
        (jmp-none f0 miss)               ; no lane hit the box
        (<- tri-end (& (as-u (blk BK-NODE (+ nb 7))) 7))
        (<- f1 (= tri-end 0))
        (jmp-if f1 inner)
        (<- tri (>> (as-u (blk BK-NODE (+ nb 7))) 3))
        (<- tri-end (+ tri-end tri))
        (<- skip (as-u (blk BK-NODE (+ nb 3))))
        (jmp to-tri)
        (bind! inner)                    ; interior: first child is next
        (<- node (+ node 1))
        (jmp top)
        (bind! miss)                     ; missed: follow the skip link
        (<- node (as-u (blk BK-NODE (+ nb 3))))
        (jmp top)))

    ; leaf: test each triangle. Its first 64 bytes decide the hit; the
    ; shading half is only loaded when a lane hits.
    (bind! to-tri)
    (let ((tri-loop (here)) (done (new-label)))
      (when kdebug (<- dbg-tris (+ dbg-tris 1)))
      (<- taddr (<< tri 3))
      (load BK-TRI taddr (as-u (uni U-TRI-BASE)) 0)
      ; triangle record (Havel-Herout planes):
      ;   TRI 0-3 N, d   (plane: N·x = d, N = e1 × e2)
      ;   TRI 4-7 N1, d1 (u = N1·x + d1)       TRI 8-11 N2, d2 (v = N2·x + d2)
      ;   TRI 12-14 n0   SHADE 0-2 dn1  SHADE 4-6 dn2  SHADE 8-10 geometric normal
      ; Everything is kept multiplied by det = N·D so no division is needed
      ; to decide a hit; t, u, v are divided out only when some lane hits.
      (let ((no-hit (new-label)))
        ; (scheduled in independent groups, see the slab test)
        ; det = N·D → t0;  t' = d - N·O → t1
        (<- t0 (* rd.x (blk BK-TRI 0))) (<- t1 (* ro.x (blk BK-TRI 0)))
        (<- t7 (* rd.y (blk BK-TRI 1))) (<- t9 (* ro.y (blk BK-TRI 1)))
        (<- t8 (* rd.z (blk BK-TRI 2))) (<- t10 (* ro.z (blk BK-TRI 2)))
        (<- t0 (+ t0 t7)) (<- t1 (+ t1 t9))
        (<- t0 (+ t0 t8)) (<- t1 (+ t1 t10))
        (<- t1 (+ (- t1) (blk BK-TRI 3)))
        ; P' = O det + D t' (= hit point * det) → t2 t3 t4
        (<- t2 (* ro.x t0)) (<- t3 (* ro.y t0)) (<- t4 (* ro.z t0))
        (<- t7 (* rd.x t1)) (<- t8 (* rd.y t1)) (<- t9 (* rd.z t1))
        (<- t2 (+ t2 t7)) (<- t3 (+ t3 t8)) (<- t4 (+ t4 t9))
        ; u' = N1·P' + d1 det → t5;  v' = N2·P' + d2 det → t6
        (<- t5 (* t2 (blk BK-TRI 4))) (<- t6 (* t2 (blk BK-TRI 8)))
        (<- t7 (* t3 (blk BK-TRI 5))) (<- t8 (* t3 (blk BK-TRI 9)))
        (<- t9 (* t4 (blk BK-TRI 6))) (<- t10 (* t4 (blk BK-TRI 10)))
        (<- t5 (+ t5 t7)) (<- t6 (+ t6 t8))
        (<- t7 (* t0 (blk BK-TRI 7))) (<- t8 (* t0 (blk BK-TRI 11)))
        (<- t5 (+ t5 t9)) (<- t6 (+ t6 t10))
        (<- t5 (+ t5 t7)) (<- t6 (+ t6 t8))
        ; hit iff u, v, 1-u-v, t > 0 and t < T; each scaled by det² (sign-safe):
        ;   u'det, v'det, (det-u'-v')det, t'det, (T det - t')det
        (<- t7 (* t5 t0)) (<- t8 (* t6 t0))
        (<- t9 (- t0 t5)) (<- t10 (* t1 t0)) (<- t2 (* ht t0))
        (<- t9 (- t9 t6)) (<- t2 (- t2 t1))
        (<- t7 (min t7 t8))
        (<- t9 (* t9 t0)) (<- t2 (* t2 t0))
        (<- t7 (min t7 t10))
        (<- t9 (min t9 t2))
        (<- t8 (min t7 t9))
        (<- f0 (> t8 0.0))
        (jmp-none f0 no-hit)
        (<- t0 (inv t0))
        (<- (at ht f0) (* t1 t0))
        (if shadow
            (<- (at ht f0) -1.0)
            (do
              (<- t5 (* t5 t0)) (<- t6 (* t6 t0))   ; u, v
              (load BK-SHADE taddr (as-u (uni U-TRI-BASE)) 4)
              (for (a 0 3)
                (<- t7 (* t6 (blk BK-SHADE (+ 4 a))))               ; v * dn2
                (<- (at (c3 hn a) f0) (* t5 (blk BK-SHADE a)))      ; u * dn1
                (<- (at (c3 hn a) f0) (+ (c3 hn a) (blk BK-TRI (+ 12 a))))   ; + n0
                (<- (at (c3 hn a) f0) (+ (c3 hn a) t7))
                (<- (at (c3 hg a) f0) (blk BK-SHADE (+ 8 a))))))
        (<- (at hm f0) 1.0)
        (bind! no-hit))
      (<- tri (+ tri 1))
      (<- f1 (< tri tri-end))
      (jmp-if f1 tri-loop)
      (when shadow
        (<- f0 (< ht 0.0))
        (jmp-all f0 done))               ; every lane occluded
      (<- node skip)                     ; leaf done: follow its skip link
      (jmp top)

      (bind! end-walk)
      (if shadow
          (bind! done)
          (do
            ; floor plane y = 0: t = -oy/dy
            (<- t0 (- rq.y))
            (<- t1 (+ t0 -1e-4))
            (<- t2 (- ht t0))
            (<- t1 (min t1 t2))
            (<- f0 (> t1 0.0))
            (<- (at ht f0) t0)
            (<- (at hm f0) 2.0)
            (<- (at hn.x f0) 0.0) (<- (at hn.y f0) 1.0) (<- (at hn.z f0) 0.0)
            (<- (at hg.x f0) 0.0) (<- (at hg.y f0) 1.0) (<- (at hg.z f0) 0.0))))))

;; ------------------------------------------------------------- shading ---

; sky(dir) → d, using s as scratch
(defn- sky3 ((d reg3) (ydir operand) (s operand)) void
  (<- s (max ydir 0.0))
  (<- s (sqrt s))
  (for (a 0 3)
    (<- (c3 d a) (* s (- (axis scene::sky-zenith a) (axis scene::sky-horizon a))))
    (<- (c3 d a) (+ (c3 d a) (axis scene::sky-horizon a)))))

(defn- bounce ((last bool) (stratified bool)) void
  (<- ht 1e6)
  (traverse false)
  (when stratified                       ; primary bounce: keep the depth,
    (<- dep ht)                          ; negative on the (glossy) teapot
    (<- f0 (= hm 1.0))
    (<- (at dep f0) (- ht)))

  ; miss lanes: add sky, kill throughput
  (let ((sk (r3 t1 t2 t3)))
    (<- f0 (= hm 0.0))
    (<- t0 0.0) (<- (at t0 f0) 1.0)
    (sky3 sk rd.y t4)
    (for (a 0 3)
      (<- t5 (* (c3 thr a) t0)) (<- t5 (* t5 (c3 sk a)))
      (<- (c3 acc a) (+ (c3 acc a) t5))))
  (<- t0 (+ (- t0) 1.0))
  (for (a 0 3) (<- (c3 thr a) (* (c3 thr a) t0)))

  ; hit point, oriented normals
  (for (a 0 3)
    (<- (c3 sp a) (* (c3 rd a) ht)) (<- (c3 sp a) (+ (c3 sp a) (c3 ro a)))
    (<- (c3 sd a) (c3 rd a))
    (<- (c3 sg a) (c3 hg a)))
  (<- sm hm)
  (normalize3 sn hn t0)
  (dot3 t0 sg sd t1)
  (<- f0 (> t0 0.0))
  (for (a 0 3) (<- (at (c3 sg a) f0) (- (c3 sg a))))
  (dot3 t0 sn sg t1)
  (<- f0 (< t0 0.0))
  (for (a 0 3) (<- (at (c3 sn a) f0) (- (c3 sn a))))

  ; shadow ray towards a jittered point on a square area light
  (if stratified
      (r2-sample t1 t0 0.31 t2)          ; swapped dims: decorrelate from pixel jitter
      (do (rand01 t0 t2) (rand01 t1 t2)))
  (<- t0 (+ t0 -0.5)) (<- t0 (* t0 (uni U-LSIZE)))
  (<- t1 (+ t1 -0.5)) (<- t1 (* t1 (uni U-LSIZE)))
  (<- rd.x (+ t0 (uni U-LIGHT))) (<- rd.x (- rd.x sp.x))
  (<- rd.y (+ (- sp.y) (uni (+ U-LIGHT 1))))
  (<- rd.z (+ t1 (uni (+ U-LIGHT 2)))) (<- rd.z (- rd.z sp.z))
  (dot3 t0 rd rd t1)
  (<- t1 (rsq t0))
  (for (a 0 3) (<- (c3 rd a) (* (c3 rd a) t1)))
  (<- ht (* t0 t1))                      ; distance to the light sample
  (for (a 0 3) (<- t2 (* (c3 sg a) 2e-3)) (<- (c3 ro a) (+ (c3 sp a) t2)))
  ; no shadow ray for lanes facing away from the light or that hit nothing
  (dot3 t0 sg rd t1)
  (<- f0 (<= t0 0.0)) (<- (at ht f0) -1.0)
  (<- f0 (= sm 0.0)) (<- (at ht f0) -1.0)
  (traverse true)

  ; local shading: vis → hn.x, n·l → hn.y
  (let ((vis hn.x) (ndl hn.y) (h (r3 t0 t1 t2)) (alb (r3 t8 t9 t10)) (skn hg)
        (dark (vec3 0.16 0.17 0.20)) (light (vec3 0.78 0.77 0.74)) (red (vec3 0.62 0.07 0.05))
        (fog (vec3 0.80 0.84 0.91)))
    (<- f0 (> ht 0.0))
    (<- vis 0.0) (<- (at vis f0) 1.0)
    (dot3 ndl sn rd t0)
    (<- ndl (max ndl 0.0))
    ; half vector
    (for (a 0 3) (<- (c3 h a) (- (c3 rd a) (c3 sd a))))
    (normalize3 h h t3)
    (dot3 t3 sn h t4)
    (<- t3 (max t3 0.0))
    ; material parameters (flag 0 = teapot)
    (<- f0 (= sm 1.0))
    (<- t4 60.0) (<- (at t4 f0) 180.0)
    (<- t3 (pow t3 t4))
    (<- t5 (/ (* 0.25 68.0) (* 8.0 PI))) (<- (at t5 f0) (/ (* 0.9 188.0) (* 8.0 PI)))
    (<- t3 (* t3 t5))                    ; specular
    (<- t6 0.09) (<- (at t6 f0) 0.05)    ; F0
    ; albedo: checkerboard floor, red teapot → t8 t9 t10
    (<- t7 (floor sp.x)) (<- t8 (floor sp.z)) (<- t7 (+ t7 t8))
    (<- t7 (* t7 0.5)) (<- t7 (frac t7)) (<- t7 (* t7 2.0))
    (for (a 0 3)
      (<- (c3 alb a) (* t7 (- (axis light a) (axis dark a))))
      (<- (c3 alb a) (+ (c3 alb a) (axis dark a))))
    (for (a 0 3) (<- (at (c3 alb a) f0) (axis red a)))
    ; direct = (albedo * n·l + spec) * vis * light colour → t0 t1 t2
    (for (a 0 3)
      (<- (tn a) (* (c3 alb a) ndl)) (<- (tn a) (+ (tn a) t3))
      (<- (tn a) (* (tn a) vis)) (<- (tn a) (* (tn a) (axis scene::light-col a))))
    ; ambient sky light: albedo * sky(n) * (0.231 + 0.154 n.y)
    (sky3 skn sn.y t4)
    (<- t5 (* sn.y 0.154)) (<- t5 (+ t5 0.231))
    (for (a 0 3)
      (<- t7 (* (c3 skn a) (c3 alb a))) (<- t7 (* t7 t5))
      (<- (tn a) (+ (tn a) t7)))
    ; Fresnel (Schlick) → t6
    (dot3 t4 sn sd t5)
    (<- t4 (min t4 0.0))
    (<- t4 (+ t4 1.0))                   ; 1 - cos
    (<- t5 (* t4 t4)) (<- t5 (* t5 t5)) (<- t5 (* t5 t4))
    (<- t7 (+ (- t6) 1.0)) (<- t7 (* t7 t5)) (<- t6 (+ t6 t7))
    ; distance fog towards the horizon colour
    (<- t7 (- sp.x (uni U-CAM))) (<- t7 (* t7 t7))
    (<- t8 (- sp.y (uni (+ U-CAM 1)))) (<- t8 (* t8 t8)) (<- t7 (+ t7 t8))
    (<- t8 (- sp.z (uni (+ U-CAM 2)))) (<- t8 (* t8 t8)) (<- t7 (+ t7 t8))
    (<- t7 (sqrt t7))
    (<- t7 (* t7 (* -0.02 1.442695)))
    (<- f0 (= sm 1.0))
    (<- (at t7 f0) 0.0)                  ; fog only on the floor
    (<- t7 (exp2 t7))                    ; e = exp(-0.02 d)
    ; acc += thr * (1 - F) * lerp(fog, local, e);  thr *= F
    (<- t9 (+ (- t6) 1.0))
    (for (a 0 3)
      (<- (tn a) (+ (tn a) (- (axis fog a)))) (<- (tn a) (* (tn a) t7)) (<- (tn a) (+ (tn a) (axis fog a)))
      (<- (tn a) (* (tn a) t9)) (<- (tn a) (* (tn a) (c3 thr a)))
      (<- (c3 acc a) (+ (c3 acc a) (tn a)))
      (<- (c3 thr a) (* (c3 thr a) t6))))

  ; reflected ray
  (dot3 t0 sd sn t1)
  (<- t0 (* t0 -2.0))
  (for (a 0 3)
    (<- t1 (* (c3 sn a) t0)) (<- (c3 rd a) (+ (c3 sd a) t1))
    (<- t2 (* (c3 sg a) 2e-3)) (<- (c3 ro a) (+ (c3 sp a) t2)))
  (when last
    (let ((sr (r3 t1 t2 t3)))
      (sky3 sr rd.y t4)
      (for (a 0 3) (<- t5 (* (c3 thr a) (c3 sr a))) (<- (c3 acc a) (+ (c3 acc a) t5))))))

(defn- kernel-main () void
  ; tile origin as float scalars
  (<- gx group-x)
  (<- gy group-y)
  (<- gx (* gx 4.0))
  (<- gy (* gy 4.0))

  ; per-pixel seed = pixel index (exact in float below 2^24), RNG state = hash of it
  (<- t0 (+ lane-x gx))
  (<- t1 (+ lane-y gy))
  (<- t1 (* t1 4096.0)) (<- t0 (+ t0 t1))
  (<- (as-u t0) t0)
  (hash01 (as-f rng) t0 0x9e3779b9 t1)
  (<- rng (^ rng (as-u (uni U-SEED))))   ; fresh random stream per pass
  (<- rng (| rng 1))                     ; xorshift state must be non-zero

  (for (a 0 3) (<- (c3 acc a) 0.0))
  (<- sample 0)
  (<- r2 (uni U-SAMPLE-BASE))            ; R2 sample index continues across passes

  (let ((sample-loop (here)))
    ; camera ray through a jittered point of the pixel
    (r2-sample t2 t3 0.0 t4)
    (<- t0 (+ lane-x gx)) (<- t0 (+ t0 t2))
    (<- t1 (+ lane-y gy)) (<- t1 (+ t1 t3))
    (<- t0 (* t0 (uni U-INV-W))) (<- t0 (+ t0 -1.0)) (<- t0 (* t0 (uni U-TAN-X)))
    (<- t1 (* t1 (uni U-INV-H))) (<- t1 (+ (- t1) 1.0)) (<- t1 (* t1 (uni U-TAN-Y)))
    (for (a 0 3)
      (<- (c3 rd a) (* t0 (uni (+ U-RIGHT a))))
      (<- t2 (* t1 (uni (+ U-UP a)))) (<- (c3 rd a) (+ (c3 rd a) t2))
      (<- (c3 rd a) (+ (c3 rd a) (uni (+ U-FWD a))))
      (<- (c3 ro a) (uni (+ U-CAM a)))
      (<- (c3 thr a) 1.0))
    (normalize3 rd rd t0)

    (if kdebug
        (do
          ; debug view: red = material, green = nodes visited / 4, blue = triangles tested
          (<- dbg-nodes 0) (<- dbg-tris 0)
          (<- ht 1e6)
          (traverse false)
          (<- t0 (* hm 100.0)) (<- (as-u t4) t0)
          (<- dbg-nodes (>> dbg-nodes 2))
          (<- (as-u t5) dbg-nodes) (<- (as-u t6) dbg-tris))
        (do
          (bounce false true)
          (bounce true false)

          (<- sample (+ sample 1))
          (<- r2 (+ r2 1.0))
          (<- f1 (< sample (as-u (uni U-SPP))))
          (jmp-if f1 sample-loop)

          ; average, ACES tone curve, gamma, pack to XRGB8888
          (for (a 0 3)
            (let ((c (c3 acc a)))
              (<- c (* c (uni U-INV-SPP)))
              (<- c (* c 0.9))
              (<- t0 (* c 2.51)) (<- t0 (+ t0 0.03)) (<- t0 (* t0 c))
              (<- t1 (* c 2.43)) (<- t1 (+ t1 0.59)) (<- t1 (* t1 c)) (<- t1 (+ t1 0.14))
              (<- t1 (inv t1))
              (<- t0 (sat (* t0 t1)))
              (<- t2 (/ 1.0 2.2))
              (<- t0 (pow t0 t2))
              (<- t0 (* t0 255.0)) (<- t0 (+ t0 0.5))
              (<- (as-u (tn (+ a 4))) t0))))))
  (let ((px (as-u t10)))
    (<- px (<< (as-u t4) 16))
    (<- (as-u t5) (<< (as-u t5) 8))
    (<- px (| px (as-u t5)))
    (<- px (| px (as-u t6)))
    (store BUF-RGB px)
    (store BUF-DEPTH dep)
    (end)))

; the kernel as a fresh program in `prog`; debug = traversal heat map
(defn build ((debug bool)) void
  (set prog (make program))
  (set kdebug debug)
  (alloc-registers)
  (kernel-main))

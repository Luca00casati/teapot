; scene.lsp — the tessellated Newell teapot on a checkerboard floor, its
; binned-SAH BVH, the GPU scene layout, the camera, and the look (light, sky)
; shared with the GPU kernel.

(import "libc")
(import "math")
(import "prelude")
(import "vec")
(import "std/sort")
(using libc prelude vec)

;; Classic 10 patch / 127 control point form; rim, body, lid and bottom are
;; mirrored into 4 quadrants, handle and spout into 2 → 32 bicubic patches.
(private
  (def patches (array (array i32 16) 10)
    {(array (array i32 16) 10)
     {(array i32 16) 102 103 104 105 4 5 6 7 8 9 10 11 12 13 14 15}    ; rim
     {(array i32 16) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27}    ; body
     {(array i32 16) 24 25 26 27 29 30 31 32 33 34 35 36 37 38 39 40}
     {(array i32 16) 96 96 96 96 97 98 99 100 101 101 101 101 0 1 2 3}    ; lid
     {(array i32 16) 0 1 2 3 106 107 108 109 110 111 112 113 114 115 116 117}
     {(array i32 16) 118 118 118 118 126 125 124 123 122 120 119 121 40 39 38 37}    ; bottom
     {(array i32 16) 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56}    ; handle
     {(array i32 16) 53 54 55 56 57 58 59 60 61 62 63 64 28 65 66 67}
     {(array i32 16) 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83}    ; spout
     {(array i32 16) 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95}})

  ; as written in the original data: decimal → f64 → f32 at use, as C rounds a
  ; double literal into a float array
  (def cpdata (array (array f64 3) 127)
    {(array (array f64 3) 127)
     {(array f64 3) 0.2 0.0 2.7} {(array f64 3) 0.2 -0.112 2.7}
     {(array f64 3) 0.112 -0.2 2.7} {(array f64 3) 0.0 -0.2 2.7}
     {(array f64 3) 1.3375 0.0 2.53125} {(array f64 3) 1.3375 -0.749 2.53125}
     {(array f64 3) 0.749 -1.3375 2.53125} {(array f64 3) 0.0 -1.3375 2.53125}
     {(array f64 3) 1.4375 0.0 2.53125} {(array f64 3) 1.4375 -0.805 2.53125}
     {(array f64 3) 0.805 -1.4375 2.53125} {(array f64 3) 0.0 -1.4375 2.53125}
     {(array f64 3) 1.5 0.0 2.4} {(array f64 3) 1.5 -0.84 2.4}
     {(array f64 3) 0.84 -1.5 2.4} {(array f64 3) 0.0 -1.5 2.4}
     {(array f64 3) 1.75 0.0 1.875} {(array f64 3) 1.75 -0.98 1.875}
     {(array f64 3) 0.98 -1.75 1.875} {(array f64 3) 0.0 -1.75 1.875}
     {(array f64 3) 2.0 0.0 1.35} {(array f64 3) 2.0 -1.12 1.35}
     {(array f64 3) 1.12 -2.0 1.35} {(array f64 3) 0.0 -2.0 1.35}
     {(array f64 3) 2.0 0.0 0.9} {(array f64 3) 2.0 -1.12 0.9}
     {(array f64 3) 1.12 -2.0 0.9} {(array f64 3) 0.0 -2.0 0.9}
     {(array f64 3) -2.0 0.0 0.9} {(array f64 3) 2.0 0.0 0.45}
     {(array f64 3) 2.0 -1.12 0.45} {(array f64 3) 1.12 -2.0 0.45}
     {(array f64 3) 0.0 -2.0 0.45} {(array f64 3) 1.5 0.0 0.225}
     {(array f64 3) 1.5 -0.84 0.225} {(array f64 3) 0.84 -1.5 0.225}
     {(array f64 3) 0.0 -1.5 0.225} {(array f64 3) 1.5 0.0 0.15}
     {(array f64 3) 1.5 -0.84 0.15} {(array f64 3) 0.84 -1.5 0.15}
     {(array f64 3) 0.0 -1.5 0.15} {(array f64 3) -1.6 0.0 2.025}
     {(array f64 3) -1.6 -0.3 2.025} {(array f64 3) -1.5 -0.3 2.25}
     {(array f64 3) -1.5 0.0 2.25} {(array f64 3) -2.3 0.0 2.025}
     {(array f64 3) -2.3 -0.3 2.025} {(array f64 3) -2.5 -0.3 2.25}
     {(array f64 3) -2.5 0.0 2.25} {(array f64 3) -2.7 0.0 2.025}
     {(array f64 3) -2.7 -0.3 2.025} {(array f64 3) -3.0 -0.3 2.25}
     {(array f64 3) -3.0 0.0 2.25} {(array f64 3) -2.7 0.0 1.8}
     {(array f64 3) -2.7 -0.3 1.8} {(array f64 3) -3.0 -0.3 1.8}
     {(array f64 3) -3.0 0.0 1.8} {(array f64 3) -2.7 0.0 1.575}
     {(array f64 3) -2.7 -0.3 1.575} {(array f64 3) -3.0 -0.3 1.35}
     {(array f64 3) -3.0 0.0 1.35} {(array f64 3) -2.5 0.0 1.125}
     {(array f64 3) -2.5 -0.3 1.125} {(array f64 3) -2.65 -0.3 0.9375}
     {(array f64 3) -2.65 0.0 0.9375} {(array f64 3) -2.0 -0.3 0.9}
     {(array f64 3) -1.9 -0.3 0.6} {(array f64 3) -1.9 0.0 0.6}
     {(array f64 3) 1.7 0.0 1.425} {(array f64 3) 1.7 -0.66 1.425}
     {(array f64 3) 1.7 -0.66 0.6} {(array f64 3) 1.7 0.0 0.6}
     {(array f64 3) 2.6 0.0 1.425} {(array f64 3) 2.6 -0.66 1.425}
     {(array f64 3) 3.1 -0.66 0.825} {(array f64 3) 3.1 0.0 0.825}
     {(array f64 3) 2.3 0.0 2.1} {(array f64 3) 2.3 -0.25 2.1}
     {(array f64 3) 2.4 -0.25 2.025} {(array f64 3) 2.4 0.0 2.025}
     {(array f64 3) 2.7 0.0 2.4} {(array f64 3) 2.7 -0.25 2.4}
     {(array f64 3) 3.3 -0.25 2.4} {(array f64 3) 3.3 0.0 2.4}
     {(array f64 3) 2.8 0.0 2.475} {(array f64 3) 2.8 -0.25 2.475}
     {(array f64 3) 3.525 -0.25 2.49375} {(array f64 3) 3.525 0.0 2.49375}
     {(array f64 3) 2.9 0.0 2.475} {(array f64 3) 2.9 -0.15 2.475}
     {(array f64 3) 3.45 -0.15 2.5125} {(array f64 3) 3.45 0.0 2.5125}
     {(array f64 3) 2.8 0.0 2.4} {(array f64 3) 2.8 -0.15 2.4}
     {(array f64 3) 3.2 -0.15 2.4} {(array f64 3) 3.2 0.0 2.4}
     {(array f64 3) 0.0 0.0 3.15} {(array f64 3) 0.8 0.0 3.15}
     {(array f64 3) 0.8 -0.45 3.15} {(array f64 3) 0.45 -0.8 3.15}
     {(array f64 3) 0.0 -0.8 3.15} {(array f64 3) 0.0 0.0 2.85}
     {(array f64 3) 1.4 0.0 2.4} {(array f64 3) 1.4 -0.784 2.4}
     {(array f64 3) 0.784 -1.4 2.4} {(array f64 3) 0.0 -1.4 2.4}
     {(array f64 3) 0.4 0.0 2.55} {(array f64 3) 0.4 -0.224 2.55}
     {(array f64 3) 0.224 -0.4 2.55} {(array f64 3) 0.0 -0.4 2.55}
     {(array f64 3) 1.3 0.0 2.55} {(array f64 3) 1.3 -0.728 2.55}
     {(array f64 3) 0.728 -1.3 2.55} {(array f64 3) 0.0 -1.3 2.55}
     {(array f64 3) 1.3 0.0 2.4} {(array f64 3) 1.3 -0.728 2.4}
     {(array f64 3) 0.728 -1.3 2.4} {(array f64 3) 0.0 -1.3 2.4}
     {(array f64 3) 0.0 0.0 0.0} {(array f64 3) 1.5 -0.84 0.075}
     {(array f64 3) 0.84 -1.5 0.075} {(array f64 3) 1.5 0.0 0.075}
     {(array f64 3) 0.0 -1.5 0.075} {(array f64 3) 1.425 0.0 0.0}
     {(array f64 3) 1.425 -0.798 0.0} {(array f64 3) 0.798 -1.425 0.0}
     {(array f64 3) 0.0 -1.425 0.0}})

  (const TESS 24))                          ; quads per patch edge

(struct tri [v0 v3] [e1 v3] [e2 v3] [i0 u32] [i1 u32] [i2 u32])
; count = 0: an interior node whose children are at first and first + 1
(struct node [bmin v3] [bmax v3] [first u32] [count u32])

(struct model
  [vpos (ptr v3)] [vnrm (ptr v3)] [nv u32]
  [tris (ptr tri)] [ntri u32]
  [nodes (ptr node)] [nnode u32]
  [tidx (ptr u32)])                      ; triangle indices in leaf order

; Bernstein basis and its derivative at t
(private
  (defn bern [t f32] [b (ptr f32)] [d (ptr f32)] void
    (let [s (- 1.0 t)]
      (set b[0] (* s s s))
      (set b[1] (* 3.0 t s s))
      (set b[2] (* 3.0 t t s))
      (set b[3] (* t t t))
      (set d[0] (* -3.0 s s))
      (set d[1] (- (* 3.0 s s) (* 6.0 t s)))
      (set d[2] (- (* 6.0 t s) (* 3.0 t t)))
      (set d[3] (* 3.0 t t))))

  ; point and the two tangents of a bicubic patch (cp: 4x4 control points)
  (defn eval-patch [cp (ptr v3)] [u f32] [v f32] [p (ptr v3)] [du (ptr v3)] [dv (ptr v3)] void
    (let [bu (array f32 4)] [du_ (array f32 4)] [bv (array f32 4)] [dv_ (array f32 4)]
         [pp (vec3 0.0 0.0 0.0)] [pu pp] [pv pp]
      (bern u bu du_)
      (bern v bv dv_)
      (for [i 0 4]
        (for [j 0 4]
          (let [c cp[(+ (* i 4) j)]]
            (set pp (+ pp (* c (* bu[i] bv[j]))))
            (set pu (+ pu (* c (* du_[i] bv[j]))))
            (set pv (+ pv (* c (* bu[i] dv_[j])))))))
      (set (deref p) pp)
      (set (deref du) pu)
      (set (deref dv) pv)))

  (defn centroid [t (ptr tri)] v3
    (+ t.v0 (* (+ t.e1 t.e2) (/ 1.0 3.0))))

  (defn build-mesh [s (ptr model)] void
    (let [npatch u64 (+ (* 6 4) (* 4 2))]
         [quad {(array f32 8) 1.0 1.0 1.0 -1.0 -1.0 -1.0 -1.0 1.0}]
      (set s.vpos (cast (ptr v3) (malloc (* (sizeof v3) npatch (+ TESS 1) (+ TESS 1)))))
      (set s.vnrm (cast (ptr v3) (malloc (* (sizeof v3) npatch (+ TESS 1) (+ TESS 1)))))
      (set s.tris (cast (ptr tri) (malloc (* (sizeof tri) npatch TESS TESS 2))))
      (for [p 0 10]
        (for [q 0 (if (< p 6) 4 2)]
          (let [sx quad[(* q 2)]] [sy quad[(+ (* q 2) 1)]]
               [cp (array v3 16)]
               [base s.nv]
            (for [i 0 16]
              (let [d cpdata[patches[p][i]]]
                   [d0 (cast f32 d[0])] [d1 (cast f32 d[1])] [d2 (cast f32 d[2])]
                ; teapot data is z-up; rotate into y-up
                (set cp[i] (vec3 (* sx d0) d2 (* (- sy) d1)))))
            (for [i u32 0 (+ TESS 1)]
              (for [j u32 0 (+ TESS 1)]
                (let [u (/ (cast f32 i) (cast f32 TESS))] [v (/ (cast f32 j) (cast f32 TESS))]
                     [pt v3] [du v3] [dv v3] [n v3]
                  (eval-patch cp u v (addr pt) (addr du) (addr dv))
                  (set n (cross du dv))
                  (when (< (length n) 1e-6)
                    ; degenerate pole (lid knob, bottom centre)
                    (let [cu (math::fminf (math::fmaxf u 1e-3) (- 1.0 1e-3))] [cv (math::fminf (math::fmaxf v 1e-3) (- 1.0 1e-3))] [q0 v3]
                      (eval-patch cp cu cv (addr q0) (addr du) (addr dv))
                      (set n (cross du dv))))
                  (set s.vpos[s.nv] pt)
                  (set s.vnrm[s.nv] (normalize n))
                  (+= s.nv 1))))
            (for [i u32 0 TESS]
              (for [j u32 0 TESS]
                (let [a (+ base (* i (+ TESS 1)) j)] [b (+ a 1)] [c (+ a TESS 1)] [d (+ c 1)]
                     [ks {(array u32 6) a c b b c d}]
                  (for [k 0 2]
                    (let [k0 ks[(* k 3)]] [k1 ks[(+ (* k 3) 1)]] [k2 ks[(+ (* k 3) 2)]]
                         [v0 s.vpos[k0]]
                         [t {tri :v0 v0 :e1 (- s.vpos[k1] v0) :e2 (- s.vpos[k2] v0) :i0 k0 :i1 k1 :i2 k2}]
                      (when (> (length (cross t.e1 t.e2)) 1e-9)
                        (set s.tris[s.ntri] t)
                        (+= s.ntri 1))))))))))))

  ;; -------------------------------------------------------------- BVH ---

  (const BINS 16)

  (defn bin-of [c f32] [lo f32] [span f32] i32
    (let [b (cast i32 (* (/ (- c lo) span) (cast f32 BINS)))]
      (if (< b 0) 0 (if (> b (- BINS 1)) (- BINS 1) b))))

  (defn area [e v3] f32 (+ (* e.x e.y) (* e.y e.z) (* e.z e.x)))

  (defn build-node [s (ptr model)] [ni u32] [first u32] [count u32] void
    (let [big (vec3 1e30 1e30 1e30)] [bmin big] [bmax (- big)] [cmin bmin] [cmax bmax]
      (for [k first (+ first count)]
        (let [t (addr s.tris[s.tidx[k]])]
             [a t.v0] [b (+ t.v0 t.e1)] [c (+ t.v0 t.e2)] [ce (centroid t)]
          (set bmin (vmin bmin (vmin a (vmin b c))))
          (set bmax (vmax bmax (vmax a (vmax b c))))
          (set cmin (vmin cmin ce))
          (set cmax (vmax cmax ce))))
      (set s.nodes[ni] {node :bmin bmin :bmax bmax :first first :count count})
      (when (<= count 4) (return))

      ; binned surface-area heuristic: pick the axis/plane minimising A_l*N_l + A_r*N_r
      (let [best f32 1e30] [best-axis -1] [best-bin 0] [mid u32 0]
        (for [ax 0 3]
          (let [lo (axis cmin ax)] [span (- (axis cmax ax) (axis cmin ax))]
            (unless (< span 1e-6)
              (let [cnt (array u32 16)] [bmn (array v3 16)] [bmx (array v3 16)] [right-cost (array f32 16)]
                   [rmn big] [rmx (- big)] [rn u32 0] [lmn big] [lmx (- big)] [ln u32 0]
                (for [b 0 BINS] (set bmn[b] big) (set bmx[b] (- big)))
                (for [k first (+ first count)]
                  (let [t (addr s.tris[s.tidx[k]])]
                       [b (bin-of (axis (centroid t) ax) lo span)]
                       [p0 t.v0] [p1 (+ t.v0 t.e1)] [p2 (+ t.v0 t.e2)]
                    (+= cnt[b] 1)
                    (set bmn[b] (vmin bmn[b] (vmin p0 (vmin p1 p2))))
                    (set bmx[b] (vmax bmx[b] (vmax p0 (vmax p1 p2))))))
                (let [b (- BINS 1)]
                  (while (> b 0)
                    (set rmn (vmin rmn bmn[b]))
                    (set rmx (vmax rmx bmx[b]))
                    (+= rn cnt[b])
                    (set right-cost[b] (if (> rn 0) (* (area (- rmx rmn)) (cast f32 rn)) 0.0))
                    (-= b 1)))
                (for [b 0 (- BINS 1)]
                  (set lmn (vmin lmn bmn[b]))
                  (set lmx (vmax lmx bmx[b]))
                  (+= ln cnt[b])
                  (unless (or (= ln 0) (= ln count))
                    (let [c (+ (* (area (- lmx lmn)) (cast f32 ln)) right-cost[(+ b 1)])]
                      (when (< c best)
                        (set best c)
                        (set best-axis ax)
                        (set best-bin b)))))))))
        (if (>= best-axis 0)
            (let [lo (axis cmin best-axis)] [span (- (axis cmax best-axis) (axis cmin best-axis))]
                 [i (cast i64 first)] [j (- (+ (cast i64 first) (cast i64 count)) 1)]
              (while (<= i j)
                (if (<= (bin-of (axis (centroid (addr s.tris[s.tidx[i]])) best-axis) lo span) best-bin)
                    (+= i 1)
                    (let [tmp s.tidx[i]]
                      (set s.tidx[i] s.tidx[j])
                      (set s.tidx[j] tmp)
                      (-= j 1))))
              (set mid (cast u32 (- i (cast i64 first)))))
            ; all centroids coincide: plain median split
            (let [e (- cmax cmin)]
              (let [ax (if (and (> e.x e.y) (> e.x e.z)) 0 (if (> e.y e.z) 1 2))]
                (sort::sort-by {(slice u32) :data (addr s.tidx[first]) :len (cast u64 count)}
                               (fn [a u32] [b u32] bool
                                 (< (axis (centroid (addr s.tris[a])) ax) (axis (centroid (addr s.tris[b])) ax)))))
              (set mid (/ count 2))))
        (let [l s.nnode]
          (+= s.nnode 2)
          (set (. s.nodes[ni] first) l)
          (set (. s.nodes[ni] count) 0)
          (build-node s l first mid)
          (build-node s (+ l 1) (+ first mid) (- count mid)))))))

; tessellate the teapot and build the SAH BVH
(defn build [] model
  (let [s {model}]
    (build-mesh (addr s))
    (set s.tidx (cast (ptr u32) (malloc (* (sizeof u32) (cast u64 s.ntri)))))
    (for [i u32 0 s.ntri] (set s.tidx[i] i))
    (set s.nodes (cast (ptr node) (malloc (* (sizeof node) (cast u64 s.ntri) 2))))
    (set s.nnode 1)
    (build-node (addr s) 0 0 s.ntri)
    s))

;; --------------------------------------------------- GPU scene layout ---
; 8 depth-first copies of the BVH with "skip" links (one per ray-direction
; octant, nearer child first) so a walk needs no stack, then one 128-byte
; record per triangle in leaf order.
;   node (32 B):  bmin.xyz, skip (u32) | bmax.xyz, first_tri << 3 | count (u32, 0 = interior)
;   triangle:     Havel-Herout planes N,d | N1,d1 | N2,d2 | n0 | n1-n0 | n2-n0 | geometric normal

(defn layout-bytes [s (ptr model)] u64 (* (cast u64 s.nnode) 32))
(defn blob-bytes [s (ptr model)] u64 (+ (* 8 (layout-bytes s)) (* (cast u64 s.ntri) 128)))

(private
  (defn put3 [r (ptr u32)] [at u32] [v v3] void
    (set r[at] (bitcast u32 v.x))
    (set r[(+ at 1)] (bitcast u32 v.y))
    (set r[(+ at 2)] (bitcast u32 v.z))))

(defn write-blob [s (ptr model)] [dst (ptr u32)] void
  (let [n s.nnode] [layout (cast u32 (/ (layout-bytes s) 4))] [nodes s.nodes]
       [order (cast (ptr u32) (malloc (* 4 (cast u64 n))))] [skip (cast (ptr u32) (malloc (* 4 (cast u64 n))))]
       [size (cast (ptr u32) (malloc (* 4 (cast u64 n))))] [stack (cast (ptr u32) (malloc (* 4 (cast u64 n))))]
    (for [oct u32 0 8]
      (let [sp 1] [k u32 0]
        (set stack[0] 0)
        (while (> sp 0)
          (-= sp 1)
          (let [ni stack[sp]]
            (set order[k] ni)
            (+= k 1)
            (when (= (. nodes[ni] count) 0)
              (let [l (. nodes[ni] first)] [r (+ l 1)]
                   ; split axis ≈ largest separation of the child centres
                   [d (- (+ (. nodes[r] bmin) (. nodes[r] bmax)) (+ (. nodes[l] bmin) (. nodes[l] bmax)))]
                   [x (math::fabsf d.x)] [y (math::fabsf d.y)] [z (math::fabsf d.z)]
                   [ax (if (and (> x y) (> x z)) 0 (if (> y z) 1 2))]
                   [neg-dir (!= (& (>> oct ax) 1) 0)]
                   [near (if (!= (> (axis d ax) 0.0) neg-dir) l r)]   ; child a ray in this octant meets first
                (set stack[sp] (if (= near l) r l))
                (set stack[(+ sp 1)] near)
                (+= sp 2)))))
        (let [i (cast i64 n)]                ; skip(i) = i + size of subtree(i)
          (while (> i 0)
            (-= i 1)
            (let [nd (addr nodes[order[i]])]
              (set size[order[i]] (+ 1 (if (!= nd.count 0) 0 (+ size[nd.first] size[(+ nd.first 1)]))))
              (set skip[i] (+ (cast u32 i) size[order[i]])))))
        (for [i u32 0 n]
          (let [nd (addr nodes[order[i]])] [r (+ dst (* oct layout) (* i 8))]
            (put3 r 0 nd.bmin)
            (set r[3] skip[i])
            (put3 r 4 nd.bmax)
            (set r[7] (if (!= nd.count 0) (| (<< nd.first 3) nd.count) 0))))))   ; count <= 4
    (for [k u32 0 s.ntri]
      (let [tr (addr s.tris[s.tidx[k]])] [r (+ dst (* 8 layout) (* k 32))]
           [n0 s.vnrm[tr.i0]] [n1 s.vnrm[tr.i1]] [n2 s.vnrm[tr.i2]]
           [nv (cross tr.e1 tr.e2)] [ng (normalize nv)] [nn (dot nv nv)]
           [n1v (* (cross tr.e2 nv) (/ 1.0 nn))] [n2v (* (cross nv tr.e1) (/ 1.0 nn))]
        (memset r 0 128)
        (put3 r 0 nv) (set r[3] (bitcast u32 (dot nv tr.v0)))
        (put3 r 4 n1v) (set r[7] (bitcast u32 (- (dot n1v tr.v0))))
        (put3 r 8 n2v) (set r[11] (bitcast u32 (- (dot n2v tr.v0))))
        (put3 r 12 n0)
        (put3 r 16 (- n1 n0))
        (put3 r 20 (- n2 n0))
        (put3 r 24 ng)))
    (free order) (free skip) (free size) (free stack)))

;; ------------------------------------------------------------ camera ---

(struct camera [pos v3] [fwd v3] [right v3] [up v3] [tan f32])   ; tan = tan(fov_y / 2)

; on a circle around the teapot; 0 = the original viewpoint
(defn orbit [angle f32] camera
  (let [centre (vec3 0.2 0.0 0.0)] [look (vec3 0.2 1.25 0.0)]
       [radius f32 9.44] [height f32 4.4] [a0 f32 0.635]          ; a0 = atan2(5.6, 7.6)
       [pos (vec3 (+ centre.x (* radius (math::sinf (+ a0 angle)))) height (+ centre.z (* radius (math::cosf (+ a0 angle)))))]
       [fwd (normalize (- look pos))]
       [right (normalize (cross fwd (vec3 0.0 1.0 0.0)))]
    {camera :pos pos :fwd fwd :right right :up (cross right fwd) :tan (math::tanf (/ (* 34.0 PI) 360.0))}))

; anywhere around the teapot: yaw as orbit's angle, pitch up from orbit's
; height, the distance scaled by zoom (yaw 0, pitch 0, zoom 1 = (orbit 0.0))
(defn view [yaw f32] [pitch f32] [zoom f32] camera
  (let [look (vec3 0.2 1.25 0.0)]
       [a0 f32 0.635] [dist (* zoom (math::sqrtf (+ (* 9.44 9.44) (* 3.15 3.15))))]   ; orbit's radius, height above look
       [el (+ (math::atan2f 3.15 9.44) pitch)]
       [pos (+ look (vec3 (* dist (math::cosf el) (math::sinf (+ a0 yaw))) (* dist (math::sinf el)) (* dist (math::cosf el) (math::cosf (+ a0 yaw)))))]
       [fwd (normalize (- look pos))]
       [right (normalize (cross fwd (vec3 0.0 1.0 0.0)))]
    {camera :pos pos :fwd fwd :right right :up (cross right fwd) :tan (math::tanf (/ (* 34.0 PI) 360.0))}))

; the look, shared with the GPU kernel
(def light-pos v3 {v3 :x -6.0 :y 8.0 :z 0.5})
(const LIGHT-RADIUS 1.6)
(def light-col v3 {v3 :x 1.25 :y 1.15 :z 1.0})
(def sky-horizon v3 {v3 :x 0.82 :y 0.86 :z 0.93})
(def sky-zenith v3 {v3 :x 0.30 :y 0.45 :z 0.78})

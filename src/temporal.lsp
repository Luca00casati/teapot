; temporal.lsp — temporal accumulation of animation frames: each new frame is
; blended with the previous ones, reprojected through the per-pixel depth (with
; depth rejection), on several threads over rows.

(import "libc")
(import "thread")
(import "math")
(import "prelude")
(import "vec")
(import "scene")
(import "tone")
(using libc prelude vec)

(struct- view [pos v3] [fwd v3] [right v3] [up v3] [tx f32] [ty f32])

(struct state
  ; history: linear RGB + sample count, and view-space depth (1e9 = sky);
  ; the frame being blended writes `next`, which then becomes `hist`
  [hist (ptr f32)] [hist-depth (ptr f32)] [next (ptr f32)] [next-depth (ptr f32)]
  [sx (ptr f32)] [sy (ptr f32)]          ; pixel-centre ray offsets
  [il (ptr f32)]                         ; 1 / ray length
  [hist-view view]
  [ok bool])

; one worker's share: rows y0..y1
(struct- rows-job
  [t (ptr state)] [frame (ptr u32)] [w u32] [h u32] [depth (ptr f32)] [v view]
  [y0 u32] [y1 u32] [thread thread::pthread_t])

(defn- blend-rows [job (ptr rows-job)] void
  (let [w job.w] [h job.h] [v job.v] [pv job.t.hist-view] [ok job.t.ok]
       [ph job.t.hist] [pd job.t.hist-depth] [nh job.t.next] [nd job.t.next-depth]
       [dp (- v.pos pv.pos)]
       [itx (/ (* 0.5 (cast f32 w)) pv.tx)]
       [ity (/ (* 0.5 (cast f32 h)) pv.ty)]
    (for [y job.y0 job.y1]
      (let [base (+ v.fwd (* v.up job.t.sy[y]))]
        (for [x u32 0 w]
          (let [i (+ (* y w) x)]
               [p job.frame[i]]
               [c {(array f32 3) tone::lin8[(& (>> p 16) 255)] tone::lin8[(& (>> p 8) 255)] tone::lin8[(& p 255)]}]
               [t job.depth[i]]
               [il job.t.il[i]]
               [hc (array f32 3)] [n f32 0.0] [wsum f32 0.0]
               [glossy (< t 0.0)]       ; the GPU marks teapot pixels with a negative depth
               [sky (>= t 1e5)]
            (set t (math::fabsf t))
            (when ok
              ; world point of this pixel relative to the previous camera
              (let [s (* (if sky 1e4 t) il)]
                   [q (+ dp (* (+ base (* v.right job.t.sx[x])) s))]
                   [z (dot q pv.fwd)]
                (when (> z 1e-3)
                  (let [rz (/ 1.0 z)]
                       [fx (- (+ (* (cast f32 w) 0.5) (* (dot q pv.right) rz itx)) 0.5)]
                       [fy (- (- (* (cast f32 h) 0.5) (* (dot q pv.up) rz ity)) 0.5)]
                       [ix (cast i32 (math::floorf fx))] [iy (cast i32 (math::floorf fy))]
                       [ax (- fx (cast f32 ix))] [ay (- fy (cast f32 iy))]
                       [tol (+ (* 0.02 z) 0.02)]
                    ; bilinear, but only over taps that saw the same surface
                    (for [k 0 4]
                      (let [tx (+ ix (& k 1))] [ty (+ iy (>> k 1))]
                        (unless (or (< tx 0) (< ty 0) (>= tx (cast i32 w)) (>= ty (cast i32 h)))
                          (let [j (+ (* (cast u32 ty) w) (cast u32 tx))]
                               [hd pd[j]]
                            (unless (if sky (< hd 1e8) (> (math::fabsf (- hd z)) tol))
                              (let [wt (+ (* (if (!= (& k 1) 0) ax (- 1.0 ax)) (if (!= (>> k 1) 0) ay (- 1.0 ay))) 1e-4)]
                                   [hp (+ ph (* j 4))]
                                (+= hc[0] (* hp[0] wt))
                                (+= hc[1] (* hp[1] wt))
                                (+= hc[2] (* hp[2] wt))
                                (+= n (* hp[3] wt))
                                (+= wsum wt)))))))))))
            (let [alpha f32 1.0]
              (if (> wsum 0.05)
                  (let [iw (/ 1.0 wsum)]
                    (*= hc[0] iw)
                    (*= hc[1] iw)
                    (*= hc[2] iw)
                    (*= n iw)
                    ; running mean, then an exponential average; reflections and highlights on
                    ; the teapot move with the view, so it keeps much less history than the floor
                    (set alpha (math::fmaxf (/ 1.0 (+ n 1.0)) (if glossy 0.5 0.2))))
                  (set n 0.0))
              (let [out (+ nh (* i 4))]
                (for [a 0 3] (set out[a] (+ hc[a] (* (- c[a] hc[a]) alpha))))
                (set out[3] (+ n 1.0))
                (set nd[i] (if sky 1e9 (* t il)))
                (let [oc {(array f32 3) (math::fminf out[0] 1.0) (math::fminf out[1] 1.0) (math::fminf out[2] 1.0)}]
                  (set job.frame[i] (tone::encode oc 4095.0)))))))))))

(defn- blend-thread [arg (ptr void)] (ptr void)
  (blend-rows (cast (ptr rows-job) arg))
  null)

(defn- falloc [n u64] (ptr f32) (cast (ptr f32) (calloc n (sizeof f32))))

; blend `frame` (XRGB8888, w x h, rendered from `cam` with primary hit
; distances `depth`, negative on the glossy teapot) with the history, in place
(defn blend [t (ptr state)] [frame (ptr u32)] [w u32] [h u32] [depth (ptr f32)] [cam (ptr scene::camera)] [threads u32] void
  (let [n (cast u64 (* w h))]
       [v {view :pos cam.pos :fwd cam.fwd :right cam.right :up cam.up
                :tx (/ (* cam.tan (cast f32 w)) (cast f32 h)) :ty cam.tan}]
    (when (= t.hist null)
      (set t.hist (falloc (* n 4)))
      (set t.next (falloc (* n 4)))
      (set t.hist-depth (falloc n))
      (set t.next-depth (falloc n))
      ; the ray length only depends on the pixel, not on the camera
      (set t.sx (falloc (cast u64 w)))
      (set t.sy (falloc (cast u64 h)))
      (set t.il (falloc n))
      (for [x u32 0 w] (set t.sx[x] (* (- (/ (* (+ (cast f32 x) 0.5) 2.0) (cast f32 w)) 1.0) v.tx)))
      (for [y u32 0 h] (set t.sy[y] (* (- 1.0 (/ (* (+ (cast f32 y) 0.5) 2.0) (cast f32 h))) v.ty)))
      (for [i u64 0 n]
        (let [sx t.sx[(% i (cast u64 w))]] [sy t.sy[(/ i (cast u64 w))]]
          (set t.il[i] (/ 1.0 (math::sqrtf (+ 1.0 (* sx sx) (* sy sy))))))))
    (let [jobs (cast (ptr rows-job) (calloc (cast u64 threads) (sizeof rows-job)))]
      (for [k u32 0 threads]
        (let [job (+ jobs k)]
          (set (deref job) {rows-job :t t :frame frame :w w :h h :depth depth :v v
                                     :y0 (/ (* h k) threads) :y1 (/ (* h (+ k 1)) threads)})
          (thread::pthread_create (addr job.thread) null blend-thread job)))
      (for [k u32 0 threads] (thread::pthread_join (. jobs[k] thread) null))
      (free jobs))
    (let [tmp t.hist] (set t.hist t.next) (set t.next tmp))
    (let [tmp t.hist-depth] (set t.hist-depth t.next-depth) (set t.next-depth tmp))
    (set t.hist-view v)
    (set t.ok true)))

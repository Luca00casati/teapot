; gpu.lsp — the GPU backend, without Mesa or any userspace driver:
;
;   rt-kernel.lsp   the ray tracer, written once in the kernel IR (kir.lsp)
;        │
;   gen75.lsp       hardware: IR → machine code, command stream
;        │
;   i915.lsp        driver: kernel driver ioctls
;
; gpu::open picks a driver that finds a GPU some hardware layer accepts,
; compiles the kernel, uploads the scene, and fills the uniforms per pass.
; Passes render asynchronously: gpu::submit queues `samples` per pixel of the
; camera's view and returns, gpu::collect waits and hands back XRGB8888 pixels
; plus the primary hit distance per pixel (negative on the teapot, 1e6 for sky).
;
; lisc -DTEAPOT_GPU_DEBUG renders a heat map of the traversal instead.

(import "libc")
(import "prelude")
(import "layers")
(import "kir")
(import "vec")
(import "scene")
(import "rt-kernel" :as rt)
(import "gen75")
(import "i915")
(using libc prelude layers vec)

(struct device
  [hw (ptr hardware)] [drv (ptr driver)]
  [bufs (array buffer 3)]                ; scene, rgb, depth
  [w u32] [h u32] [w4 u32]               ; w4: image row length on the GPU (whole tiles)
  [layout u64] [nnodes u32])

; the hardware layers, tried in order on the chip a driver finds
(def- accepted (ptr hardware) null)
(defn- accept-chip [id u32] [desc (ptr void)] bool
  (set accepted (gen75::probe id (cast (ptr i8) desc)))
  (!= accepted null))

; null if no supported GPU is found
(defn open [s (ptr scene::model)] [w u32] [h u32] (ptr device)
  (let [path (array i8 64)] [desc (array i8 64)] [drv (i915::connect accept-chip desc path)]
    (when (= drv null)
      (fprintf stderr c"device: no supported GPU found\n")
      (return null))
    (let [hw accepted] [g (cast (ptr device) (calloc 1 (sizeof device)))]
         [tx (/ (+ w 3) 4)] [ty (/ (+ h 3) 4)] [img (* (cast u64 tx) (cast u64 ty) 16 4)]
      (fprintf stderr c"device: %s %s (%s, %s)\n" path desc drv.name hw.name)
      (rt::build (defined TEAPOT_GPU_DEBUG))   ; debug: heat map of the traversal
      (unless (hw.compile hw (addr kir::prog)) (return null))
      (unless (and (drv.alloc drv (scene::blob-bytes s) (addr g.bufs[rt::BUF-SCENE]))
                   (drv.alloc drv img (addr g.bufs[rt::BUF-RGB]))
                   (drv.alloc drv img (addr g.bufs[rt::BUF-DEPTH])))
        (return null))
      (drv.sync drv (addr g.bufs[rt::BUF-SCENE]) true)
      (scene::write-blob s (cast (ptr u32) (. g.bufs[rt::BUF-SCENE] map)))
      (unless (hw.setup hw drv tx ty (addr g.bufs[0])) (return null))
      (set g.hw hw)
      (set g.drv drv)
      (set g.w w)
      (set g.h h)
      (set g.w4 (* tx 4))
      (set g.layout (scene::layout-bytes s))
      (set g.nnodes s.nnode)
      g)))

(defn- put-f [u (ptr u32)] [i i32] [v f32] void (set u[i] (bitcast u32 v)))
(defn- put-v3 [u (ptr u32)] [i i32] [v v3] void (put-f u i v.x) (put-f u (+ i 1) v.y) (put-f u (+ i 2) v.z))

; Callers keep passes short (a few samples) so a batch stays far below the
; kernel's hang detection.
(defn submit [g (ptr device)] [cam (ptr scene::camera)] [samples u32] [seed u32] [sample-base u32] bool
  (let [u (array u32 32)]
    (put-v3 u rt::U-CAM cam.pos) (put-v3 u rt::U-FWD cam.fwd) (put-v3 u rt::U-RIGHT cam.right) (put-v3 u rt::U-UP cam.up)
    (put-v3 u rt::U-LIGHT scene::light-pos)
    (put-f u rt::U-TAN-X (* cam.tan (/ (cast f32 g.w) (cast f32 g.h))))
    (put-f u rt::U-TAN-Y cam.tan)
    (put-f u rt::U-INV-W (/ 2.0 (cast f32 g.w)))
    (put-f u rt::U-INV-H (/ 2.0 (cast f32 g.h)))
    (put-f u rt::U-LSIZE (* 2.0 scene::LIGHT-RADIUS))
    (put-f u rt::U-INV-SPP (/ 1.0 (cast f32 samples)))
    (put-f u rt::U-SAMPLE-BASE (cast f32 sample-base))
    (set u[rt::U-TRI-BASE] (cast u32 (/ (* 8 g.layout) 16)))
    (set u[rt::U-NNODES] g.nnodes)
    (for [a 0 3] (set u[(+ rt::U-STRIDE a)] (cast u32 (/ (<< g.layout a) 16))))
    (set u[rt::U-SPP] samples)
    (set u[rt::U-SEED] seed)
    (g.hw.run g.hw u)))

(defn collect [g (ptr device)] [rgb (ptr u32)] [depth (ptr f32)] bool
  (g.drv.sync g.drv (addr g.bufs[rt::BUF-RGB]) false)   ; waits for the GPU
  (when (g.drv.hung g.drv)
    (fprintf stderr c"device: kernel hung the GPU (reset by the driver), giving up on the GPU path\n")
    (return false))
  (let [src (cast (ptr u32) (. g.bufs[rt::BUF-RGB] map))]
    (for [y u32 0 g.h] (memcpy (+ rgb (* y g.w)) (+ src (* y g.w4)) (* (cast u64 g.w) 4))))
  (when (!= depth null)
    (g.drv.sync g.drv (addr g.bufs[rt::BUF-DEPTH]) false)
    (let [src (cast (ptr f32) (. g.bufs[rt::BUF-DEPTH] map))]
      (for [y u32 0 g.h] (memcpy (+ depth (* y g.w)) (+ src (* y g.w4)) (* (cast u64 g.w) 4)))))
  true)

; main.lsp — Utah teapot ray tracer that talks to the display only through
; kernel interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa,
; no X11/Wayland/GL/Vulkan), or a PPM file.
;
;   teapot                  camera orbits the teapot on the first DRM output (run from a VT);
;                           built with make SDL=1, in a desktop window instead (see window.lsp)
;   teapot -x               the desktop window (make SDL=1): drag to turn, fps in the title
;   teapot -p               still image instead, refined until Enter is pressed
;   teapot -o out.ppm       render a still to a file (-w/-h set the size)
;   teapot -o f.ppm -n 120  render a 120-frame orbit to f_0000.ppm ... f_0119.ppm
;   options: -s <spp> samples per pixel (per frame when animating), -r <n> render at
;            1/n resolution on screen, -b no temporal accumulation of animation frames,
;            -t <sec> stop after this long, -d <card>
;
; Rendering runs on the GPU (gpu.lsp: the ray tracer in a small kernel IR,
; compiled by our own hardware layer and submitted with raw kernel-driver
; ioctls, no Mesa; today Intel Haswell on i915).

(import "libc")
(import "posix")
(import "io")
(import "signal")
(import "time")
(import "prelude")
(import "scene")
(import "tone")
(import "temporal")
(import "gpu")
(import "display")
(if-defined TEAPOT_SDL (import "window"))
(using libc posix io prelude)

; set by the signal handler, read by the loops: atomic, so no load is cached
(def quit i32 0)
(defn on-signal [sig i32] (atomic-store quit 1 :relaxed))
(defn quitting [] -> bool (return (!= (atomic-load quit :relaxed) 0)))

(defn now [] -> f64
  (let [ts {time::timespec}]
    (time::clock_gettime time::CLOCK_MONOTONIC (addr ts))
    (return (+ (cast f64 ts.tv_sec) (* (cast f64 ts.tv_nsec) 1e-9)))))

(defn key-pressed [] -> bool
  (let [p {pollfd :fd 0 :events POLLIN}]
    (return (> (poll (addr p) 1 0) 0))))
(defn read-key [] -> i64 (let [c u8 0] (return (read 0 (addr c) 1))))

(const MAX-PASS 8)                       ; samples per submitted pass (keeps every GPU batch short)
(const ORBIT-SECONDS 12.0)               ; one revolution of the camera on screen

(defn pass-seed [i u32] -> u32
  (let [s (+ (* i 0x9E3779B9) 0x7f4a7c15)]
    (set s (^ s (>> s 15)))
    (set s (* s 0x2c1b3c6d))
    (return (^ s (>> s 12)))))

;; passes of a few samples each, averaged in linear light
(struct accum [rgb (ptr u32)] [acc (ptr f32)] [n u32] [samples u32])

(defn accum-new [n u32] -> accum
  (return {accum :rgb (cast (ptr u32) (calloc (cast u64 n) 4)) :acc (cast (ptr f32) (calloc (* 3 (cast u64 n)) 4)) :n n}))
(defn accum-free [a (ptr accum)] (free a.rgb) (free a.acc))

(defn accum-add [a (ptr accum)] [g (ptr gpu::device)] [cam (ptr scene::camera)] [k u32] [seed u32] [sample-base u32] [depth (ptr f32)] -> bool
  (unless (and (gpu::submit g cam k seed sample-base) (gpu::collect g a.rgb depth)) (return false))
  (for [i u32 0 a.n]
    (let [p a.rgb[i]]
      (for [c u32 0 3]
        (+= a.acc[(+ (* 3 i) c)] (* tone::lin8[(& (>> p (- 16 (* 8 c))) 255)] (cast f32 k))))))
  (+= a.samples k)
  (return true))

(defn accum-resolve [a (ptr accum)] [t display::target]
  (let [k (/ 4095.0 (cast f32 a.samples))]
    (for [y u32 0 t.h]
      (let [row (display::trow t y)]
        (for [x u32 0 t.w] (set row[x] (tone::encode (+ a.acc (* (+ (* y t.w) x) 3)) k)))))))

; The progressive still image, drawn into tgt (the display or the file
; buffer). With `refine` (on screen) it keeps adding passes until a key is
; pressed; *key-stop then tells the caller.
(defn still-image [g (ptr gpu::device)] [cam (ptr scene::camera)] [tgt display::target] [disp (ptr display::drm)] [spp (ptr u32)] [refine bool] [key-stop (ptr bool)] -> bool
  (let [per-pass u32 4] [acc (accum-new (* tgt.w tgt.h))] [ok true] [pass u32 0]
    (defer (accum-free (addr acc)))
    (while (!= (% (deref spp) per-pass) 0) (-= per-pass 1))
    (while (not (quitting))
      (when (>= acc.samples (deref spp))
        ; on screen: keep refining until a key is pressed (or 1024 samples)
        (when (or (not refine) (>= acc.samples 1024)) (break))
        (when (key-pressed)
          (set (deref key-stop) true)
          (break)))
      (set ok (accum-add (addr acc) g cam per-pass (pass-seed pass) acc.samples null))
      (unless ok (break))
      (accum-resolve (addr acc) tgt)
      (when (!= disp null) (display::dirty disp))
      (when (and refine (>= acc.samples (deref spp)))
        (fprintf stderr c"\r%u samples/pixel, press Enter to exit " acc.samples))
      (+= pass 1))
    (when refine (fprintf stderr c"\n"))
    (set (deref spp) acc.samples)
    (return ok)))

; one animation frame with more samples than a single pass
(defn frame-passes [g (ptr gpu::device)] [cam (ptr scene::camera)] [frame display::target] [spp u32] [f u32] [depth (ptr f32)] -> bool
  (let [acc (accum-new (* frame.w frame.h))] [ok true]
    (defer (accum-free (addr acc)))
    (while (and ok (< acc.samples spp))
      (let [done acc.samples] [k (if (< (- spp done) MAX-PASS) (- spp done) MAX-PASS)]
        (set ok (accum-add (addr acc) g cam k (pass-seed (+ (* f 1024) done)) (+ (* f spp) done) depth))))
    (when ok (accum-resolve (addr acc) frame))
    (return ok)))

; out.ppm → out_0007.ppm; a name with a % in it is a printf pattern for the frame number
(defn frame-name [o (ptr i8)] [f u32] [name (ptr i8)] [n u64]
  (let [dot (strrchr o 46)]            ; '.'
    (cond ((!= (strchr o 37) null) (snprintf name n o f))   ; '%'
          ((!= dot null) (snprintf name n c"%.*s_%04u%s" (cast i32 (- dot o)) o f dot))
          (else (snprintf name n c"%s_%04u.ppm" o f)))))

(def disp display::drm {display::drm :fd -1})

(defn run [argc i32] [argv (ptr (ptr i8))] -> i32
  (let [out (ptr i8) null] [card (ptr i8) null]
       [W u32 1280] [H u32 720] [hold -1] [nframes u32 0] [rscale u32 1] [spp-arg 0]
       [still false] [no-blend false] [in-window false] [opt 0]
    (while (!= (set-opt (addr opt) (getopt argc argv c"o:d:w:h:s:t:pn:r:bx")) -1)
      (case opt (#\o (set out optarg))
        (#\d (set card optarg))
        (#\w (set W (cast u32 (atoi optarg))))
        (#\h (set H (cast u32 (atoi optarg))))
        (#\s (set spp-arg (atoi optarg)))
        (#\t (set hold (atoi optarg)))
        (#\p (set still true))
        (#\n (set nframes (cast u32 (atoi optarg))))
        (#\r (set rscale (if (> (atoi optarg) 0) (cast u32 (atoi optarg)) 1)))
        (#\b (set no-blend true))
        (#\x (set in-window true))
        (else
         (fprintf stderr c"usage: %s [-o file.ppm [-n frames]] [-p] [-x] [-d /dev/dri/cardN]\n       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds]\n"
                  argv[0])
         (return 1))))

    (tone::init)
    (let [t0 (now)] [scene (scene::build)] [filebuf (ptr u32) null] [tgt {display::target}]
      (fprintf stderr c"teapot: %u triangles, %u BVH nodes (%.0f ms)\n" scene.ntri scene.nnode (* (- (now) t0) 1e3))
      ; a desktop window: asked for, or (built with SDL) the screen belongs to a desktop session
      (if-defined TEAPOT_SDL
        (let [desktop (or (!= (getenv c"DISPLAY") null) (!= (getenv c"WAYLAND_DISPLAY") null))]
          (when (and (= out null) (or in-window (and desktop (= card null) (not still))))
            (signal::signal signal::SIGINT on-signal)
            (signal::signal signal::SIGTERM on-signal)
            (return (run-window (addr scene) W H rscale (if (> spp-arg 0) (cast u32 spp-arg) 1) no-blend hold))))
        (when in-window
          (fprintf stderr c"-x: this teapot was built without SDL2; build it with make SDL=1\n")
          (return 1)))
      (if (!= out null)
          (do (set filebuf (cast (ptr u32) (calloc (* (cast u64 W) H) 4)))
              (set tgt (display::xrgb filebuf W H)))
          (do
            (if (!= card null)
                (display::open-card (addr disp) card)
                (for [i 0 8]
                  (let [p (array i8 32)]
                    (snprintf p 32 c"/dev/dri/card%d" i)
                    (when (and (= (access p F_OK) 0) (display::open-card (addr disp) p)) (break)))))
            (when (< disp.fd 0)
              (fprintf stderr c"no display available; use -o out.ppm\n")
              ; a runtime check, not a build switch: is a desktop session holding the screen?
              (when (or (!= (getenv c"DISPLAY") null) (!= (getenv c"WAYLAND_DISPLAY") null))
                (fprintf stderr c"a desktop session owns the screen: switch to a text console (Ctrl+Alt+F3), log in and run it there\n"))
              (return 1))
            (set tgt (display::front (addr disp)))))

      (signal::signal signal::SIGINT on-signal)
      (signal::signal signal::SIGTERM on-signal)
      (let [nthreads (sysconf _SC_NPROCESSORS_ONLN)] [threads u32 (if (< nthreads 1) 1 (cast u32 nthreads))]
           [auto-spp (<= spp-arg 0)] [spp u32 (if (> spp-arg 0) (cast u32 spp-arg) 0)]

        (when (if (!= out null) (> nframes 0) (not still))
          ;; ---- animation: the camera orbits the teapot ----
          (let [scale (if (!= out null) 1 rscale)]
               [fw (/ (- (+ tgt.w scale) 1) scale)] [fh (/ (- (+ tgt.h scale) 1) scale)]
               [framebuf (cast (ptr u32) (calloc (* (cast u64 fw) fh) 4))]
               [frame (display::xrgb framebuf fw fh)]
               [g (gpu::open (addr scene) fw fh)]
            (when (= g null) (return 1))
            (when auto-spp (set spp (if (!= out null) 16 1)))
            ; on screen the GPU renders few samples per frame and accumulates them over time
            (let [blend (and (= out null) (not no-blend))]
                 [history {temporal::state}]
                 [depth (ptr f32) (if blend (cast (ptr f32) (calloc (* (cast u64 fw) fh) 4)) null)]
                 [start (now)] [tick (now)] [frames u32 0] [tick-frames u32 0]
                 ; light frames are pipelined: frame f+1 renders while f is blended and shown
                 [pipelined (<= spp MAX-PASS)]
                 [next (scene::orbit 0.0)] [f u32 0]
              (fprintf stderr c"animating %ux%u @ %u spp%s on the gpu%s\n" fw fh spp
                       (if blend c" + temporal accumulation" c"") (if (!= out null) c"" c" (Enter or Ctrl-C to stop)"))
              (set start (now))
              (set tick start)
              (set next (scene::orbit (angle-of out f start nframes)))
              (when (and pipelined (not (gpu::submit g (addr next) spp (pass-seed 0) 0))) (return 1))
              (while (not (quitting))
                (let [cam next] [ok true]
                  (if pipelined
                      (do
                        (set ok (gpu::collect g framebuf depth))
                        (when (and ok (or (= out null) (< (+ f 1) nframes)))
                          (set next (scene::orbit (angle-of out (+ f 1) start nframes)))
                          (set ok (gpu::submit g (addr next) spp (pass-seed (+ f 1)) (* (+ f 1) spp)))))
                      (do
                        (set cam (scene::orbit (angle-of out f start nframes)))
                        (set ok (frame-passes g (addr cam) frame spp f depth))))
                  (unless ok (return 1))
                  (when (quitting) (break))
                  (when blend (temporal::blend (addr history) framebuf fw fh depth (addr cam) threads))
                  (+= frames 1)
                  (+= tick-frames 1)
                  (if (!= out null)
                      (let [name (array i8 600)]
                        (frame-name out f name 600)
                        (unless (display::write-ppm name frame) (return 1))
                        (fprintf stderr c"\r%s (%u/%u)" name (+ f 1) nframes)
                        (when (>= (+ f 1) nframes)
                          (fprintf stderr c"\n")
                          (break)))
                      (let [t (now)]
                        (display::wait-flip (addr disp))
                        (display::blit frame (display::back (addr disp)) scale)
                        (display::flip (addr disp))
                        (when (>= (- t tick) 1.0)
                          (fprintf stderr c"\r%.1f fps " (/ (cast f64 tick-frames) (- t tick)))
                          (set tick t)
                          (set tick-frames 0))
                        (when (and (>= hold 0) (>= (- t start) (cast f64 hold))) (break))
                        (when (and (= (isatty 0) 1) (key-pressed))
                          (discard (read-key))   ; any key stops
                          (break))))
                  (+= f 1)))
              (let [secs (- (now) start)]
                (fprintf stderr c"\n%u frames in %.1f s (%.1f fps) on the gpu\n" frames secs (/ (cast f64 frames) secs)))
              (return 0))))

        ;; ---- still image ----
        (let [t1 (now)] [refine (and (= out null) (< hold 0) (= (isatty 0) 1))] [key-stop false]
             [cam (scene::orbit 0.0)]
          (when auto-spp (set spp 32))
          (let [g (gpu::open (addr scene) tgt.w tgt.h)]
            (when (= g null) (return 1))
            (unless (still-image g (addr cam) tgt (if (= out null) (addr disp) null) (addr spp) refine (addr key-stop))
              (return 1)))
          (let [secs (- (now) t1)]
            (when (!= out null)
              (fprintf stderr c"rendered %ux%u @ %u spp on the gpu in %.2f s\n" W H spp secs)
              (return (if (display::write-ppm out tgt) 0 1)))
            ; hold the image: until Enter / Ctrl-C, or for -t seconds
            (cond (key-stop (read-key))         ; the key only ends the hold
                  ((not (quitting))
                   (let [ms (cond ((>= hold 0) (* hold 1000)) ((= (isatty 0) 1) -1) (else 10000))]
                        [p {pollfd :fd 0 :events POLLIN}]
                     (when (and (> (poll (addr p) 1 ms) 0) (!= (isatty 0) 1) (= (read-key) 0))
                       ; stdin at EOF: don't flash and exit
                       (sleep (if (>= hold 0) 0 10))))))
            (fprintf stderr c"rendered %ux%u @ %u spp on the gpu in %.2f s\n" tgt.w tgt.h spp secs)
            (return 0)))))))

; The real-time view in a desktop window: the GPU renders frame f+1 while
; frame f is blended and shown; the mouse and keys move the camera.
(if-defined TEAPOT_SDL
  (defn run-window [s (ptr scene::model)] [W u32] [H u32] [rscale u32] [spp-arg u32] [no-blend bool] [hold i32] -> i32
    (let [fw (/ (- (+ W rscale) 1) rscale)] [fh (/ (- (+ H rscale) 1) rscale)]
         [spp (if (> spp-arg MAX-PASS) MAX-PASS spp-arg)]   ; (one pass per frame)
         [framebuf (cast (ptr u32) (calloc (* (cast u64 fw) fh) 4))]
         [frame (display::xrgb framebuf fw fh)]
         [g (gpu::open s fw fh)]
      (defer (free framebuf))
      (when (= g null) (return 1))
      (let [wd (window::open c"teapot" W H fw fh)]
        (when (= wd null) (return 1))
        (defer (window::close wd))
        (let [blend (not no-blend)] [history {temporal::state}]
             [depth (ptr f32) (if blend (cast (ptr f32) (calloc (* (cast u64 fw) fh) 4)) null)]
             [nthreads (sysconf _SC_NPROCESSORS_ONLN)] [threads u32 (if (< nthreads 1) 1 (cast u32 nthreads))]
             [v (window::start-view)] [cam (scene::view 0.0 0.0 1.0)]
             [start (now)] [last start] [tick start] [tick-frames u32 0] [frames u32 0] [f u32 0]
             [title (array i8 160)]
          (defer (free depth))
          (fprintf stderr c"window %ux%u, rendering %ux%u @ %u spp%s on the gpu\n" W H fw fh spp (if blend c" + temporal accumulation" c""))
          (unless (gpu::submit g (addr cam) spp (pass-seed 0) 0) (return 1))
          (while (not (quitting))
            (unless (gpu::collect g framebuf depth) (return 1))
            (let [shown cam] [t (now)]
              ; the next frame's camera, from the input so far; it renders while this one is shown
              (unless (window::poll wd (addr v) (cast f32 (- t last))) (break))
              (set last t)
              (set cam (scene::view v.yaw v.pitch v.zoom))
              (unless (gpu::submit g (addr cam) spp (pass-seed (+ f 1)) (* (+ f 1) spp)) (return 1))
              (when blend (temporal::blend (addr history) framebuf fw fh depth (addr shown) threads))
              (window::show wd frame)
              (+= frames 1)
              (+= tick-frames 1)
              (+= f 1)
              (when (>= (- t tick) 0.5)
                (snprintf title 160 c"teapot  -  %.1f fps  -  %ux%u @ %u spp  -  drag to turn, wheel to zoom, space: %s"
                          (/ (cast f64 tick-frames) (- t tick)) fw fh spp (if v.auto c"stop" c"orbit"))
                (window::set-title wd title)
                (set tick t)
                (set tick-frames 0))
              (when (and (>= hold 0) (>= (- t start) (cast f64 hold))) (break))))
          (let [secs (- (now) start)]
            (fprintf stderr c"%u frames in %.1f s (%.1f fps) on the gpu\n" frames secs (/ (cast f64 frames) secs)))
          (return 0))))))

; the camera angle of animation frame f: evenly around the orbit for a file,
; by the clock on screen
(defn angle-of [out (ptr i8)] [f u32] [start f64] [nframes u32] -> f32
  (if (!= out null)
      (return (/ (* 2.0 PI (cast f32 f)) (cast f32 nframes)))
      (return (cast f32 (/ (* (- (now) start) 2.0 3.14159265358979323846) ORBIT-SECONDS)))))

; getopt's result, kept for the loop condition
(defn set-opt [p (ptr i32)] [v i32] -> i32 (set (deref p) v) (return v))

(defn main [argc i32] [argv (ptr (ptr i8))] -> i32
  ; run returns instead of exiting so the display is restored on the way out
  (let [r (run argc argv)]
    (display::close-card (addr disp))
    (return r)))

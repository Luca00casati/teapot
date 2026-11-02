; main.lsp — Utah teapot ray tracer that talks to the display only through
; kernel interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa,
; no X11/Wayland/GL/Vulkan), or a PPM file.
;
;   teapot                  camera orbits the teapot on the first DRM output (run from a VT)
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
(using libc posix io prelude)

; set by the signal handler, read by the loops: atomic, so no load is cached
(def quit i32 0)
(defn on-signal ((sig i32)) void (atomic-store quit 1 :relaxed))
(defn quitting () bool (!= (atomic-load quit :relaxed) 0))

(defn now () f64
  (let ((ts (make time::timespec)))
    (time::clock_gettime time::CLOCK_MONOTONIC (addr ts))
    (+ (cast f64 ts.tv_sec) (* (cast f64 ts.tv_nsec) 1e-9))))

(defn key-pressed () bool
  (let ((p (make pollfd :fd 0 :events POLLIN)))
    (> (poll (addr p) 1 0) 0)))
(defn read-key () i64 (let ((c u8 0)) (read 0 (addr c) 1)))

(const MAX-PASS 8)                       ; samples per submitted pass (keeps every GPU batch short)
(const ORBIT-SECONDS 12.0)               ; one revolution of the camera on screen

(defn pass-seed ((i u32)) u32
  (let ((s (+ (* i 0x9E3779B9) 0x7f4a7c15)))
    (set s (^ s (>> s 15)))
    (set s (* s 0x2c1b3c6d))
    (^ s (>> s 12))))

;; passes of a few samples each, averaged in linear light
(struct accum (rgb (ptr u32)) (acc (ptr f32)) (n u32) (samples u32))

(defn accum-new ((n u32)) accum
  (make accum :rgb (cast (ptr u32) (calloc (cast u64 n) 4)) :acc (cast (ptr f32) (calloc (* 3 (cast u64 n)) 4)) :n n))
(defn accum-free ((a (ptr accum))) void (free a.rgb) (free a.acc))

(defn accum-add ((a (ptr accum)) (g (ptr gpu::device)) (cam (ptr scene::camera)) (k u32) (seed u32) (sample-base u32) (depth (ptr f32))) bool
  (unless (and (gpu::submit g cam k seed sample-base) (gpu::collect g a.rgb depth)) (return false))
  (for (i u32 0 a.n)
    (let ((p (aref a.rgb i)))
      (for (c u32 0 3)
        (+= (aref a.acc (+ (* 3 i) c)) (* (aref tone::lin8 (& (>> p (- 16 (* 8 c))) 255)) (cast f32 k))))))
  (+= a.samples k)
  true)

(defn accum-resolve ((a (ptr accum)) (t display::target)) void
  (let ((k (/ 4095.0 (cast f32 a.samples))))
    (for (y u32 0 t.h)
      (let ((row (display::trow t y)))
        (for (x u32 0 t.w) (set (aref row x) (tone::encode (+ a.acc (* (+ (* y t.w) x) 3)) k)))))))

; The progressive still image, drawn into tgt (the display or the file
; buffer). With `refine` (on screen) it keeps adding passes until a key is
; pressed; *key-stop then tells the caller.
(defn still-image ((g (ptr gpu::device)) (cam (ptr scene::camera)) (tgt display::target) (disp (ptr display::drm)) (spp (ptr u32)) (refine bool) (key-stop (ptr bool))) bool
  (let ((per-pass u32 4) (acc (accum-new (* tgt.w tgt.h))) (ok true) (pass u32 0))
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
        (fprintf stderr "\r%u samples/pixel, press Enter to exit " acc.samples))
      (+= pass 1))
    (when refine (fprintf stderr "\n"))
    (set (deref spp) acc.samples)
    (accum-free (addr acc))
    ok))

; one animation frame with more samples than a single pass
(defn frame-passes ((g (ptr gpu::device)) (cam (ptr scene::camera)) (frame display::target) (spp u32) (f u32) (depth (ptr f32))) bool
  (let ((acc (accum-new (* frame.w frame.h))) (ok true))
    (while (and ok (< acc.samples spp))
      (let ((done acc.samples) (k (if (< (- spp done) MAX-PASS) (- spp done) MAX-PASS)))
        (set ok (accum-add (addr acc) g cam k (pass-seed (+ (* f 1024) done)) (+ (* f spp) done) depth))))
    (when ok (accum-resolve (addr acc) frame))
    (accum-free (addr acc))
    ok))

; out.ppm → out_0007.ppm; a name with a % in it is a printf pattern for the frame number
(defn frame-name ((o (ptr i8)) (f u32) (name (ptr i8)) (n u64)) void
  (let ((dot (strrchr o 46)))            ; '.'
    (cond ((!= (strchr o 37) null) (snprintf name n o f))   ; '%'
          ((!= dot null) (snprintf name n "%.*s_%04u%s" (cast i32 (- dot o)) o f dot))
          (else (snprintf name n "%s_%04u.ppm" o f)))))

(def disp display::drm (make display::drm :fd -1))

(defn run ((argc i32) (argv (ptr (ptr i8)))) i32
  (let ((out (ptr i8) null) (card (ptr i8) null)
        (W u32 1280) (H u32 720) (hold -1) (nframes u32 0) (rscale u32 1) (spp-arg 0)
        (still false) (no-blend false) (opt 0))
    (while (!= (set-opt (addr opt) (getopt argc argv "o:d:w:h:s:t:pn:r:b")) -1)
      (cond ((= opt 111) (set out optarg))                          ; o
            ((= opt 100) (set card optarg))                         ; d
            ((= opt 119) (set W (cast u32 (atoi optarg))))          ; w
            ((= opt 104) (set H (cast u32 (atoi optarg))))          ; h
            ((= opt 115) (set spp-arg (atoi optarg)))               ; s
            ((= opt 116) (set hold (atoi optarg)))                  ; t
            ((= opt 112) (set still true))                          ; p
            ((= opt 110) (set nframes (cast u32 (atoi optarg))))    ; n
            ((= opt 114) (set rscale (if (> (atoi optarg) 0) (cast u32 (atoi optarg)) 1)))   ; r
            ((= opt 98) (set no-blend true))                        ; b
            (else
             (fprintf stderr "usage: %s [-o file.ppm [-n frames]] [-p] [-d /dev/dri/cardN]\n       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds]\n"
                      (aref argv 0))
             (return 1))))

    (tone::init)
    (let ((t0 (now)) (scene (scene::build)) (filebuf (ptr u32) null) (tgt (make display::target)))
      (fprintf stderr "teapot: %u triangles, %u BVH nodes (%.0f ms)\n" scene.ntri scene.nnode (* (- (now) t0) 1e3))
      (if (!= out null)
          (do (set filebuf (cast (ptr u32) (calloc (* (cast u64 W) H) 4)))
              (set tgt (display::xrgb filebuf W H)))
          (do
            (if (!= card null)
                (display::open-card (addr disp) card)
                (for (i 0 8)
                  (let ((p (array i8 32)))
                    (snprintf p 32 "/dev/dri/card%d" i)
                    (when (and (= (access p F_OK) 0) (display::open-card (addr disp) p)) (break)))))
            (when (< disp.fd 0)
              (fprintf stderr "no display available; use -o out.ppm\n")
              ; a runtime check, not a build switch: is a desktop session holding the screen?
              (when (or (!= (getenv "DISPLAY") null) (!= (getenv "WAYLAND_DISPLAY") null))
                (fprintf stderr "a desktop session owns the screen: switch to a text console (Ctrl+Alt+F3), log in and run it there\n"))
              (return 1))
            (set tgt (display::front (addr disp)))))

      (signal::signal signal::SIGINT on-signal)
      (signal::signal signal::SIGTERM on-signal)
      (let ((nthreads (sysconf _SC_NPROCESSORS_ONLN)) (threads u32 (if (< nthreads 1) 1 (cast u32 nthreads)))
            (auto-spp (<= spp-arg 0)) (spp u32 (if (> spp-arg 0) (cast u32 spp-arg) 0)))

        (when (if (!= out null) (> nframes 0) (not still))
          ;; ---- animation: the camera orbits the teapot ----
          (let ((scale (if (!= out null) 1 rscale))
                (fw (/ (- (+ tgt.w scale) 1) scale)) (fh (/ (- (+ tgt.h scale) 1) scale))
                (framebuf (cast (ptr u32) (calloc (* (cast u64 fw) fh) 4)))
                (frame (display::xrgb framebuf fw fh))
                (g (gpu::open (addr scene) fw fh)))
            (when (= g null) (return 1))
            (when auto-spp (set spp (if (!= out null) 16 1)))
            ; on screen the GPU renders few samples per frame and accumulates them over time
            (let ((blend (and (= out null) (not no-blend)))
                  (history (make temporal::state))
                  (depth (ptr f32) (if blend (cast (ptr f32) (calloc (* (cast u64 fw) fh) 4)) null))
                  (start (now)) (tick (now)) (frames u32 0) (tick-frames u32 0)
                  ; light frames are pipelined: frame f+1 renders while f is blended and shown
                  (pipelined (<= spp MAX-PASS))
                  (next (scene::orbit 0.0)) (f u32 0))
              (fprintf stderr "animating %ux%u @ %u spp%s on the gpu%s\n" fw fh spp
                       (if blend " + temporal accumulation" "") (if (!= out null) "" " (Enter or Ctrl-C to stop)"))
              (set start (now))
              (set tick start)
              (set next (scene::orbit (angle-of out f start nframes)))
              (when (and pipelined (not (gpu::submit g (addr next) spp (pass-seed 0) 0))) (return 1))
              (while (not (quitting))
                (let ((cam next) (ok true))
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
                      (let ((name (array i8 600)))
                        (frame-name out f name 600)
                        (unless (display::write-ppm name frame) (return 1))
                        (fprintf stderr "\r%s (%u/%u)" name (+ f 1) nframes)
                        (when (>= (+ f 1) nframes)
                          (fprintf stderr "\n")
                          (break)))
                      (let ((t (now)))
                        (display::wait-flip (addr disp))
                        (display::blit frame (display::back (addr disp)) scale)
                        (display::flip (addr disp))
                        (when (>= (- t tick) 1.0)
                          (fprintf stderr "\r%.1f fps " (/ (cast f64 tick-frames) (- t tick)))
                          (set tick t)
                          (set tick-frames 0))
                        (when (and (>= hold 0) (>= (- t start) (cast f64 hold))) (break))
                        (when (and (= (isatty 0) 1) (key-pressed))
                          (read-key)             ; any key stops
                          (break))))
                  (+= f 1)))
              (let ((secs (- (now) start)))
                (fprintf stderr "\n%u frames in %.1f s (%.1f fps) on the gpu\n" frames secs (/ (cast f64 frames) secs)))
              (return 0))))

        ;; ---- still image ----
        (let ((t1 (now)) (refine (and (= out null) (< hold 0) (= (isatty 0) 1))) (key-stop false)
              (cam (scene::orbit 0.0)))
          (when auto-spp (set spp 32))
          (let ((g (gpu::open (addr scene) tgt.w tgt.h)))
            (when (= g null) (return 1))
            (unless (still-image g (addr cam) tgt (if (= out null) (addr disp) null) (addr spp) refine (addr key-stop))
              (return 1)))
          (let ((secs (- (now) t1)))
            (when (!= out null)
              (fprintf stderr "rendered %ux%u @ %u spp on the gpu in %.2f s\n" W H spp secs)
              (return (if (display::write-ppm out tgt) 0 1)))
            ; hold the image: until Enter / Ctrl-C, or for -t seconds
            (cond (key-stop (read-key))         ; the key only ends the hold
                  ((not (quitting))
                   (let ((ms (cond ((>= hold 0) (* hold 1000)) ((= (isatty 0) 1) -1) (else 10000)))
                         (p (make pollfd :fd 0 :events POLLIN)))
                     (when (and (> (poll (addr p) 1 ms) 0) (!= (isatty 0) 1) (= (read-key) 0))
                       ; stdin at EOF: don't flash and exit
                       (sleep (if (>= hold 0) 0 10))))))
            (fprintf stderr "rendered %ux%u @ %u spp on the gpu in %.2f s\n" tgt.w tgt.h spp secs)
            0))))))

; the camera angle of animation frame f: evenly around the orbit for a file,
; by the clock on screen
(defn angle-of ((out (ptr i8)) (f u32) (start f64) (nframes u32)) f32
  (if (!= out null)
      (/ (* 2.0 PI (cast f32 f)) (cast f32 nframes))
      (cast f32 (/ (* (- (now) start) 2.0 3.14159265358979323846) ORBIT-SECONDS))))

; getopt's result, kept for the loop condition
(defn set-opt ((p (ptr i32)) (v i32)) i32 (set (deref p) v) v)

(defn main ((argc i32) (argv (ptr (ptr i8)))) i32
  ; run returns instead of exiting so the display is restored on the way out
  (let ((r (run argc argv)))
    (display::close-card (addr disp))
    r))

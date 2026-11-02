; host.lsp — checks the host side against the previous implementations:
; the GPU scene data and the frame blending must give the same checksums as
; a strict-float (no -ffast-math, no FMA contraction) C++/Rust build.
(import "libc")
(import "../src/prelude")
(import "../src/scene")
(import "../src/tone")
(import "../src/temporal")
(using libc prelude)

(defn fnv [h u64] [v u32] u64 (* (^ h (cast u64 v)) 1099511628211))

(def failed i32 0)
(defn check [what (ptr i8)] [got u64] [want u64] void
  (printf c"%-9s %016lx %s\n" what got (if (= got want) c"ok" c"MISMATCH"))
  (when (!= got want) (set failed 1)))

(defn main [] i32
  (tone::init)
  (let [s (scene::build)] [n (scene::blob-bytes (addr s))] [blob (cast (ptr u32) (calloc 1 n))] [h u64 1469598103934665603]
    (scene::write-blob (addr s) blob)
    (for [i u64 0 (/ n 4)] (set h (fnv h blob[i])))
    (check c"scene" h 0xcaef619fa0c611bb))
  ; the same input as the Rust test and the C++ harness
  (let [w u32 320] [h u32 180] [px (cast (ptr u32) (calloc (cast u64 (* w h)) 4))] [dep (cast (ptr f32) (calloc (cast u64 (* w h)) 4))]
       [t {temporal::state}] [hash u64 1469598103934665603] [s u32 1]
    (for [f 0 4]
      (for [i u32 0 (* w h)]
        (set s (+ (* s 1664525) 1013904223))
        (set px[i] (>> s 8))
        (set dep[i] (if (= (% i 7) 0) 2e6 (* (if (!= (% i 3) 0) 1.0 -1.0) (+ 5.0 (* (cast f32 (% i w)) 0.01))))))
      (let [cam (scene::orbit (* (cast f32 f) 0.05))]
        (temporal::blend (addr t) px w h dep (addr cam) 4))
      (for [i u32 0 (* w h)] (set hash (fnv hash px[i]))))
    (check c"temporal" hash 0x203872f244413a09))
  failed)

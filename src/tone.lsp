; tone.lsp — passes and frames are averaged in linear light: 8-bit display
; values in through lin8, 12-bit linear back out through enc12.


(import "math")
(def lin8 (array f32 256))
(def enc12 (array u8 4096))

(defn init [] void
  (for [i 0 256] (set lin8[i] (math::powf (/ (cast f32 i) 255.0) 2.2)))
  (for [i 0 4096] (set enc12[i] (cast u8 (+ (* (math::powf (/ (cast f32 i) 4095.0) (/ 1.0 2.2)) 255.0) 0.5)))))

; XRGB8888 from linear c[0..3] in [0, 1], scaled by k / 4095
(defn encode [c (ptr f32)] [k f32] u32
  (| (<< (cast u32 enc12[(cast u64 (* c[0] k))]) 16)
     (<< (cast u32 enc12[(cast u64 (* c[1] k))]) 8)
     (cast u32 enc12[(cast u64 (* c[2] k))])))

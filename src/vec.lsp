; vec.lsp — 3-component float vector used by the scene and the host side of the GPU path.

(import "math")
(struct v3 [x f32] [y f32] [z f32])

(defn vec3 [x f32] [y f32] [z f32] -> v3 (return {v3 :x x :y y :z z}))

(defop + [a v3] [b v3] -> v3 (return (vec3 (+ a.x b.x) (+ a.y b.y) (+ a.z b.z))))
(defop - [a v3] [b v3] -> v3 (return (vec3 (- a.x b.x) (- a.y b.y) (- a.z b.z))))
(defop * [a v3] [s f32] -> v3 (return (vec3 (* a.x s) (* a.y s) (* a.z s))))
(defop - [a v3] -> v3 (return (* a -1.0)))

; component a (0, 1, 2)
(defn axis [v v3] [a i32] -> f32 (if (= a 0) (return v.x) (if (= a 1) (return v.y) (return v.z))))

(defn dot [a v3] [b v3] -> f32 (return (+ (* a.x b.x) (* a.y b.y) (* a.z b.z))))
(defn cross [a v3] [b v3] -> v3
  (return (vec3 (- (* a.y b.z) (* a.z b.y)) (- (* a.z b.x) (* a.x b.z)) (- (* a.x b.y) (* a.y b.x)))))
(defn length [a v3] -> f32 (return (math::sqrtf (dot a a))))
(defn normalize [a v3] -> v3
  (let [l (length a)]
    (if (> l 0.0) (return (* a (/ 1.0 l))) (return a))))
(defn vmin [a v3] [b v3] -> v3 (return (vec3 (math::fminf a.x b.x) (math::fminf a.y b.y) (math::fminf a.z b.z))))
(defn vmax [a v3] [b v3] -> v3 (return (vec3 (math::fmaxf a.x b.x) (math::fmaxf a.y b.y) (math::fmaxf a.z b.z))))

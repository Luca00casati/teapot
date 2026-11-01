; vec.lsp — 3-component float vector used by the scene and the host side of the GPU path.

(struct v3 (x f32) (y f32) (z f32))

(defn vec3 ((x f32) (y f32) (z f32)) v3 (make v3 :x x :y y :z z))

(defop + ((a v3) (b v3)) v3 (vec3 (+ a.x b.x) (+ a.y b.y) (+ a.z b.z)))
(defop - ((a v3) (b v3)) v3 (vec3 (- a.x b.x) (- a.y b.y) (- a.z b.z)))
(defop * ((a v3) (s f32)) v3 (vec3 (* a.x s) (* a.y s) (* a.z s)))
(defop - ((a v3)) v3 (* a -1.0))

; component a (0, 1, 2)
(defn axis ((v v3) (a i32)) f32 (if (= a 0) v.x (if (= a 1) v.y v.z)))

(defn dot ((a v3) (b v3)) f32 (+ (* a.x b.x) (* a.y b.y) (* a.z b.z)))
(defn cross ((a v3) (b v3)) v3
  (vec3 (- (* a.y b.z) (* a.z b.y)) (- (* a.z b.x) (* a.x b.z)) (- (* a.x b.y) (* a.y b.x))))
(defn length ((a v3)) f32 (sqrtf (dot a a)))
(defn normalize ((a v3)) v3
  (let ((l (length a)))
    (if (> l 0.0) (* a (/ 1.0 l)) a)))
(defn vmin ((a v3) (b v3)) v3 (vec3 (fminf a.x b.x) (fminf a.y b.y) (fminf a.z b.z)))
(defn vmax ((a v3) (b v3)) v3 (vec3 (fmaxf a.x b.x) (fmaxf a.y b.y) (fmaxf a.z b.z)))

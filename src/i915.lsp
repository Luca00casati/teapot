; i915.lsp — the driver for the i915 kernel driver: finds a render node,
; allocates GEM buffers, relocates and executes a batch, waits, and detects
; GPU resets. Raw ioctls only, with the structs and numbers of the kernel's
; own <drm/i915_drm.h>. It knows nothing about the hardware's command stream
; or instruction set, so it serves every Intel generation i915 drives.

(import "libc")
(import "posix")
(import "io")
(import-c "drm/i915_drm.h" :as i915h)
(import "prelude")
(import "layers")
(using libc posix io i915h prelude layers)

(struct- i915
  [fd i32]
  ; the batch is built once and executed many times, so relocations are kept
  ; and re-applied on every exec (presumed_offset ~0 forces it)
  [rel (ptr drm_i915_gem_relocation_entry)] [nrel u32]
  [obj (array drm_i915_gem_exec_object2 16)])

(defn- i915-of [d (ptr driver)] (ptr i915) (cast (ptr i915) d.state))

(defn- i915-close-handle [s (ptr i915)] [handle u32] void
  (let [c {drm_gem_close :handle handle}]
    (ioctl s.fd DRM_IOCTL_GEM_CLOSE (addr c))))

(defn- i915-alloc [d (ptr driver)] [size u64] [b (ptr buffer)] bool
  (let [s (i915-of d)]
       [c {drm_i915_gem_create :size (& (+ size 4095) (~ (cast u64 4095)))}]
    (when (!= (ioctl s.fd DRM_IOCTL_I915_GEM_CREATE (addr c)) 0)
      (perror c"GEM_CREATE")
      (return false))
    (let [mo {drm_i915_gem_mmap_offset :handle c.handle :flags I915_MMAP_OFFSET_WB}]
      (when (!= (ioctl s.fd DRM_IOCTL_I915_GEM_MMAP_OFFSET (addr mo)) 0)
        (perror c"GEM_MMAP_OFFSET")
        (i915-close-handle s c.handle)
        (return false))
      (let [m (mmap null c.size (| PROT_READ PROT_WRITE) MAP_SHARED s.fd (cast i64 mo.offset))]
        (when (= (cast i64 m) -1)
          (perror c"mmap bo")
          (i915-close-handle s c.handle)
          (return false))
        (set (deref b) {buffer :drv d :handle c.handle :size c.size :map (cast (ptr u8) m)})
        true))))

(defn- i915-release [d (ptr driver)] [b (ptr buffer)] void
  (munmap b.map b.size)
  (i915-close-handle (i915-of d) b.handle)
  (set (deref b) {buffer}))

(defn- i915-sync [d (ptr driver)] [b (ptr buffer)] [write bool] void
  (let [sd {drm_i915_gem_set_domain :handle b.handle :read_domains I915_GEM_DOMAIN_CPU
                                          :write_domain (if write I915_GEM_DOMAIN_CPU 0)}]
    (ioctl (. (i915-of d) fd) DRM_IOCTL_I915_GEM_SET_DOMAIN (addr sd))))

(defn- i915-reloc [d (ptr driver)] [batch (ptr buffer)] [offset u32] [target (ptr buffer)] [delta u32] [write bool] void
  (let [s (i915-of d)]
    (set s.rel (cast (ptr drm_i915_gem_relocation_entry)
                     (realloc s.rel (* (sizeof drm_i915_gem_relocation_entry) (cast u64 (+ s.nrel 1))))))
    (set s.rel[s.nrel]
         {drm_i915_gem_relocation_entry :target_handle target.handle :delta delta :offset (cast u64 offset)
               :presumed_offset (~ (cast u64 0)) :read_domains I915_GEM_DOMAIN_RENDER
               :write_domain (if write I915_GEM_DOMAIN_RENDER 0)})
    (+= s.nrel 1)
    (set (deref (cast (ptr u32) (+ batch.map offset))) delta)))

(defn- i915-exec [d (ptr driver)] [batch (ptr buffer)] [len u32] bool
  (let [s (i915-of d)] [n u32 0]
    (for [i u32 0 s.nrel]                ; every relocation target once, the batch last
      (let [r (addr s.rel[i])] [j u32 0]
        (set r.presumed_offset (~ (cast u64 0)))
        (unless (= r.target_handle batch.handle)
          (while (and (< j n) (!= (. s.obj[j] handle) r.target_handle)) (+= j 1))
          (when (= j n)
            (set s.obj[n] {drm_i915_gem_exec_object2 :handle r.target_handle})
            (+= n 1))
          (when (!= r.write_domain 0)
            (set (. s.obj[j] flags) (| (. s.obj[j] flags) EXEC_OBJECT_WRITE))))))
    (set s.obj[n] {drm_i915_gem_exec_object2 :handle batch.handle :relocation_count s.nrel
                              :relocs_ptr (cast u64 s.rel)})
    (+= n 1)
    (let [eb {drm_i915_gem_execbuffer2 :buffers_ptr (cast u64 (addr s.obj[0])) :buffer_count n
                    :batch_len len :flags I915_EXEC_RENDER}]
      (when (!= (ioctl s.fd DRM_IOCTL_I915_GEM_EXECBUFFER2 (addr eb)) 0)
        (perror c"EXECBUFFER2")
        (return false)))
    true))

(defn- i915-hung [d (ptr driver)] bool
  (let [rs {drm_i915_reset_stats}]
    (and (= (ioctl (. (i915-of d) fd) DRM_IOCTL_I915_GET_RESET_STATS (addr rs)) 0)
         (or (!= rs.batch_active 0) (!= rs.batch_pending 0)))))

; open the first render node whose chip id `accept` takes; its path goes to path[64]
(defn connect [accept (ptr (fn (u32 (ptr void)) bool))] [ctx (ptr void)] [path (ptr i8)] (ptr driver)
  (for [i 128 136]
    (snprintf path 64 c"/dev/dri/renderD%d" i)
    (let [fd (open path (| O_RDWR O_CLOEXEC))]
      (when (>= fd 0)
        (let [id i32 0] [gp {drm_i915_getparam :param I915_PARAM_CHIPSET_ID :value (addr id)}]
          (when (and (= (ioctl fd DRM_IOCTL_I915_GETPARAM (addr gp)) 0) (accept (cast u32 id) ctx))
            (let [s (cast (ptr i915) (calloc 1 (sizeof i915)))] [d (cast (ptr driver) (calloc 1 (sizeof driver)))]
              (set s.fd fd)
              (set (deref d) {driver :name c"i915" :state s :alloc i915-alloc :release i915-release :sync i915-sync
                                          :reloc i915-reloc :exec i915-exec :hung i915-hung})
              (return d))))
        (close fd))))
  null)

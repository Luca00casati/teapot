; display.lsp — images, PPM output, and DRM/KMS dumb buffers through raw
; ioctls (no libdrm), with the structs of the kernel's <drm/drm_mode.h>.

(import "libc")
(import "posix")
(import "io")
(import-c "drm/drm.h" "drm/drm_mode.h" :as drmh)
(import "prelude")
(using libc posix io drmh prelude)

; an XRGB8888 image in memory someone else owns (a buffer or a DRM mapping)
(struct target [base (ptr u8)] [w u32] [h u32] [pitch u32])

(defn xrgb [buf (ptr u32)] [w u32] [h u32] target {target :base (cast (ptr u8) buf) :w w :h h :pitch (* w 4)})
(defn trow [t target] [y u32] (ptr u32) (cast (ptr u32) (+ t.base (* (cast u64 y) (cast u64 t.pitch)))))

; copy src to dst, upscaling by an integer factor
(defn blit [src target] [dst target] [scale u32] void
  (for [y u32 0 dst.h]
    (let [row (trow src (if (< (/ y scale) src.h) (/ y scale) (- src.h 1)))] [drow (trow dst y)]
      (for [x u32 0 dst.w]
        (set drow[x] row[(if (< (/ x scale) src.w) (/ x scale) (- src.w 1))])))))

(defn write-ppm [path (ptr i8)] [t target] bool
  (let [f (fopen path c"wb")]
    (when (= f null)
      (perror path)
      (return false))
    (fprintf f c"P6\n%u %u\n255\n" t.w t.h)
    (let [line (cast (ptr u8) (malloc (* 3 (cast u64 t.w))))]
      (for [y u32 0 t.h]
        (let [row (trow t y)]
          (for [x u32 0 t.w]
            (let [p row[x]]
              (set line[(* 3 x)] (truncate u8 (>> p 16)))
              (set line[(+ (* 3 x) 1)] (truncate u8 (>> p 8)))
              (set line[(+ (* 3 x) 2)] (truncate u8 p)))))
        (fwrite line 1 (* 3 (cast u64 t.w)) f))
      (free line))
    (= (fclose f) 0)))

;; ---------------------------------------------------------- DRM / KMS ---

; The first connected output of a DRM card, with two dumb buffers: still
; images draw into the front one, animation flips between them. close-card
; restores the CRTC and frees the buffers.
(struct drm
  [fd i32] [conn u32] [crtc u32]
  [fb (array u32 2)] [handle (array u32 2)] [size u64] [map (array (ptr u8) 2)]
  [front u32]                            ; buffer currently scanned out (or queued to be)
  [flip-pending bool]                    ; a page flip was queued and its completion event not yet read
  [saved drm_mode_crtc]                  ; the CRTC as we found it
  [tgt target])

(defn- card-target [d (ptr drm)] [k u32] target (let [t d.tgt] (set t.base d.map[k]) t))
(defn front [d (ptr drm)] target (card-target d d.front))
(defn back [d (ptr drm)] target (card-target d (- 1 d.front)))

(defn- dirty-buf [d (ptr drm)] [k u32] void
  (let [c {drm_mode_fb_dirty_cmd :fb_id d.fb[k]}]
    (ioctl d.fd DRM_IOCTL_MODE_DIRTYFB (addr c))))
; the front buffer changed (needed by virtio/udl style drivers)
(defn dirty [d (ptr drm)] void (dirty-buf d d.front))

; until the last queued flip happened, so the back buffer is free
(defn wait-flip [d (ptr drm)] void
  (when d.flip-pending
    (let [ev (array u8 1024)] [p {pollfd :fd d.fd :events POLLIN}]
      (when (> (poll (addr p) 1 100) 0) (read d.fd ev 1024)))   ; event content unused
    (set d.flip-pending false)))

; queue the back buffer for the next vblank (does not wait)
(defn flip [d (ptr drm)] void
  (let [back (- 1 d.front)]
       [fl {drm_mode_crtc_page_flip :crtc_id d.crtc :fb_id d.fb[back] :flags DRM_MODE_PAGE_FLIP_EVENT}]
    (dirty-buf d back)
    (if (= (ioctl d.fd DRM_IOCTL_MODE_PAGE_FLIP (addr fl)) 0)
        (set d.flip-pending true)
        ; no page flip support: just switch the CRTC over
        (let [c {drm_mode_crtc :crtc_id d.crtc}]
          (ioctl d.fd DRM_IOCTL_MODE_GETCRTC (addr c))
          (set c.fb_id d.fb[back])
          (set c.set_connectors_ptr (cast u64 (addr d.conn)))
          (set c.count_connectors 1)
          (ioctl d.fd DRM_IOCTL_MODE_SETCRTC (addr c))))
    (set d.front back)))

(defn close-card [d (ptr drm)] void
  (when (< d.fd 0) (return))
  (wait-flip d)
  (when (!= d.saved.mode_valid 0)
    (set d.saved.set_connectors_ptr (cast u64 (addr d.conn)))
    (set d.saved.count_connectors 1)
    (ioctl d.fd DRM_IOCTL_MODE_SETCRTC (addr d.saved)))
  (for [k 0 2]
    (when (!= d.map[k] null) (munmap d.map[k] d.size))
    (when (!= d.fb[k] 0) (ioctl d.fd DRM_IOCTL_MODE_RMFB (addr d.fb[k])))
    (when (!= d.handle[k] 0)
      (let [dd {drm_mode_destroy_dumb :handle d.handle[k]}]
        (ioctl d.fd DRM_IOCTL_MODE_DESTROY_DUMB (addr dd)))))
  (close d.fd)
  (set (deref d) {drm :fd -1}))

; set up the first connected output of the card at path; false (with d closed) if none
(defn open-card [d (ptr drm)] [path (ptr i8)] bool
  (set (deref d) {drm :fd -1})
  (let [fd (open path (| O_RDWR O_CLOEXEC))]
    (when (< fd 0) (return false))
    (ioctl fd DRM_IOCTL_SET_MASTER 0)   ; implicit on first open from a VT; harmless otherwise
    (let [res {drm_mode_card_res}] [conns (array u32 64)] [crtcs (array u32 32)] [encs (array u32 64)]
      (when (!= (ioctl fd DRM_IOCTL_MODE_GETRESOURCES (addr res)) 0)
        (close fd)
        (return false))
      (when (> res.count_connectors 64) (set res.count_connectors 64))
      (when (> res.count_crtcs 32) (set res.count_crtcs 32))
      (when (> res.count_encoders 64) (set res.count_encoders 64))
      (set res.count_fbs 0)
      (set res.connector_id_ptr (cast u64 (addr conns[0])))
      (set res.crtc_id_ptr (cast u64 (addr crtcs[0])))
      (set res.encoder_id_ptr (cast u64 (addr encs[0])))
      (when (!= (ioctl fd DRM_IOCTL_MODE_GETRESOURCES (addr res)) 0)
        (close fd)
        (return false))
      (set d.fd fd)                      ; close-card cleans up from here on

      (for [i u32 0 res.count_connectors]
        (let [c {drm_mode_get_connector :connector_id conns[i]}]
          (when (and (= (ioctl fd DRM_IOCTL_MODE_GETCONNECTOR (addr c)) 0)   ; (probes modes)
                     (= c.connection 1) (!= c.count_modes 0))
            (let [nmodes c.count_modes]
                 [modes (cast (ptr drm_mode_modeinfo) (calloc (cast u64 nmodes) (sizeof drm_mode_modeinfo)))]
                 [cenc (cast (ptr u32) (calloc (cast u64 (+ c.count_encoders 1)) 4))]
              (set c.modes_ptr (cast u64 modes))
              (set c.encoders_ptr (cast u64 cenc))
              (set c.count_props 0)
              (set c.props_ptr 0)
              (set c.prop_values_ptr 0)
              (when (and (= (ioctl fd DRM_IOCTL_MODE_GETCONNECTOR (addr c)) 0) (!= c.count_modes 0))
                (when (< c.count_modes nmodes) (set nmodes c.count_modes))
                (let [mode modes[0]] [crtc u32 0]
                  (for [m u32 0 nmodes]
                    (when (!= (& (. modes[m] type) DRM_MODE_TYPE_PREFERRED) 0)
                      (set mode modes[m])
                      (break)))
                  (when (!= c.encoder_id 0)
                    (let [e {drm_mode_get_encoder :encoder_id c.encoder_id}]
                      (when (= (ioctl fd DRM_IOCTL_MODE_GETENCODER (addr e)) 0) (set crtc e.crtc_id))))
                  (for [k u32 0 c.count_encoders]
                    (when (!= crtc 0) (break))
                    (let [e {drm_mode_get_encoder :encoder_id cenc[k]}]
                      (when (= (ioctl fd DRM_IOCTL_MODE_GETENCODER (addr e)) 0)
                        (for [j u32 0 res.count_crtcs]
                          (when (!= (& e.possible_crtcs (<< 1 j)) 0)
                            (set crtc crtcs[j])
                            (break))))))
                  (when (!= crtc 0)
                    (let [cd {drm_mode_create_dumb}]
                      (for [k 0 2]
                        (set cd {drm_mode_create_dumb :width mode.hdisplay :height mode.vdisplay :bpp 32})
                        (when (!= (ioctl fd DRM_IOCTL_MODE_CREATE_DUMB (addr cd)) 0)
                          (perror c"CREATE_DUMB") (close-card d) (return false))
                        (set d.handle[k] cd.handle)
                        (set d.size cd.size)
                        (let [fb {drm_mode_fb_cmd :width cd.width :height cd.height :pitch cd.pitch
                                                  :bpp 32 :depth 24 :handle cd.handle}]
                          (when (!= (ioctl fd DRM_IOCTL_MODE_ADDFB (addr fb)) 0)
                            (perror c"ADDFB") (close-card d) (return false))
                          (set d.fb[k] fb.fb_id))
                        (let [md {drm_mode_map_dumb :handle cd.handle}]
                          (when (!= (ioctl fd DRM_IOCTL_MODE_MAP_DUMB (addr md)) 0)
                            (perror c"MAP_DUMB") (close-card d) (return false))
                          (let [m (mmap null cd.size (| PROT_READ PROT_WRITE) MAP_SHARED fd (cast i64 md.offset))]
                            (when (= (cast i64 m) -1)
                              (perror c"mmap dumb") (close-card d) (return false))
                            (memset m 0 cd.size)
                            (set d.map[k] (cast (ptr u8) m)))))

                      (set d.saved {drm_mode_crtc :crtc_id crtc})
                      (ioctl fd DRM_IOCTL_MODE_GETCRTC (addr d.saved))
                      (set d.conn conns[i])
                      (let [sc {drm_mode_crtc :set_connectors_ptr (cast u64 (addr d.conn)) :count_connectors 1
                                              :crtc_id crtc :fb_id d.fb[0] :mode_valid 1 :mode mode}]
                        (when (!= (ioctl fd DRM_IOCTL_MODE_SETCRTC (addr sc)) 0)
                          (perror c"SETCRTC (is a compositor holding DRM master? run from a text VT)")
                          (set d.saved.mode_valid 0)
                          (close-card d)
                          (return false)))
                      (set d.crtc crtc)
                      (set d.front 0)
                      (set d.tgt {target :w cd.width :h cd.height :pitch cd.pitch})
                      (fprintf stderr c"drm: %s connector %u crtc %u %ux%u@%u\n" path d.conn crtc
                               (cast u32 mode.hdisplay) (cast u32 mode.vdisplay) mode.vrefresh)
                      (free modes) (free cenc)
                      (return true)))))
              (free modes) (free cenc))))))
    (fprintf stderr c"drm: %s has no usable connected output\n" path)
    (close-card d)
    false))

; layers.lsp — the two layers under the kernel IR, as structs of functions:
;
;   hardware   one per GPU generation: IR → machine code, and the command
;              stream that dispatches it (gen75.lsp)
;   driver     one per kernel driver: device, memory, submission (i915.lsp)
;
; Supporting another GPU means another hardware (and, if its kernel driver
; differs, another driver); the ray tracer itself does not change.

(import "kir")

; GPU memory, mapped write-back into the CPU; freed through its driver
(struct buffer (drv (ptr driver)) (handle u32) (size u64) (map (ptr u8)))

(struct driver
  (name (ptr i8))
  (state (ptr void))                                     ; the driver's own
  (alloc (ptr (fn ((ptr driver) u64 (ptr buffer)) bool)))    ; zeroed and mapped
  (release (ptr (fn ((ptr driver) (ptr buffer)) void)))
  (sync (ptr (fn ((ptr driver) (ptr buffer) bool) void)))    ; make the CPU view coherent; waits for the GPU
  ; a GPU address inside target (+ delta) is written into batch at byte offset
  (reloc (ptr (fn ((ptr driver) (ptr buffer) u32 (ptr buffer) u32 bool) void)))
  (exec (ptr (fn ((ptr driver) (ptr buffer) u32) bool)))     ; queue the batch, return at once
  (hung (ptr (fn ((ptr driver)) bool))))                     ; was one of our batches reset?

(struct hardware
  (name (ptr i8))
  (state (ptr void))
  (compile (ptr (fn ((ptr hardware) (ptr kir::program)) bool)))
  ; bufs[0] scene, bufs[1..2] images of tiles_x*4 by tiles_y*4 dwords
  (setup (ptr (fn ((ptr hardware) (ptr driver) u32 u32 (ptr buffer)) bool)))
  (run (ptr (fn ((ptr hardware) (ptr u32)) bool))))          ; queue the whole grid, return at once

(defn buffer-free ((b (ptr buffer))) void
  (when (!= b.drv null) (b.drv.release b.drv b)))

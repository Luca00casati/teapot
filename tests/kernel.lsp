; kernel.lsp — writes the GPU kernel's machine code to a file, without a
; GPU, for disassemblers and for comparing builds (argv[1]: path, argv[2]: "debug")
(import "libc")
(import "../src/kir")
(import "../src/rt-kernel" :as rt)
(import "../src/gen75")
(using libc)

(defn main [argc i32] [argv (ptr (ptr i8))] -> i32
  (rt::build (and (> argc 2) (= (strcmp argv[2] c"debug") 0)))
  (unless (gen75::compile (addr kir::prog)) (return 1))
  (let [f (fopen argv[1] c"wb")]
    (fwrite gen75::enc.code 16 (cast u64 gen75::enc.n) f)
    (fclose f))
  (printf c"%u IR instructions, %u vector registers, %u EU instructions\n" kir::prog.n kir::prog.nvec gen75::enc.n)
  (return 0))

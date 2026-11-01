; kernel.lsp — writes the GPU kernel's machine code to a file, without a
; GPU, for disassemblers and for comparing builds (argv[1]: path, argv[2]: "debug")
(c-include "stdio.h" "string.h")
(include "../src/rt-kernel.lsp" "../src/gen75.lsp")

(defn main ((argc i32) (argv (ptr (ptr i8)))) i32
  (rt-build (and (> argc 2) (= (strcmp (aref argv 2) "debug") 0)))
  (unless (gen75-compile (addr prog)) (return 1))
  (let ((f (fopen (aref argv 1) "wb")))
    (fwrite enc.code 16 (cast u64 enc.n) f)
    (fclose f))
  (printf "%u IR instructions, %u vector registers, %u EU instructions\n" prog.n prog.nvec enc.n)
  0)

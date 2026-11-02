; prelude.lsp — small macros used everywhere.

; (cond (test body...) ... (else body...)): an if / else-if chain
(defmacro cond (&rest clauses)
  (if (empty? clauses)
      '(do)
      (let ((c (first clauses)))
        (if (eq (first c) 'else)
            `(do ,@(rest c))
            `(if ,(first c) (do ,@(rest c)) (cond ,@(rest clauses)))))))

(const PI 3.14159265358979323846)        ; untyped: an f32 where an f32 is wanted

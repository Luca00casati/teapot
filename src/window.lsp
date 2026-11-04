; window.lsp — the real-time view in a desktop window, through SDL2 (built
; with make SDL=1). The GPU renders as everywhere else; SDL only shows the
; frames, reads the mouse and keyboard, and the title bar shows the frame rate.
;
;   drag with the left button   turn the camera around the teapot (and up/down)
;   wheel, + / -                closer / farther
;   arrow keys                  turn
;   space                       start / stop the automatic orbit
;   r                           back to the start view
;   Esc, q                      quit

(import "libc")
(import "math")
(import-c "SDL2/SDL.h" :as sdl)
(import "display")
(using libc)

(struct window
  [win (ptr sdl::SDL_Window)]
  [ren (ptr sdl::SDL_Renderer)]
  [tex (ptr sdl::SDL_Texture)]
  [w u32] [h u32])                       ; the frames' size (the window may be scaled)

; where the camera is: yaw and pitch around the teapot, and the distance
; as a factor of the default one
(struct view [yaw f32] [pitch f32] [zoom f32] [auto bool])

(defn start-view [] -> view (return {view :yaw 0.0 :pitch 0.0 :zoom 1.0 :auto true}))

(private
  (const WINDOWPOS-CENTERED 0x2FFF0000)  ; SDL_WINDOWPOS_CENTERED (a C macro expression)
  (const BUTTON-LMASK 1)                 ; SDL_BUTTON_LMASK
  ; pitch, from the default height (0.32 rad above the horizon): down to just
  ; above the floor's horizon, up to nearly overhead
  (const PITCH-MIN -0.29)
  (const PITCH-MAX 1.2)

  (defn turn [v (ptr view)] [dyaw f32] [dpitch f32]
    (+= v.yaw dyaw)
    (set v.pitch (math::fminf PITCH-MAX (math::fmaxf PITCH-MIN (+ v.pitch dpitch)))))
  (defn scale-zoom [v (ptr view)] [k f32]
    (set v.zoom (math::fminf 3.0 (math::fmaxf 0.4 (* v.zoom k))))))

; a window of win-w x win-h showing frames of w x h
(defn open [title (ptr i8)] [win-w u32] [win-h u32] [w u32] [h u32] -> (ptr window)
  (when (!= (sdl::SDL_Init sdl::SDL_INIT_VIDEO) 0)
    (fprintf stderr c"window: %s\n" (sdl::SDL_GetError))
    (return null))
  (let [wd (cast (ptr window) (calloc 1 (sizeof window)))]
    (set wd.w w)
    (set wd.h h)
    (set wd.win (sdl::SDL_CreateWindow title WINDOWPOS-CENTERED WINDOWPOS-CENTERED (cast i32 win-w) (cast i32 win-h)
                                       sdl::SDL_WINDOW_RESIZABLE))
    (when (!= wd.win null)
      (set wd.ren (sdl::SDL_CreateRenderer wd.win -1 0)))
    (when (!= wd.ren null)
      (sdl::SDL_SetHint c"SDL_RENDER_SCALE_QUALITY" c"linear")
      (set wd.tex (sdl::SDL_CreateTexture wd.ren sdl::SDL_PIXELFORMAT_XRGB8888 sdl::SDL_TEXTUREACCESS_STREAMING
                                          (cast i32 w) (cast i32 h))))
    (when (= wd.tex null)
      (fprintf stderr c"window: %s\n" (sdl::SDL_GetError))
      (close wd)
      (return null))
    (return wd)))

(defn close [wd (ptr window)]
  (when (!= wd.tex null) (sdl::SDL_DestroyTexture wd.tex))
  (when (!= wd.ren null) (sdl::SDL_DestroyRenderer wd.ren))
  (when (!= wd.win null) (sdl::SDL_DestroyWindow wd.win))
  (free wd)
  (sdl::SDL_Quit))

; show an XRGB8888 frame, stretched to the window
(defn show [wd (ptr window)] [frame display::target]
  (sdl::SDL_UpdateTexture wd.tex null (cast (ptr void) frame.base) (cast i32 frame.pitch))
  (sdl::SDL_RenderClear wd.ren)
  (sdl::SDL_RenderCopy wd.ren wd.tex null null)
  (sdl::SDL_RenderPresent wd.ren))

(defn set-title [wd (ptr window)] [text (ptr i8)] (sdl::SDL_SetWindowTitle wd.win text))

; the events since the last call, applied to v; false once the window is closed
(defn poll [wd (ptr window)] [v (ptr view)] [dt f32] -> bool
  (let [ev sdl::SDL_Event undefined] [open true]
    (while (!= (sdl::SDL_PollEvent (addr ev)) 0)
      (let [t ev.type]
        (cond ((= t sdl::SDL_QUIT) (set open false))
              ((= t sdl::SDL_KEYDOWN)
               (let [k ev.key.keysym.sym]
                 (cond ((or (= k sdl::SDLK_ESCAPE) (= k sdl::SDLK_q)) (set open false))
                       ((= k sdl::SDLK_SPACE) (set v.auto (not v.auto)))
                       ((= k sdl::SDLK_r) (set (deref v) (start-view)))
                       ((or (= k sdl::SDLK_PLUS) (= k sdl::SDLK_EQUALS) (= k sdl::SDLK_KP_PLUS)) (scale-zoom v 0.9))
                       ((or (= k sdl::SDLK_MINUS) (= k sdl::SDLK_KP_MINUS)) (scale-zoom v 1.1)))))
              ((and (= t sdl::SDL_MOUSEMOTION) (!= (& ev.motion.state BUTTON-LMASK) 0))
               (set v.auto false)
               (turn v (* (cast f32 ev.motion.xrel) -0.008) (* (cast f32 ev.motion.yrel) 0.006)))
              ((= t sdl::SDL_MOUSEWHEEL)
               (scale-zoom v (if (> ev.wheel.y 0) 0.9 1.1))))))
    ; held arrow keys turn smoothly, at a speed independent of the frame rate
    (let [keys (sdl::SDL_GetKeyboardState null)] [speed (* 1.5 dt)]
      (when (!= keys[sdl::SDL_SCANCODE_LEFT] 0) (set v.auto false) (turn v speed 0.0))
      (when (!= keys[sdl::SDL_SCANCODE_RIGHT] 0) (set v.auto false) (turn v (- speed) 0.0))
      (when (!= keys[sdl::SDL_SCANCODE_UP] 0) (turn v 0.0 (* speed 0.6)))
      (when (!= keys[sdl::SDL_SCANCODE_DOWN] 0) (turn v 0.0 (* speed -0.6))))
    (when v.auto (+= v.yaw (* dt (/ (* 2.0 3.14159265) 12.0))))   ; a turn in 12 s, as on the console
    (return open)))

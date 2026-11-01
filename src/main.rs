//! Utah teapot ray tracer that talks to the display only through kernel
//! interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa, no
//! X11/Wayland/GL/Vulkan), or a PPM file.
//!
//!   teapot                  camera orbits the teapot on the first DRM output (run from a VT)
//!   teapot -p               still image instead, refined until Enter is pressed
//!   teapot -o out.ppm       render a still to a file (-w/-h set the size)
//!   teapot -o f.ppm -n 120  render a 120-frame orbit to f_0000.ppm ... f_0119.ppm
//!   options: -s <spp> samples per pixel (per frame when animating), -r <n> render at
//!            1/n resolution on screen, -b no temporal accumulation of animation frames,
//!            -t <sec> stop after this long, -d <card>
//!
//! Rendering runs on the GPU (gpu/: the ray tracer in a small kernel IR,
//! compiled by our own hardware layer and submitted with raw kernel-driver
//! ioctls, no Mesa; today Intel Haswell on i915).

mod display;
mod gpu;
mod scene;
mod sys;
mod temporal;
mod tone;
mod vec;

use std::f32::consts::PI;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use display::{blit, write_ppm, Drm, Target};
use gpu::Gpu;
use scene::{Camera, Scene};
use temporal::Temporal;
use tone::tone;

static QUIT: AtomicBool = AtomicBool::new(false);
extern "C" fn on_signal(_: i32) {
    QUIT.store(true, Ordering::Relaxed);
}
fn quit() -> bool {
    QUIT.load(Ordering::Relaxed)
}

const STDIN: i32 = 0;
fn key_pressed() -> bool {
    sys::poll_in(STDIN, 0)
}
fn read_key() -> isize {
    sys::read(STDIN, &mut [0u8; 1])
}
fn stdin_tty() -> bool {
    std::io::stdin().is_terminal()
}

const MAX_PASS: u32 = 8; // samples per submitted pass (keeps every GPU batch short)
const ORBIT_SECONDS: f64 = 12.0; // one revolution of the camera on screen

fn pass_seed(i: u32) -> u32 {
    let mut s = i.wrapping_mul(0x9E37_79B9).wrapping_add(0x7f4a_7c15);
    s ^= s >> 15;
    s = s.wrapping_mul(0x2c1b_3c6d);
    s ^= s >> 12;
    s
}

/// passes of a few samples each, averaged in linear light
struct Accum {
    rgb: Vec<u32>,
    acc: Vec<f32>,
    samples: u32,
}

impl Accum {
    fn new(n: usize) -> Accum {
        Accum { rgb: vec![0; n], acc: vec![0.0; n * 3], samples: 0 }
    }
    fn add(&mut self, gpu: &mut Gpu, cam: &Camera, k: u32, seed: u32, sample_base: u32, depth: Option<&mut [f32]>) -> bool {
        if !gpu.submit(cam, k, seed, sample_base) || !gpu.collect(&mut self.rgb, depth) {
            return false;
        }
        let lin8 = &tone().lin8;
        for (&p, acc) in self.rgb.iter().zip(self.acc.as_chunks_mut::<3>().0) {
            for (a, c) in acc.iter_mut().enumerate() {
                *c += lin8[(p >> (16 - 8 * a) & 255) as usize] * k as f32;
            }
        }
        self.samples += k;
        true
    }
    fn resolve(&self, t: &Target) {
        let k = 4095.0 / self.samples as f32;
        for y in 0..t.h {
            for (x, p) in t.row(y).iter_mut().enumerate() {
                *p = tone().encode(&self.acc[(y * t.w + x) * 3..], k);
            }
        }
    }
}

/// The progressive still image, drawn into tgt (the display or the file
/// buffer). With `refine` (on screen) it keeps adding passes until a key is
/// pressed; *key_stop then tells the caller.
fn still_image(gpu: &mut Gpu, cam: &Camera, tgt: &Target, drm: Option<&Drm>, spp: &mut u32, refine: bool, key_stop: &mut bool) -> bool {
    let mut per_pass = 4;
    while !spp.is_multiple_of(per_pass) {
        per_pass -= 1;
    }
    let mut acc = Accum::new(tgt.w * tgt.h);
    let mut ok = true;
    let mut pass = 0;
    while !quit() {
        if acc.samples >= *spp {
            // on screen: keep refining until a key is pressed (or 1024 samples)
            if !refine || acc.samples >= 1024 {
                break;
            }
            if key_pressed() {
                *key_stop = true;
                break;
            }
        }
        let base = acc.samples;
        ok = acc.add(gpu, cam, per_pass, pass_seed(pass), base, None);
        if !ok {
            break;
        }
        acc.resolve(tgt);
        if let Some(d) = drm {
            d.dirty();
        }
        if refine && acc.samples >= *spp {
            eprint!("\r{} samples/pixel, press Enter to exit ", acc.samples);
        }
        pass += 1;
    }
    if refine {
        eprintln!();
    }
    *spp = acc.samples;
    ok
}

/// one animation frame with more samples than a single pass
fn frame_passes(gpu: &mut Gpu, cam: &Camera, frame: &Target, spp: u32, f: u32, mut depth: Option<&mut [f32]>) -> bool {
    let mut acc = Accum::new(frame.w * frame.h);
    while acc.samples < spp {
        let done = acc.samples;
        let k = (spp - done).min(MAX_PASS);
        if !acc.add(gpu, cam, k, pass_seed(f * 1024 + done), f * spp + done, depth.as_deref_mut()) {
            return false;
        }
    }
    acc.resolve(frame);
    true
}

/// out.ppm → out_0007.ppm; a name with a % in it is a printf-style pattern
/// for the frame number (%d, %u, %i with optional 0 flag and width; %% = %)
fn frame_name(o: &str, f: u32) -> String {
    if o.contains('%') {
        let mut s = String::new();
        let mut it = o.chars().peekable();
        while let Some(c) = it.next() {
            if c != '%' {
                s.push(c);
                continue;
            }
            let mut spec = String::new();
            while let Some(&d) = it.peek().filter(|d| d.is_ascii_digit()) {
                spec.push(d);
                it.next();
            }
            match it.next() {
                Some('d' | 'u' | 'i') => {
                    let width: usize = spec.parse().unwrap_or(0);
                    if spec.starts_with('0') { s += &format!("{f:0width$}") } else { s += &format!("{f:width$}") }
                }
                Some(other) => {
                    s.push('%');
                    s += &spec;
                    if other != '%' || !spec.is_empty() {
                        s.push(other);
                    }
                }
                None => s.push('%'),
            }
        }
        return s;
    }
    match o.rfind('.') {
        Some(dot) => format!("{}_{f:04}{}", &o[..dot], &o[dot..]),
        None => format!("{o}_{f:04}.ppm"),
    }
}

/// C's atoi: leading whitespace, sign and digits; anything else ends it
fn atoi(s: &str) -> i32 {
    let s = s.trim_start();
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let v = digits.bytes().take_while(u8::is_ascii_digit).fold(0i32, |v, d| v.wrapping_mul(10).wrapping_add((d - b'0') as i32));
    if neg { -v } else { v }
}

struct Opts {
    out: Option<String>,
    card: Option<String>,
    w: usize,
    h: usize,
    hold: i32,
    nframes: u32,
    rscale: usize,
    spp: i32,
    still: bool,
    no_blend: bool,
}

/// getopt-style: "o:d:w:h:s:t:pn:r:b" (flags may be grouped, -ofile or -o file)
fn parse_args(args: &[String]) -> Option<Opts> {
    let mut o = Opts { out: None, card: None, w: 1280, h: 720, hold: -1, nframes: 0, rscale: 1, spp: 0, still: false, no_blend: false };
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if a == "--" {
            break;
        }
        if !a.starts_with('-') || a.len() == 1 {
            continue;
        }
        let mut j = 1;
        while j < a.len() {
            let c = a.as_bytes()[j];
            j += 1;
            match c {
                b'p' => o.still = true,
                b'b' => o.no_blend = true,
                b'o' | b'd' | b'w' | b'h' | b's' | b't' | b'n' | b'r' => {
                    let v = if j < a.len() {
                        let v = a[j..].to_string();
                        j = a.len();
                        v
                    } else if i < args.len() {
                        i += 1;
                        args[i - 1].clone()
                    } else {
                        eprintln!("{}: option requires an argument -- '{}'", args[0], c as char);
                        return None;
                    };
                    match c {
                        b'o' => o.out = Some(v),
                        b'd' => o.card = Some(v),
                        b'w' => o.w = atoi(&v).max(0) as usize,
                        b'h' => o.h = atoi(&v).max(0) as usize,
                        b's' => o.spp = atoi(&v),
                        b't' => o.hold = atoi(&v),
                        b'n' => o.nframes = atoi(&v).max(0) as u32,
                        _ => o.rscale = atoi(&v).max(1) as usize,
                    }
                }
                _ => {
                    eprintln!("{}: invalid option -- '{}'", args[0], c as char);
                    return None;
                }
            }
        }
    }
    Some(o)
}

fn main() {
    // run() returns instead of exiting so the display is restored on the way out
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let Some(o) = parse_args(&args) else {
        eprintln!(
            "usage: {} [-o file.ppm [-n frames]] [-p] [-d /dev/dri/cardN]\n       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds]",
            args[0]
        );
        return 1;
    };
    let out = o.out.as_deref();

    let t0 = Instant::now();
    let scene = Scene::build();
    eprintln!("teapot: {} triangles, {} BVH nodes ({:.0} ms)", scene.tris.len(), scene.nodes.len(), t0.elapsed().as_secs_f64() * 1e3);

    let mut drm = None;
    let mut filebuf = vec![0u32; if out.is_some() { o.w * o.h } else { 0 }];
    let tgt = if out.is_some() {
        Target::xrgb(&mut filebuf, o.w, o.h)
    } else {
        drm = match &o.card {
            Some(card) => Drm::open(card),
            None => (0..8).find_map(|i| {
                let p = format!("/dev/dri/card{i}");
                if std::path::Path::new(&p).exists() { Drm::open(&p) } else { None }
            }),
        };
        let Some(d) = &drm else {
            eprintln!("no display available; use -o out.ppm");
            // a runtime check, not a build switch: is a desktop session holding the screen?
            if std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some() {
                eprintln!("a desktop session owns the screen: switch to a text console (Ctrl+Alt+F3), log in and run it there");
            }
            return 1;
        };
        d.front()
    };

    sys::signal(sys::SIGINT, on_signal);
    sys::signal(sys::SIGTERM, on_signal);
    let nthreads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let auto_spp = o.spp <= 0;
    let mut spp = o.spp.max(0) as u32;
    let (hold, nframes) = (o.hold, o.nframes);

    if if out.is_some() { nframes > 0 } else { !o.still } {
        /* ---- animation: the camera orbits the teapot ---- */
        let scale = if out.is_some() { 1 } else { o.rscale };
        let (fw, fh) = (tgt.w.div_ceil(scale), tgt.h.div_ceil(scale));
        let mut framebuf = vec![0u32; fw * fh];
        let Some(mut gpu) = Gpu::open(&scene, fw, fh) else { return 1 };
        if auto_spp {
            spp = if out.is_some() { 16 } else { 1 };
        }
        // on screen the GPU renders few samples per frame and accumulates them over time
        let blend = out.is_none() && !o.no_blend;
        let mut temporal = Temporal::default();
        let mut depth = vec![0f32; if blend { fw * fh } else { 0 }];

        eprintln!(
            "animating {fw}x{fh} @ {spp} spp{} on the gpu{}",
            if blend { " + temporal accumulation" } else { "" },
            if out.is_some() { "" } else { " (Enter or Ctrl-C to stop)" }
        );
        let start = Instant::now();
        let mut tick = start;
        let (mut frames, mut tick_frames) = (0, 0);
        let angle = |i: u32| -> f32 {
            if out.is_some() {
                2.0 * PI * i as f32 / nframes as f32
            } else {
                (start.elapsed().as_secs_f64() * 2.0 * std::f64::consts::PI / ORBIT_SECONDS) as f32
            }
        };
        // light frames are pipelined: frame f+1 renders while f is blended and shown
        let pipelined = spp <= MAX_PASS;
        let mut next = Camera::orbit(angle(0));
        if pipelined && !gpu.submit(&next, spp, pass_seed(0), 0) {
            return 1;
        }
        let mut f = 0u32;
        while !quit() {
            let mut cam = next;
            let depth_out = if blend { Some(&mut depth[..]) } else { None };
            let ok = if pipelined {
                let mut ok = gpu.collect(&mut framebuf, depth_out);
                if ok && (out.is_none() || f + 1 < nframes) {
                    next = Camera::orbit(angle(f + 1));
                    ok = gpu.submit(&next, spp, pass_seed(f + 1), (f + 1) * spp);
                }
                ok
            } else {
                cam = Camera::orbit(angle(f));
                frame_passes(&mut gpu, &cam, &Target::xrgb(&mut framebuf, fw, fh), spp, f, depth_out)
            };
            if !ok {
                return 1;
            }
            if quit() {
                break;
            }
            if blend {
                temporal.blend(&mut framebuf, fw, fh, &depth, &cam, nthreads);
            }
            frames += 1;
            tick_frames += 1;
            let frame = Target::xrgb(&mut framebuf, fw, fh);

            if let Some(out) = out {
                let name = frame_name(out, f);
                if !write_ppm(&name, &frame) {
                    return 1;
                }
                eprint!("\r{name} ({}/{nframes})", f + 1);
                if f + 1 >= nframes {
                    eprintln!();
                    break;
                }
                f += 1;
                continue;
            }
            let d = drm.as_mut().unwrap();
            d.wait_flip();
            blit(&frame, &d.back(), scale);
            d.flip();
            let t = Instant::now();
            let dt = (t - tick).as_secs_f64();
            if dt >= 1.0 {
                eprint!("\r{:.1} fps ", tick_frames as f64 / dt);
                tick = t;
                tick_frames = 0;
            }
            if hold >= 0 && (t - start).as_secs_f64() >= hold as f64 {
                break;
            }
            if stdin_tty() && key_pressed() {
                read_key(); // any key stops
                break;
            }
            f += 1;
        }
        let secs = start.elapsed().as_secs_f64();
        eprintln!("\n{frames} frames in {secs:.1} s ({:.1} fps) on the gpu", frames as f64 / secs);
        return 0;
    }

    /* ---- still image ---- */
    let t0 = Instant::now();
    let refine = out.is_none() && hold < 0 && stdin_tty();
    let mut key_stop = false;
    if auto_spp {
        spp = 32;
    }
    {
        let Some(mut gpu) = Gpu::open(&scene, tgt.w, tgt.h) else { return 1 };
        if !still_image(&mut gpu, &Camera::orbit(0.0), &tgt, drm.as_ref(), &mut spp, refine, &mut key_stop) {
            return 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    if let Some(out) = out {
        eprintln!("rendered {}x{} @ {spp} spp on the gpu in {secs:.2} s", o.w, o.h);
        return if write_ppm(out, &tgt) { 0 } else { 1 };
    }

    // hold the image: until Enter / Ctrl-C, or for -t seconds
    if key_stop {
        read_key(); // the key only ends the hold
    } else if !quit() {
        let ms = if hold >= 0 { hold * 1000 } else if stdin_tty() { -1 } else { 10000 };
        if sys::poll_in(STDIN, ms) && !stdin_tty() && read_key() == 0 {
            // stdin at EOF: don't flash and exit
            std::thread::sleep(Duration::from_secs(if hold >= 0 { 0 } else { 10 }));
        }
    }
    eprintln!("rendered {}x{} @ {spp} spp on the gpu in {secs:.2} s", tgt.w, tgt.h);
    0
}

// teapot.cpp — Utah teapot ray tracer that talks to the display only through
// kernel interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa,
// no X11/Wayland/GL/Vulkan), or a PPM file.
//
//   ./teapot                 camera orbits the teapot on the first DRM output (run from a VT)
//   ./teapot -p              still image instead, refined until Enter is pressed
//   ./teapot -o out.ppm      render a still to a file (-w/-h set the size)
//   ./teapot -o f.ppm -n 120 render a 120-frame orbit to f_0000.ppm ... f_0119.ppm
//   options: -s <spp> samples per pixel (per frame when animating), -r <n> render at
//            1/n resolution on screen, -b no temporal accumulation of animation frames,
//            -t <sec> stop after this long, -d <card>
//
// Rendering runs on the GPU (gpu/: the ray tracer in a small kernel IR,
// compiled by our own hardware layer and submitted with raw kernel-driver
// ioctls, no Mesa; today Intel Haswell on i915).

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <poll.h>
#include <unistd.h>

#include "backend.h"
#include "display.h"
#include "scene.h"
#include "temporal.h"
#include "tone.h"

namespace {

volatile sig_atomic_t quit;
void on_signal(int) { quit = 1; }

double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool key_pressed() {
    pollfd pfd = {STDIN_FILENO, POLLIN, 0};
    return poll(&pfd, 1, 0) > 0;
}

constexpr int MAX_PASS = 8;               // samples per submitted pass (keeps every GPU batch short)
constexpr double ORBIT_SECONDS = 12.0;    // one revolution of the camera on screen

uint32_t pass_seed(uint32_t i) {
    uint32_t s = i * 0x9E3779B9u + 0x7f4a7c15u;
    s ^= s >> 15; s *= 0x2c1b3c6du; s ^= s >> 12;
    return s;
}

// passes of a few samples each, averaged in linear light
struct Accum {
    ml::Vec<uint32_t> rgb;
    ml::Vec<float> acc;
    int samples = 0;

    explicit Accum(size_t n) : rgb(n), acc(n * 3) {}
    bool add(Backend &b, const Camera &cam, int k, uint32_t seed, int sample_base, float *depth) {
        if (!b.submit(cam, k, seed, sample_base) || !b.collect(rgb.data(), depth)) return false;
        const ToneLuts &lut = tone();
        for (size_t i = 0; i < rgb.size(); i++)
            for (int a = 0; a < 3; a++) acc[3 * i + a] += lut.lin8[rgb[i] >> (16 - 8 * a) & 255] * k;
        samples += k;
        return true;
    }
    void resolve(const Target &t) const {
        float k = 4095.0f / samples;
        for (int y = 0; y < t.h; y++) {
            uint32_t *row = t.row(y);
            for (int x = 0; x < t.w; x++) row[x] = tone().encode(&acc[((size_t)y * t.w + x) * 3], k);
        }
    }
};

// The progressive still image, drawn into tgt (the display or the file
// buffer). With `refine` (on screen) it keeps adding passes until a key is
// pressed; *key_stop then tells the caller.
bool still_image(Backend &b, const Camera &cam, const Target &tgt, Drm *drm, int &spp, bool refine, bool &key_stop) {
    int per_pass = 4;
    while (spp % per_pass) per_pass--;
    Accum acc((size_t)tgt.w * tgt.h);
    bool ok = true;
    for (int pass = 0; !quit; pass++) {
        if (acc.samples >= spp) {
            // on screen: keep refining until a key is pressed (or 1024 samples)
            if (!refine || acc.samples >= 1024) break;
            if (key_pressed()) { key_stop = true; break; }
        }
        if (!(ok = acc.add(b, cam, per_pass, pass_seed(pass), acc.samples, nullptr))) break;
        acc.resolve(tgt);
        if (drm) drm->dirty();
        if (refine && acc.samples >= spp) fprintf(stderr, "\r%d samples/pixel, press Enter to exit ", acc.samples);
    }
    if (refine) fputc('\n', stderr);
    spp = acc.samples;
    return ok;
}

// one animation frame with more samples than a single pass
bool frame_passes(Backend &b, const Camera &cam, const Target &tgt, int spp, uint32_t frame, float *depth) {
    Accum acc((size_t)tgt.w * tgt.h);
    while (acc.samples < spp) {
        int done = acc.samples, k = spp - done < MAX_PASS ? spp - done : MAX_PASS;
        if (!acc.add(b, cam, k, pass_seed(frame * 1024 + done), frame * spp + done, depth)) return false;
    }
    acc.resolve(tgt);
    return true;
}

// out.ppm → out_0007.ppm; a name with a % in it is a printf pattern for the frame number
ml::Str frame_name(const char *o, uint32_t f) {
    if (strchr(o, '%')) return ml::Str::fmt(o, f);
    const char *dot = strrchr(o, '.');
    if (!dot) return ml::Str::fmt("%s_%04u.ppm", o, f);
    return ml::Str::fmt("%.*s_%04u%s", (int)(dot - o), o, f, dot);
}

}  // namespace

int main(int argc, char **argv) {
    const char *out = nullptr, *card = nullptr;
    int W = 1280, H = 720, hold = -1, nframes = 0, rscale = 1, spp = 0, opt;
    bool still = false, no_blend = false;
    while ((opt = getopt(argc, argv, "o:d:w:h:s:t:pn:r:b")) != -1) {
        switch (opt) {
        case 'o': out = optarg; break;
        case 'd': card = optarg; break;
        case 'w': W = atoi(optarg); break;
        case 'h': H = atoi(optarg); break;
        case 's': spp = atoi(optarg); break;
        case 't': hold = atoi(optarg); break;
        case 'p': still = true; break;
        case 'n': nframes = atoi(optarg); break;
        case 'r': rscale = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'b': no_blend = true; break;
        default:
            fprintf(stderr, "usage: %s [-o file.ppm [-n frames]] [-p] [-d /dev/dri/cardN]\n"
                            "       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds]\n",
                    argv[0]);
            return 1;
        }
    }

    double t0 = now();
    Scene scene;
    scene.build();
    fprintf(stderr, "teapot: %zu triangles, %zu BVH nodes (%.0f ms)\n", scene.tris.size(), scene.nodes.size(),
            (now() - t0) * 1e3);

    Drm drm;
    ml::Vec<uint32_t> filebuf;
    Target tgt;
    if (out) {
        filebuf.assign((size_t)W * H, 0);
        tgt = Target::xrgb((uint8_t *)filebuf.data(), W, H);
    } else {
        if (card) drm.open(card);
        else
            for (int i = 0; !drm && i < 8; i++) {
                ml::Str p = ml::Str::fmt("/dev/dri/card%d", i);
                if (!access(p.c_str(), F_OK)) drm.open(p.c_str());
            }
        if (!drm) {
            fprintf(stderr, "no display available; use -o out.ppm\n");
            if (getenv("DISPLAY") || getenv("WAYLAND_DISPLAY"))
                fprintf(stderr, "a desktop session owns the screen: switch to a text console "
                                "(Ctrl+Alt+F3), log in and run it there\n");
            return 1;
        }
        tgt = drm.front();
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1) nthreads = 1;
    bool auto_spp = spp <= 0;

    if (out ? nframes > 0 : !still) {
        /* ---- animation: the camera orbits the teapot ---- */
        int scale = out ? 1 : rscale;
        int fw = (tgt.w + scale - 1) / scale, fh = (tgt.h + scale - 1) / scale;
        ml::Vec<uint32_t> framebuf((size_t)fw * fh);
        Target frame = Target::xrgb((uint8_t *)framebuf.data(), fw, fh);
        auto gpu = open_gpu(scene, fw, fh);
        if (!gpu) return 1;
        if (auto_spp) spp = out ? 16 : 1;
        // on screen the GPU renders few samples per frame and accumulates them over time
        bool blend = !out && !no_blend;
        Temporal temporal;
        ml::Vec<float> depth(blend ? (size_t)fw * fh : 0);
        float *depth_out = blend ? depth.data() : nullptr;

        fprintf(stderr, "animating %dx%d @ %d spp%s on the gpu%s\n", fw, fh, spp,
                blend ? " + temporal accumulation" : "", out ? "" : " (Enter or Ctrl-C to stop)");
        double start = now(), tick = start;
        int frames = 0, tick_frames = 0;
        auto angle = [&](uint32_t i) {
            return out ? 2 * (float)M_PI * i / nframes : (float)((now() - start) * 2 * M_PI / ORBIT_SECONDS);
        };
        // light frames are pipelined: frame f+1 renders while f is blended and shown
        bool pipelined = spp <= MAX_PASS;
        Camera next = Camera::orbit(angle(0));
        if (pipelined && !gpu->submit(next, spp, pass_seed(0), 0)) return 1;
        for (uint32_t f = 0; !quit; f++) {
            Camera cam = next;
            bool ok;
            if (pipelined) {
                ok = gpu->collect((uint32_t *)frame.base, depth_out);
                if (ok && (!out || (int)f + 1 < nframes)) {
                    next = Camera::orbit(angle(f + 1));
                    ok = gpu->submit(next, spp, pass_seed(f + 1), (f + 1) * spp);
                }
            } else {
                cam = Camera::orbit(angle(f));
                ok = frame_passes(*gpu, cam, frame, spp, f, depth_out);
            }
            if (!ok) return 1;
            if (quit) break;
            if (blend) temporal.blend(frame, depth_out, cam, nthreads);
            frames++; tick_frames++;

            if (out) {
                ml::Str name = frame_name(out, f);
                if (write_ppm(name.c_str(), frame)) return 1;
                fprintf(stderr, "\r%s (%u/%d)", name.c_str(), f + 1, nframes);
                if ((int)f + 1 >= nframes) { fputc('\n', stderr); break; }
                continue;
            }
            drm.wait_flip();
            blit(frame, drm.back(), scale);
            drm.flip();
            double t = now();
            if (t - tick >= 1) {
                fprintf(stderr, "\r%.1f fps ", tick_frames / (t - tick));
                tick = t; tick_frames = 0;
            }
            if (hold >= 0 && t - start >= hold) break;
            if (isatty(STDIN_FILENO) && key_pressed()) {
                char c;
                if (read(STDIN_FILENO, &c, 1) < 0) { /* any key stops */ }
                break;
            }
        }
        double secs = now() - start;
        fprintf(stderr, "\n%d frames in %.1f s (%.1f fps) on the gpu\n", frames, secs, frames / secs);
        return 0;
    }

    /* ---- still image ---- */
    t0 = now();
    bool refine = !out && hold < 0 && isatty(STDIN_FILENO), key_stop = false;
    if (auto_spp) spp = 32;
    {
        auto gpu = open_gpu(scene, tgt.w, tgt.h);
        if (!gpu || !still_image(*gpu, Camera::orbit(0), tgt, out ? nullptr : &drm, spp, refine, key_stop)) return 1;
    }
    double secs = now() - t0;

    if (out) {
        fprintf(stderr, "rendered %dx%d @ %d spp on the gpu in %.2f s\n", W, H, spp, secs);
        return write_ppm(out, tgt) ? 1 : 0;
    }

    // hold the image: until Enter / Ctrl-C, or for -t seconds
    if (key_stop) {
        char c;
        if (read(STDIN_FILENO, &c, 1) < 0) { /* the key only ends the hold */ }
    } else if (!quit) {
        pollfd pfd = {STDIN_FILENO, POLLIN, 0};
        int ms = hold >= 0 ? hold * 1000 : -1;
        if (!isatty(STDIN_FILENO) && hold < 0) ms = 10000;
        int r = poll(&pfd, 1, ms);
        if (r > 0 && !isatty(STDIN_FILENO)) {  // stdin at EOF: don't flash and exit
            char c;
            if (read(STDIN_FILENO, &c, 1) == 0) usleep(hold >= 0 ? 0 : 10000000);
        }
    }
    fprintf(stderr, "rendered %dx%d @ %d spp on the gpu in %.2f s\n", tgt.w, tgt.h, spp, secs);
    return 0;
}

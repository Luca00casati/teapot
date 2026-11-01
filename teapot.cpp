// teapot.cpp — Utah teapot ray tracer that talks to the display only through
// kernel interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa,
// no X11/Wayland/GL/Vulkan), falling back to the fbdev device, or a PPM file.
//
//   ./teapot                 camera orbits the teapot on the first DRM output (run from a VT)
//   ./teapot -p              still image instead, refined until Enter is pressed
//   ./teapot -f /dev/fb0     force fbdev
//   ./teapot -o out.ppm      render a still to a file (-w/-h set the size)
//   ./teapot -o f.ppm -n 120 render a 120-frame orbit to f_0000.ppm ... f_0119.ppm
//   options: -s <spp> samples per pixel (per frame when animating), -r <n> render at
//            1/n resolution on screen, -b no temporal accumulation of animation frames,
//            -t <sec> stop after this long, -d <card>,
//            -g auto|gpu|cpu  rendering backend (-c = -g cpu)
//
// Backends: the GPU (gpu/: the ray tracer in a small kernel IR, compiled by
// our own hardware layer and submitted with raw kernel-driver ioctls, no Mesa;
// today Intel Haswell on i915), or all CPU cores (cpu_render.cpp). "auto"
// tries them in that order.

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include "backend.h"
#include "cpu_render.h"
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

// The progressive still image: passes of a few samples, averaged into tgt (the
// display or the file buffer). With `refine` (on screen) it keeps adding
// passes until a key is pressed; *key_stop then tells the caller.
bool gpu_still(Backend &b, const Camera &cam, const Target &tgt, Drm *drm, int &spp, bool refine, bool &key_stop) {
    const ToneLuts &lut = tone();
    int w = tgt.w, h = tgt.h, per_pass = 4;
    while (spp % per_pass) per_pass--;
    int passes = spp / per_pass, pass = 0;
    bool ok = true;
    std::vector<uint32_t> rgb((size_t)w * h);
    std::vector<float> acc((size_t)w * h * 3);
    for (;; pass++) {
        if (quit) break;
        if (pass >= passes) {
            // on screen: keep refining until a key is pressed (or 1024 samples)
            if (!refine || pass * per_pass >= 1024) break;
            if (key_pressed()) { key_stop = true; break; }
        }
        if (!(ok = b.submit(cam, per_pass, pass_seed(pass), pass * per_pass) && b.collect(rgb.data(), nullptr))) break;
        for (size_t i = 0; i < rgb.size(); i++)
            for (int a = 0; a < 3; a++) acc[3 * i + a] += lut.lin8[rgb[i] >> (16 - 8 * a) & 255];
        float k = 4095.0f / (pass + 1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint32_t p = lut.encode(&acc[((size_t)y * w + x) * 3], k);
                tgt.put_rgb8(x, y, p >> 16, p >> 8 & 255, p & 255);
            }
        if (drm) drm->dirty();
        if (refine && pass + 1 >= passes) fprintf(stderr, "\r%d samples/pixel, press Enter to exit ", (pass + 1) * per_pass);
    }
    if (refine) fputc('\n', stderr);
    spp = pass * per_pass;
    return ok && !quit;
}

// one animation frame with more samples than a single pass: several passes, averaged
bool gpu_frame_passes(Backend &b, const Camera &cam, const Target &tgt, int spp, uint32_t frame, float *depth) {
    const ToneLuts &lut = tone();
    size_t n = (size_t)tgt.w * tgt.h;
    std::vector<uint32_t> rgb(n);
    std::vector<float> acc(n * 3);
    for (int done = 0; done < spp;) {
        int k = spp - done < MAX_PASS ? spp - done : MAX_PASS;
        if (!b.submit(cam, k, pass_seed(frame * 1024 + done), frame * spp + done) || !b.collect(rgb.data(), depth))
            return false;
        for (size_t i = 0; i < n; i++)
            for (int a = 0; a < 3; a++) acc[3 * i + a] += lut.lin8[rgb[i] >> (16 - 8 * a) & 255] * k;
        done += k;
    }
    uint32_t *out = (uint32_t *)tgt.base;
    for (size_t i = 0; i < n; i++) out[i] = lut.encode(&acc[3 * i], 4095.0f / spp);
    return true;
}

std::unique_ptr<Backend> open_backend(const std::string &want, const Scene &scene, int w, int h) {
    if (want == "auto" || want == "gpu") return open_gpu(scene, w, h);
    return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
    const char *out = nullptr, *fbdev = nullptr, *card = nullptr;
    int W = 1280, H = 720, hold = -1, nframes = 0, rscale = 1, spp = 0, opt;
    bool still = false, no_blend = false;
    std::string backend = "auto";
    while ((opt = getopt(argc, argv, "o:f:d:w:h:s:t:cpn:r:bg:")) != -1) {
        switch (opt) {
        case 'o': out = optarg; break;
        case 'f': fbdev = optarg; break;
        case 'd': card = optarg; break;
        case 'w': W = atoi(optarg); break;
        case 'h': H = atoi(optarg); break;
        case 's': spp = atoi(optarg); break;
        case 't': hold = atoi(optarg); break;
        case 'c': backend = "cpu"; break;
        case 'g': backend = optarg; break;
        case 'p': still = true; break;
        case 'n': nframes = atoi(optarg); break;
        case 'r': rscale = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'b': no_blend = true; break;
        default:
            fprintf(stderr, "usage: %s [-o file.ppm [-n frames]] [-p] [-f /dev/fbN] [-d /dev/dri/cardN]\n"
                            "       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds] [-g auto|gpu|cpu] [-c]\n",
                    argv[0]);
            return 1;
        }
    }
    bool any_backend = backend == "auto" || backend == "cpu";

    double t0 = now();
    Scene scene;
    scene.build();
    fprintf(stderr, "teapot: %zu triangles, %zu BVH nodes (%.0f ms)\n", scene.tris.size(), scene.nodes.size(),
            (now() - t0) * 1e3);

    enum { OUT_FILE, OUT_DRM, OUT_FB } mode;
    Drm drm;
    Fbdev fb;
    std::vector<uint32_t> filebuf;
    Target tgt;
    if (out) {
        mode = OUT_FILE;
        filebuf.assign((size_t)W * H, 0);
        tgt = Target::xrgb((uint8_t *)filebuf.data(), W, H);
    } else {
        if (!fbdev) {
            if (card) drm.open(card);
            for (int i = 0; !drm && !card && i < 8; i++) {
                std::string p = "/dev/dri/card" + std::to_string(i);
                if (!access(p.c_str(), F_OK)) drm.open(p.c_str());
            }
        }
        if (drm) {
            mode = OUT_DRM;
            tgt = drm.front();
        } else if (fb.open(fbdev ? fbdev : "/dev/fb0")) {
            mode = OUT_FB;
            tgt = fb.target();
        } else {
            fprintf(stderr, "no display available; use -o out.ppm\n");
            if (getenv("DISPLAY") || getenv("WAYLAND_DISPLAY"))
                fprintf(stderr, "a desktop session owns the screen: switch to a text console "
                                "(Ctrl+Alt+F3), log in and run it there\n");
            return 1;
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1) nthreads = 1;
    bool auto_spp = spp <= 0;
    const char *device = "cpu";
    CpuRender cpu(scene);

    if (mode == OUT_FILE ? nframes > 0 : !still) {
        /* ---- animation: the camera orbits the teapot ---- */
        Target disp = tgt;
        int scale = mode == OUT_FILE ? 1 : rscale;
        int fw = mode == OUT_FILE ? W : (disp.w + scale - 1) / scale;
        int fh = mode == OUT_FILE ? H : (disp.h + scale - 1) / scale;
        std::vector<uint32_t> framebuf((size_t)fw * fh);
        Target frame = Target::xrgb((uint8_t *)framebuf.data(), fw, fh);
        auto gpu = open_backend(backend, scene, fw, fh);
        if (gpu) device = gpu->name();
        else if (!any_backend) { fprintf(stderr, "backend %s unavailable\n", backend.c_str()); return 1; }
        if (auto_spp) spp = gpu ? (mode == OUT_FILE ? 16 : 1) : (mode == OUT_FILE ? 4 : 1);
        // on screen the GPU renders few samples per frame and accumulates them over time
        bool blend = gpu && mode != OUT_FILE && !no_blend;
        Temporal temporal;
        std::vector<float> depth(blend ? (size_t)fw * fh : 0);
        float *depth_out = blend ? depth.data() : nullptr;

        std::string pattern;
        if (mode == OUT_FILE) {
            std::string o = out;
            size_t dot = o.rfind('.');
            if (o.find('%') != std::string::npos) pattern = o;
            else if (dot != std::string::npos) pattern = o.substr(0, dot) + "_%04d" + o.substr(dot);
            else pattern = o + "_%04d.ppm";
        }
        fprintf(stderr, "animating %dx%d @ %d spp%s on the %s%s\n", fw, fh, spp,
                blend ? " + temporal accumulation" : "", device, mode == OUT_FILE ? "" : " (Enter or Ctrl-C to stop)");
        double start = now(), tick = start;
        int frames = 0, tick_frames = 0;
        auto angle = [&](uint32_t i) {
            return mode == OUT_FILE ? 2 * (float)M_PI * i / nframes : (float)((now() - start) * 2 * M_PI / ORBIT_SECONDS);
        };
        auto gpu_failed = [&](uint32_t f) {
            fprintf(stderr, "%s failed, falling back to CPU rendering\n", gpu->name());
            gpu.reset();
            blend = false; depth_out = nullptr; device = "cpu";
            if (auto_spp) spp = mode == OUT_FILE ? 4 : 1;
            cpu.render(Camera::orbit(angle(f)), frame, spp, f, nthreads, &quit);
        };
        // light frames are pipelined: frame f+1 renders while f is blended and shown
        bool pipelined = gpu && spp <= MAX_PASS, queued = false;
        Camera cam{}, queued_cam{};
        for (uint32_t f = 0; !quit; f++) {
            if (gpu && pipelined) {
                if (!queued) {                              // first frame
                    queued_cam = Camera::orbit(angle(f));
                    queued = gpu->submit(queued_cam, spp, pass_seed(f), f * spp);
                }
                cam = queued_cam;
                bool ok = queued && gpu->collect((uint32_t *)frame.base, depth_out);
                queued = false;
                if (ok && (mode != OUT_FILE || (int)f + 1 < nframes)) {
                    queued_cam = Camera::orbit(angle(f + 1));
                    ok = queued = gpu->submit(queued_cam, spp, pass_seed(f + 1), (f + 1) * spp);
                }
                if (!ok) gpu_failed(f);
            } else {
                cam = Camera::orbit(angle(f));
                if (gpu && !gpu_frame_passes(*gpu, cam, frame, spp, f, depth_out)) gpu_failed(f);
                else if (!gpu) cpu.render(cam, frame, spp, f, nthreads, &quit);
            }
            if (quit) break;
            if (blend) temporal.blend(frame, depth_out, cam, nthreads);
            frames++; tick_frames++;

            if (mode == OUT_FILE) {
                char name[600];
                snprintf(name, sizeof name, pattern.c_str(), f);
                if (write_ppm(name, frame)) break;
                fprintf(stderr, "\r%s (%u/%d)", name, f + 1, nframes);
                if ((int)f + 1 >= nframes) { fputc('\n', stderr); break; }
                continue;
            }
            if (mode == OUT_DRM) {
                drm.wait_flip();
                blit(frame, drm.back(), scale);
                drm.flip();
            } else {
                blit(frame, disp, scale);
            }
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
        fprintf(stderr, "\n%d frames in %.1f s (%.1f fps) on the %s\n", frames, secs, frames / secs, device);
        return 0;
    }

    /* ---- still image ---- */
    t0 = now();
    bool refine = mode != OUT_FILE && hold < 0 && isatty(STDIN_FILENO), key_stop = false;
    if (auto_spp) spp = 32;
    Camera cam = Camera::orbit(0);
    auto gpu = open_backend(backend, scene, tgt.w, tgt.h);
    if (!gpu && !any_backend) { fprintf(stderr, "backend %s unavailable\n", backend.c_str()); return 1; }
    Drm *dirty = mode == OUT_DRM ? &drm : nullptr;
    if (gpu && gpu_still(*gpu, cam, tgt, dirty, spp, refine, key_stop)) {
        device = gpu->name();
    } else if (!quit) {
        if (gpu) fprintf(stderr, "%s failed, falling back to CPU rendering\n", gpu->name());
        if (auto_spp) spp = 9;
        cpu.start(cam, tgt, spp, 0, nthreads, &quit);
        while (!quit && cpu.rows_done() < tgt.h) {
            usleep(100000);
            if (dirty) dirty->dirty();
            if (mode == OUT_FILE) fprintf(stderr, "\r%3d%%", cpu.rows_done() * 100 / tgt.h);
        }
        cpu.join();
    }
    gpu.reset();
    if (dirty) dirty->dirty();
    double secs = now() - t0;

    if (mode == OUT_FILE) {
        fprintf(stderr, "\rrendered %dx%d @ %d spp on the %s in %.2f s\n", W, H, spp, device, secs);
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
    fprintf(stderr, "rendered %dx%d @ %d spp on the %s in %.2f s\n", tgt.w, tgt.h, spp, device, secs);
    return 0;
}

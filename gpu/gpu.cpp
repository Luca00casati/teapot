// gpu.cpp — the GPU backend, without Mesa or any userspace driver:
//
//   rt_kernel.cpp   the ray tracer, written once in the kernel IR (kir.h)
//        │
//   gen75.cpp       Hardware: IR → machine code, command stream
//        │
//   i915.cpp        Driver: kernel driver ioctls
//
// This file picks a driver that finds a GPU some hardware layer accepts,
// compiles the kernel, uploads the scene, and fills the uniforms per pass.

#include <cstdio>
#include <cstring>
#include <vector>

#include "../backend.h"
#include "layers.h"
#include "rt_kernel.h"

namespace gpu {
namespace {

class GpuBackend final : public Backend {
public:
    GpuBackend(const Scene &s, int w, int h) : w_(w), h_(h), layout_(s.layout_bytes()), nnodes_((uint32_t)s.nodes.size()) {}
    bool init(const Scene &s);
    bool submit(const Camera &cam, int samples, uint32_t seed, int sample_base) override;
    bool collect(uint32_t *rgb, float *depth) override;

private:
    // destroyed in reverse: buffers, then the hardware (its batch), then the driver
    std::unique_ptr<Driver> drv_;
    std::unique_ptr<Hardware> hw_;
    Buffer buf_[3];            // scene, rgb, depth
    int w_, h_, w4_ = 0;
    size_t layout_;
    uint32_t nnodes_;
};

bool GpuBackend::init(const Scene &s) {
    std::unique_ptr<Driver> drivers[] = {make_i915()};
    std::unique_ptr<Hardware> layers[] = {make_gen75()};
    uint32_t chip = 0;
    std::string path, desc;
    for (auto &d : drivers) {
        auto accept = [&](uint32_t id) {
            for (auto &l : layers)
                if (l && l->probe(id, desc)) { hw_ = std::move(l); return true; }
            return false;
        };
        if (d->open(accept, chip, path)) { drv_ = std::move(d); break; }
    }
    if (!drv_) { fprintf(stderr, "gpu: no supported GPU found\n"); return false; }
    fprintf(stderr, "gpu: %s %s (%s, %s)\n", path.c_str(), desc.c_str(), drv_->name(), hw_->name());

    kir::Program prog;
#ifdef TEAPOT_GPU_DEBUG
    rt::build(prog, true);                 // heat-map view of the traversal
#else
    rt::build(prog, false);
#endif
    if (!hw_->compile(prog)) return false;

    int tx = (w_ + 3) / 4, ty = (h_ + 3) / 4;
    w4_ = tx * 4;
    uint64_t img = (uint64_t)tx * ty * 16 * 4;
    if (!drv_->alloc(buf_[rt::BUF_SCENE], s.blob_bytes()) || !drv_->alloc(buf_[rt::BUF_RGB], img) ||
        !drv_->alloc(buf_[rt::BUF_DEPTH], img))
        return false;
    drv_->sync(buf_[rt::BUF_SCENE], true);
    s.write_blob(buf_[rt::BUF_SCENE].map);
    return hw_->setup(*drv_, tx, ty, buf_);
}

// Callers keep passes short (a few samples) so a batch stays far below the
// kernel's hang detection.
bool GpuBackend::submit(const Camera &cam, int samples, uint32_t seed, int sample_base) {
    using namespace rt;
    union { float f; uint32_t u; } u[U_COUNT] = {};
    const v3 vs[] = {cam.pos, cam.fwd, cam.right, cam.up, look::light_pos};
    const int at[] = {U_CAM, U_FWD, U_RIGHT, U_UP, U_LIGHT};
    for (int i = 0; i < 5; i++)
        for (int a = 0; a < 3; a++) u[at[i] + a].f = vs[i][a];
    u[U_TAN_X].f = cam.tan * ((float)w_ / h_); u[U_TAN_Y].f = cam.tan;
    u[U_INV_W].f = 2.0f / w_; u[U_INV_H].f = 2.0f / h_;
    u[U_LSIZE].f = 2 * look::light_radius;
    u[U_TRI_BASE].u = (uint32_t)(8 * layout_ / 16);
    u[U_NNODES].u = nnodes_;
    for (int a = 0; a < 3; a++) u[U_STRIDE + a].u = (uint32_t)((layout_ << a) / 16);
    u[U_SPP].u = (uint32_t)samples; u[U_INV_SPP].f = 1.0f / samples;
    u[U_SEED].u = seed; u[U_SAMPLE_BASE].f = (float)sample_base;
    return hw_->run(&u[0].u);
}

bool GpuBackend::collect(uint32_t *rgb, float *depth) {
    drv_->sync(buf_[rt::BUF_RGB], false);          // waits for the GPU
    if (drv_->hung()) {
        fprintf(stderr, "gpu: kernel hung the GPU (reset by the driver), giving up on the GPU path\n");
        return false;
    }
    const uint32_t *src = (const uint32_t *)buf_[rt::BUF_RGB].map;
    for (int y = 0; y < h_; y++) memcpy(rgb + (size_t)y * w_, src + (size_t)y * w4_, (size_t)w_ * 4);
    if (depth) {
        drv_->sync(buf_[rt::BUF_DEPTH], false);
        const float *dep = (const float *)buf_[rt::BUF_DEPTH].map;
        for (int y = 0; y < h_; y++) memcpy(depth + (size_t)y * w_, dep + (size_t)y * w4_, (size_t)w_ * 4);
    }
    return true;
}

}  // namespace
}  // namespace gpu

std::unique_ptr<Backend> open_gpu(const Scene &scene, int w, int h) {
    auto b = std::unique_ptr<gpu::GpuBackend>(new gpu::GpuBackend(scene, w, h));
    if (!b->init(scene)) return nullptr;
    return b;
}

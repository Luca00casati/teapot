// backend.h — an accelerator that renders whole passes asynchronously:
// submit() queues `samples` per pixel of the camera's view and returns,
// collect() waits and hands back XRGB8888 pixels plus the primary hit
// distance per pixel (negative on the teapot, 1e6 for sky).
#pragma once

#include <cstdint>
#include <memory>

#include "scene.h"

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    virtual bool submit(const Camera &cam, int samples, uint32_t seed, int sample_base) = 0;
    virtual bool collect(uint32_t *rgb, float *depth) = 0;
};

// the GPU path (gpu/): nullptr if no supported GPU is found
std::unique_ptr<Backend> open_gpu(const Scene &scene, int w, int h);

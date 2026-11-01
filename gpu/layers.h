// layers.h — the two layers under the kernel IR:
//
//   Hardware   one per GPU generation: IR → machine code, and the command
//              stream that dispatches it (gen75.cpp)
//   Driver     one per kernel driver: device, memory, submission (i915.cpp)
//
// Supporting another GPU means another Hardware (and, if its kernel driver
// differs, another Driver); the ray tracer itself does not change.
#pragma once

#include <stdint.h>

#include "../minlib.h"
#include "kir.h"

namespace gpu {

class Driver;

// GPU memory, mapped write-back into the CPU; freed through its driver
class Buffer {
public:
    Buffer() = default;
    Buffer(Buffer &&o) noexcept { *this = ml::move(o); }
    Buffer &operator=(Buffer &&o) noexcept;
    ~Buffer();
    Driver *drv = nullptr;
    uint32_t handle = 0;
    uint64_t size = 0;
    void *map = nullptr;
};

class Driver {
public:
    virtual ~Driver() = default;
    virtual const char *name() const = 0;
    // open the first device whose chip id `accept` takes
    virtual bool open(const ml::FnRef<bool(uint32_t)> &accept, uint32_t &chip_id, char (&path)[64]) = 0;
    virtual bool alloc(Buffer &b, uint64_t size) = 0;         // zeroed and mapped
    virtual void release(Buffer &b) = 0;                      // (called by ~Buffer)
    virtual void sync(const Buffer &b, bool write) = 0;       // make the CPU view coherent; waits for the GPU
    // a GPU address inside `target` (+ delta) is written into `batch` at byte `offset`
    virtual void reloc(Buffer &batch, uint32_t offset, const Buffer &target, uint32_t delta, bool write) = 0;
    virtual bool exec(const Buffer &batch, uint32_t len) = 0; // queue the batch, return at once
    virtual bool hung() = 0;                                  // was one of our batches reset?
};

class Hardware {
public:
    virtual ~Hardware() = default;
    virtual const char *name() const = 0;
    virtual bool probe(uint32_t chip_id, char (&desc)[64]) = 0;  // does this layer drive the chip?
    virtual bool compile(const kir::Program &p) = 0;
    // bufs[0] scene, bufs[1..2] images of tiles_x*4 by tiles_y*4 dwords
    virtual bool setup(Driver &drv, int tiles_x, int tiles_y, Buffer *bufs) = 0;
    virtual bool run(const uint32_t *uniforms) = 0;              // queue the whole grid, return at once
};

ml::Box<Driver> make_i915();
ml::Box<Hardware> make_gen75();

inline Buffer &Buffer::operator=(Buffer &&o) noexcept {
    if (this != &o) {
        if (drv) drv->release(*this);
        drv = o.drv; handle = o.handle; size = o.size; map = o.map;
        o.drv = nullptr; o.handle = 0; o.size = 0; o.map = nullptr;
    }
    return *this;
}
inline Buffer::~Buffer() { if (drv) drv->release(*this); }

}  // namespace gpu

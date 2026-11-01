// i915.cpp — the Driver for the i915 kernel driver: finds a render node,
// allocates GEM buffers, relocates and executes a batch, waits, and detects
// GPU resets. Raw ioctls only. It knows nothing about the hardware's command
// stream or instruction set, so it serves every Intel generation i915 drives.

#include <cstdio>
#include <cstring>
#include <vector>

#include <drm/i915_drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "layers.h"

namespace gpu {
namespace {

class I915 final : public Driver {
public:
    ~I915() override { if (fd_ >= 0) close(fd_); }
    const char *name() const override { return "i915"; }

    bool open(const std::function<bool(uint32_t)> &accept, uint32_t &chip_id, std::string &path) override {
        for (int i = 128; i < 136; i++) {
            std::string p = "/dev/dri/renderD" + std::to_string(i);
            int f = ::open(p.c_str(), O_RDWR | O_CLOEXEC);
            if (f < 0) continue;
            int id = 0;
            drm_i915_getparam gp = {I915_PARAM_CHIPSET_ID, &id};
            if (!ioctl(f, DRM_IOCTL_I915_GETPARAM, &gp) && accept((uint32_t)id)) {
                fd_ = f;
                chip_id = (uint32_t)id;
                path = p;
                return true;
            }
            close(f);
        }
        return false;
    }

    bool alloc(Buffer &b, uint64_t size) override {
        b = Buffer();
        drm_i915_gem_create c = {};
        c.size = (size + 4095) & ~4095ull;
        if (ioctl(fd_, DRM_IOCTL_I915_GEM_CREATE, &c)) { perror("GEM_CREATE"); return false; }
        b.drv = this; b.handle = c.handle; b.size = c.size;
        drm_i915_gem_mmap_offset mo = {};
        mo.handle = c.handle; mo.flags = I915_MMAP_OFFSET_WB;
        if (ioctl(fd_, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &mo)) { perror("GEM_MMAP_OFFSET"); return false; }
        void *m = mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, mo.offset);
        if (m == MAP_FAILED) { perror("mmap bo"); return false; }
        b.map = m;
        return true;
    }

    void release(Buffer &b) override {
        if (b.map) munmap(b.map, b.size);
        if (b.handle) {
            drm_gem_close c = {};
            c.handle = b.handle;
            ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &c);
        }
        b.drv = nullptr; b.handle = 0; b.size = 0; b.map = nullptr;
    }

    void sync(const Buffer &b, bool write) override {
        drm_i915_gem_set_domain sd = {b.handle, I915_GEM_DOMAIN_CPU, write ? I915_GEM_DOMAIN_CPU : 0u};
        ioctl(fd_, DRM_IOCTL_I915_GEM_SET_DOMAIN, &sd);
    }

    // the batch is built once and executed many times, so relocations are kept
    // and re-applied on every exec (presumed_offset ~0 forces it)
    void reloc(Buffer &batch, uint32_t offset, const Buffer &target, uint32_t delta, bool write) override {
        drm_i915_gem_relocation_entry r = {};
        r.target_handle = target.handle; r.delta = delta; r.offset = offset; r.presumed_offset = ~0ull;
        r.read_domains = I915_GEM_DOMAIN_RENDER; r.write_domain = write ? I915_GEM_DOMAIN_RENDER : 0;
        rel_.push_back(r);
        *(uint32_t *)((char *)batch.map + offset) = delta;
    }

    bool exec(const Buffer &batch, uint32_t len) override {
        obj_.clear();
        for (auto &r : rel_) {                   // every relocation target once, the batch last
            r.presumed_offset = ~0ull;
            if (r.target_handle == batch.handle) continue;
            size_t j = 0;
            while (j < obj_.size() && obj_[j].handle != r.target_handle) j++;
            if (j == obj_.size()) {
                obj_.push_back({});
                obj_[j].handle = r.target_handle;
            }
            if (r.write_domain) obj_[j].flags |= EXEC_OBJECT_WRITE;
        }
        drm_i915_gem_exec_object2 b = {};
        b.handle = batch.handle; b.relocation_count = (uint32_t)rel_.size(); b.relocs_ptr = (uintptr_t)rel_.data();
        obj_.push_back(b);
        drm_i915_gem_execbuffer2 eb = {};
        eb.buffers_ptr = (uintptr_t)obj_.data(); eb.buffer_count = (uint32_t)obj_.size();
        eb.batch_len = len; eb.flags = I915_EXEC_RENDER;
        if (ioctl(fd_, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb)) { perror("EXECBUFFER2"); return false; }
        return true;
    }

    bool hung() override {
        drm_i915_reset_stats rs = {};
        return !ioctl(fd_, DRM_IOCTL_I915_GET_RESET_STATS, &rs) && (rs.batch_active || rs.batch_pending);
    }

private:
    int fd_ = -1;
    std::vector<drm_i915_gem_relocation_entry> rel_;
    std::vector<drm_i915_gem_exec_object2> obj_;
};

}  // namespace

std::unique_ptr<Driver> make_i915() { return std::make_unique<I915>(); }

}  // namespace gpu

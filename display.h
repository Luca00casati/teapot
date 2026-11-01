// display.h — images, PPM output, and DRM/KMS dumb buffers through raw
// ioctls (no libdrm).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <drm/drm.h>
#include <drm/drm_mode.h>

// an XRGB8888 image in memory
struct Target {
    uint8_t *base;
    int w, h, pitch;
    static Target xrgb(uint8_t *base, int w, int h) { Target t = {base, w, h, w * 4}; return t; }
    uint32_t *row(int y) const { return (uint32_t *)(base + (size_t)y * pitch); }
};

// copy src to dst, upscaling by an integer factor
void blit(const Target &src, const Target &dst, int scale);
int write_ppm(const char *path, const Target &t);

// The first connected output of a DRM card, with two dumb buffers: still
// images draw into the front one, animation flips between them.
class Drm {
public:
    ~Drm() { close(); }
    bool open(const char *path);
    void close();
    Target front() const { return target(front_); }
    Target back() const { return target(!front_); }
    void dirty() { dirty(front_); }  // the front buffer changed (needed by virtio/udl style drivers)
    void wait_flip();             // until the last queued flip happened, so the back buffer is free
    void flip();                  // queue the back buffer for the next vblank (does not wait)
    explicit operator bool() const { return fd_ >= 0; }

private:
    void dirty(int k);
    Target target(int k) const { Target t = tgt_; t.base = (uint8_t *)map_[k]; return t; }
    int fd_ = -1;
    uint32_t conn_ = 0, crtc_ = 0, fb_[2] = {}, handle_[2] = {};
    uint64_t size_ = 0;
    void *map_[2] = {};
    int front_ = 0;               // buffer currently scanned out (or queued to be)
    bool flip_pending_ = false;   // a page flip was queued and its completion event not yet read
    drm_mode_crtc saved_ = {};
    Target tgt_ = {};
};

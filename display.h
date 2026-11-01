// display.h — pixel targets and the two kernel display interfaces:
// DRM/KMS dumb buffers (raw ioctls, no libdrm) and the fbdev device.
#pragma once

#include <cstddef>
#include <cstdint>

#include <drm/drm.h>
#include <drm/drm_mode.h>

struct Target {
    uint8_t *base = nullptr;
    int w = 0, h = 0, pitch = 0, bpp = 32;
    int ro = 16, rl = 8, go = 8, gl = 8, bo = 0, bl = 8;   // channel offsets / lengths

    static Target xrgb(uint8_t *base, int w, int h) { return {base, w, h, w * 4}; }
    uint32_t *row(int y) const { return (uint32_t *)(base + (size_t)y * pitch); }
    bool is_xrgb() const { return bpp == 32 && ro == 16 && go == 8 && bo == 0; }

    void put_rgb8(int x, int y, uint32_t r, uint32_t g, uint32_t b) const {
        uint32_t px = (r >> (8 - rl)) << ro | (g >> (8 - gl)) << go | (b >> (8 - bl)) << bo;
        uint8_t *p = base + (size_t)y * pitch + (size_t)x * (bpp / 8);
        switch (bpp) {
        case 32: *(volatile uint32_t *)p = px; break;
        case 24: p[0] = px; p[1] = px >> 8; p[2] = px >> 16; break;
        case 16: *(volatile uint16_t *)p = px; break;
        }
    }
};

// copy an XRGB8888 frame to a display target, upscaling by an integer factor
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
    void dirty();                 // the front buffer changed (needed by virtio/udl style drivers)
    void wait_flip();             // until the last queued flip happened, so the back buffer is free
    void flip();                  // queue the back buffer for the next vblank (does not wait)
    explicit operator bool() const { return fd_ >= 0; }

private:
    Target target(int k) const { Target t = tgt_; t.base = (uint8_t *)map_[k]; return t; }
    int fd_ = -1;
    uint32_t conn_ = 0, crtc_ = 0, fb_[2] = {}, handle_[2] = {};
    uint64_t size_ = 0;
    void *map_[2] = {};
    int front_ = 0;               // buffer currently scanned out (or queued to be)
    bool flip_pending_ = false;   // a page flip was queued and its completion event not yet read
    drm_mode_crtc saved_ = {};
    Target tgt_;
};

class Fbdev {
public:
    ~Fbdev() { close(); }
    bool open(const char *path);
    void close();
    const Target &target() const { return tgt_; }
    explicit operator bool() const { return fd_ >= 0; }

private:
    int fd_ = -1;
    bool kd_set_ = false;
    void *map_ = nullptr;
    size_t len_ = 0;
    Target tgt_;
};

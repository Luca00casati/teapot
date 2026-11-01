// display.cpp — DRM/KMS through raw ioctls, plus PPM output.

#include "display.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

void blit(const Target &src, const Target &dst, int scale) {
    for (int y = 0; y < dst.h; y++) {
        const uint32_t *row = src.row(std::min(y / scale, src.h - 1));
        uint32_t *drow = dst.row(y);
        for (int x = 0; x < dst.w; x++) drow[x] = row[std::min(x / scale, src.w - 1)];
    }
}

int write_ppm(const char *path, const Target &t) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    fprintf(f, "P6\n%d %d\n255\n", t.w, t.h);
    std::vector<uint8_t> line(t.w * 3);
    for (int y = 0; y < t.h; y++) {
        const uint32_t *row = t.row(y);
        for (int x = 0; x < t.w; x++) {
            line[3 * x] = row[x] >> 16; line[3 * x + 1] = row[x] >> 8; line[3 * x + 2] = row[x];
        }
        fwrite(line.data(), 1, line.size(), f);
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------ DRM / KMS ------ */

bool Drm::open(const char *path) {
    int fd = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    ioctl(fd, DRM_IOCTL_SET_MASTER, 0);  // implicit on first open from a VT; harmless otherwise

    drm_mode_card_res res = {};
    uint32_t conns[64], crtcs[32], encs[64];
    if (ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &res) < 0) { ::close(fd); return false; }
    if (res.count_connectors > 64) res.count_connectors = 64;
    if (res.count_crtcs > 32) res.count_crtcs = 32;
    if (res.count_encoders > 64) res.count_encoders = 64;
    res.count_fbs = 0;
    res.connector_id_ptr = (uintptr_t)conns;
    res.crtc_id_ptr = (uintptr_t)crtcs;
    res.encoder_id_ptr = (uintptr_t)encs;
    if (ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &res) < 0) { ::close(fd); return false; }
    fd_ = fd;                                      // close() cleans up from here on

    for (uint32_t i = 0; i < res.count_connectors; i++) {
        drm_mode_get_connector c = {};
        c.connector_id = conns[i];
        if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &c) < 0) continue;  // probes modes
        if (c.connection != 1 || !c.count_modes) continue;
        std::vector<drm_mode_modeinfo> modes(c.count_modes);
        std::vector<uint32_t> cenc(c.count_encoders + 1);
        c.modes_ptr = (uintptr_t)modes.data();
        c.encoders_ptr = (uintptr_t)cenc.data();
        c.count_props = 0; c.props_ptr = c.prop_values_ptr = 0;
        if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &c) < 0 || !c.count_modes) continue;

        drm_mode_modeinfo mode = modes[0];
        for (uint32_t m = 0; m < c.count_modes; m++)
            if (modes[m].type & DRM_MODE_TYPE_PREFERRED) { mode = modes[m]; break; }

        uint32_t crtc = 0;
        if (c.encoder_id) {
            drm_mode_get_encoder e = {};
            e.encoder_id = c.encoder_id;
            if (!ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &e)) crtc = e.crtc_id;
        }
        for (uint32_t k = 0; !crtc && k < c.count_encoders; k++) {
            drm_mode_get_encoder e = {};
            e.encoder_id = cenc[k];
            if (ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &e)) continue;
            for (uint32_t j = 0; j < res.count_crtcs; j++)
                if (e.possible_crtcs & (1u << j)) { crtc = crtcs[j]; break; }
        }
        if (!crtc) continue;

        drm_mode_create_dumb cd = {};
        for (int k = 0; k < 2; k++) {
            cd = {};
            cd.width = mode.hdisplay; cd.height = mode.vdisplay; cd.bpp = 32;
            if (ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd) < 0) { perror("CREATE_DUMB"); close(); return false; }
            handle_[k] = cd.handle; size_ = cd.size;
            drm_mode_fb_cmd fb = {};
            fb.width = cd.width; fb.height = cd.height; fb.pitch = cd.pitch;
            fb.bpp = 32; fb.depth = 24; fb.handle = cd.handle;
            if (ioctl(fd, DRM_IOCTL_MODE_ADDFB, &fb) < 0) { perror("ADDFB"); close(); return false; }
            fb_[k] = fb.fb_id;
            drm_mode_map_dumb md = {};
            md.handle = cd.handle;
            if (ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md) < 0) { perror("MAP_DUMB"); close(); return false; }
            void *map = mmap(nullptr, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, md.offset);
            if (map == MAP_FAILED) { perror("mmap dumb"); close(); return false; }
            memset(map, 0, cd.size);
            map_[k] = map;
        }

        saved_ = {};
        saved_.crtc_id = crtc;
        ioctl(fd, DRM_IOCTL_MODE_GETCRTC, &saved_);

        conn_ = conns[i];
        drm_mode_crtc set = {};
        set.set_connectors_ptr = (uintptr_t)&conn_;
        set.count_connectors = 1;
        set.crtc_id = crtc;
        set.fb_id = fb_[0];
        set.mode_valid = 1;
        set.mode = mode;
        if (ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &set) < 0) {
            perror("SETCRTC (is a compositor holding DRM master? run from a text VT)");
            saved_.mode_valid = 0;
            close();
            return false;
        }
        crtc_ = crtc; front_ = 0;
        tgt_ = {nullptr, (int)cd.width, (int)cd.height, (int)cd.pitch};
        fprintf(stderr, "drm: %s connector %u crtc %u %ux%u@%u\n", path, conn_, crtc,
                mode.hdisplay, mode.vdisplay, mode.vrefresh);
        return true;
    }
    fprintf(stderr, "drm: %s has no usable connected output\n", path);
    close();
    return false;
}

void Drm::dirty(int k) {
    drm_mode_fb_dirty_cmd d = {};
    d.fb_id = fb_[k];
    ioctl(fd_, DRM_IOCTL_MODE_DIRTYFB, &d);
}

void Drm::wait_flip() {
    if (!flip_pending_) return;
    char ev[1024];
    pollfd pfd = {fd_, POLLIN, 0};
    if (poll(&pfd, 1, 100) > 0 && read(fd_, ev, sizeof ev) < 0) { /* event content unused */ }
    flip_pending_ = false;
}

void Drm::flip() {
    int back = !front_;
    dirty(back);
    drm_mode_crtc_page_flip fl = {};
    fl.crtc_id = crtc_; fl.fb_id = fb_[back]; fl.flags = DRM_MODE_PAGE_FLIP_EVENT;
    if (!ioctl(fd_, DRM_IOCTL_MODE_PAGE_FLIP, &fl)) {
        flip_pending_ = true;
    } else {  // no page flip support: just switch the CRTC over
        drm_mode_crtc c = {};
        c.crtc_id = crtc_;
        ioctl(fd_, DRM_IOCTL_MODE_GETCRTC, &c);
        c.fb_id = fb_[back];
        c.set_connectors_ptr = (uintptr_t)&conn_;
        c.count_connectors = 1;
        ioctl(fd_, DRM_IOCTL_MODE_SETCRTC, &c);
    }
    front_ = back;
}

void Drm::close() {
    if (fd_ < 0) return;
    wait_flip();
    if (saved_.mode_valid) {
        saved_.set_connectors_ptr = (uintptr_t)&conn_;
        saved_.count_connectors = 1;
        ioctl(fd_, DRM_IOCTL_MODE_SETCRTC, &saved_);
    }
    for (int k = 0; k < 2; k++) {
        if (map_[k]) munmap(map_[k], size_);
        if (fb_[k]) ioctl(fd_, DRM_IOCTL_MODE_RMFB, &fb_[k]);
        if (handle_[k]) {
            drm_mode_destroy_dumb dd = {};
            dd.handle = handle_[k];
            ioctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
        }
        map_[k] = nullptr; fb_[k] = handle_[k] = 0;
    }
    ::close(fd_);
    fd_ = -1;
    saved_ = {};
}

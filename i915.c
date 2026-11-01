// i915.c — the submit layer for the i915 kernel driver: finds a render node,
// allocates GEM buffers, relocates and executes a batch, waits, and detects
// GPU resets. Raw ioctls only. It knows nothing about the hardware's command
// stream or instruction set (that is the hardware layer, e.g. gen75.c), so the
// same file serves every Intel generation i915 drives.
//
// A different kernel driver (xe, amdgpu, ...) is another file filling the
// gpu_drv table of gpu.c.

#include <drm/i915_drm.h>

#define I915_MAX_RELOC 16

static struct {
    int fd;
    struct drm_i915_gem_relocation_entry rel[I915_MAX_RELOC];
    struct drm_i915_gem_exec_object2 obj[I915_MAX_RELOC + 1];
    int nrel;
} I = {.fd = -1};

static int i915_open(int (*accept)(uint32_t), uint32_t *chip_id, char *path, size_t n) {
    for (int i = 128; i < 136; i++) {
        snprintf(path, n, "/dev/dri/renderD%d", i);
        int f = open(path, O_RDWR | O_CLOEXEC);
        if (f < 0) continue;
        int id = 0;
        struct drm_i915_getparam gp = {.param = I915_PARAM_CHIPSET_ID, .value = &id};
        if (!ioctl(f, DRM_IOCTL_I915_GETPARAM, &gp) && accept((uint32_t)id) == 0) {
            I.fd = f; I.nrel = 0;
            *chip_id = (uint32_t)id;
            return 0;
        }
        close(f);
    }
    return -1;
}

static int i915_buf_new(gpu_buf *b, uint64_t size) {
    size = (size + 4095) & ~4095ull;
    struct drm_i915_gem_create c = {.size = size};
    if (ioctl(I.fd, DRM_IOCTL_I915_GEM_CREATE, &c)) { perror("GEM_CREATE"); return -1; }
    b->handle = c.handle; b->size = c.size;
    struct drm_i915_gem_mmap_offset mo = {.handle = c.handle, .flags = I915_MMAP_OFFSET_WB};
    if (ioctl(I.fd, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &mo)) { perror("GEM_MMAP_OFFSET"); return -1; }
    b->map = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED, I.fd, mo.offset);
    if (b->map == MAP_FAILED) { b->map = NULL; perror("mmap bo"); return -1; }
    return 0;
}

static void i915_buf_free(gpu_buf *b) {
    if (b->map) munmap(b->map, b->size);
    if (b->handle) {
        struct drm_gem_close c = {.handle = b->handle};
        ioctl(I.fd, DRM_IOCTL_GEM_CLOSE, &c);
    }
    memset(b, 0, sizeof *b);
}

static void i915_buf_sync(gpu_buf *b, int write) {
    struct drm_i915_gem_set_domain sd = {.handle = b->handle, .read_domains = I915_GEM_DOMAIN_CPU,
                                         .write_domain = write ? I915_GEM_DOMAIN_CPU : 0};
    ioctl(I.fd, DRM_IOCTL_I915_GEM_SET_DOMAIN, &sd);
}

// the batch is built once and executed many times, so relocations are kept
// and re-applied on every exec (presumed_offset ~0 forces it)
static void i915_reloc(gpu_buf *batch, uint32_t offset, gpu_buf *target, uint32_t delta, int write) {
    if (I.nrel == I915_MAX_RELOC) { fprintf(stderr, "i915: too many relocations\n"); exit(1); }
    I.rel[I.nrel++] = (struct drm_i915_gem_relocation_entry){
        .target_handle = target->handle, .delta = delta, .offset = offset, .presumed_offset = ~0ull,
        .read_domains = I915_GEM_DOMAIN_RENDER, .write_domain = write ? I915_GEM_DOMAIN_RENDER : 0};
    *(uint32_t *)((char *)batch->map + offset) = delta;
}

static int i915_exec(gpu_buf *batch, uint32_t len) {
    int nobj = 0;
    for (int i = 0; i < I.nrel; i++) {            // every relocation target once, the batch last
        I.rel[i].presumed_offset = ~0ull;
        if (I.rel[i].target_handle == batch->handle) continue;
        int j = 0;
        while (j < nobj && I.obj[j].handle != I.rel[i].target_handle) j++;
        if (j == nobj) I.obj[nobj++] = (struct drm_i915_gem_exec_object2){.handle = I.rel[i].target_handle};
        if (I.rel[i].write_domain) I.obj[j].flags |= EXEC_OBJECT_WRITE;
    }
    I.obj[nobj++] = (struct drm_i915_gem_exec_object2){.handle = batch->handle, .relocation_count = I.nrel,
                                                        .relocs_ptr = (uintptr_t)I.rel};
    struct drm_i915_gem_execbuffer2 eb = {.buffers_ptr = (uintptr_t)I.obj, .buffer_count = nobj,
                                          .batch_len = len, .flags = I915_EXEC_RENDER};
    if (ioctl(I.fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb)) { perror("EXECBUFFER2"); return -1; }
    return 0;
}

static int i915_hung(void) {
    struct drm_i915_reset_stats rs = {0};
    return !ioctl(I.fd, DRM_IOCTL_I915_GET_RESET_STATS, &rs) && (rs.batch_active || rs.batch_pending);
}

static void i915_close(void) {
    if (I.fd >= 0) close(I.fd);
    I.fd = -1; I.nrel = 0;
}

static const gpu_drv drv_i915 = {"i915", i915_open, i915_buf_new, i915_buf_free, i915_buf_sync,
                                 i915_reloc, i915_exec, i915_hung, i915_close};

// gpu.c — the GPU backend, without Mesa or any userspace driver. Layers:
//
//   rt_kernel.c   the ray tracer, written once in the kernel IR (kir.c)
//        │
//   gen75.c       hardware layer: IR → machine code, command stream    (gpu_hw)
//        │        another GPU generation = another file like it
//   i915.c        submit layer: kernel driver ioctls                   (gpu_drv)
//                 another kernel driver = another file like it
//
// This file defines the two layer interfaces and drives them: pick a driver
// that finds a GPU some hardware layer accepts, compile, upload the scene, and
// implement the backend interface of teapot.c (gpu_init / gpu_submit /
// gpu_collect / gpu_fini).

#include "kir.c"
#include "rt_kernel.c"

typedef struct { uint32_t handle; uint64_t size; void *map; } gpu_buf;

// submit layer: one per kernel driver
typedef struct {
    const char *name;
    // open the first device whose chip id `accept` takes (returns 0)
    int (*open)(int (*accept)(uint32_t chip_id), uint32_t *chip_id, char *path, size_t n);
    int (*buf_new)(gpu_buf *b, uint64_t size);       // zeroed, CPU-mapped (write-back)
    void (*buf_free)(gpu_buf *b);
    void (*buf_sync)(gpu_buf *b, int write);         // make the CPU view coherent; waits for the GPU
    // a GPU address inside `target` is written into `batch` at byte `offset`
    void (*reloc)(gpu_buf *batch, uint32_t offset, gpu_buf *target, uint32_t delta, int write);
    int (*exec)(gpu_buf *batch, uint32_t len);       // queue the batch, return at once
    int (*hung)(void);                               // did one of our batches get reset?
    void (*close)(void);
} gpu_drv;

// hardware layer: one per GPU generation
typedef struct {
    const char *name;
    int (*probe)(uint32_t chip_id, char *desc, size_t n);  // 0 if this layer drives the chip
    int (*compile)(void);                                  // encode kir[]
    // bufs: 0 scene, 1 and 2 images of tiles_x*4 by tiles_y*4 dwords
    int (*setup)(const gpu_drv *drv, int tiles_x, int tiles_y, gpu_buf *bufs);
    int (*run)(const uint32_t *uniforms);                  // queue the grid, return at once
    void (*fini)(void);
} gpu_hw;

#include "i915.c"
#include "gen75.c"

static const gpu_drv *const gpu_drvs[] = {&drv_i915};
static const gpu_hw *const gpu_hws[] = {&hw_gen75};

static struct {
    const gpu_drv *drv;
    const gpu_hw *hw;
    gpu_buf buf[3];            // scene, rgb, depth
    int w, h, w4;
    char desc[96];
} GP;

static int gpu_accept(uint32_t chip_id) {
    for (size_t i = 0; i < sizeof gpu_hws / sizeof gpu_hws[0]; i++)
        if (gpu_hws[i]->probe(chip_id, GP.desc, sizeof GP.desc) == 0) { GP.hw = gpu_hws[i]; return 0; }
    return -1;
}

static void gpu_fini(void) {
    if (!GP.drv) return;
    if (GP.hw) GP.hw->fini();
    for (int i = 0; i < 3; i++) GP.drv->buf_free(&GP.buf[i]);
    GP.drv->close();
    memset(&GP, 0, sizeof GP);
}

static int gpu_init(int w, int h) {
    uint32_t chip;
    char path[64];
    for (size_t i = 0; i < sizeof gpu_drvs / sizeof gpu_drvs[0] && !GP.drv; i++)
        if (gpu_drvs[i]->open(gpu_accept, &chip, path, sizeof path) == 0) GP.drv = gpu_drvs[i];
    if (!GP.drv) { fprintf(stderr, "gpu: no supported GPU found\n"); return -1; }
    fprintf(stderr, "gpu: %s %s (%s, %s)\n", path, GP.desc, GP.drv->name, GP.hw->name);

    rt_dbg = getenv("TEAPOT_GPU_DEBUG") != NULL;  // heat-map view of the traversal
    rt_build();
    if (GP.hw->compile()) goto fail;

    int tx = (w + 3) / 4, ty = (h + 3) / 4;
    GP.w = w; GP.h = h; GP.w4 = tx * 4;
    uint64_t img = (uint64_t)tx * ty * 16 * 4;
    if (GP.drv->buf_new(&GP.buf[0], scene_blob_bytes()) || GP.drv->buf_new(&GP.buf[1], img) ||
        GP.drv->buf_new(&GP.buf[2], img))
        goto fail;
    GP.drv->buf_sync(&GP.buf[0], 1);
    write_scene_blob(GP.buf[0].map);
    if (GP.hw->setup(GP.drv, tx, ty, GP.buf)) goto fail;
    return 0;
fail:
    gpu_fini();
    return -1;
}

// Queue one pass of `samples` per pixel with the current camera and return at
// once; gpu_collect waits for it. Callers keep passes short (a few samples)
// so a batch stays far below the kernel's hang detection.
static int gpu_submit(int samples, uint32_t seed, int sample_base) {
    union { float f; uint32_t u; } u[U_COUNT] = {{0}};
    float aspect = (float)GP.w / GP.h;
    v3 vs[] = {cam_pos, cam_fwd, cam_right, cam_up, light_pos};
    int at[] = {U_CAM, U_FWD, U_RIGHT, U_UP, U_LIGHT};
    for (int i = 0; i < 5; i++)
        for (int a = 0; a < 3; a++) u[at[i] + a].f = axis(vs[i], a);
    u[U_TAN_X].f = cam_tan * aspect; u[U_TAN_Y].f = cam_tan;
    u[U_INV_W].f = 2.0f / GP.w; u[U_INV_H].f = 2.0f / GP.h;
    u[U_LSIZE].f = 2 * light_radius;
    size_t layout = scene_layout_bytes();
    u[U_TRI_BASE].u = (uint32_t)(8 * layout / 16);
    u[U_NNODES].u = (uint32_t)nnodes;
    for (int a = 0; a < 3; a++) u[U_STRIDE + a].u = (uint32_t)((layout << a) / 16);
    u[U_SPP].u = (uint32_t)samples; u[U_INV_SPP].f = 1.0f / samples;
    u[U_SEED].u = seed; u[U_SAMPLE_BASE].f = (float)sample_base;
    return GP.hw->run(&u[0].u);
}

// wait for the pass; XRGB8888 pixels to rgb, primary depth (negative on the teapot) to depth
static int gpu_collect(uint32_t *rgb, float *depth) {
    GP.drv->buf_sync(&GP.buf[1], 0);               // waits for the GPU
    if (GP.drv->hung()) {
        fprintf(stderr, "gpu: kernel hung the GPU (reset by the driver), giving up on the GPU path\n");
        return -1;
    }
    const uint32_t *src = GP.buf[1].map;
    for (int y = 0; y < GP.h; y++) memcpy(rgb + (size_t)y * GP.w, src + (size_t)y * GP.w4, (size_t)GP.w * 4);
    if (depth) {
        GP.drv->buf_sync(&GP.buf[2], 0);
        const float *dep = GP.buf[2].map;
        for (int y = 0; y < GP.h; y++) memcpy(depth + (size_t)y * GP.w, dep + (size_t)y * GP.w4, (size_t)GP.w * 4);
    }
    return 0;
}

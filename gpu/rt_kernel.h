// rt_kernel.h — the teapot ray tracer as a kernel IR program, and the
// uniforms the host fills for it.
#pragma once

#include "kir.h"

namespace rt {

// uniform slots (32 dwords)
enum Uniform {
    U_CAM = 0,        // 0-2 camera  3 tan*aspect  4-6 forward  7 tan
    U_TAN_X = 3, U_FWD = 4, U_TAN_Y = 7,
    U_RIGHT = 8,      // 8-10 right  11 2/W  12-14 up  15 2/H
    U_INV_W = 11, U_UP = 12, U_INV_H = 15,
    U_LIGHT = 16,     // 16-18 light centre  19 light size
    U_LSIZE = 19,
    U_TRI_BASE = 20,  // triangle records, in 16-byte units
    U_SPP = 21, U_SEED = 22, U_NNODES = 23,
    U_INV_SPP = 24,
    U_SAMPLE_BASE = 26,  // first sample index of this pass (float)
    U_STRIDE = 28,    // 28-30 node layout stride for x, y, z octant bits (16-byte units)
    U_COUNT = 32,
};

// buffers: 0 = scene (Scene::write_blob), 1 = XRGB8888 image, 2 = primary hit
// distance (negative on the glossy teapot)
enum Buffer { BUF_SCENE, BUF_RGB, BUF_DEPTH };

// `samples` (U_SPP) per pixel of each packet's tile; debug = traversal heat map
void build(kir::Program &p, bool debug);

}  // namespace rt

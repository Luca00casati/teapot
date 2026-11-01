// temporal.h — temporal accumulation of animation frames: each new frame is
// blended with the previous ones, reprojected through the per-pixel depth.
#pragma once

#include "display.h"
#include "minlib.h"
#include "scene.h"

class Temporal {
public:
    // blend `frame` (XRGB8888, rendered from `cam` with primary hit distances
    // `depth`, negative on the glossy teapot) with the history, in place
    void blend(const Target &frame, const float *depth, const Camera &cam, int threads);

private:
    struct View { v3 pos, fwd, right, up; float tx, ty; };
    void rows(int y0, int y1);

    Target frame_;
    const float *depth_ = nullptr;
    View view_{}, hist_view_{};
    // history: linear RGB + sample count, and view-space depth (1e9 = sky)
    ml::Vec<float> hist_[2], hist_depth_[2];
    ml::Vec<float> sx_, sy_, il_;  // pixel-centre ray offsets and 1 / ray length
    int cur_ = 0;
    bool ok_ = false;
};

// temporal.cpp — reprojection with depth rejection, multithreaded over rows.

#include "temporal.h"

#include <thread>

#include "tone.h"

void Temporal::blend(const Target &frame, const float *depth, const Camera &cam, int threads) {
    int W = frame.w, H = frame.h;
    size_t n = (size_t)W * H;
    frame_ = frame; depth_ = depth;
    view_ = {cam.pos, cam.fwd, cam.right, cam.up, cam.tan * W / H, cam.tan};
    if (hist_[0].empty()) {
        for (int k = 0; k < 2; k++) { hist_[k].assign(n * 4, 0); hist_depth_[k].assign(n, 0); }
        // the ray length only depends on the pixel, not on the camera
        sx_.resize(W); sy_.resize(H); il_.resize(n);
        for (int x = 0; x < W; x++) sx_[x] = ((x + 0.5f) * 2 / W - 1) * view_.tx;
        for (int y = 0; y < H; y++) sy_[y] = (1 - (y + 0.5f) * 2 / H) * view_.ty;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) il_[(size_t)y * W + x] = 1 / sqrtf(1 + sx_[x] * sx_[x] + sy_[y] * sy_[y]);
    }
    std::vector<std::thread> th;
    for (int i = 0; i < threads; i++) th.emplace_back(&Temporal::rows, this, H * i / threads, H * (i + 1) / threads);
    for (auto &t : th) t.join();
    cur_ = !cur_;
    hist_view_ = view_;
    ok_ = true;
}

void Temporal::rows(int y0, int y1) {
    const ToneLuts &lut = tone();
    int W = frame_.w, H = frame_.h;
    const float *ph = hist_[cur_].data(), *pd = hist_depth_[cur_].data();
    float *nh = hist_[!cur_].data(), *nd = hist_depth_[!cur_].data();
    const View v = view_, pv = hist_view_;
    const v3 dp = v.pos - pv.pos;
    const float itx = 0.5f * W / pv.tx, ity = 0.5f * H / pv.ty;
    for (int y = y0; y < y1; y++) {
        uint32_t *row = frame_.row(y);
        v3 base = v.fwd + v.up * sy_[y];
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            uint32_t p = row[x];
            float c[3] = {lut.lin8[p >> 16 & 255], lut.lin8[p >> 8 & 255], lut.lin8[p & 255]};
            float t = depth_[i], il = il_[i], h[3] = {0, 0, 0}, n = 0, wsum = 0;
            bool glossy = t < 0, sky = t >= 1e5f;   // the GPU marks teapot pixels with a negative depth
            t = fabsf(t);
            if (ok_) {
                // world point of this pixel relative to the previous camera
                float s = (sky ? 1e4f : t) * il;
                v3 q = dp + (base + v.right * sx_[x]) * s;
                float z = dot(q, pv.fwd);
                if (z > 1e-3f) {
                    float rz = 1 / z;
                    float fx = W * 0.5f + dot(q, pv.right) * rz * itx - 0.5f;
                    float fy = H * 0.5f - dot(q, pv.up) * rz * ity - 0.5f;
                    int ix = (int)floorf(fx), iy = (int)floorf(fy);
                    float ax = fx - ix, ay = fy - iy, tol = 0.02f * z + 0.02f;
                    // bilinear, but only over taps that saw the same surface
                    for (int k = 0; k < 4; k++) {
                        int tx = ix + (k & 1), ty = iy + (k >> 1);
                        if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;
                        size_t j = (size_t)ty * W + tx;
                        float hd = pd[j];
                        if (sky ? hd < 1e8f : fabsf(hd - z) > tol) continue;
                        float w = ((k & 1) ? ax : 1 - ax) * ((k >> 1) ? ay : 1 - ay) + 1e-4f;
                        const float *hp = ph + j * 4;
                        h[0] += hp[0] * w; h[1] += hp[1] * w; h[2] += hp[2] * w;
                        n += hp[3] * w;
                        wsum += w;
                    }
                }
            }
            float alpha = 1;
            if (wsum > 0.05f) {
                float iw = 1 / wsum;
                h[0] *= iw; h[1] *= iw; h[2] *= iw; n *= iw;
                // running mean, then an exponential average; reflections and highlights on
                // the teapot move with the view, so it keeps much less history than the floor
                alpha = fmaxf(1 / (n + 1), glossy ? 0.5f : 0.2f);
            } else {
                n = 0;
            }
            float *o = nh + i * 4;
            for (int a = 0; a < 3; a++) o[a] = h[a] + (c[a] - h[a]) * alpha;
            o[3] = n + 1;
            nd[i] = sky ? 1e9f : t * il;
            float oc[3] = {fminf(o[0], 1), fminf(o[1], 1), fminf(o[2], 1)};
            row[x] = lut.encode(oc);
        }
    }
}

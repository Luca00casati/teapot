// cpu_render.cpp — recursive Whitted-style tracer with soft shadows and
// ambient occlusion, stack-based BVH traversal and Möller–Trumbore triangles.

#include "cpu_render.h"

namespace {

struct Ray { v3 o, d; };
struct Hit { float t; v3 p, n, ng; int mat; };

enum { MAT_NONE, MAT_TEAPOT, MAT_FLOOR };

inline bool box_hit(const Node &n, v3 o, v3 inv, float tmax) {
    float tx1 = (n.bmin.x - o.x) * inv.x, tx2 = (n.bmax.x - o.x) * inv.x;
    float tmin = fminf(tx1, tx2), tm = fmaxf(tx1, tx2);
    float ty1 = (n.bmin.y - o.y) * inv.y, ty2 = (n.bmax.y - o.y) * inv.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2)); tm = fminf(tm, fmaxf(ty1, ty2));
    float tz1 = (n.bmin.z - o.z) * inv.z, tz2 = (n.bmax.z - o.z) * inv.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2)); tm = fminf(tm, fmaxf(tz1, tz2));
    return tm >= fmaxf(tmin, 0) && tmin < tmax;
}

// Möller–Trumbore; returns t or -1
inline float tri_hit(const Tri &t, v3 o, v3 d, float &bu, float &bv) {
    v3 p = cross(d, t.e2);
    float det = dot(t.e1, p);
    if (fabsf(det) < 1e-12f) return -1;
    float id = 1 / det;
    v3 s = o - t.v0;
    float u = dot(s, p) * id;
    if (u < 0 || u > 1) return -1;
    v3 q = cross(s, t.e1);
    float v = dot(d, q) * id;
    if (v < 0 || u + v > 1) return -1;
    bu = u; bv = v;
    return dot(t.e2, q) * id;
}

// any_hit: stop at the first occluder closer than tmax (shadow / AO rays)
bool intersect(const Scene &sc, Ray r, float tmax, Hit &h, bool any_hit) {
    h.t = tmax; h.mat = MAT_NONE;
    v3 inv(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
    int stack[64], sp = 0, best = -1;
    float bu = 0, bv = 0;
    stack[sp++] = 0;
    while (sp) {
        const Node &n = sc.nodes[stack[--sp]];
        if (!box_hit(n, r.o, inv, h.t)) continue;
        if (n.count) {
            for (int i = n.first; i < n.first + n.count; i++) {
                float u, v, t = tri_hit(sc.tris[sc.tidx[i]], r.o, r.d, u, v);
                if (t > 1e-4f && t < h.t) {
                    h.t = t; best = sc.tidx[i]; bu = u; bv = v;
                    if (any_hit) return true;
                }
            }
        } else {
            stack[sp++] = n.first;
            stack[sp++] = n.first + 1;
        }
    }
    if (best >= 0) {
        const Tri &t = sc.tris[best];
        h.mat = MAT_TEAPOT;
        h.ng = normalize(cross(t.e1, t.e2));
        h.n = normalize(sc.vnrm[t.i0] * (1 - bu - bv) + sc.vnrm[t.i1] * bu + sc.vnrm[t.i2] * bv);
    }
    if (r.d.y < 0 && r.o.y > 0) {  // floor plane y = 0
        float t = -r.o.y / r.d.y;
        if (t < h.t) {
            if (any_hit) return true;
            h.t = t; h.mat = MAT_FLOOR; h.n = h.ng = v3(0, 1, 0);
        }
    }
    if (h.mat == MAT_NONE) return false;
    h.p = r.o + r.d * h.t;
    if (dot(h.ng, r.d) > 0) h.ng = -h.ng;  // two-sided surfaces
    if (dot(h.n, h.ng) < 0) h.n = -h.n;
    return true;
}

v3 sky(v3 d) { return lerp(look::sky_horizon, look::sky_zenith, sqrtf(fmaxf(d.y, 0))); }

v3 trace(const Scene &sc, Ray r, int depth, uint64_t &rng, float sx, float sy) {
    using namespace look;
    Hit h;
    if (!intersect(sc, r, 1e30f, h, false)) return sky(r.d);

    v3 n = h.n, P = h.p + h.ng * (2e-4f * fmaxf(1, h.t));
    v3 V = -r.d;
    v3 albedo;
    float f0, shin, ks;
    if (h.mat == MAT_TEAPOT) {
        albedo = v3(0.62f, 0.07f, 0.05f); f0 = 0.05f; shin = 180; ks = 0.9f;
    } else {
        int c = ((int)floorf(h.p.x) + (int)floorf(h.p.z)) & 1;
        albedo = c ? v3(0.78f, 0.77f, 0.74f) : v3(0.16f, 0.17f, 0.20f);
        f0 = 0.03f; shin = 60; ks = 0.25f;
    }

    // soft shadow: one sample on a disk light, stratified with the pixel sample
    v3 ld = normalize(light_pos - P);
    v3 lt = normalize(cross(ld, fabsf(ld.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0))), lb = cross(ld, lt);
    float a = 2 * (float)M_PI * sx, rr = light_radius * sqrtf(sy);
    v3 lp = light_pos + (lt * (rr * cosf(a)) + lb * (rr * sinf(a)));
    v3 L = lp - P;
    float ldist = length(L);
    L = L * (1 / ldist);
    float ndl = dot(n, L), vis = 0;
    Hit sh;
    if (ndl > 0 && dot(h.ng, L) > 0) vis = intersect(sc, {P, L}, ldist, sh, true) ? 0 : 1;

    v3 H = normalize(L + V);
    float spec = ks * powf(fmaxf(dot(n, H), 0), shin) * (shin + 8) / (8 * (float)M_PI);
    v3 direct = (albedo * fmaxf(ndl, 0) + v3(spec, spec, spec)) * vis;
    direct = direct * light_col;

    // sky light with one ambient-occlusion ray (primary hits only)
    float ao = 0.7f;
    if (depth == 0) {
        float r1 = rnd(rng), r2 = rnd(rng), phi = 2 * (float)M_PI * r1, sr = sqrtf(r2);
        v3 t1 = normalize(cross(n, fabsf(n.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0))), t2 = cross(n, t1);
        v3 d = normalize(t1 * (sr * cosf(phi)) + t2 * (sr * sinf(phi)) + n * sqrtf(1 - r2));
        ao = intersect(sc, {P, d}, 4.0f, sh, true) ? 0.15f : 1.0f;
    }
    v3 amb = albedo * sky(n) * (0.55f * ao * (0.6f + 0.4f * n.y));

    v3 col = direct + amb;

    if (depth < 4) {
        float cs = fmaxf(dot(n, V), 0);
        float F = f0 + (1 - f0) * powf(1 - cs, 5);
        if (h.mat == MAT_FLOOR) F = fminf(F + 0.06f, 1);
        v3 R = normalize(reflect(r.d, n));
        if (dot(R, h.ng) < 0) R = normalize(reflect(r.d, h.ng));
        col = col * (1 - F) + trace(sc, {P, R}, depth + 1, rng, sx, sy) * F;
    }
    if (h.mat == MAT_FLOOR) col = lerp(col, sky(v3(0, 0.05f, 0)), 1 - expf(-h.t * 0.02f));
    return col;
}

inline float tonemap(float x) {
    x *= 0.9f;
    x = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);  // ACES fit
    return powf(fminf(fmaxf(x, 0), 1), 1 / 2.2f);
}

}  // namespace

void CpuRender::start(const Camera &cam, const Target &t, int spp, uint32_t frame, int threads,
                      const volatile sig_atomic_t *quit) {
    join();
    cam_ = cam; tgt_ = t; spp_ = spp; frame_ = frame; quit_ = quit;
    next_row_ = 0; rows_done_ = 0;
    for (int i = 0; i < threads; i++) threads_.emplace_back(&CpuRender::worker, this, i + 1);
}

void CpuRender::join() {
    for (auto &th : threads_) th.join();
    threads_.clear();
}

void CpuRender::worker(int id) {
    uint64_t rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t)id * 0xD1B54A32D192ED03ull) ^ ((uint64_t)frame_ << 32);
    int grid = (int)sqrtf((float)spp_);
    if (grid < 1) grid = 1;
    int n = grid * grid;
    float aspect = (float)tgt_.w / tgt_.h;
    for (;;) {
        int y = next_row_.fetch_add(1);
        if (y >= tgt_.h || *quit_) break;
        for (int x = 0; x < tgt_.w; x++) {
            v3 acc;
            for (int s = 0; s < n; s++) {
                float jx = ((s % grid) + rnd(rng)) / grid, jy = ((s / grid) + rnd(rng)) / grid;
                float px = (2 * (x + jx) / tgt_.w - 1) * cam_.tan * aspect;
                float py = (1 - 2 * (y + jy) / tgt_.h) * cam_.tan;
                v3 d = normalize(cam_.fwd + (cam_.right * px + cam_.up * py));
                // decorrelated stratum for the light sample
                int ls = (s * 7 + 3) % n;
                float lx = ((ls % grid) + rnd(rng)) / grid, ly = ((ls / grid) + rnd(rng)) / grid;
                acc += trace(scene_, {cam_.pos, d}, 0, rng, lx, ly);
            }
            acc = acc * (1.0f / n);
            tgt_.put_rgb8(x, y, (uint32_t)(tonemap(acc.x) * 255 + 0.5f), (uint32_t)(tonemap(acc.y) * 255 + 0.5f),
                          (uint32_t)(tonemap(acc.z) * 255 + 0.5f));
        }
        rows_done_.fetch_add(1);
    }
}

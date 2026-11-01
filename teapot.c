// teapot.c — Utah teapot ray tracer that talks to the display only through
// kernel interfaces: DRM/KMS dumb buffers via raw ioctls (no libdrm, no Mesa,
// no X11/Wayland/GL/Vulkan), falling back to the fbdev device, or a PPM file.
//
//   ./teapot                 camera orbits the teapot on the first DRM output (run from a VT)
//   ./teapot -p              still image instead, refined until Enter is pressed
//   ./teapot -f /dev/fb0     force fbdev
//   ./teapot -o out.ppm      render a still to a file (-w/-h set the size)
//   ./teapot -o f.ppm -n 120 render a 120-frame orbit to f_0000.ppm ... f_0119.ppm
//   options: -s <spp> samples per pixel (per frame when animating), -r <n> render at
//            1/n resolution on screen, -b no temporal accumulation of animation frames,
//            -t <sec> stop after this long, -d <card>,
//            -g auto|gpu|cpu  rendering backend (-c = -g cpu)
//
// Backends: the GPU (gpu.c: the ray tracer in a small kernel IR, compiled by
// our own hardware layer and submitted with raw kernel-driver ioctls, no Mesa;
// today Intel Haswell on i915), or all CPU cores. "auto" tries them in that
// order.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <linux/fb.h>
#include <linux/kd.h>

/* ---------------------------------------------------------------- math --- */

typedef struct { float x, y, z; } v3;

static inline v3 V(float x, float y, float z) { return (v3){x, y, z}; }
static inline v3 add(v3 a, v3 b) { return V(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline v3 sub(v3 a, v3 b) { return V(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline v3 mul(v3 a, float s) { return V(a.x * s, a.y * s, a.z * s); }
static inline v3 vmul(v3 a, v3 b) { return V(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline v3 cross(v3 a, v3 b) {
    return V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float len(v3 a) { return sqrtf(dot(a, a)); }
static inline v3 norm(v3 a) { float l = len(a); return l > 0 ? mul(a, 1.0f / l) : a; }
static inline v3 lerp(v3 a, v3 b, float t) { return add(mul(a, 1 - t), mul(b, t)); }
static inline float axis(v3 a, int i) { return i == 0 ? a.x : i == 1 ? a.y : a.z; }
static inline v3 vmin(v3 a, v3 b) { return V(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
static inline v3 vmax(v3 a, v3 b) { return V(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
static inline v3 reflect(v3 d, v3 n) { return sub(d, mul(n, 2 * dot(d, n))); }

static inline float rnd(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return (float)(*s >> 40) / (float)(1 << 24);
}

/* ------------------------------------------------- Newell teapot data --- */
// Classic 10 patch / 127 control point form; rim, body, lid and bottom are
// mirrored into 4 quadrants, handle and spout into 2 → 32 bicubic patches.

static const int patches[10][16] = {
    {102, 103, 104, 105, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},       // rim
    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},     // body
    {24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
    {96, 96, 96, 96, 97, 98, 99, 100, 101, 101, 101, 101, 0, 1, 2, 3},    // lid
    {0, 1, 2, 3, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117},
    {118, 118, 118, 118, 126, 125, 124, 123, 122, 120, 119, 121, 40, 39, 38, 37}, // bottom
    {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},     // handle
    {53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 28, 65, 66, 67},
    {68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83},     // spout
    {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
};

static const float cpdata[127][3] = {
    {0.2, 0, 2.7}, {0.2, -0.112, 2.7}, {0.112, -0.2, 2.7}, {0, -0.2, 2.7},
    {1.3375, 0, 2.53125}, {1.3375, -0.749, 2.53125}, {0.749, -1.3375, 2.53125}, {0, -1.3375, 2.53125},
    {1.4375, 0, 2.53125}, {1.4375, -0.805, 2.53125}, {0.805, -1.4375, 2.53125}, {0, -1.4375, 2.53125},
    {1.5, 0, 2.4}, {1.5, -0.84, 2.4}, {0.84, -1.5, 2.4}, {0, -1.5, 2.4},
    {1.75, 0, 1.875}, {1.75, -0.98, 1.875}, {0.98, -1.75, 1.875}, {0, -1.75, 1.875},
    {2, 0, 1.35}, {2, -1.12, 1.35}, {1.12, -2, 1.35}, {0, -2, 1.35},
    {2, 0, 0.9}, {2, -1.12, 0.9}, {1.12, -2, 0.9}, {0, -2, 0.9},
    {-2, 0, 0.9},
    {2, 0, 0.45}, {2, -1.12, 0.45}, {1.12, -2, 0.45}, {0, -2, 0.45},
    {1.5, 0, 0.225}, {1.5, -0.84, 0.225}, {0.84, -1.5, 0.225}, {0, -1.5, 0.225},
    {1.5, 0, 0.15}, {1.5, -0.84, 0.15}, {0.84, -1.5, 0.15}, {0, -1.5, 0.15},
    {-1.6, 0, 2.025}, {-1.6, -0.3, 2.025}, {-1.5, -0.3, 2.25}, {-1.5, 0, 2.25},
    {-2.3, 0, 2.025}, {-2.3, -0.3, 2.025}, {-2.5, -0.3, 2.25}, {-2.5, 0, 2.25},
    {-2.7, 0, 2.025}, {-2.7, -0.3, 2.025}, {-3, -0.3, 2.25}, {-3, 0, 2.25},
    {-2.7, 0, 1.8}, {-2.7, -0.3, 1.8}, {-3, -0.3, 1.8}, {-3, 0, 1.8},
    {-2.7, 0, 1.575}, {-2.7, -0.3, 1.575}, {-3, -0.3, 1.35}, {-3, 0, 1.35},
    {-2.5, 0, 1.125}, {-2.5, -0.3, 1.125}, {-2.65, -0.3, 0.9375}, {-2.65, 0, 0.9375},
    {-2, -0.3, 0.9}, {-1.9, -0.3, 0.6}, {-1.9, 0, 0.6},
    {1.7, 0, 1.425}, {1.7, -0.66, 1.425}, {1.7, -0.66, 0.6}, {1.7, 0, 0.6},
    {2.6, 0, 1.425}, {2.6, -0.66, 1.425}, {3.1, -0.66, 0.825}, {3.1, 0, 0.825},
    {2.3, 0, 2.1}, {2.3, -0.25, 2.1}, {2.4, -0.25, 2.025}, {2.4, 0, 2.025},
    {2.7, 0, 2.4}, {2.7, -0.25, 2.4}, {3.3, -0.25, 2.4}, {3.3, 0, 2.4},
    {2.8, 0, 2.475}, {2.8, -0.25, 2.475}, {3.525, -0.25, 2.49375}, {3.525, 0, 2.49375},
    {2.9, 0, 2.475}, {2.9, -0.15, 2.475}, {3.45, -0.15, 2.5125}, {3.45, 0, 2.5125},
    {2.8, 0, 2.4}, {2.8, -0.15, 2.4}, {3.2, -0.15, 2.4}, {3.2, 0, 2.4},
    {0, 0, 3.15}, {0.8, 0, 3.15}, {0.8, -0.45, 3.15}, {0.45, -0.8, 3.15}, {0, -0.8, 3.15},
    {0, 0, 2.85},
    {1.4, 0, 2.4}, {1.4, -0.784, 2.4}, {0.784, -1.4, 2.4}, {0, -1.4, 2.4},
    {0.4, 0, 2.55}, {0.4, -0.224, 2.55}, {0.224, -0.4, 2.55}, {0, -0.4, 2.55},
    {1.3, 0, 2.55}, {1.3, -0.728, 2.55}, {0.728, -1.3, 2.55}, {0, -1.3, 2.55},
    {1.3, 0, 2.4}, {1.3, -0.728, 2.4}, {0.728, -1.3, 2.4}, {0, -1.3, 2.4},
    {0, 0, 0},
    {1.5, -0.84, 0.075}, {0.84, -1.5, 0.075}, {1.5, 0, 0.075}, {0, -1.5, 0.075},
    {1.425, 0, 0}, {1.425, -0.798, 0}, {0.798, -1.425, 0}, {0, -1.425, 0},
};

/* --------------------------------------------- tessellation + BVH ------- */

#define TESS 24  // quads per patch edge

typedef struct { v3 v0, e1, e2; int i0, i1, i2; } tri_t;
typedef struct { v3 bmin, bmax; int first, count; } node_t;  // count==0: children at first, first+1

static v3 *vpos, *vnrm;
static tri_t *tris;
static int nverts, ntris;
static node_t *nodes;
static int nnodes;
static int *tidx;

static void bern(float t, float b[4], float d[4]) {
    float s = 1 - t;
    b[0] = s * s * s; b[1] = 3 * t * s * s; b[2] = 3 * t * t * s; b[3] = t * t * t;
    d[0] = -3 * s * s; d[1] = 3 * s * s - 6 * t * s; d[2] = 6 * t * s - 3 * t * t; d[3] = 3 * t * t;
}

static void eval_patch(v3 cp[4][4], float u, float v, v3 *p, v3 *du, v3 *dv) {
    float bu[4], du_[4], bv[4], dv_[4];
    bern(u, bu, du_); bern(v, bv, dv_);
    *p = *du = *dv = V(0, 0, 0);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            *p = add(*p, mul(cp[i][j], bu[i] * bv[j]));
            *du = add(*du, mul(cp[i][j], du_[i] * bv[j]));
            *dv = add(*dv, mul(cp[i][j], bu[i] * dv_[j]));
        }
}

static void build_mesh(void) {
    static const int quad[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    int npatch = 6 * 4 + 4 * 2;
    int vp = (TESS + 1) * (TESS + 1);
    vpos = malloc(sizeof(v3) * npatch * vp);
    vnrm = malloc(sizeof(v3) * npatch * vp);
    tris = malloc(sizeof(tri_t) * npatch * TESS * TESS * 2);

    for (int p = 0; p < 10; p++) {
        int copies = p < 6 ? 4 : 2;
        for (int c = 0; c < copies; c++) {
            float sx = quad[c][0], sy = quad[c][1];
            v3 cp[4][4];
            for (int i = 0; i < 16; i++) {
                const float *d = cpdata[patches[p][i]];
                // teapot data is z-up; rotate into y-up
                cp[i / 4][i % 4] = V(sx * d[0], d[2], -sy * d[1]);
            }
            int base = nverts;
            for (int i = 0; i <= TESS; i++)
                for (int j = 0; j <= TESS; j++) {
                    float u = (float)i / TESS, v = (float)j / TESS;
                    v3 pt, du, dv;
                    eval_patch(cp, u, v, &pt, &du, &dv);
                    v3 n = cross(du, dv);
                    if (len(n) < 1e-6f) {  // degenerate pole (lid knob, bottom centre)
                        v3 q;
                        eval_patch(cp, fminf(fmaxf(u, 1e-3f), 1 - 1e-3f),
                                   fminf(fmaxf(v, 1e-3f), 1 - 1e-3f), &q, &du, &dv);
                        n = cross(du, dv);
                    }
                    vpos[nverts] = pt;
                    vnrm[nverts++] = norm(n);
                }
            for (int i = 0; i < TESS; i++)
                for (int j = 0; j < TESS; j++) {
                    int a = base + i * (TESS + 1) + j, b = a + 1;
                    int cc = a + TESS + 1, d = cc + 1;
                    int t[2][3] = {{a, cc, b}, {b, cc, d}};
                    for (int k = 0; k < 2; k++) {
                        tri_t T = {vpos[t[k][0]], sub(vpos[t[k][1]], vpos[t[k][0]]),
                                   sub(vpos[t[k][2]], vpos[t[k][0]]), t[k][0], t[k][1], t[k][2]};
                        if (len(cross(T.e1, T.e2)) > 1e-9f) tris[ntris++] = T;
                    }
                }
        }
    }
}

static int sort_axis;
static v3 centroid(int i) { return add(tris[i].v0, mul(add(tris[i].e1, tris[i].e2), 1.0f / 3)); }
static int cmp_tri(const void *a, const void *b) {
    float ca = axis(centroid(*(const int *)a), sort_axis);
    float cb = axis(centroid(*(const int *)b), sort_axis);
    return (ca > cb) - (ca < cb);
}

static void build_node(int ni, int first, int count) {
    node_t *n = &nodes[ni];
    n->bmin = V(1e30f, 1e30f, 1e30f); n->bmax = V(-1e30f, -1e30f, -1e30f);
    v3 cmin = n->bmin, cmax = n->bmax;
    for (int i = first; i < first + count; i++) {
        tri_t *t = &tris[tidx[i]];
        v3 a = t->v0, b = add(a, t->e1), c = add(a, t->e2);
        n->bmin = vmin(n->bmin, vmin(a, vmin(b, c)));
        n->bmax = vmax(n->bmax, vmax(a, vmax(b, c)));
        v3 ce = centroid(tidx[i]);
        cmin = vmin(cmin, ce); cmax = vmax(cmax, ce);
    }
    if (count <= 4) { n->first = first; n->count = count; return; }

    // binned surface-area heuristic: pick the axis/plane minimising A_l*N_l + A_r*N_r
    enum { BINS = 16 };
    float best = 1e30f;
    int best_axis = -1, best_bin = 0;
    for (int ax = 0; ax < 3; ax++) {
        float lo = axis(cmin, ax), span = axis(cmax, ax) - lo;
        if (span < 1e-6f) continue;
        int cnt[BINS] = {0};
        v3 bmn[BINS], bmx[BINS];
        for (int b = 0; b < BINS; b++) bmn[b] = V(1e30f, 1e30f, 1e30f), bmx[b] = V(-1e30f, -1e30f, -1e30f);
        for (int i = first; i < first + count; i++) {
            tri_t *t = &tris[tidx[i]];
            int b = (int)((axis(centroid(tidx[i]), ax) - lo) / span * BINS);
            b = b < 0 ? 0 : b >= BINS ? BINS - 1 : b;
            v3 p0 = t->v0, p1 = add(p0, t->e1), p2 = add(p0, t->e2);
            cnt[b]++;
            bmn[b] = vmin(bmn[b], vmin(p0, vmin(p1, p2)));
            bmx[b] = vmax(bmx[b], vmax(p0, vmax(p1, p2)));
        }
        float right_cost[BINS];
        v3 rmn = V(1e30f, 1e30f, 1e30f), rmx = V(-1e30f, -1e30f, -1e30f);
        int rn = 0;
        for (int b = BINS - 1; b > 0; b--) {
            rmn = vmin(rmn, bmn[b]); rmx = vmax(rmx, bmx[b]); rn += cnt[b];
            v3 e = sub(rmx, rmn);
            right_cost[b] = rn ? (e.x * e.y + e.y * e.z + e.z * e.x) * rn : 0;
        }
        v3 lmn = V(1e30f, 1e30f, 1e30f), lmx = V(-1e30f, -1e30f, -1e30f);
        int ln = 0;
        for (int b = 0; b < BINS - 1; b++) {
            lmn = vmin(lmn, bmn[b]); lmx = vmax(lmx, bmx[b]); ln += cnt[b];
            if (!ln || ln == count) continue;
            v3 e = sub(lmx, lmn);
            float c = (e.x * e.y + e.y * e.z + e.z * e.x) * ln + right_cost[b + 1];
            if (c < best) { best = c; best_axis = ax; best_bin = b; }
        }
    }

    int mid;
    if (best_axis >= 0) {
        float lo = axis(cmin, best_axis), span = axis(cmax, best_axis) - lo;
        int i = first, j = first + count - 1;
        while (i <= j) {
            int b = (int)((axis(centroid(tidx[i]), best_axis) - lo) / span * BINS);
            if (b <= best_bin) i++;
            else { int tmp = tidx[i]; tidx[i] = tidx[j]; tidx[j--] = tmp; }
        }
        mid = i - first;
    } else {  // all centroids coincide: plain median split
        v3 ext = sub(cmax, cmin);
        sort_axis = ext.x > ext.y && ext.x > ext.z ? 0 : ext.y > ext.z ? 1 : 2;
        qsort(tidx + first, count, sizeof(int), cmp_tri);
        mid = count / 2;
    }
    int l = nnodes;
    nnodes += 2;
    n->first = l; n->count = 0;
    build_node(l, first, mid);
    build_node(l + 1, first + mid, count - mid);
}

static void build_bvh(void) {
    tidx = malloc(sizeof(int) * ntris);
    for (int i = 0; i < ntris; i++) tidx[i] = i;
    nodes = malloc(sizeof(node_t) * ntris * 2);
    nnodes = 1;
    build_node(0, 0, ntris);
}

/* ------------------------------------------------------------ tracing --- */

typedef struct { v3 o, d; } ray_t;
typedef struct { float t; v3 p, n, ng; int mat; } hit_t;

enum { MAT_NONE, MAT_TEAPOT, MAT_FLOOR };

static inline int box_hit(const node_t *n, v3 o, v3 inv, float tmax) {
    float tx1 = (n->bmin.x - o.x) * inv.x, tx2 = (n->bmax.x - o.x) * inv.x;
    float tmin = fminf(tx1, tx2), tm = fmaxf(tx1, tx2);
    float ty1 = (n->bmin.y - o.y) * inv.y, ty2 = (n->bmax.y - o.y) * inv.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2)); tm = fminf(tm, fmaxf(ty1, ty2));
    float tz1 = (n->bmin.z - o.z) * inv.z, tz2 = (n->bmax.z - o.z) * inv.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2)); tm = fminf(tm, fmaxf(tz1, tz2));
    return tm >= fmaxf(tmin, 0) && tmin < tmax;
}

// Möller–Trumbore; returns t or -1
static inline float tri_hit(const tri_t *t, v3 o, v3 d, float *bu, float *bv) {
    v3 p = cross(d, t->e2);
    float det = dot(t->e1, p);
    if (fabsf(det) < 1e-12f) return -1;
    float id = 1 / det;
    v3 s = sub(o, t->v0);
    float u = dot(s, p) * id;
    if (u < 0 || u > 1) return -1;
    v3 q = cross(s, t->e1);
    float v = dot(d, q) * id;
    if (v < 0 || u + v > 1) return -1;
    *bu = u; *bv = v;
    return dot(t->e2, q) * id;
}

// any_hit: stop at first occluder closer than tmax (shadow / AO rays)
static int intersect(ray_t r, float tmax, hit_t *h, int any_hit) {
    h->t = tmax; h->mat = MAT_NONE;
    v3 inv = V(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
    int stack[64], sp = 0, best = -1;
    float bu = 0, bv = 0;
    stack[sp++] = 0;
    while (sp) {
        const node_t *n = &nodes[stack[--sp]];
        if (!box_hit(n, r.o, inv, h->t)) continue;
        if (n->count) {
            for (int i = n->first; i < n->first + n->count; i++) {
                float u, v, t = tri_hit(&tris[tidx[i]], r.o, r.d, &u, &v);
                if (t > 1e-4f && t < h->t) {
                    h->t = t; best = tidx[i]; bu = u; bv = v;
                    if (any_hit) return 1;
                }
            }
        } else {
            stack[sp++] = n->first;
            stack[sp++] = n->first + 1;
        }
    }
    if (best >= 0) {
        const tri_t *t = &tris[best];
        h->mat = MAT_TEAPOT;
        h->ng = norm(cross(t->e1, t->e2));
        h->n = norm(add(add(mul(vnrm[t->i0], 1 - bu - bv), mul(vnrm[t->i1], bu)), mul(vnrm[t->i2], bv)));
    }
    if (r.d.y < 0 && r.o.y > 0) {  // floor plane y = 0
        float t = -r.o.y / r.d.y;
        if (t < h->t) {
            if (any_hit) return 1;
            h->t = t; h->mat = MAT_FLOOR; h->n = h->ng = V(0, 1, 0);
        }
    }
    if (h->mat == MAT_NONE) return 0;
    h->p = add(r.o, mul(r.d, h->t));
    if (dot(h->ng, r.d) > 0) h->ng = mul(h->ng, -1);  // two-sided surfaces
    if (dot(h->n, h->ng) < 0) h->n = mul(h->n, -1);
    return 1;
}

static const v3 light_pos = {-6.0f, 8.0f, 0.5f};
static const float light_radius = 1.6f;
static const v3 light_col = {1.25f, 1.15f, 1.0f};

static v3 sky(v3 d) {
    float t = fmaxf(d.y, 0);
    return lerp(V(0.82f, 0.86f, 0.93f), V(0.30f, 0.45f, 0.78f), sqrtf(t));
}

static v3 trace(ray_t r, int depth, uint64_t *rng, float sx, float sy) {
    hit_t h;
    if (!intersect(r, 1e30f, &h, 0)) return sky(r.d);

    v3 n = h.n, P = add(h.p, mul(h.ng, 2e-4f * fmaxf(1, h.t)));
    v3 V_ = mul(r.d, -1);
    v3 albedo;
    float f0, shin, ks;
    if (h.mat == MAT_TEAPOT) {
        albedo = V(0.62f, 0.07f, 0.05f); f0 = 0.05f; shin = 180; ks = 0.9f;
    } else {
        int c = ((int)floorf(h.p.x) + (int)floorf(h.p.z)) & 1;
        albedo = c ? V(0.78f, 0.77f, 0.74f) : V(0.16f, 0.17f, 0.20f);
        f0 = 0.03f; shin = 60; ks = 0.25f;
    }

    // soft shadow: one sample on a disk light, stratified with the pixel sample
    v3 lc = sub(light_pos, P), ld = norm(lc);
    v3 lt = norm(cross(ld, fabsf(ld.y) < 0.9f ? V(0, 1, 0) : V(1, 0, 0))), lb = cross(ld, lt);
    float a = 2 * (float)M_PI * sx, rr = light_radius * sqrtf(sy);
    v3 lp = add(light_pos, add(mul(lt, rr * cosf(a)), mul(lb, rr * sinf(a))));
    v3 L = sub(lp, P);
    float ldist = len(L);
    L = mul(L, 1 / ldist);
    float ndl = dot(n, L), vis = 0;
    hit_t sh;
    if (ndl > 0 && dot(h.ng, L) > 0) vis = intersect((ray_t){P, L}, ldist, &sh, 1) ? 0 : 1;

    v3 H = norm(add(L, V_));
    float spec = ks * powf(fmaxf(dot(n, H), 0), shin) * (shin + 8) / (8 * (float)M_PI);
    v3 direct = mul(add(mul(albedo, fmaxf(ndl, 0)), V(spec, spec, spec)), vis);
    direct = vmul(direct, light_col);

    // sky light with one ambient-occlusion ray (primary hits only)
    float ao = 0.7f;
    if (depth == 0) {
        float r1 = rnd(rng), r2 = rnd(rng), phi = 2 * (float)M_PI * r1, sr = sqrtf(r2);
        v3 t1 = norm(cross(n, fabsf(n.y) < 0.9f ? V(0, 1, 0) : V(1, 0, 0))), t2 = cross(n, t1);
        v3 d = norm(add(add(mul(t1, sr * cosf(phi)), mul(t2, sr * sinf(phi))), mul(n, sqrtf(1 - r2))));
        ao = intersect((ray_t){P, d}, 4.0f, &sh, 1) ? 0.15f : 1.0f;
    }
    v3 amb = mul(vmul(albedo, sky(n)), 0.55f * ao * (0.6f + 0.4f * n.y));

    v3 col = add(direct, amb);

    if (depth < 4) {
        float cs = fmaxf(dot(n, V_), 0);
        float F = f0 + (1 - f0) * powf(1 - cs, 5);
        if (h.mat == MAT_FLOOR) F = fminf(F + 0.06f, 1);
        v3 R = norm(reflect(r.d, n));
        if (dot(R, h.ng) < 0) R = norm(reflect(r.d, h.ng));
        col = add(mul(col, 1 - F), mul(trace((ray_t){P, R}, depth + 1, rng, sx, sy), F));
    }
    if (h.mat == MAT_FLOOR) col = lerp(col, sky(V(0, 0.05f, 0)), 1 - expf(-h.t * 0.02f));
    return col;
}

/* ---------------------------------------------------- render target ---- */

typedef struct {
    uint8_t *base;
    int w, h, pitch, bpp;
    int ro, rl, go, gl, bo, bl;
} target_t;

static inline float tonemap(float x) {
    x *= 0.9f;
    x = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);  // ACES fit
    return powf(fminf(fmaxf(x, 0), 1), 1 / 2.2f);
}

static void put_rgb8(const target_t *t, int x, int y, uint32_t r, uint32_t g, uint32_t b) {
    uint32_t px = (r >> (8 - t->rl)) << t->ro | (g >> (8 - t->gl)) << t->go | (b >> (8 - t->bl)) << t->bo;
    uint8_t *p = t->base + (size_t)y * t->pitch + (size_t)x * (t->bpp / 8);
    switch (t->bpp) {
    case 32: *(volatile uint32_t *)p = px; break;
    case 24: p[0] = px; p[1] = px >> 8; p[2] = px >> 16; break;
    case 16: *(volatile uint16_t *)p = px; break;
    }
}

static void put_pixel(const target_t *t, int x, int y, v3 c) {
    put_rgb8(t, x, y, (uint32_t)(tonemap(c.x) * 255 + 0.5f), (uint32_t)(tonemap(c.y) * 255 + 0.5f),
             (uint32_t)(tonemap(c.z) * 255 + 0.5f));
}

/* ------------------------------------------------------ render loop ---- */

static target_t tgt;
static int spp = 0;  // 0 = default: 32 on the GPU, 9 on the CPU
static atomic_int next_row, rows_done;
static volatile sig_atomic_t quit;
static v3 cam_pos, cam_fwd, cam_right, cam_up;
static float cam_tan;
static uint32_t frame_no;  // varies the CPU sampling noise between animation frames

// camera on a circle around the teapot; angle 0 is the original viewpoint
static void set_camera(float angle) {
    const v3 centre = {0.2f, 0, 0}, look = {0.2f, 1.25f, 0};
    const float radius = 9.44f, height = 4.4f, a0 = 0.635f;  // a0 = atan2(5.6, 7.6)
    cam_pos = V(centre.x + radius * sinf(a0 + angle), height, centre.z + radius * cosf(a0 + angle));
    cam_fwd = norm(sub(look, cam_pos));
    cam_right = norm(cross(cam_fwd, V(0, 1, 0)));
    cam_up = cross(cam_right, cam_fwd);
    cam_tan = tanf(34.0f * (float)M_PI / 360.0f);
}

static void *worker(void *arg) {
    uint64_t rng = 0x9E3779B97F4A7C15ull ^ ((uintptr_t)arg * 0xD1B54A32D192ED03ull) ^
                   ((uint64_t)frame_no << 32);
    int grid = (int)sqrtf((float)spp);
    if (grid < 1) grid = 1;
    int n = grid * grid;
    float aspect = (float)tgt.w / tgt.h;
    for (;;) {
        int y = atomic_fetch_add(&next_row, 1);
        if (y >= tgt.h || quit) break;
        for (int x = 0; x < tgt.w; x++) {
            v3 acc = V(0, 0, 0);
            for (int s = 0; s < n; s++) {
                float jx = ((s % grid) + rnd(&rng)) / grid, jy = ((s / grid) + rnd(&rng)) / grid;
                float px = (2 * (x + jx) / tgt.w - 1) * cam_tan * aspect;
                float py = (1 - 2 * (y + jy) / tgt.h) * cam_tan;
                v3 d = norm(add(cam_fwd, add(mul(cam_right, px), mul(cam_up, py))));
                // decorrelated stratum for the light sample
                int ls = (s * 7 + 3) % n;
                float lx = ((ls % grid) + rnd(&rng)) / grid, ly = ((ls / grid) + rnd(&rng)) / grid;
                acc = add(acc, trace((ray_t){cam_pos, d}, 0, &rng, lx, ly));
            }
            put_pixel(&tgt, x, y, mul(acc, 1.0f / n));
        }
        atomic_fetch_add(&rows_done, 1);
    }
    return NULL;
}

/* ------------------------------------------------------ DRM / KMS ------ */

static int drm_fd = -1;
static uint32_t drm_conn, drm_crtc, drm_fb[2], drm_handle[2];
static uint64_t drm_size;
static void *drm_map[2];
static int drm_front;         // buffer currently scanned out (or queued to be)
static int drm_flip_pending;  // a page flip was queued and its completion event not yet read
static struct drm_mode_crtc drm_saved;

static int drm_open(const char *path) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    ioctl(fd, DRM_IOCTL_SET_MASTER, 0);  // implicit on first open from a VT; harmless otherwise

    struct drm_mode_card_res res = {0};
    uint32_t conns[64], crtcs[32], encs[64];
    if (ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &res) < 0) goto fail;
    if (res.count_connectors > 64) res.count_connectors = 64;
    if (res.count_crtcs > 32) res.count_crtcs = 32;
    if (res.count_encoders > 64) res.count_encoders = 64;
    res.count_fbs = 0;
    res.connector_id_ptr = (uintptr_t)conns;
    res.crtc_id_ptr = (uintptr_t)crtcs;
    res.encoder_id_ptr = (uintptr_t)encs;
    if (ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &res) < 0) goto fail;

    for (uint32_t i = 0; i < res.count_connectors; i++) {
        struct drm_mode_get_connector c = {.connector_id = conns[i]};
        if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &c) < 0) continue;  // probes modes
        if (c.connection != 1 || !c.count_modes) continue;
        struct drm_mode_modeinfo modes[c.count_modes];
        uint32_t cenc[c.count_encoders + 1];
        c.modes_ptr = (uintptr_t)modes;
        c.encoders_ptr = (uintptr_t)cenc;
        c.count_props = 0; c.props_ptr = c.prop_values_ptr = 0;
        if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &c) < 0 || !c.count_modes) continue;

        struct drm_mode_modeinfo mode = modes[0];
        for (uint32_t m = 0; m < c.count_modes; m++)
            if (modes[m].type & DRM_MODE_TYPE_PREFERRED) { mode = modes[m]; break; }

        uint32_t crtc = 0;
        if (c.encoder_id) {
            struct drm_mode_get_encoder e = {.encoder_id = c.encoder_id};
            if (!ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &e)) crtc = e.crtc_id;
        }
        for (uint32_t k = 0; !crtc && k < c.count_encoders; k++) {
            struct drm_mode_get_encoder e = {.encoder_id = cenc[k]};
            if (ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &e)) continue;
            for (uint32_t j = 0; j < res.count_crtcs; j++)
                if (e.possible_crtcs & (1u << j)) { crtc = crtcs[j]; break; }
        }
        if (!crtc) continue;

        // two dumb buffers: still images draw into the front one, animation flips between them
        struct drm_mode_create_dumb cd;
        for (int k = 0; k < 2; k++) {
            cd = (struct drm_mode_create_dumb){.width = mode.hdisplay, .height = mode.vdisplay, .bpp = 32};
            if (ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd) < 0) { perror("CREATE_DUMB"); goto fail; }
            struct drm_mode_fb_cmd fb = {.width = cd.width, .height = cd.height, .pitch = cd.pitch,
                                         .bpp = 32, .depth = 24, .handle = cd.handle};
            if (ioctl(fd, DRM_IOCTL_MODE_ADDFB, &fb) < 0) { perror("ADDFB"); goto fail; }
            struct drm_mode_map_dumb md = {.handle = cd.handle};
            if (ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md) < 0) { perror("MAP_DUMB"); goto fail; }
            void *map = mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, md.offset);
            if (map == MAP_FAILED) { perror("mmap dumb"); goto fail; }
            memset(map, 0, cd.size);
            drm_fb[k] = fb.fb_id; drm_handle[k] = cd.handle; drm_map[k] = map;
        }

        drm_saved = (struct drm_mode_crtc){.crtc_id = crtc};
        ioctl(fd, DRM_IOCTL_MODE_GETCRTC, &drm_saved);

        drm_conn = conns[i];
        struct drm_mode_crtc set = {.crtc_id = crtc, .fb_id = drm_fb[0],
                                    .set_connectors_ptr = (uintptr_t)&drm_conn,
                                    .count_connectors = 1, .mode = mode, .mode_valid = 1};
        if (ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &set) < 0) {
            perror("SETCRTC (is a compositor holding DRM master? run from a text VT)");
            goto fail;
        }
        drm_fd = fd; drm_crtc = crtc; drm_size = cd.size; drm_front = 0;
        tgt = (target_t){drm_map[0], cd.width, cd.height, cd.pitch, 32, 16, 8, 8, 8, 0, 8};
        fprintf(stderr, "drm: %s connector %u crtc %u %ux%u@%u\n", path, drm_conn, crtc,
                mode.hdisplay, mode.vdisplay, mode.vrefresh);
        return 0;
    }
    fprintf(stderr, "drm: %s has no usable connected output\n", path);
fail:
    close(fd);
    return -1;
}

static void drm_dirty(void) {
    struct drm_mode_fb_dirty_cmd d = {.fb_id = drm_fb[drm_front]};  // needed by virtio/udl style drivers
    ioctl(drm_fd, DRM_IOCTL_MODE_DIRTYFB, &d);
}

// wait until the last queued flip has happened, so the other buffer is free to draw into
static void drm_wait_flip(void) {
    if (!drm_flip_pending) return;
    char ev[1024];
    struct pollfd pfd = {.fd = drm_fd, .events = POLLIN};
    if (poll(&pfd, 1, 100) > 0 && read(drm_fd, ev, sizeof ev) < 0) { /* event content unused */ }
    drm_flip_pending = 0;
}

// queue buffer `back` for display at the next vblank (does not wait)
static void drm_flip(int back) {
    struct drm_mode_fb_dirty_cmd d = {.fb_id = drm_fb[back]};
    ioctl(drm_fd, DRM_IOCTL_MODE_DIRTYFB, &d);
    struct drm_mode_crtc_page_flip fl = {.crtc_id = drm_crtc, .fb_id = drm_fb[back],
                                         .flags = DRM_MODE_PAGE_FLIP_EVENT};
    if (!ioctl(drm_fd, DRM_IOCTL_MODE_PAGE_FLIP, &fl)) {
        drm_flip_pending = 1;
    } else {  // no page flip support: just switch the CRTC over
        struct drm_mode_crtc c = {.crtc_id = drm_crtc};
        ioctl(drm_fd, DRM_IOCTL_MODE_GETCRTC, &c);
        c.fb_id = drm_fb[back];
        c.set_connectors_ptr = (uintptr_t)&drm_conn;
        c.count_connectors = 1;
        ioctl(drm_fd, DRM_IOCTL_MODE_SETCRTC, &c);
    }
    drm_front = back;
}

static void drm_close(void) {
    drm_wait_flip();
    if (drm_saved.mode_valid) {
        drm_saved.set_connectors_ptr = (uintptr_t)&drm_conn;
        drm_saved.count_connectors = 1;
        ioctl(drm_fd, DRM_IOCTL_MODE_SETCRTC, &drm_saved);
    }
    for (int k = 0; k < 2; k++) {
        munmap(drm_map[k], drm_size);
        ioctl(drm_fd, DRM_IOCTL_MODE_RMFB, &drm_fb[k]);
        struct drm_mode_destroy_dumb dd = {.handle = drm_handle[k]};
        ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
    }
    close(drm_fd);
}

/* ---------------------------------------------------------- fbdev ------ */

static int fb_fd = -1, kd_set;
static void *fb_map;
static size_t fb_len;

static int fb_open(const char *path) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror(path); return -1; }
    struct fb_var_screeninfo vi;
    struct fb_fix_screeninfo fi;
    if (ioctl(fd, FBIOGET_VSCREENINFO, &vi) < 0 || ioctl(fd, FBIOGET_FSCREENINFO, &fi) < 0) {
        perror("fbdev ioctl"); close(fd); return -1;
    }
    if (vi.bits_per_pixel != 32 && vi.bits_per_pixel != 24 && vi.bits_per_pixel != 16) {
        fprintf(stderr, "fbdev: unsupported %u bpp\n", vi.bits_per_pixel); close(fd); return -1;
    }
    void *map = mmap(NULL, fi.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap fb"); close(fd); return -1; }
    fb_fd = fd; fb_map = map; fb_len = fi.smem_len;
    uint8_t *base = (uint8_t *)map + vi.yoffset * fi.line_length + vi.xoffset * vi.bits_per_pixel / 8;
    tgt = (target_t){base, vi.xres, vi.yres, fi.line_length, vi.bits_per_pixel,
                     vi.red.offset, vi.red.length, vi.green.offset, vi.green.length,
                     vi.blue.offset, vi.blue.length};
    if (!ioctl(STDIN_FILENO, KDSETMODE, KD_GRAPHICS)) kd_set = 1;  // hide the console on a VT
    fprintf(stderr, "fbdev: %s %ux%u %ubpp\n", path, vi.xres, vi.yres, vi.bits_per_pixel);
    return 0;
}

static void fb_close(void) {
    if (kd_set) ioctl(STDIN_FILENO, KDSETMODE, KD_TEXT);
    munmap(fb_map, fb_len);
    close(fb_fd);
}

/* ----------------------------------------------------------- main ------ */

static void on_signal(int s) { (void)s; quit = 1; }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------- GPU scene layout --- */
// Shared by the GPU backends: 8 depth-first copies of the BVH with "skip"
// links (one per ray-direction octant, nearer child first) so a walk needs no
// stack, then one 128-byte record per triangle in leaf order.
//   node (32 B):  bmin.xyz, skip (u32) | bmax.xyz, first_tri << 3 | count (u32, 0 = interior)
//   triangle:     Havel-Herout planes N,d | N1,d1 | N2,d2 | n0 | n1-n0 | n2-n0 | geometric normal
static size_t scene_layout_bytes(void) { return (size_t)nnodes * 32; }
static size_t scene_blob_bytes(void) { return 8 * scene_layout_bytes() + (size_t)ntris * 128; }

static void write_scene_blob(void *dst) {
    int n = nnodes;
    size_t layout_bytes = scene_layout_bytes(), node_bytes = 8 * layout_bytes;
    int *order = malloc(sizeof(int) * n), *skip = malloc(sizeof(int) * n), *size = malloc(sizeof(int) * n);
    for (int oct = 0; oct < 8; oct++) {
        int stack[128], sp = 0, k = 0;
        stack[sp++] = 0;
        while (sp) {
            int ni = stack[--sp];
            order[k++] = ni;
            if (nodes[ni].count) continue;
            int l = nodes[ni].first, r = l + 1;
            // split axis ≈ largest separation of the child centres
            v3 d = sub(add(nodes[r].bmin, nodes[r].bmax), add(nodes[l].bmin, nodes[l].bmax));
            int ax = fabsf(d.x) > fabsf(d.y) && fabsf(d.x) > fabsf(d.z) ? 0 : fabsf(d.y) > fabsf(d.z) ? 1 : 2;
            int neg_dir = oct >> ax & 1;
            int near = (axis(d, ax) > 0) != neg_dir ? l : r;   // child a ray in this octant meets first
            stack[sp++] = near == l ? r : l;
            stack[sp++] = near;
        }
        for (int i = n - 1; i >= 0; i--) {  // skip(i) = i + size of subtree(i)
            node_t *nd = &nodes[order[i]];
            size[order[i]] = 1 + (nd->count ? 0 : size[nd->first] + size[nd->first + 1]);
            skip[i] = i + size[order[i]];
        }
        float *nf = (float *)((char *)dst + oct * layout_bytes);
        for (int i = 0; i < n; i++) {
            node_t *nd = &nodes[order[i]];
            float *r = nf + i * 8;
            uint32_t *u = (uint32_t *)r;
            r[0] = nd->bmin.x; r[1] = nd->bmin.y; r[2] = nd->bmin.z; u[3] = skip[i];
            r[4] = nd->bmax.x; r[5] = nd->bmax.y; r[6] = nd->bmax.z;
            u[7] = nd->count ? (uint32_t)nd->first << 3 | nd->count : 0;  // count <= 4
        }
    }
    free(size);
    float *tf = (float *)((char *)dst + node_bytes);
    for (int k = 0; k < ntris; k++) {
        tri_t *tr = &tris[tidx[k]];
        float *r = tf + k * 32;
        memset(r, 0, 128);
        v3 n0 = vnrm[tr->i0], n1 = vnrm[tr->i1], n2 = vnrm[tr->i2];
        v3 N = cross(tr->e1, tr->e2), ng = norm(N);
        float nn = dot(N, N);
        v3 N1 = mul(cross(tr->e2, N), 1 / nn), N2 = mul(cross(N, tr->e1), 1 / nn);
        v3 vals[7] = {N, N1, N2, n0, sub(n1, n0), sub(n2, n0), ng};
        float w[3] = {dot(N, tr->v0), -dot(N1, tr->v0), -dot(N2, tr->v0)};
        for (int j = 0; j < 7; j++) {
            r[j * 4] = vals[j].x; r[j * 4 + 1] = vals[j].y; r[j * 4 + 2] = vals[j].z;
            if (j < 3) r[j * 4 + 3] = w[j];
        }
    }
    free(order); free(skip);
}

/* ------------------------------------------------- GPU backends ------- */
// A backend renders whole passes of the current camera (cam_*) asynchronously:
// submit() queues `samples` per pixel and returns, collect() waits and hands
// back XRGB8888 pixels plus the primary hit distance per pixel (negative on
// the teapot, 1e6 for sky).

#include "gpu.c"

typedef struct {
    const char *name;
    int (*init)(int w, int h);
    int (*submit)(int samples, uint32_t seed, int sample_base);
    int (*collect)(uint32_t *rgb, float *depth);
    void (*fini)(void);
} backend_t;

static const backend_t be_gpu = {"gpu", gpu_init, gpu_submit, gpu_collect, gpu_fini};

#define MAX_PASS 8  // samples per submitted pass (keeps every GPU batch short)

// "auto" / "gpu": the GPU if one is supported; "cpu" → NULL
static const backend_t *open_backend(const char *want, int w, int h) {
    if ((!strcmp(want, "auto") || !strcmp(want, "gpu")) && be_gpu.init(w, h) == 0) return &be_gpu;
    return NULL;
}

static uint32_t pass_seed(uint32_t i) {
    uint32_t s = i * 0x9E3779B9u + 0x7f4a7c15u;
    s ^= s >> 15; s *= 0x2c1b3c6du; s ^= s >> 12;
    return s;
}

// passes are averaged in linear light: 8-bit sRGB-ish in, 12-bit LUT back out
static float lin8[256];
static uint8_t enc12[4096];

static void tone_luts(void) {
    for (int i = 0; i < 256; i++) lin8[i] = powf(i / 255.0f, 2.2f);
    for (int i = 0; i < 4096; i++) enc12[i] = (uint8_t)(powf(i / 4095.0f, 1 / 2.2f) * 255 + 0.5f);
}

static void accumulate(float *acc, const uint32_t *rgb, size_t n) {
    for (size_t i = 0; i < n; i++) {
        acc[3 * i] += lin8[rgb[i] >> 16 & 255];
        acc[3 * i + 1] += lin8[rgb[i] >> 8 & 255];
        acc[3 * i + 2] += lin8[rgb[i] & 255];
    }
}

static int gpu_refine;    // still image on screen: keep adding passes until a key is pressed
static int gpu_key_stop;  // set when that key press ended the render

// progressive still image into tgt (the display or the file buffer)
static int gpu_still(const backend_t *b, int drm) {
    int w = tgt.w, h = tgt.h, per_pass = 4;
    while (spp % per_pass) per_pass--;
    int passes = spp / per_pass, pass = 0, err = 0;
    uint32_t *rgb = malloc((size_t)w * h * 4);
    float *acc = calloc((size_t)w * h * 3, sizeof(float));
    tone_luts();
    for (;; pass++) {
        if (quit) break;
        if (pass >= passes) {
            // on screen: keep refining until a key is pressed (or 1024 samples)
            if (!gpu_refine || pass * per_pass >= 1024) break;
            struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
            if (poll(&pfd, 1, 0) > 0) { gpu_key_stop = 1; break; }
        }
        if ((err = b->submit(per_pass, pass_seed(pass), pass * per_pass)) || (err = b->collect(rgb, NULL))) break;
        accumulate(acc, rgb, (size_t)w * h);
        float k = 4095.0f / (pass + 1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const float *a = acc + ((size_t)y * w + x) * 3;
                put_rgb8(&tgt, x, y, enc12[(int)(a[0] * k)], enc12[(int)(a[1] * k)], enc12[(int)(a[2] * k)]);
            }
        if (drm) drm_dirty();
        if (gpu_refine && pass + 1 >= passes)
            fprintf(stderr, "\r%d samples/pixel, press Enter to exit ", (pass + 1) * per_pass);
    }
    if (gpu_refine) fputc('\n', stderr);
    spp = pass * per_pass;
    free(rgb); free(acc);
    return quit || err ? -1 : 0;
}

// one animation frame with more samples than a single pass: several passes, averaged
static int gpu_frame_passes(const backend_t *b, uint32_t frame, float *depth) {
    size_t n = (size_t)tgt.w * tgt.h;
    uint32_t *rgb = malloc(n * 4);
    float *acc = calloc(n * 3, sizeof(float));
    int done = 0, err = 0;
    while (done < spp && !err) {
        int k = spp - done < MAX_PASS ? spp - done : MAX_PASS;
        err = b->submit(k, pass_seed(frame * 1024 + done), frame * spp + done) || b->collect(rgb, depth);
        for (size_t i = 0; i < n && !err; i++)
            for (int a = 0; a < 3; a++) acc[3 * i + a] += lin8[rgb[i] >> (16 - 8 * a) & 255] * k;
        done += k;
    }
    for (size_t i = 0; i < n && !err; i++) {
        const float *a = acc + 3 * i;
        float s = 4095.0f / spp;
        ((uint32_t *)tgt.base)[i] = (uint32_t)enc12[(int)(a[0] * s)] << 16 |
                                    (uint32_t)enc12[(int)(a[1] * s)] << 8 | enc12[(int)(a[2] * s)];
    }
    free(rgb); free(acc);
    return err;
}

#define ORBIT_SECONDS 12.0  // one revolution of the camera on screen

static long nthreads;

/* ---- temporal accumulation: reproject the previous frames with the depth ---- */

typedef struct { v3 pos, fwd, right, up; float tx, ty; } view_t;

static view_t current_view(void) {
    return (view_t){cam_pos, cam_fwd, cam_right, cam_up, cam_tan * tgt.w / tgt.h, cam_tan};
}

// history: linear RGB + sample count, and view-space depth (1e9 = sky)
static float *hist[2], *hist_depth[2], *frame_depth_buf, *pix_sx, *pix_sy, *pix_il;
static int hist_cur, hist_ok;
static view_t hist_view, frame_view;

static void *temporal_rows(void *arg) {
    long part = (long)arg;
    int W = tgt.w, H = tgt.h, y0 = H * part / nthreads, y1 = H * (part + 1) / nthreads;
    const float *ph = hist[hist_cur], *pd = hist_depth[hist_cur];
    float *nh = hist[!hist_cur], *nd = hist_depth[!hist_cur];
    const view_t v = frame_view, pv = hist_view;
    const v3 dp = sub(v.pos, pv.pos);
    const float itx = 0.5f * W / pv.tx, ity = 0.5f * H / pv.ty;
    for (int y = y0; y < y1; y++) {
        uint32_t *row = (uint32_t *)(tgt.base + (size_t)y * tgt.pitch);
        v3 base = add(v.fwd, mul(v.up, pix_sy[y]));
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            uint32_t p = row[x];
            float c[3] = {lin8[p >> 16 & 255], lin8[p >> 8 & 255], lin8[p & 255]};
            float t = frame_depth_buf[i], il = pix_il[i], h[3] = {0, 0, 0}, n = 0, wsum = 0;
            int glossy = t < 0, sky = t >= 1e5f;   // the GPU marks teapot pixels with a negative depth
            t = fabsf(t);
            if (hist_ok) {
                // world point of this pixel relative to the previous camera
                float s = (sky ? 1e4f : t) * il;
                v3 q = add(dp, mul(add(base, mul(v.right, pix_sx[x])), s));
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
            o[0] = h[0] + (c[0] - h[0]) * alpha;
            o[1] = h[1] + (c[1] - h[1]) * alpha;
            o[2] = h[2] + (c[2] - h[2]) * alpha;
            o[3] = n + 1;
            nd[i] = sky ? 1e9f : t * il;
            row[x] = (uint32_t)enc12[(int)(fminf(o[0], 1) * 4095)] << 16 |
                     (uint32_t)enc12[(int)(fminf(o[1], 1) * 4095)] << 8 | enc12[(int)(fminf(o[2], 1) * 4095)];
        }
    }
    return NULL;
}

// blend the frame just rendered into tgt (+ frame_depth_buf) with the history; result goes to tgt
static void temporal(void) {
    size_t n = (size_t)tgt.w * tgt.h;
    if (!hist[0]) {
        for (int k = 0; k < 2; k++) {
            hist[k] = calloc(n * 4, sizeof(float));
            hist_depth[k] = calloc(n, sizeof(float));
        }
        // pixel-centre ray offsets; the ray length only depends on them, not on the camera
        float tx = cam_tan * tgt.w / tgt.h, ty = cam_tan;
        pix_sx = malloc(sizeof(float) * tgt.w); pix_sy = malloc(sizeof(float) * tgt.h);
        pix_il = malloc(sizeof(float) * n);
        for (int x = 0; x < tgt.w; x++) pix_sx[x] = ((x + 0.5f) * 2 / tgt.w - 1) * tx;
        for (int y = 0; y < tgt.h; y++) pix_sy[y] = (1 - (y + 0.5f) * 2 / tgt.h) * ty;
        for (int y = 0; y < tgt.h; y++)
            for (int x = 0; x < tgt.w; x++)
                pix_il[(size_t)y * tgt.w + x] = 1 / sqrtf(1 + pix_sx[x] * pix_sx[x] + pix_sy[y] * pix_sy[y]);
        tone_luts();
    }
    pthread_t th[nthreads];
    for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, temporal_rows, (void *)i);
    for (long i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    hist_cur = !hist_cur;
    hist_view = frame_view;
    hist_ok = 1;
}

// render tgt on all CPU cores (blocking)
static void cpu_frame(void) {
    atomic_store(&next_row, 0);
    atomic_store(&rows_done, 0);
    pthread_t th[nthreads];
    for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
    for (long i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}

static int write_ppm(const char *path, const target_t *t) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    fprintf(f, "P6\n%d %d\n255\n", t->w, t->h);
    for (int y = 0; y < t->h; y++)
        for (int x = 0; x < t->w; x++) {
            uint32_t px = ((uint32_t *)(t->base + (size_t)y * t->pitch))[x];
            uint8_t rgb[3] = {px >> 16, px >> 8, px};
            fwrite(rgb, 1, 3, f);
        }
    fclose(f);
    return 0;
}

// copy an XRGB8888 frame to a display target, upscaling by an integer factor
static void blit(const target_t *src, const target_t *dst, int scale) {
    int direct = dst->bpp == 32 && dst->ro == 16 && dst->go == 8 && dst->bo == 0;
    for (int y = 0; y < dst->h; y++) {
        int sy = y / scale < src->h ? y / scale : src->h - 1;
        const uint32_t *row = (const uint32_t *)(src->base + (size_t)sy * src->pitch);
        uint32_t *drow = (uint32_t *)(dst->base + (size_t)y * dst->pitch);
        for (int x = 0; x < dst->w; x++) {
            int sx = x / scale < src->w ? x / scale : src->w - 1;
            uint32_t p = row[sx];
            if (direct) drow[x] = p;
            else put_rgb8(dst, x, y, p >> 16 & 255, p >> 8 & 255, p & 255);
        }
    }
}

int main(int argc, char **argv) {
    const char *out = NULL, *fbdev = NULL, *card = NULL;
    int W = 1280, H = 720, hold = -1, still = 0, nframes = 0, rscale = 1, no_blend = 0, opt;
    const char *backend = "auto";
    while ((opt = getopt(argc, argv, "o:f:d:w:h:s:t:cpn:r:bg:")) != -1) {
        switch (opt) {
        case 'o': out = optarg; break;
        case 'f': fbdev = optarg; break;
        case 'd': card = optarg; break;
        case 'w': W = atoi(optarg); break;
        case 'h': H = atoi(optarg); break;
        case 's': spp = atoi(optarg); break;
        case 't': hold = atoi(optarg); break;
        case 'c': backend = "cpu"; break;
        case 'g': backend = optarg; break;
        case 'p': still = 1; break;
        case 'n': nframes = atoi(optarg); break;
        case 'r': rscale = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'b': no_blend = 1; break;
        default:
            fprintf(stderr, "usage: %s [-o file.ppm [-n frames]] [-p] [-f /dev/fbN] [-d /dev/dri/cardN]\n"
                            "       [-w W -h H] [-s spp] [-r scale] [-b] [-t seconds] [-g auto|gpu|cpu] [-c]\n",
                    argv[0]);
            return 1;
        }
    }

    double t0 = now();
    build_mesh();
    build_bvh();
    fprintf(stderr, "teapot: %d triangles, %d BVH nodes (%.0f ms)\n", ntris, nnodes, (now() - t0) * 1e3);

    enum { OUT_FILE, OUT_DRM, OUT_FB } mode;
    uint8_t *filebuf = NULL;
    if (out) {
        mode = OUT_FILE;
        filebuf = calloc((size_t)W * H, 4);
        tgt = (target_t){filebuf, W, H, W * 4, 32, 16, 8, 8, 8, 0, 8};
    } else {
        int ok = -1;
        if (!fbdev) {
            if (card) ok = drm_open(card);
            for (int i = 0; ok < 0 && !card && i < 8; i++) {
                char p[32];
                snprintf(p, sizeof p, "/dev/dri/card%d", i);
                if (!access(p, F_OK)) ok = drm_open(p);
            }
        }
        if (ok == 0) mode = OUT_DRM;
        else if (fb_open(fbdev ? fbdev : "/dev/fb0") == 0) mode = OUT_FB;
        else {
            fprintf(stderr, "no display available; use -o out.ppm\n");
            if (getenv("DISPLAY") || getenv("WAYLAND_DISPLAY"))
                fprintf(stderr, "a desktop session owns the screen: switch to a text console "
                                "(Ctrl+Alt+F3), log in and run it there\n");
            return 1;
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    set_camera(0);
    nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1) nthreads = 1;
    int auto_spp = spp <= 0;
    const char *device = "cpu";

    if (mode == OUT_FILE ? nframes > 0 : !still) {
        /* ---- animation: the camera orbits the teapot ---- */
        target_t disp = tgt;
        int scale = mode == OUT_FILE ? 1 : rscale;
        int fw = mode == OUT_FILE ? W : (disp.w + scale - 1) / scale;
        int fh = mode == OUT_FILE ? H : (disp.h + scale - 1) / scale;
        uint8_t *frame = calloc((size_t)fw * fh, 4);
        tgt = (target_t){frame, fw, fh, fw * 4, 32, 16, 8, 8, 8, 0, 8};
        const backend_t *gpu = open_backend(backend, fw, fh);
        if (gpu) device = gpu->name;
        else if (strcmp(backend, "cpu") && strcmp(backend, "auto")) { fprintf(stderr, "backend %s unavailable\n", backend); return 1; }
        if (auto_spp) spp = gpu ? (mode == OUT_FILE ? 16 : 1) : (mode == OUT_FILE ? 4 : 1);
        tone_luts();
        // on screen the GPU renders few samples per frame and accumulates them over time
        int blend = gpu && mode != OUT_FILE && !no_blend;
        if (blend) frame_depth_buf = malloc((size_t)fw * fh * sizeof(float));

        char pattern[512];
        if (mode == OUT_FILE) {
            const char *dot = strrchr(out, '.');
            if (strchr(out, '%')) snprintf(pattern, sizeof pattern, "%s", out);
            else if (dot) snprintf(pattern, sizeof pattern, "%.*s_%%04d%s", (int)(dot - out), out, dot);
            else snprintf(pattern, sizeof pattern, "%s_%%04d.ppm", out);
        }
        fprintf(stderr, "animating %dx%d @ %d spp%s on the %s%s\n", fw, fh, spp,
                blend ? " + temporal accumulation" : "", device,
                mode == OUT_FILE ? "" : " (Enter or Ctrl-C to stop)");
        double start = now(), tick = start;
        int frames = 0, tick_frames = 0;
        // light frames are pipelined: frame f+1 renders while f is blended and shown
        int pipelined = gpu && spp <= MAX_PASS, queued = 0;
        view_t queued_view;
#define ANGLE(i) (mode == OUT_FILE ? 2 * (float)M_PI * (i) / nframes \
                                   : (float)((now() - start) * 2 * M_PI / ORBIT_SECONDS))
        for (uint32_t f = 0; !quit; f++) {
            if (pipelined && !queued) {             // first frame
                set_camera(ANGLE(f));
                queued_view = current_view();
                if (gpu->submit(spp, pass_seed(f), f * spp)) goto gpu_failed;
                queued = 1;
            }
            if (pipelined) {
                queued = 0;
                if (gpu->collect((uint32_t *)tgt.base, frame_depth_buf)) goto gpu_failed;
                frame_view = queued_view;
                if (mode != OUT_FILE || (int)f + 1 < nframes) {
                    set_camera(ANGLE(f + 1));
                    queued_view = current_view();
                    if (gpu->submit(spp, pass_seed(f + 1), (f + 1) * spp)) goto gpu_failed;
                    queued = 1;
                }
            } else {
                set_camera(ANGLE(f));
                frame_view = current_view();
                if (gpu && gpu_frame_passes(gpu, f, frame_depth_buf)) {
                gpu_failed:
                    fprintf(stderr, "%s failed, falling back to CPU rendering\n", gpu->name);
                    gpu->fini();
                    gpu = NULL; pipelined = queued = blend = 0; device = "cpu";
                    if (auto_spp) spp = mode == OUT_FILE ? 4 : 1;
                    set_camera(ANGLE(f));
                }
                if (!gpu) { frame_no = f; cpu_frame(); }
            }
#undef ANGLE
            if (quit) break;
            if (blend) temporal();
            frames++; tick_frames++;

            if (mode == OUT_FILE) {
                char name[600];
                snprintf(name, sizeof name, pattern, f);
                if (write_ppm(name, &tgt)) break;
                fprintf(stderr, "\r%s (%u/%d)", name, f + 1, nframes);
                if ((int)f + 1 >= nframes) { fputc('\n', stderr); break; }
                continue;
            }
            if (mode == OUT_DRM) {
                drm_wait_flip();
                target_t back = disp;
                back.base = drm_map[!drm_front];
                blit(&tgt, &back, scale);
                drm_flip(!drm_front);
            } else {
                blit(&tgt, &disp, scale);
            }
            double t = now();
            if (t - tick >= 1) {
                fprintf(stderr, "\r%.1f fps ", tick_frames / (t - tick));
                tick = t; tick_frames = 0;
            }
            if (hold >= 0 && t - start >= hold) break;
            struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
            if (isatty(STDIN_FILENO) && poll(&pfd, 1, 0) > 0) {
                char c;
                if (read(STDIN_FILENO, &c, 1) < 0) { /* any key stops */ }
                break;
            }
        }
        double secs = now() - start;
        if (gpu) gpu->fini();
        tgt = disp;
        if (mode == OUT_DRM) drm_close();
        else if (mode == OUT_FB) fb_close();
        fprintf(stderr, "\n%d frames in %.1f s (%.1f fps) on the %s\n", frames, secs, frames / secs, device);
        return 0;
    }

    /* ---- still image ---- */
    t0 = now();
    gpu_refine = mode != OUT_FILE && hold < 0 && isatty(STDIN_FILENO);
    if (auto_spp) spp = 32;
    const backend_t *gpu = open_backend(backend, tgt.w, tgt.h);
    if (!gpu && strcmp(backend, "cpu") && strcmp(backend, "auto")) { fprintf(stderr, "backend %s unavailable\n", backend); return 1; }
    int ok = gpu && gpu_still(gpu, mode == OUT_DRM) == 0;
    if (gpu) gpu->fini();
    if (ok) {
        device = gpu->name;
    } else if (!quit) {
        if (gpu) fprintf(stderr, "%s failed, falling back to CPU rendering\n", gpu->name);
        if (auto_spp) spp = 9;
        atomic_store(&rows_done, 0);
        atomic_store(&next_row, 0);
        pthread_t th[nthreads];
        for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
        while (!quit && atomic_load(&rows_done) < tgt.h) {
            usleep(100000);
            if (mode == OUT_DRM) drm_dirty();
            if (mode == OUT_FILE)
                fprintf(stderr, "\r%3d%%", atomic_load(&rows_done) * 100 / tgt.h);
        }
        for (long i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    }
    if (mode == OUT_DRM) drm_dirty();
    double secs = now() - t0;

    if (mode == OUT_FILE) {
        fprintf(stderr, "\rrendered %dx%d @ %d spp on the %s in %.2f s\n", W, H, spp, device, secs);
        return write_ppm(out, &tgt) ? 1 : 0;
    }

    // hold the image: until Enter / Ctrl-C, or for -t seconds
    if (gpu_key_stop) {
        char c;
        if (read(STDIN_FILENO, &c, 1) < 0) { /* the key only ends the hold */ }
    } else if (!quit) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        int ms = hold >= 0 ? hold * 1000 : -1;
        if (!isatty(STDIN_FILENO) && hold < 0) ms = 10000;
        int r = poll(&pfd, 1, ms);
        if (r > 0 && !isatty(STDIN_FILENO)) {  // stdin at EOF: don't flash and exit
            char c;
            if (read(STDIN_FILENO, &c, 1) == 0) usleep(hold >= 0 ? 0 : 10000000);
        }
    }

    if (mode == OUT_DRM) drm_close();
    else fb_close();
    fprintf(stderr, "rendered %dx%d @ %d spp on the %s in %.2f s\n", tgt.w, tgt.h, spp, device, secs);
    return 0;
}

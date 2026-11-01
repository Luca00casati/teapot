// scene.cpp — teapot tessellation, binned-SAH BVH, and the GPU scene layout.

#include "scene.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

constexpr int TESS = 24;  // quads per patch edge

static void bern(float t, float b[4], float d[4]) {
    float s = 1 - t;
    b[0] = s * s * s; b[1] = 3 * t * s * s; b[2] = 3 * t * t * s; b[3] = t * t * t;
    d[0] = -3 * s * s; d[1] = 3 * s * s - 6 * t * s; d[2] = 6 * t * s - 3 * t * t; d[3] = 3 * t * t;
}

static void eval_patch(const v3 cp[4][4], float u, float v, v3 &p, v3 &du, v3 &dv) {
    float bu[4], du_[4], bv[4], dv_[4];
    bern(u, bu, du_); bern(v, bv, dv_);
    p = du = dv = v3();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            p += cp[i][j] * (bu[i] * bv[j]);
            du += cp[i][j] * (du_[i] * bv[j]);
            dv += cp[i][j] * (bu[i] * dv_[j]);
        }
}

void Scene::build_mesh() {
    static const int quad[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    int npatch = 6 * 4 + 4 * 2;
    vpos.reserve(npatch * (TESS + 1) * (TESS + 1));
    vnrm.reserve(vpos.capacity());
    tris.reserve(npatch * TESS * TESS * 2);

    for (int p = 0; p < 10; p++) {
        int copies = p < 6 ? 4 : 2;
        for (int c = 0; c < copies; c++) {
            float sx = quad[c][0], sy = quad[c][1];
            v3 cp[4][4];
            for (int i = 0; i < 16; i++) {
                const float *d = cpdata[patches[p][i]];
                // teapot data is z-up; rotate into y-up
                cp[i / 4][i % 4] = v3(sx * d[0], d[2], -sy * d[1]);
            }
            int base = (int)vpos.size();
            for (int i = 0; i <= TESS; i++)
                for (int j = 0; j <= TESS; j++) {
                    float u = (float)i / TESS, v = (float)j / TESS;
                    v3 pt, du, dv;
                    eval_patch(cp, u, v, pt, du, dv);
                    v3 n = cross(du, dv);
                    if (length(n) < 1e-6f) {  // degenerate pole (lid knob, bottom centre)
                        v3 q;
                        eval_patch(cp, fminf(fmaxf(u, 1e-3f), 1 - 1e-3f), fminf(fmaxf(v, 1e-3f), 1 - 1e-3f), q, du, dv);
                        n = cross(du, dv);
                    }
                    vpos.push_back(pt);
                    vnrm.push_back(normalize(n));
                }
            for (int i = 0; i < TESS; i++)
                for (int j = 0; j < TESS; j++) {
                    int a = base + i * (TESS + 1) + j, b = a + 1;
                    int cc = a + TESS + 1, d = cc + 1;
                    int t[2][3] = {{a, cc, b}, {b, cc, d}};
                    for (auto &k : t) {
                        Tri T = {vpos[k[0]], vpos[k[1]] - vpos[k[0]], vpos[k[2]] - vpos[k[0]], k[0], k[1], k[2]};
                        if (length(cross(T.e1, T.e2)) > 1e-9f) tris.push_back(T);
                    }
                }
        }
    }
}

static v3 centroid(const Tri &t) { return t.v0 + (t.e1 + t.e2) * (1.0f / 3); }

static const Scene *sort_scene;
static int sort_axis;
static int cmp_tri(const void *a, const void *b) {
    float ca = centroid(sort_scene->tris[*(const int *)a])[sort_axis];
    float cb = centroid(sort_scene->tris[*(const int *)b])[sort_axis];
    return (ca > cb) - (ca < cb);
}

void Scene::build_node(int ni, int first, int count) {
    const v3 big(1e30f, 1e30f, 1e30f);
    Node &n = nodes[ni];
    n.bmin = big; n.bmax = -big;
    v3 cmin = n.bmin, cmax = n.bmax;
    for (int i = first; i < first + count; i++) {
        const Tri &t = tris[tidx[i]];
        v3 a = t.v0, b = a + t.e1, c = a + t.e2;
        n.bmin = vmin(n.bmin, vmin(a, vmin(b, c)));
        n.bmax = vmax(n.bmax, vmax(a, vmax(b, c)));
        v3 ce = centroid(t);
        cmin = vmin(cmin, ce); cmax = vmax(cmax, ce);
    }
    if (count <= 4) { n.first = first; n.count = count; return; }

    // binned surface-area heuristic: pick the axis/plane minimising A_l*N_l + A_r*N_r
    constexpr int BINS = 16;
    float best = 1e30f;
    int best_axis = -1, best_bin = 0;
    for (int ax = 0; ax < 3; ax++) {
        float lo = cmin[ax], span = cmax[ax] - lo;
        if (span < 1e-6f) continue;
        int cnt[BINS] = {0};
        v3 bmn[BINS], bmx[BINS];
        for (int b = 0; b < BINS; b++) bmn[b] = big, bmx[b] = -big;
        for (int i = first; i < first + count; i++) {
            const Tri &t = tris[tidx[i]];
            int b = (int)((centroid(t)[ax] - lo) / span * BINS);
            b = b < 0 ? 0 : b >= BINS ? BINS - 1 : b;
            v3 p0 = t.v0, p1 = p0 + t.e1, p2 = p0 + t.e2;
            cnt[b]++;
            bmn[b] = vmin(bmn[b], vmin(p0, vmin(p1, p2)));
            bmx[b] = vmax(bmx[b], vmax(p0, vmax(p1, p2)));
        }
        float right_cost[BINS];
        v3 rmn = big, rmx = -big;
        int rn = 0;
        for (int b = BINS - 1; b > 0; b--) {
            rmn = vmin(rmn, bmn[b]); rmx = vmax(rmx, bmx[b]); rn += cnt[b];
            v3 e = rmx - rmn;
            right_cost[b] = rn ? (e.x * e.y + e.y * e.z + e.z * e.x) * rn : 0;
        }
        v3 lmn = big, lmx = -big;
        int ln = 0;
        for (int b = 0; b < BINS - 1; b++) {
            lmn = vmin(lmn, bmn[b]); lmx = vmax(lmx, bmx[b]); ln += cnt[b];
            if (!ln || ln == count) continue;
            v3 e = lmx - lmn;
            float c = (e.x * e.y + e.y * e.z + e.z * e.x) * ln + right_cost[b + 1];
            if (c < best) { best = c; best_axis = ax; best_bin = b; }
        }
    }

    int mid;
    if (best_axis >= 0) {
        float lo = cmin[best_axis], span = cmax[best_axis] - lo;
        int i = first, j = first + count - 1;
        while (i <= j) {
            int b = (int)((centroid(tris[tidx[i]])[best_axis] - lo) / span * BINS);
            if (b <= best_bin) i++;
            else { int tmp = tidx[i]; tidx[i] = tidx[j]; tidx[j--] = tmp; }
        }
        mid = i - first;
    } else {  // all centroids coincide: plain median split
        v3 ext = cmax - cmin;
        sort_scene = this;
        sort_axis = ext.x > ext.y && ext.x > ext.z ? 0 : ext.y > ext.z ? 1 : 2;
        qsort(&tidx[first], count, sizeof(int), cmp_tri);
        mid = count / 2;
    }
    int l = (int)nodes.size();
    nodes.resize(l + 2);              // (invalidates n)
    nodes[ni].first = l; nodes[ni].count = 0;
    build_node(l, first, mid);
    build_node(l + 1, first + mid, count - mid);
}

void Scene::build() {
    build_mesh();
    tidx.resize(tris.size());
    for (size_t i = 0; i < tris.size(); i++) tidx[i] = (int)i;
    nodes.reserve(tris.size() * 2);
    nodes.resize(1);
    build_node(0, 0, (int)tris.size());
}

/* ------------------------------------------------- GPU scene layout --- */

void Scene::write_blob(void *dst) const {
    int n = (int)nodes.size();
    size_t layout = layout_bytes();
    ml::Vec<int> order(n), skip(n), size(n);
    for (int oct = 0; oct < 8; oct++) {
        int stack[128], sp = 0, k = 0;
        stack[sp++] = 0;
        while (sp) {
            int ni = stack[--sp];
            order[k++] = ni;
            if (nodes[ni].count) continue;
            int l = nodes[ni].first, r = l + 1;
            // split axis ≈ largest separation of the child centres
            v3 d = (nodes[r].bmin + nodes[r].bmax) - (nodes[l].bmin + nodes[l].bmax);
            int ax = fabsf(d.x) > fabsf(d.y) && fabsf(d.x) > fabsf(d.z) ? 0 : fabsf(d.y) > fabsf(d.z) ? 1 : 2;
            int neg_dir = oct >> ax & 1;
            int near = (d[ax] > 0) != neg_dir ? l : r;   // child a ray in this octant meets first
            stack[sp++] = near == l ? r : l;
            stack[sp++] = near;
        }
        for (int i = n - 1; i >= 0; i--) {  // skip(i) = i + size of subtree(i)
            const Node &nd = nodes[order[i]];
            size[order[i]] = 1 + (nd.count ? 0 : size[nd.first] + size[nd.first + 1]);
            skip[i] = i + size[order[i]];
        }
        float *nf = (float *)((char *)dst + oct * layout);
        for (int i = 0; i < n; i++) {
            const Node &nd = nodes[order[i]];
            float *r = nf + i * 8;
            uint32_t *u = (uint32_t *)r;
            r[0] = nd.bmin.x; r[1] = nd.bmin.y; r[2] = nd.bmin.z; u[3] = skip[i];
            r[4] = nd.bmax.x; r[5] = nd.bmax.y; r[6] = nd.bmax.z;
            u[7] = nd.count ? (uint32_t)nd.first << 3 | nd.count : 0;  // count <= 4
        }
    }
    float *tf = (float *)((char *)dst + 8 * layout);
    for (size_t k = 0; k < tris.size(); k++) {
        const Tri &tr = tris[tidx[k]];
        float *r = tf + k * 32;
        memset(r, 0, 128);
        v3 n0 = vnrm[tr.i0], n1 = vnrm[tr.i1], n2 = vnrm[tr.i2];
        v3 N = cross(tr.e1, tr.e2), ng = normalize(N);
        float nn = dot(N, N);
        v3 N1 = cross(tr.e2, N) * (1 / nn), N2 = cross(N, tr.e1) * (1 / nn);
        v3 vals[7] = {N, N1, N2, n0, n1 - n0, n2 - n0, ng};
        float w[3] = {dot(N, tr.v0), -dot(N1, tr.v0), -dot(N2, tr.v0)};
        for (int j = 0; j < 7; j++) {
            r[j * 4] = vals[j].x; r[j * 4 + 1] = vals[j].y; r[j * 4 + 2] = vals[j].z;
            if (j < 3) r[j * 4 + 3] = w[j];
        }
    }
}

Camera Camera::orbit(float angle) {
    const v3 centre(0.2f, 0, 0), look(0.2f, 1.25f, 0);
    const float radius = 9.44f, height = 4.4f, a0 = 0.635f;  // a0 = atan2(5.6, 7.6)
    Camera c;
    c.pos = v3(centre.x + radius * sinf(a0 + angle), height, centre.z + radius * cosf(a0 + angle));
    c.fwd = normalize(look - c.pos);
    c.right = normalize(cross(c.fwd, v3(0, 1, 0)));
    c.up = cross(c.right, c.fwd);
    c.tan = tanf(34.0f * (float)M_PI / 360.0f);
    return c;
}

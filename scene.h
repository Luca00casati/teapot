// scene.h — the tessellated Newell teapot on a checkerboard floor, its BVH,
// the camera, and the look (light, sky, materials) shared by the CPU tracer
// and the GPU kernel.
#pragma once

#include <cstddef>
#include <vector>

#include "vec.h"

struct Tri { v3 v0, e1, e2; int i0, i1, i2; };
struct Node { v3 bmin, bmax; int first, count; };  // count == 0: children at first, first + 1

struct Scene {
    std::vector<v3> vpos, vnrm;
    std::vector<Tri> tris;
    std::vector<Node> nodes;
    std::vector<int> tidx;       // triangle indices in leaf order

    void build();                // tessellate the teapot, build the SAH BVH

    // GPU layout: 8 depth-first copies of the BVH with "skip" links (one per
    // ray-direction octant, nearer child first) so a walk needs no stack, then
    // one 128-byte record per triangle in leaf order.
    //   node (32 B):  bmin.xyz, skip (u32) | bmax.xyz, first_tri << 3 | count (u32, 0 = interior)
    //   triangle:     Havel-Herout planes N,d | N1,d1 | N2,d2 | n0 | n1-n0 | n2-n0 | geometric normal
    size_t layout_bytes() const { return nodes.size() * 32; }
    size_t blob_bytes() const { return 8 * layout_bytes() + tris.size() * 128; }
    void write_blob(void *dst) const;

private:
    void build_mesh();
    void build_node(int ni, int first, int count);
};

struct Camera {
    v3 pos, fwd, right, up;
    float tan;                   // tan(fov_y / 2)
    static Camera orbit(float angle);  // on a circle around the teapot; 0 = the original viewpoint
};

namespace look {
constexpr v3 light_pos{-6.0f, 8.0f, 0.5f};
constexpr float light_radius = 1.6f;
constexpr v3 light_col{1.25f, 1.15f, 1.0f};
constexpr v3 sky_horizon{0.82f, 0.86f, 0.93f}, sky_zenith{0.30f, 0.45f, 0.78f};
}  // namespace look

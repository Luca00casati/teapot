// vec.h — 3-component float vector used by the scene and the host side of the GPU path.
#pragma once

#include <math.h>
#include <stdint.h>

struct v3 {
    float x = 0, y = 0, z = 0;
    constexpr v3() = default;
    constexpr v3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr float operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};

constexpr v3 operator+(v3 a, v3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr v3 operator-(v3 a, v3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr v3 operator*(v3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
constexpr v3 operator-(v3 a) { return a * -1; }
inline v3 &operator+=(v3 &a, v3 b) { return a = a + b; }

constexpr float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr v3 cross(v3 a, v3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(v3 a) { return sqrtf(dot(a, a)); }
inline v3 normalize(v3 a) { float l = length(a); return l > 0 ? a * (1.0f / l) : a; }
inline v3 vmin(v3 a, v3 b) { return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
inline v3 vmax(v3 a, v3 b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }


// tone.h — passes and frames are averaged in linear light: 8-bit display
// values in through lin8, 12-bit linear back out through enc12.
#pragma once

#include <cmath>
#include <cstdint>

struct ToneLuts {
    float lin8[256];
    uint8_t enc12[4096];
    ToneLuts() {
        for (int i = 0; i < 256; i++) lin8[i] = powf(i / 255.0f, 2.2f);
        for (int i = 0; i < 4096; i++) enc12[i] = (uint8_t)(powf(i / 4095.0f, 1 / 2.2f) * 255 + 0.5f);
    }
    // XRGB8888 from linear [0, 1] (scaled by k / 4095)
    uint32_t encode(const float *c, float k = 4095.0f) const {
        return (uint32_t)enc12[(int)(c[0] * k)] << 16 | (uint32_t)enc12[(int)(c[1] * k)] << 8 | enc12[(int)(c[2] * k)];
    }
};

inline const ToneLuts &tone() {
    static const ToneLuts t;
    return t;
}

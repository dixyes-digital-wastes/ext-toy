#include "toy/q8_0.h"

#include <math.h>
#include <string.h>

float toy_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t shift = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3FFu;
            bits = sign | ((113u - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t toy_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp >= 0x1F) {
        if (((bits >> 23) & 0xFFu) == 0xFFu && mant != 0) {
            return (uint16_t)(sign | 0x7E00u | (mant >> 13));
        }
        return (uint16_t)(sign | 0x7C00u);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half & 1u) != 0)) {
            half++;
        }
        return (uint16_t)(sign | half);
    }

    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u) != 0)) {
        half++;
    }
    return (uint16_t)(sign | half);
}

size_t toy_q8_0_row_bytes(uint32_t k) {
    return (size_t)(k / TOY_Q8_0_BLOCK) * TOY_Q8_0_BLOCK_BYTES;
}

void toy_q8_0_quantize_rows(uint32_t rows, uint32_t k, const float *src, void *dst) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    struct toy_q8_0_block *out = dst;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const float *x = src + ((size_t)r * blocks + b) * TOY_Q8_0_BLOCK;
            float amax = 0.0f;
            for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
                const float v = fabsf(x[i]);
                if (v > amax) {
                    amax = v;
                }
            }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            out->d = toy_f32_to_f16(d);
            for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
                int q = (int)roundf(x[i] * id);
                if (q > 127) {
                    q = 127;
                }
                if (q < -127) {
                    q = -127;
                }
                out->qs[i] = (int8_t)q;
            }
            out++;
        }
    }
}

void toy_q8_0_dequantize_rows(uint32_t rows, uint32_t k, const void *src, float *dst) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct toy_q8_0_block *in = src;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const float d = toy_f16_to_f32(in->d);
            float *x = dst + ((size_t)r * blocks + b) * TOY_Q8_0_BLOCK;
            for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
                x[i] = (float)in->qs[i] * d;
            }
            in++;
        }
    }
}

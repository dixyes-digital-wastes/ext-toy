/*
 * Scalar f32 reference kernels.
 *
 * dot, matvec and the shaped kernels sum over k in increasing order, following
 * ggml's scalar dot product, with fmaf mirroring the single-rounded FMA that
 * the generated kernels use. Row blocking only hides FMA latency; it does not
 * change the order of any individual sum.
 *
 * ref_matvec_avx2_order reproduces the order of the AVX2 matvec kernel, which
 * keeps every eighth element of a row in one lane and folds the lanes with a
 * fixed tree. The extension uses it when the generator rejects a shape, so an
 * AVX2 host returns the same bytes for a row at any m and row offset.
 *
 * Adapted from llama.cpp/ggml (MIT License):
 *   Copyright (c) 2023-2026 The ggml authors
 *   https://github.com/ggml-org/llama.cpp
 *   commit f5027fb21307eb264aed483242603d68c87b0fe3
 *   ggml/src/ggml-cpu/vec.cpp, ggml_vec_dot_f32 scalar fallback
 *
 * The full license text is in THIRD_PARTY_NOTICES.md.
 */

#include "toy/ref.h"

#include <math.h>

#include "toy/q8_0.h"

void ref_dot(uint32_t k, const float *x, const float *y, float *out) {
    float sum = 0.0f;
    for (uint32_t p = 0; p < k; p++) {
        sum = fmaf(x[p], y[p], sum);
    }
    *out = sum;
}

void ref_add_f32(const float *a, const float *b, float *c, size_t count) {
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

void ref_mul_f32(const float *a, const float *b, float *c, size_t count) {
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] * b[i];
    }
}

void ref_scale_f32(const float *a, float factor, float *c, size_t count) {
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] * factor;
    }
}

/* SiLU: c = a * (1 / (1 + exp(-a))) */
void ref_silu_f32(const float *a, float *c, size_t count) {
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] * (1.0f / (1.0f + expf(-a[i])));
    }
}

void ref_rmsnorm_f32(const float *x, const float *w, float *out, uint32_t n, float eps) {
    float squares = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        squares = fmaf(x[i], x[i], squares);
    }

    const float scale = 1.0f / sqrtf(squares / (float) n + eps);
    for (uint32_t i = 0; i < n; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

void ref_matvec(uint32_t m, uint32_t k, const float *w, const float *x, float *y) {
    uint32_t i = 0;
    for (; i + 8 <= m; i += 8) {
        const float *w0 = w + (size_t)(i + 0) * k;
        const float *w1 = w + (size_t)(i + 1) * k;
        const float *w2 = w + (size_t)(i + 2) * k;
        const float *w3 = w + (size_t)(i + 3) * k;
        const float *w4 = w + (size_t)(i + 4) * k;
        const float *w5 = w + (size_t)(i + 5) * k;
        const float *w6 = w + (size_t)(i + 6) * k;
        const float *w7 = w + (size_t)(i + 7) * k;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        float s4 = 0.0f;
        float s5 = 0.0f;
        float s6 = 0.0f;
        float s7 = 0.0f;
        for (uint32_t p = 0; p < k; p++) {
            const float xv = x[p];
            s0 = fmaf(w0[p], xv, s0);
            s1 = fmaf(w1[p], xv, s1);
            s2 = fmaf(w2[p], xv, s2);
            s3 = fmaf(w3[p], xv, s3);
            s4 = fmaf(w4[p], xv, s4);
            s5 = fmaf(w5[p], xv, s5);
            s6 = fmaf(w6[p], xv, s6);
            s7 = fmaf(w7[p], xv, s7);
        }
        y[i + 0] = s0;
        y[i + 1] = s1;
        y[i + 2] = s2;
        y[i + 3] = s3;
        y[i + 4] = s4;
        y[i + 5] = s5;
        y[i + 6] = s6;
        y[i + 7] = s7;
    }
    for (; i < m; i++) {
        y[i] = 0.0f;
        const float *wp = w + (size_t)i * k;
        for (uint32_t p = 0; p < k; p++) {
            y[i] = fmaf(wp[p], x[p], y[i]);
        }
    }
}

/*
 * Same accumulation order as the AVX2 matvec kernel: lane l of the ymm
 * accumulator takes w[p + l] * x[p + l] for p in steps of 8, the lane tree
 * follows the horizontal sum, and the k tail adds in increasing order
 */
void ref_matvec_avx2_order(uint32_t m, uint32_t k, const float *w, const float *x,
                           float *y) {
    const uint32_t k8 = k & ~7u;
    for (uint32_t i = 0; i < m; i++) {
        const float *wp = w + (size_t) i * k;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        float s4 = 0.0f;
        float s5 = 0.0f;
        float s6 = 0.0f;
        float s7 = 0.0f;
        for (uint32_t p = 0; p < k8; p += 8) {
            s0 = fmaf(wp[p + 0], x[p + 0], s0);
            s1 = fmaf(wp[p + 1], x[p + 1], s1);
            s2 = fmaf(wp[p + 2], x[p + 2], s2);
            s3 = fmaf(wp[p + 3], x[p + 3], s3);
            s4 = fmaf(wp[p + 4], x[p + 4], s4);
            s5 = fmaf(wp[p + 5], x[p + 5], s5);
            s6 = fmaf(wp[p + 6], x[p + 6], s6);
            s7 = fmaf(wp[p + 7], x[p + 7], s7);
        }
        float sum = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
        for (uint32_t p = k8; p < k; p++) {
            sum = fmaf(wp[p], x[p], sum);
        }
        y[i] = sum;
    }
}

// 8 rows x 1 column, 8 independent accumulators
static void ref_block_8x1(uint32_t k, const float *b, uint32_t j, uint32_t n,
                              const float *a0, const float *a1, const float *a2, const float *a3,
                              const float *a4, const float *a5, const float *a6, const float *a7,
                              float *c0, float *c1, float *c2, float *c3,
                              float *c4, float *c5, float *c6, float *c7) {
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;
    const float *bp = b + j;
    for (uint32_t p = 0; p < k; p++) {
        const float bv = *bp;
        bp += n;
        s0 = fmaf(a0[p], bv, s0);
        s1 = fmaf(a1[p], bv, s1);
        s2 = fmaf(a2[p], bv, s2);
        s3 = fmaf(a3[p], bv, s3);
        s4 = fmaf(a4[p], bv, s4);
        s5 = fmaf(a5[p], bv, s5);
        s6 = fmaf(a6[p], bv, s6);
        s7 = fmaf(a7[p], bv, s7);
    }
    *c0 = s0;
    *c1 = s1;
    *c2 = s2;
    *c3 = s3;
    *c4 = s4;
    *c5 = s5;
    *c6 = s6;
    *c7 = s7;
}

void ref_gemm(uint32_t m, uint32_t n, uint32_t k, const float *a, const float *b, float *c) {
    uint32_t i0 = 0;
    for (; i0 + 8 <= m; i0 += 8) {
        const float *a0 = a + (size_t)(i0 + 0) * k;
        const float *a1 = a + (size_t)(i0 + 1) * k;
        const float *a2 = a + (size_t)(i0 + 2) * k;
        const float *a3 = a + (size_t)(i0 + 3) * k;
        const float *a4 = a + (size_t)(i0 + 4) * k;
        const float *a5 = a + (size_t)(i0 + 5) * k;
        const float *a6 = a + (size_t)(i0 + 6) * k;
        const float *a7 = a + (size_t)(i0 + 7) * k;

        uint32_t j0 = 0;
        for (; j0 + 4 <= n; j0 += 4) {
            float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
            float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
            float s20 = 0.0f, s21 = 0.0f, s22 = 0.0f, s23 = 0.0f;
            float s30 = 0.0f, s31 = 0.0f, s32 = 0.0f, s33 = 0.0f;
            float s40 = 0.0f, s41 = 0.0f, s42 = 0.0f, s43 = 0.0f;
            float s50 = 0.0f, s51 = 0.0f, s52 = 0.0f, s53 = 0.0f;
            float s60 = 0.0f, s61 = 0.0f, s62 = 0.0f, s63 = 0.0f;
            float s70 = 0.0f, s71 = 0.0f, s72 = 0.0f, s73 = 0.0f;
            const float *bp = b + j0;
            for (uint32_t p = 0; p < k; p++) {
                // The strided walk over B reaches a new cache line each step, prefetch to cover the memory latency
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(bp + 32 * n, 0, 1);
#endif
                const float b0 = bp[0];
                const float b1 = bp[1];
                const float b2 = bp[2];
                const float b3 = bp[3];
                bp += n;
                const float v0 = a0[p];
                const float v1 = a1[p];
                const float v2 = a2[p];
                const float v3 = a3[p];
                const float v4 = a4[p];
                const float v5 = a5[p];
                const float v6 = a6[p];
                const float v7 = a7[p];
                s00 = fmaf(v0, b0, s00);
                s01 = fmaf(v0, b1, s01);
                s02 = fmaf(v0, b2, s02);
                s03 = fmaf(v0, b3, s03);
                s10 = fmaf(v1, b0, s10);
                s11 = fmaf(v1, b1, s11);
                s12 = fmaf(v1, b2, s12);
                s13 = fmaf(v1, b3, s13);
                s20 = fmaf(v2, b0, s20);
                s21 = fmaf(v2, b1, s21);
                s22 = fmaf(v2, b2, s22);
                s23 = fmaf(v2, b3, s23);
                s30 = fmaf(v3, b0, s30);
                s31 = fmaf(v3, b1, s31);
                s32 = fmaf(v3, b2, s32);
                s33 = fmaf(v3, b3, s33);
                s40 = fmaf(v4, b0, s40);
                s41 = fmaf(v4, b1, s41);
                s42 = fmaf(v4, b2, s42);
                s43 = fmaf(v4, b3, s43);
                s50 = fmaf(v5, b0, s50);
                s51 = fmaf(v5, b1, s51);
                s52 = fmaf(v5, b2, s52);
                s53 = fmaf(v5, b3, s53);
                s60 = fmaf(v6, b0, s60);
                s61 = fmaf(v6, b1, s61);
                s62 = fmaf(v6, b2, s62);
                s63 = fmaf(v6, b3, s63);
                s70 = fmaf(v7, b0, s70);
                s71 = fmaf(v7, b1, s71);
                s72 = fmaf(v7, b2, s72);
                s73 = fmaf(v7, b3, s73);
            }
            float *cp = c + (size_t)i0 * n + j0;
            cp[0 * n + 0] = s00;
            cp[0 * n + 1] = s01;
            cp[0 * n + 2] = s02;
            cp[0 * n + 3] = s03;
            cp[1 * n + 0] = s10;
            cp[1 * n + 1] = s11;
            cp[1 * n + 2] = s12;
            cp[1 * n + 3] = s13;
            cp[2 * n + 0] = s20;
            cp[2 * n + 1] = s21;
            cp[2 * n + 2] = s22;
            cp[2 * n + 3] = s23;
            cp[3 * n + 0] = s30;
            cp[3 * n + 1] = s31;
            cp[3 * n + 2] = s32;
            cp[3 * n + 3] = s33;
            cp[4 * n + 0] = s40;
            cp[4 * n + 1] = s41;
            cp[4 * n + 2] = s42;
            cp[4 * n + 3] = s43;
            cp[5 * n + 0] = s50;
            cp[5 * n + 1] = s51;
            cp[5 * n + 2] = s52;
            cp[5 * n + 3] = s53;
            cp[6 * n + 0] = s60;
            cp[6 * n + 1] = s61;
            cp[6 * n + 2] = s62;
            cp[6 * n + 3] = s63;
            cp[7 * n + 0] = s70;
            cp[7 * n + 1] = s71;
            cp[7 * n + 2] = s72;
            cp[7 * n + 3] = s73;
        }
        for (; j0 < n; j0++) {
            float *cp = c + (size_t)i0 * n + j0;
            ref_block_8x1(k, b, j0, n, a0, a1, a2, a3, a4, a5, a6, a7,
                              cp + 0 * n, cp + 1 * n, cp + 2 * n, cp + 3 * n,
                              cp + 4 * n, cp + 5 * n, cp + 6 * n, cp + 7 * n);
        }
    }
    for (; i0 < m; i0++) {
        const float *ap = a + (size_t)i0 * k;
        float *cp = c + (size_t)i0 * n;
        for (uint32_t j = 0; j < n; j++) {
            const float *bp = b + j;
            float sum = 0.0f;
            for (uint32_t p = 0; p < k; p++) {
                sum = fmaf(ap[p], *bp, sum);
                bp += n;
            }
            cp[j] = sum;
        }
    }
}

/* ---------- q8_0 ---------- */

// stride is the step of x along k, 1 for dot and matvec, n for matmul
static float ref_q8_0_row(const struct q8_0_block *w, uint32_t blocks,
                             const float *x, uint32_t stride) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const float *xb = x + (size_t)b * TOY_Q8_0_BLOCK * stride;
        float sum = 0.0f;
        for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
            sum = fmaf((float)w[b].qs[i], xb[(size_t)i * stride], sum);
        }
        acc = fmaf(f16_to_f32(w[b].d), sum, acc);
    }
    return acc;
}

static double ref_q8_0_row_acc64(const struct q8_0_block *w, uint32_t blocks,
                                     const float *x, uint32_t stride) {
    double acc = 0.0;
    for (uint32_t b = 0; b < blocks; b++) {
        const float *xb = x + (size_t)b * TOY_Q8_0_BLOCK * stride;
        double sum = 0.0;
        for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
            sum = fma((double)w[b].qs[i], (double)xb[(size_t)i * stride], sum);
        }
        acc = fma((double)f16_to_f32(w[b].d), sum, acc);
    }
    return acc;
}

void ref_q8_0_dot(uint32_t k, const void *w, const float *x, float *out) {
    *out = ref_q8_0_row(w, k / TOY_Q8_0_BLOCK, x, 1);
}

void ref_q8_0_matvec(uint32_t m, uint32_t k, const void *w, const float *x, float *y) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    for (uint32_t i = 0; i < m; i++) {
        y[i] = ref_q8_0_row(wp + (size_t)i * blocks, blocks, x, 1);
    }
}

void ref_q8_0_matmul(uint32_t m, uint32_t n, uint32_t k,
                         const void *w, const float *x, float *c) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            c[(size_t)i * n + j] = ref_q8_0_row(wp + (size_t)i * blocks, blocks, x + j, n);
        }
    }
}

void ref_q8_0_dot_acc64(uint32_t k, const void *w, const float *x, float *out) {
    *out = (float)ref_q8_0_row_acc64(w, k / TOY_Q8_0_BLOCK, x, 1);
}

void ref_q8_0_matvec_acc64(uint32_t m, uint32_t k, const void *w, const float *x, float *y) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    for (uint32_t i = 0; i < m; i++) {
        y[i] = (float)ref_q8_0_row_acc64(wp + (size_t)i * blocks, blocks, x, 1);
    }
}

void ref_q8_0_matmul_acc64(uint32_t m, uint32_t n, uint32_t k,
                               const void *w, const float *x, float *c) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            c[(size_t)i * n + j] =
                (float)ref_q8_0_row_acc64(wp + (size_t)i * blocks, blocks, x + j, n);
        }
    }
}

/*
 * q8_0 against q8_0: the ggml lane order, see mul_sum_i8_pairs_float and
 * hsum_float_8 in ggml/src/ggml-cpu/arch/x86/quants.c
 *
 * The int8 products of one block split into 8 lanes of 4 adjacent values, so
 * every lane is an exact int32 sum. The combined f16 scale then accumulates
 * into the lanes with one fma each, and the final reduction is the ggml tree
 *
 * The pair product follows vpsignb and vpmaddubsw: the weight byte becomes an
 * unsigned magnitude and the activation byte is negated when the weight is
 * negative, so two -128 bytes give -16384. A quantizer never emits -128, so
 * this only matters for hand-built bytes, and it keeps both paths identical
 */
static int32_t ref_q8_0_pair(int8_t weight, int8_t value) {
    const int32_t magnitude = weight < 0 ? -(int32_t)weight : (int32_t)weight;
    // The negation wraps in int8, the same as vpsignb
    const int32_t signed_value = weight < 0 ? (int8_t)(-(int32_t)value) : (int32_t)value;

    return magnitude * signed_value;
}

static float ref_q8_0_q8_0_row(const struct q8_0_block *w, const struct q8_0_block *x,
                               uint32_t blocks) {
    float lanes[8] = { 0.0f };

    for (uint32_t b = 0; b < blocks; b++) {
        const float d = f16_to_f32(w[b].d) * f16_to_f32(x[b].d);
        for (int lane = 0; lane < 8; lane++) {
            int32_t sum = 0;
            for (int i = 0; i < 4; i++) {
                const int index = 4 * lane + i;
                sum += ref_q8_0_pair(w[b].qs[index], x[b].qs[index]);
            }
            lanes[lane] = fmaf(d, (float)sum, lanes[lane]);
        }
    }

    // hsum_float_8: add the halves, then the pairs, then the last two lanes
    const float half0 = lanes[0] + lanes[4];
    const float half1 = lanes[1] + lanes[5];
    const float half2 = lanes[2] + lanes[6];
    const float half3 = lanes[3] + lanes[7];

    return (half0 + half2) + (half1 + half3);
}

void ref_q8_0_q8_0_matvec(uint32_t m, uint32_t k, const void *w, const void *x,
                          float *y) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    const struct q8_0_block *xp = x;

    for (uint32_t i = 0; i < m; i++) {
        y[i] = ref_q8_0_q8_0_row(wp + (size_t)i * blocks, xp, blocks);
    }
}

void ref_q8_0_q8_0_matmul(uint32_t m, uint32_t n, uint32_t k,
                          const void *w, const void *x, float *c) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const struct q8_0_block *wp = w;
    const struct q8_0_block *xp = x;

    for (uint32_t j = 0; j < n; j++) {
        const struct q8_0_block *column = xp + (size_t)j * blocks;
        for (uint32_t i = 0; i < m; i++) {
            c[(size_t)i * n + j] = ref_q8_0_q8_0_row(wp + (size_t)i * blocks, column, blocks);
        }
    }
}

#include "gdn.h"

#include <math.h>
#include <stdlib.h>

/*
 * The extension is compiled for the baseline target, so one function carries
 * the AVX2 and FMA target attribute and the caller runs it only after the
 * runtime check. MSVC enables the intrinsics per function instead
 */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
# include <immintrin.h>
# define TOY_GDN_X86 1
#endif

#if defined(TOY_GDN_X86)
# if defined(_MSC_VER) && !defined(__clang__)
#  define TOY_GDN_TARGET
# else
#  define TOY_GDN_TARGET __attribute__((target("avx2,fma")))
# endif
#endif

/* Lane fold of the dots, see the header comment */
static float gdn_fold_lanes(float s0, float s1, float s2, float s3,
                            float s4, float s5, float s6, float s7)
{
    const float even = (s0 + s4) + (s2 + s6);
    const float odd = (s1 + s5) + (s3 + s7);

    return even + odd;
}

/* One head over a column range, portable path */
static void gdn_columns_scalar(const gdn_problem *problem, uint32_t head,
                               uint32_t col_begin, uint32_t col_end)
{
    const uint32_t key_dim = problem->key_dim;
    const uint32_t value_dim = problem->value_dim;
    const uint32_t key8 = key_dim & ~7u;
    const float *q = problem->q + (size_t) head * key_dim;
    const float *k = problem->k + (size_t) head * key_dim;
    const float *v = problem->v + (size_t) head * value_dim;
    float *state = problem->state + (size_t) head * value_dim * key_dim;
    float *out = problem->out + (size_t) head * value_dim;
    const float decay = problem->decay[head];
    const float beta = problem->beta[head];
    const float scale = problem->scale;

    for (uint32_t col = col_begin; col < col_end; col++) {
        float *row = state + (size_t) col * key_dim;
        float d0 = 0.0f;
        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;
        float d4 = 0.0f;
        float d5 = 0.0f;
        float d6 = 0.0f;
        float d7 = 0.0f;
        uint32_t i = 0;
        for (; i < key8; i += 8) {
            const float s0 = row[i + 0] * decay;
            const float s1 = row[i + 1] * decay;
            const float s2 = row[i + 2] * decay;
            const float s3 = row[i + 3] * decay;
            const float s4 = row[i + 4] * decay;
            const float s5 = row[i + 5] * decay;
            const float s6 = row[i + 6] * decay;
            const float s7 = row[i + 7] * decay;
            d0 = fmaf(s0, k[i + 0], d0);
            d1 = fmaf(s1, k[i + 1], d1);
            d2 = fmaf(s2, k[i + 2], d2);
            d3 = fmaf(s3, k[i + 3], d3);
            d4 = fmaf(s4, k[i + 4], d4);
            d5 = fmaf(s5, k[i + 5], d5);
            d6 = fmaf(s6, k[i + 6], d6);
            d7 = fmaf(s7, k[i + 7], d7);
            row[i + 0] = s0;
            row[i + 1] = s1;
            row[i + 2] = s2;
            row[i + 3] = s3;
            row[i + 4] = s4;
            row[i + 5] = s5;
            row[i + 6] = s6;
            row[i + 7] = s7;
        }
        float sum = gdn_fold_lanes(d0, d1, d2, d3, d4, d5, d6, d7);
        for (; i < key_dim; i++) {
            const float s = row[i] * decay;
            row[i] = s;
            sum = fmaf(s, k[i], sum);
        }

        const float delta = beta * (v[col] - sum);

        d0 = 0.0f;
        d1 = 0.0f;
        d2 = 0.0f;
        d3 = 0.0f;
        d4 = 0.0f;
        d5 = 0.0f;
        d6 = 0.0f;
        d7 = 0.0f;
        i = 0;
        for (; i < key8; i += 8) {
            const float s0 = fmaf(k[i + 0], delta, row[i + 0]);
            const float s1 = fmaf(k[i + 1], delta, row[i + 1]);
            const float s2 = fmaf(k[i + 2], delta, row[i + 2]);
            const float s3 = fmaf(k[i + 3], delta, row[i + 3]);
            const float s4 = fmaf(k[i + 4], delta, row[i + 4]);
            const float s5 = fmaf(k[i + 5], delta, row[i + 5]);
            const float s6 = fmaf(k[i + 6], delta, row[i + 6]);
            const float s7 = fmaf(k[i + 7], delta, row[i + 7]);
            d0 = fmaf(s0, q[i + 0] * scale, d0);
            d1 = fmaf(s1, q[i + 1] * scale, d1);
            d2 = fmaf(s2, q[i + 2] * scale, d2);
            d3 = fmaf(s3, q[i + 3] * scale, d3);
            d4 = fmaf(s4, q[i + 4] * scale, d4);
            d5 = fmaf(s5, q[i + 5] * scale, d5);
            d6 = fmaf(s6, q[i + 6] * scale, d6);
            d7 = fmaf(s7, q[i + 7] * scale, d7);
            row[i + 0] = s0;
            row[i + 1] = s1;
            row[i + 2] = s2;
            row[i + 3] = s3;
            row[i + 4] = s4;
            row[i + 5] = s5;
            row[i + 6] = s6;
            row[i + 7] = s7;
        }
        float result = gdn_fold_lanes(d0, d1, d2, d3, d4, d5, d6, d7);
        for (; i < key_dim; i++) {
            const float s = fmaf(k[i], delta, row[i]);
            row[i] = s;
            result = fmaf(s, q[i] * scale, result);
        }
        out[col] = result;
    }
}

void gdn_recurrence_scalar(const gdn_problem *problem, const gdn_range *range)
{
    for (uint32_t head = range->head_begin; head < range->head_end; head++) {
        gdn_columns_scalar(problem, head, range->col_begin, range->col_end);
    }
}

#if defined(TOY_GDN_X86)
static TOY_GDN_TARGET float gdn_fold_ymm(__m256 value)
{
    float lane[8];

    _mm256_storeu_ps(lane, value);
    return gdn_fold_lanes(lane[0], lane[1], lane[2], lane[3],
                          lane[4], lane[5], lane[6], lane[7]);
}

/* One head over a column range; scaled_q holds q * scale for this head */
static TOY_GDN_TARGET void gdn_columns_avx2(const gdn_problem *problem, uint32_t head,
                                            uint32_t col_begin, uint32_t col_end,
                                            const float *scaled_q)
{
    const uint32_t key_dim = problem->key_dim;
    const uint32_t value_dim = problem->value_dim;
    const uint32_t key8 = key_dim & ~7u;
    const float *k = problem->k + (size_t) head * key_dim;
    const float *v = problem->v + (size_t) head * value_dim;
    float *state = problem->state + (size_t) head * value_dim * key_dim;
    float *out = problem->out + (size_t) head * value_dim;
    const float decay = problem->decay[head];
    const float beta = problem->beta[head];
    const __m256 decay8 = _mm256_set1_ps(decay);

    for (uint32_t col = col_begin; col < col_end; col++) {
        float *row = state + (size_t) col * key_dim;
        __m256 acc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i < key8; i += 8) {
            const __m256 s = _mm256_mul_ps(_mm256_loadu_ps(row + i), decay8);
            _mm256_storeu_ps(row + i, s);
            acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(k + i), acc);
        }
        float sum = gdn_fold_ymm(acc);
        for (; i < key_dim; i++) {
            const float s = row[i] * decay;
            row[i] = s;
            sum = fmaf(s, k[i], sum);
        }

        const float delta = beta * (v[col] - sum);
        const __m256 delta8 = _mm256_set1_ps(delta);

        __m256 acc_out = _mm256_setzero_ps();
        i = 0;
        for (; i < key8; i += 8) {
            const __m256 s = _mm256_fmadd_ps(_mm256_loadu_ps(k + i), delta8,
                                             _mm256_loadu_ps(row + i));
            _mm256_storeu_ps(row + i, s);
            acc_out = _mm256_fmadd_ps(s, _mm256_loadu_ps(scaled_q + i), acc_out);
        }
        float result = gdn_fold_ymm(acc_out);
        for (; i < key_dim; i++) {
            const float s = fmaf(k[i], delta, row[i]);
            row[i] = s;
            result = fmaf(s, scaled_q[i], result);
        }
        out[col] = result;
    }

    /* MSVC does not add the AVX to SSE transition itself */
    _mm256_zeroupper();
}
#endif

[[nodiscard]] bool gdn_recurrence_avx2(const gdn_problem *problem,
                                       const gdn_range *range)
{
#if defined(TOY_GDN_X86)
    const uint32_t key_dim = problem->key_dim;
    float *scaled_q = malloc((size_t) key_dim * sizeof(float));
    if (scaled_q == NULL) {
        return false;
    }

    for (uint32_t head = range->head_begin; head < range->head_end; head++) {
        const float *q = problem->q + (size_t) head * key_dim;
        for (uint32_t i = 0; i < key_dim; i++) {
            scaled_q[i] = q[i] * problem->scale;
        }
        gdn_columns_avx2(problem, head, range->col_begin, range->col_end, scaled_q);
    }

    free(scaled_q);
    return true;
#else
    (void) problem;
    (void) range;
    return false;
#endif
}

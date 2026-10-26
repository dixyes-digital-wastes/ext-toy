#ifndef GDN_H
#define GDN_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Gated delta net recurrence for one token
 *
 * A head owns a value_dim x key_dim state. A state row is one value column and
 * holds the key_dim keys, so a row is contiguous and the state tensor is
 * [heads, value_dim, key_dim], the layout of ggml_gated_delta_net. The state
 * decays, takes the delta update and then produces value_dim outputs
 *
 * The arithmetic of one value column, in order:
 *
 *   s[i] = state[col][i] * decay[head]
 *   delta = beta[head] * (v[head][col] - dot(s, k[head]))
 *   s[i] = k[head][i] * delta + s[i]
 *   out[head][col] = dot(s, q[head] * scale)
 *
 * A column reads and writes its own state row only, so a caller can split the
 * work by head or by value column without a reduction
 *
 * The dot follows the AVX2 lane order: lane l accumulates i = l, l + 8, ...,
 * the lanes fold as ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)), and the
 * key tail adds in ascending order. The scalar and the AVX2 path run that same
 * order with single-rounded FMAs, so they agree bit for bit and no result
 * depends on the head or column split
 */
typedef struct gdn_problem {
    uint32_t key_dim;
    uint32_t value_dim;
    float scale;
    const float *q;
    const float *k;
    const float *v;
    const float *decay;
    const float *beta;
    float *state;
    float *out;
} gdn_problem;

/* A half-open range of heads and value columns */
typedef struct gdn_range {
    uint32_t head_begin;
    uint32_t head_end;
    uint32_t col_begin;
    uint32_t col_end;
} gdn_range;

/* Runs the portable path and updates the state in place */
void gdn_recurrence_scalar(const gdn_problem *problem, const gdn_range *range);

/*
 * Runs the AVX2 path. Returns false when this build has no AVX2 kernel or the
 * scaled q scratch buffer cannot be allocated, and the caller then runs the
 * scalar path; the result is the same either way
 */
[[nodiscard]] bool gdn_recurrence_avx2(const gdn_problem *problem,
                                       const gdn_range *range);

#endif /* GDN_H */

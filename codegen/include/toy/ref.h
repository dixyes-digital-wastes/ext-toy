#ifndef TOY_REF_H
#define TOY_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scalar f32 reference, separate from the generator
 *
 * Each accumulation uses fmaf to match the single-rounded FMA of the
 * generated kernels
 */

void ref_gemm(uint32_t m, uint32_t n, uint32_t k,
                  const float *a, const float *b, float *c);
void ref_matvec(uint32_t m, uint32_t k, const float *w, const float *x, float *y);
/*
 * AVX2 matvec order: a lane accumulates every eighth element of a row, the
 * lane tree matches the horizontal sum of the generated kernel, and the k tail
 * adds in increasing order
 */
void ref_matvec_avx2_order(uint32_t m, uint32_t k, const float *w, const float *x,
                           float *y);
void ref_dot(uint32_t k, const float *x, const float *y, float *out);

void ref_add_f32(const float *a, const float *b, float *c, size_t count);
void ref_mul_f32(const float *a, const float *b, float *c, size_t count);
void ref_scale_f32(const float *a, float factor, float *c, size_t count);
void ref_silu_f32(const float *a, float *c, size_t count);
void ref_sigmoid_f32(const float *a, float *c, size_t count);
void ref_softplus_f32(const float *a, float *c, size_t count);
void ref_rmsnorm_f32(const float *x, const float *w, float *out, uint32_t n, float eps);

/*
 * rms norm of every row against one shared weight row
 *
 * out[r][i] = (x[r][i] * scale) * w[i], scale = 1 / sqrt(sum(x[r][i]^2) / width + eps).
 * The sum of squares is one fma chain in element order, so the generated
 * kernel can match it exactly
 */
void ref_rmsnorm_rows_f32(uint32_t rows, uint32_t width, const float *x, const float *w,
                          float *out, float eps);

/*
 * l2 norm of every row, with eps inside the square root
 *
 * out[r][i] = x[r][i] / sqrt(sum(x[r][i]^2) + eps), and the sum of squares is
 * one fma chain in element order, so the generated kernel can match it exactly
 */
void ref_l2norm_rows_f32(uint32_t rows, uint32_t width, const float *x, float *out,
                         float eps);

/*
 * Causal depthwise conv over channels, with SiLU on the result
 *
 * w holds one kernel of taps per channel, x holds the taps of every channel
 * with x[j * stride + c] for tap j, and out[c] takes the activation of the
 * channel sum. Each sum is one fma chain in increasing tap order
 */
void ref_conv1d_silu_f32(uint32_t count, uint32_t stride, uint32_t taps,
                         const float *w, const float *x, float *out);

/*
 * q8_0 reference: each block first sums q_i * x_i in f32, then the f16 scale
 * folds into the output
 * The _acc64 versions use double for the block sum and the output sum, for
 * accuracy comparison
 */
void ref_q8_0_dot(uint32_t k, const void *w, const float *x, float *out);
void ref_q8_0_matvec(uint32_t m, uint32_t k, const void *w, const float *x, float *y);
void ref_q8_0_matmul(uint32_t m, uint32_t n, uint32_t k,
                         const void *w, const float *x, float *c);

void ref_q8_0_dot_acc64(uint32_t k, const void *w, const float *x, float *out);
void ref_q8_0_matvec_acc64(uint32_t m, uint32_t k, const void *w, const float *x, float *y);
void ref_q8_0_matmul_acc64(uint32_t m, uint32_t n, uint32_t k,
                               const void *w, const float *x, float *c);

/*
 * q8_0 weights against q8_0 activations, in the ggml lane order
 *
 * Each block splits into 8 int32 lanes of 4 adjacent products, the combined
 * f16 scale feeds one fma into every lane, and a fixed tree reduces the lanes
 * at the end. The generated AVX2 kernel follows the same order, so the two
 * paths agree bit for bit
 *
 * The activation holds one series of blocks per column, at stride k / 32 blocks
 */
void ref_q8_0_q8_0_matvec(uint32_t m, uint32_t k, const void *w, const void *x,
                          float *y);
void ref_q8_0_q8_0_matmul(uint32_t m, uint32_t n, uint32_t k,
                          const void *w, const void *x, float *c);

#ifdef __cplusplus
}
#endif

#endif

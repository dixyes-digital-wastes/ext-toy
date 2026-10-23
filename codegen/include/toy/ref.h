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
 * The build must disable contraction and auto-vectorization:
 *   -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize
 * and enable hardware FMA (-mfma), so fmaf maps to one instruction
 */

void toy_ref_gemm(uint32_t m, uint32_t n, uint32_t k,
                  const float *a, const float *b, float *c);
void toy_ref_matvec(uint32_t m, uint32_t k, const float *w, const float *x, float *y);
void toy_ref_dot(uint32_t k, const float *x, const float *y, float *out);

/*
 * q8_0 reference: each block first sums q_i * x_i in f32, then the f16 scale
 * folds into the output
 * The _acc64 versions use double for the block sum and the output sum, for
 * accuracy comparison
 */
void toy_ref_q8_0_dot(uint32_t k, const void *w, const float *x, float *out);
void toy_ref_q8_0_matvec(uint32_t m, uint32_t k, const void *w, const float *x, float *y);
void toy_ref_q8_0_matmul(uint32_t m, uint32_t n, uint32_t k,
                         const void *w, const float *x, float *c);

void toy_ref_q8_0_dot_acc64(uint32_t k, const void *w, const float *x, float *out);
void toy_ref_q8_0_matvec_acc64(uint32_t m, uint32_t k, const void *w, const float *x, float *y);
void toy_ref_q8_0_matmul_acc64(uint32_t m, uint32_t n, uint32_t k,
                               const void *w, const float *x, float *c);

#ifdef __cplusplus
}
#endif

#endif

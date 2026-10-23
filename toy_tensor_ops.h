#ifndef TOY_TENSOR_OPS_H
#define TOY_TENSOR_OPS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Scalar f32 reference kernels; the caller validates the shapes and the dtype
 * Each kernel uses fused multiply-add for its accumulation to mirror the
 * generated FMA kernels
 */

void toy_matmul_f32(const float *a, const float *b, float *c,
                    uint32_t m, uint32_t n, uint32_t k);
void toy_matvec_f32(const float *w, const float *x, float *y,
                    uint32_t m, uint32_t k);
void toy_add_f32(const float *a, const float *b, float *c, size_t count);
void toy_scale_f32(const float *a, float factor, float *c, size_t count);
void toy_rmsnorm_f32(const float *x, const float *w, float *out,
                     uint32_t n, float eps);

#endif /* TOY_TENSOR_OPS_H */

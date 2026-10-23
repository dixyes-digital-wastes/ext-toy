#ifndef TOY_REF_H
#define TOY_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void toy_ref_gemm(uint32_t m, uint32_t n, uint32_t k,
                  const float *a, const float *b, float *c);
void toy_ref_matvec(uint32_t m, uint32_t k, const float *w, const float *x, float *y);
void toy_ref_dot(uint32_t k, const float *x, const float *y, float *out);

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

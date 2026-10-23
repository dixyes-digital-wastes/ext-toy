#include "toy_tensor_ops.h"

#include <math.h>

void toy_matmul_f32(const float *a, const float *b, float *c,
                    uint32_t m, uint32_t n, uint32_t k)
{
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (uint32_t p = 0; p < k; p++) {
                sum = fmaf(a[(size_t) i * k + p], b[(size_t) p * n + j], sum);
            }
            c[(size_t) i * n + j] = sum;
        }
    }
}

void toy_matvec_f32(const float *w, const float *x, float *y,
                    uint32_t m, uint32_t k)
{
    for (uint32_t i = 0; i < m; i++) {
        float sum = 0.0f;
        for (uint32_t p = 0; p < k; p++) {
            sum = fmaf(w[(size_t) i * k + p], x[p], sum);
        }
        y[i] = sum;
    }
}

void toy_add_f32(const float *a, const float *b, float *c, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

void toy_scale_f32(const float *a, float factor, float *c, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] * factor;
    }
}

void toy_rmsnorm_f32(const float *x, const float *w, float *out,
                     uint32_t n, float eps)
{
    float squares = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        squares = fmaf(x[i], x[i], squares);
    }

    float scale = 1.0f / sqrtf(squares / (float) n + eps);
    for (uint32_t i = 0; i < n; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

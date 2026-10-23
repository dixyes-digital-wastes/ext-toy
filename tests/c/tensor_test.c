/*
 * Standalone C tests for the tensor core and the reference kernels
 * This file does not link against PHP; the alias tests cover the C-only API
 *
 * Build and run:
 *   cc -std=gnu23 -Wall -Wextra -I. -Icodegen/include \
 *      tests/c/tensor_test.c toy_tensor.c codegen/src/ref.c \
 *      codegen/src/q8_0.c -lm -o tensor_test
 *   ./tensor_test
 */

#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "toy/ref.h"
#include "toy_tensor.h"

static_assert(TOY_TENSOR_ALIGNMENT == 64, "test buffer alignment");

static int failures;
static const float test_tolerance = 1e-6f;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            failures++; \
        } \
    } while (0)

static void check_near(float actual, float expected, float tolerance, const char *label)
{
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL %s: got %.9g, expected %.9g\n", label, actual, expected);
        failures++;
    }
}

static void test_dtype(void)
{
    toy_dtype dtype;

    CHECK(strcmp(toy_dtype_name(TOY_DTYPE_F32), "f32") == 0);
    CHECK(strcmp(toy_dtype_name(TOY_DTYPE_Q8_0), "q8_0") == 0);
    CHECK(strcmp(toy_dtype_name(TOY_DTYPE_INVALID), "invalid") == 0);

    CHECK(toy_dtype_from_name("f32", 3, &dtype) && dtype == TOY_DTYPE_F32);
    CHECK(toy_dtype_from_name("q8_0", 4, &dtype) && dtype == TOY_DTYPE_Q8_0);
    CHECK(toy_dtype_from_name("bf16", 4, &dtype) && dtype == TOY_DTYPE_BF16);
    CHECK(!toy_dtype_from_name("f64", 3, &dtype));
    CHECK(!toy_dtype_from_name("F32", 3, &dtype));
}

static void test_shape_bytes(void)
{
    size_t bytes;
    uint32_t dims2[4] = { 2, 3, 0, 0 };
    uint32_t dims_q8[4] = { 2, 32, 0, 0 };
    uint32_t dims_bad_q8[4] = { 2, 31, 0, 0 };
    uint32_t dims_zero[4] = { 2, 0, 0, 0 };
    uint32_t dims_over[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };

    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F32, 2, dims2, &bytes) == TOY_OK && bytes == 24);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F16, 2, dims2, &bytes) == TOY_OK && bytes == 12);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_BF16, 2, dims2, &bytes) == TOY_OK && bytes == 12);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_I8, 2, dims2, &bytes) == TOY_OK && bytes == 6);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_Q8_0, 2, dims_q8, &bytes) == TOY_OK && bytes == 68);

    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_Q8_0, 2, dims_bad_q8, &bytes) == TOY_ERROR_SHAPE);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F32, 2, dims_zero, &bytes) == TOY_ERROR_SHAPE);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F32, 0, dims2, &bytes) == TOY_ERROR_SHAPE);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F32, 5, dims2, &bytes) == TOY_ERROR_SHAPE);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_INVALID, 2, dims2, &bytes) == TOY_ERROR_DTYPE);
    CHECK(toy_tensor_shape_bytes(TOY_DTYPE_F32, 4, dims_over, &bytes) == TOY_ERROR_OVERFLOW);
}

static void test_alloc(void)
{
    uint32_t dims[4] = { 4, 8, 0, 0 };
    toy_status status = TOY_ERROR_MEMORY;
    toy_tensor *tensor = toy_tensor_alloc(TOY_DTYPE_F32, 2, dims, &status);

    CHECK(tensor != NULL);
    CHECK(status == TOY_OK);
    if (!tensor) {
        return;
    }

    CHECK(tensor->refcount == 1);
    CHECK((tensor->flags & TOY_TENSOR_FLAG_OWNS_DATA) != 0);
    CHECK((tensor->flags & TOY_TENSOR_FLAG_ALIAS) == 0);
    CHECK(tensor->buffer != NULL);
    CHECK(tensor->bytes == 128);
    CHECK(tensor->rank == 2);
    CHECK(tensor->dims[0] == 4 && tensor->dims[1] == 8);
    CHECK(toy_tensor_numel(tensor) == 32);
    CHECK(((uintptr_t) toy_tensor_data(tensor) % TOY_TENSOR_ALIGNMENT) == 0);

    const float *data = toy_tensor_data(tensor);
    for (size_t i = 0; i < 32; i++) {
        CHECK(data[i] == 0.0f);
    }

    toy_tensor_retain(tensor);
    toy_tensor_release(tensor);
    CHECK(toy_tensor_numel(tensor) == 32);

    tensor = toy_tensor_retain(tensor);
    toy_tensor_release(tensor);
    toy_tensor_release(tensor);

    uint32_t small[4] = { 1, 0, 0, 0 };
    toy_tensor *i8_tensor = toy_tensor_alloc(TOY_DTYPE_I8, 1, small, &status);
    CHECK(i8_tensor != NULL && i8_tensor->bytes == 1);
    if (i8_tensor) {
        CHECK(((uintptr_t) toy_tensor_data(i8_tensor) % TOY_TENSOR_ALIGNMENT) == 0);
        toy_tensor_release(i8_tensor);
    }
}

static void test_reshape_clone(void)
{
    uint32_t dims[4] = { 4, 8, 0, 0 };
    uint32_t wide[4] = { 2, 16, 0, 0 };
    uint32_t bad[4] = { 2, 7, 0, 0 };

    toy_tensor *tensor = toy_tensor_alloc(TOY_DTYPE_F32, 2, dims, NULL);
    CHECK(tensor != NULL);
    if (!tensor) {
        return;
    }

    float *data = toy_tensor_data(tensor);
    for (size_t i = 0; i < 32; i++) {
        data[i] = (float) i;
    }

    CHECK(toy_tensor_reshape(tensor, 2, wide) == TOY_OK);
    CHECK(tensor->dims[0] == 2 && tensor->dims[1] == 16);
    CHECK(toy_tensor_reshape(tensor, 2, bad) == TOY_ERROR_BYTES);

    toy_status status;
    toy_tensor *copy = toy_tensor_clone(tensor, &status);
    CHECK(copy != NULL);
    if (copy) {
        CHECK(copy->data != tensor->data);
        CHECK(copy->bytes == tensor->bytes);
        CHECK(memcmp(copy->data, tensor->data, tensor->bytes) == 0);
        ((float *) copy->data)[0] = -1.0f;
        CHECK(((float *) tensor->data)[0] == 0.0f);
        toy_tensor_release(copy);
    }
    toy_tensor_release(tensor);
}

static void test_alias(void)
{
    alignas(TOY_TENSOR_ALIGNMENT) unsigned char external[128];
    uint32_t dims[4] = { 4, 8, 0, 0 };
    uint32_t wide[4] = { 2, 16, 0, 0 };
    uint32_t bad_dims[4] = { 4, 9, 0, 0 };
    uint32_t q8_dims[4] = { 1, 32, 0, 0 };

    for (size_t i = 0; i < sizeof(external); i++) {
        external[i] = (unsigned char) i;
    }

    toy_tensor *alias = toy_tensor_alias(external, sizeof(external), TOY_DTYPE_F32, 2, dims);
    CHECK(alias != NULL);
    if (!alias) {
        return;
    }

    CHECK(alias->data == external);
    CHECK(alias->buffer == NULL);
    CHECK(alias->flags == TOY_TENSOR_FLAG_ALIAS);
    CHECK((alias->flags & TOY_TENSOR_FLAG_OWNS_DATA) == 0);
    CHECK(alias->bytes == sizeof(external));
    CHECK(toy_tensor_numel(alias) == 32);
    CHECK(((unsigned char *) toy_tensor_data(alias))[7] == 7);

    toy_tensor_retain(alias);
    toy_tensor_release(alias);
    CHECK(alias->refcount == 1);

    CHECK(toy_tensor_reshape(alias, 2, wide) == TOY_OK);
    CHECK(alias->dims[0] == 2 && alias->dims[1] == 16);

    toy_tensor_release(alias);

    for (size_t i = 0; i < sizeof(external); i++) {
        CHECK(external[i] == (unsigned char) i);
    }

    CHECK(toy_tensor_alias(NULL, 128, TOY_DTYPE_F32, 2, dims) == NULL);
    CHECK(toy_tensor_alias(external, 127, TOY_DTYPE_F32, 2, dims) == NULL);
    CHECK(toy_tensor_alias(external, 128, TOY_DTYPE_F32, 2, bad_dims) == NULL);
    CHECK(toy_tensor_alias(external, 128, TOY_DTYPE_INVALID, 2, dims) == NULL);
    CHECK(toy_tensor_alias(external, 128, TOY_DTYPE_F32, 0, dims) == NULL);

    toy_tensor *q8_alias = toy_tensor_alias(external, 34, TOY_DTYPE_Q8_0, 2, q8_dims);
    CHECK(q8_alias != NULL);
    if (q8_alias) {
        CHECK(q8_alias->bytes == 34);
        toy_tensor_release(q8_alias);
    }
}

static void test_kernels(void)
{
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float b[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
    float c[4] = { 0 };

    toy_ref_gemm(2, 2, 2, a, b, c);
    check_near(c[0], 19.0f, test_tolerance, "matmul[0]");
    check_near(c[1], 22.0f, test_tolerance, "matmul[1]");
    check_near(c[2], 43.0f, test_tolerance, "matmul[2]");
    check_near(c[3], 50.0f, test_tolerance, "matmul[3]");

    float x[2] = { 5.0f, 6.0f };
    float y[2] = { 0 };
    toy_ref_matvec(2, 2, a, x, y);
    check_near(y[0], 17.0f, test_tolerance, "matvec[0]");
    check_near(y[1], 39.0f, test_tolerance, "matvec[1]");

    float sum[4] = { 0 };
    ref_add_f32(a, b, sum, 4);
    check_near(sum[0], 6.0f, test_tolerance, "add[0]");
    check_near(sum[3], 12.0f, test_tolerance, "add[3]");

    ref_scale_f32(a, 2.5f, sum, 4);
    check_near(sum[0], 2.5f, test_tolerance, "scale[0]");
    check_near(sum[3], 10.0f, test_tolerance, "scale[3]");

    float norm_x[2] = { 3.0f, 4.0f };
    float norm_w[2] = { 1.0f, 2.0f };
    float norm_out[2] = { 0 };
    ref_rmsnorm_f32(norm_x, norm_w, norm_out, 2, 0.0f);
    float scale = 1.0f / sqrtf(12.5f);
    check_near(norm_out[0], 3.0f * scale, test_tolerance, "rmsnorm[0]");
    check_near(norm_out[1], 8.0f * scale, test_tolerance, "rmsnorm[1]");
}

int main(void)
{
    test_dtype();
    test_shape_bytes();
    test_alloc();
    test_reshape_clone();
    test_alias();
    test_kernels();

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}

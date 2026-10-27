/*
 * Standalone C tests for the transcendental kernels, the l2 norm and the
 * fused conv1d silu kernel
 *
 * The reference here is codegen/src/ref.c:
 *
 * - l2norm must match the reference bit for bit, the sum of squares is the
 *   same scalar fma chain on both sides
 * - sigmoid, softplus and conv1d_silu use a polynomial exp or log in the
 *   generated code, so they are compared with a tolerance
 * - every kernel must also run from pointers that are not aligned, which is
 *   why some buffers below start at an odd float
 *
 * Build and run on POSIX:
 *   cc -std=gnu17 -Wall -Wextra -Werror -O2 -I. -Icodegen/include \
 *      tests/c/ops_test.c codegen/src/emit.c codegen/src/asm.c \
 *      codegen/src/isa.c codegen/src/ref.c codegen/src/q8_0.c \
 *      -lm -o ops_test
 *   ./ops_test
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "toy/codegen.h"
#include "toy/ref.h"

static int failures;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            failures++; \
        } \
    } while (0)

static uint64_t rng_state = 0x243F6A8885A308D3ull;

static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static float rng_float(float scale)
{
    const int32_t raw = (int32_t)(rng_next() & 0xFFFF) - 32768;
    return (float) raw * scale / 32768.0f;
}

typedef struct code_block {
    void *map;
    size_t bytes;
    kernel_fn entry;
} code_block;

static code_block build(const struct spec *spec)
{
    code_block block = { 0 };
    block.bytes = TOY_EMIT_CAP;
    block.map = mmap(NULL, block.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (block.map == MAP_FAILED) {
        block.map = NULL;
        return block;
    }
    const size_t length = emit(spec, block.map, block.bytes);
    if (length == 0 || mprotect(block.map, block.bytes, PROT_READ | PROT_EXEC) != 0) {
        munmap(block.map, block.bytes);
        block.map = NULL;
        return block;
    }
    memcpy(&block.entry, &block.map, sizeof(block.entry));
    return block;
}

static void release(code_block *block)
{
    if (block->map != NULL) {
        munmap(block->map, block->bytes);
        block->map = NULL;
    }
}

/* Compares two f32 arrays within an absolute and a relative tolerance */
static void close_arrays(const char *what, const float *expected, const float *actual,
                         uint32_t count, float abs_tol, float rel_tol)
{
    float worst = 0.0f;
    uint32_t worst_index = 0;
    bool ok = true;
    for (uint32_t i = 0; i < count; i++) {
        const float diff = fabsf(expected[i] - actual[i]);
        const float scaled = fabsf(expected[i]) > 1.0f ? fabsf(expected[i]) : 1.0f;
        if (diff > abs_tol && diff > rel_tol * scaled) {
            if (ok || diff > worst) {
                worst = diff;
                worst_index = i;
            }
            ok = false;
        }
    }
    if (!ok) {
        fprintf(stderr, "FAIL %s: index %u expected %.9g got %.9g\n", what, worst_index,
                (double) expected[worst_index], (double) actual[worst_index]);
        failures++;
    }
}

static void same_bytes(const char *what, const float *expected, const float *actual,
                       uint32_t count)
{
    if (memcmp(expected, actual, (size_t) count * 4) != 0) {
        for (uint32_t i = 0; i < count; i++) {
            if (memcmp(&expected[i], &actual[i], 4) != 0) {
                fprintf(stderr, "FAIL %s: index %u expected %.9g got %.9g\n", what, i,
                        (double) expected[i], (double) actual[i]);
                break;
            }
        }
        failures++;
    }
}

/* Allocates a padded buffer; the returned pointer is one float into it */
static float *alloc_offset(uint32_t count)
{
    float *base = calloc((size_t) count + 2, sizeof(float));
    if (base == NULL) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return base + 1;
}

static void free_offset(float *offset)
{
    free(offset - 1);
}

static void test_unary(uint32_t op, const char *name, uint32_t count, bool aligned)
{
    float *input = aligned ? calloc(count, 4) : alloc_offset(count);
    float *expected = aligned ? calloc(count, 4) : alloc_offset(count);
    float *actual = aligned ? calloc(count, 4) : alloc_offset(count);
    input = input;
    for (uint32_t i = 0; i < count; i++) {
        // Mostly small values, with a few far out ones for the clamps
        const uint32_t pick = rng_next() % 16;
        if (pick == 0) {
            input[i] = rng_next() % 2 == 0 ? -200.0f : 200.0f;
        } else if (pick == 1) {
            input[i] = rng_float(1.0f);
        } else {
            input[i] = rng_float(24.0f);
        }
    }

    if (op == TOY_OP_SIGMOID) {
        ref_sigmoid_f32(input, expected, count);
    } else {
        ref_softplus_f32(input, expected, count);
    }

    struct spec spec = {
        .op = op,
        .isa = TOY_ISA_AVX2,
        .m = count,
        .n = 1,
        .k = 1,
    };
    code_block block = build(&spec);
    CHECK(block.entry != NULL);
    if (block.entry != NULL) {
        block.entry(input, NULL, actual);
        close_arrays(name, expected, actual, count, 2e-6f, 2e-6f);
    }
    release(&block);

    if (aligned) {
        free(input);
        free(expected);
        free(actual);
    } else {
        free_offset(input);
        free_offset(expected);
        free_offset(actual);
    }
}

/*
 * l2norm and rmsnorm_rows share one generated body, so both are checked the
 * same way: against the reference and bit for bit. The weight row of
 * rmsnorm_rows is read from b
 */
static void test_row_norm(uint32_t op, uint32_t rows, uint32_t width, float eps,
                          bool aligned)
{
    const uint32_t total = rows * width;
    float *weight = aligned ? calloc(width, 4) : alloc_offset(width);
    float *input = aligned ? calloc(total, 4) : alloc_offset(total);
    float *expected = aligned ? calloc(total, 4) : alloc_offset(total);
    float *actual = aligned ? calloc(total, 4) : alloc_offset(total);
    for (uint32_t i = 0; i < width; i++) {
        weight[i] = rng_float(2.0f);
    }
    for (uint32_t i = 0; i < total; i++) {
        input[i] = rng_float(8.0f);
    }

    if (op == TOY_OP_L2NORM) {
        ref_l2norm_rows_f32(rows, width, input, expected, eps);
    } else {
        ref_rmsnorm_rows_f32(rows, width, input, weight, expected, eps);
    }

    struct spec spec = {
        .op = op,
        .isa = TOY_ISA_AVX2,
        .m = rows,
        .n = 1,
        .k = width,
    };
    memcpy(&spec.param_bits, &eps, 4);
    code_block block = build(&spec);
    CHECK(block.entry != NULL);
    if (block.entry != NULL) {
        block.entry(input, weight, actual);
        same_bytes(op == TOY_OP_L2NORM ? "l2norm" : "rmsnorm_rows", expected, actual, total);
    }
    release(&block);

    if (aligned) {
        free(weight);
        free(input);
        free(expected);
        free(actual);
    } else {
        free_offset(weight);
        free_offset(input);
        free_offset(expected);
        free_offset(actual);
    }
}

/*
 * A row slice must produce the same bits as the same rows inside a full run,
 * although the row groups of the two calls are aligned differently
 */
static void test_row_norm_slice(uint32_t op, uint32_t rows, uint32_t width, uint32_t start,
                                uint32_t count)
{
    const uint32_t total = rows * width;
    float *weight = calloc(width, 4);
    float *input = calloc(total, 4);
    float *full = calloc(total, 4);
    float *part = calloc(total, 4);
    for (uint32_t i = 0; i < width; i++) {
        weight[i] = rng_float(2.0f);
    }
    for (uint32_t i = 0; i < total; i++) {
        input[i] = rng_float(8.0f);
    }

    const float eps = 1e-6f;
    struct spec whole = {
        .op = op,
        .isa = TOY_ISA_AVX2,
        .m = rows,
        .n = 1,
        .k = width,
    };
    memcpy(&whole.param_bits, &eps, 4);
    code_block full_block = build(&whole);
    CHECK(full_block.entry != NULL);
    if (full_block.entry != NULL) {
        full_block.entry(input, weight, full);
    }
    release(&full_block);

    struct spec slice = whole;
    slice.m = count;
    code_block slice_block = build(&slice);
    CHECK(slice_block.entry != NULL);
    if (slice_block.entry != NULL) {
        const size_t offset = (size_t) start * width;
        slice_block.entry(input + offset, weight, part + offset);
        same_bytes(op == TOY_OP_L2NORM ? "l2norm slice" : "rmsnorm_rows slice",
                   full + offset, part + offset, count * width);
    }
    release(&slice_block);
    free(weight);
    free(input);
    free(full);
    free(part);
}

/*
 * channels [start, start + count) come from a full reference run over the
 * same data, so the slice path and the whole path see the same values
 */
static void test_conv1d(uint32_t rows, uint32_t taps, uint32_t start, uint32_t count,
                        bool aligned)
{
    const uint32_t inputs = rows * taps;
    float *weights = aligned ? calloc(inputs, 4) : alloc_offset(inputs);
    float *x = aligned ? calloc(inputs, 4) : alloc_offset(inputs);
    float *expected = aligned ? calloc(rows, 4) : alloc_offset(rows);
    float *actual = aligned ? calloc(rows, 4) : alloc_offset(rows);
    for (uint32_t i = 0; i < inputs; i++) {
        weights[i] = rng_float(2.0f);
        x[i] = rng_float(3.0f);
    }

    ref_conv1d_silu_f32(rows, rows, taps, weights, x, expected);

    struct spec spec = {
        .op = TOY_OP_CONV1D_SILU,
        .isa = TOY_ISA_AVX2,
        .m = count,
        .n = 1,
        .k = rows,
        .param_bits = taps,
    };
    code_block block = build(&spec);
    CHECK(block.entry != NULL);
    if (block.entry != NULL) {
        block.entry(weights + (size_t) start * taps, x + start, actual + start);
        close_arrays("conv1d_silu", expected + start, actual + start, count, 2e-5f, 2e-6f);
    }
    release(&block);

    if (aligned) {
        free(weights);
        free(x);
        free(expected);
        free(actual);
    } else {
        free_offset(weights);
        free_offset(x);
        free_offset(expected);
        free_offset(actual);
    }
}

int main(void)
{
    if (!isa_available(TOY_ISA_AVX2)) {
        printf("skipped: the host has no AVX2\n");
        return 0;
    }

    // The generator must refuse a scalar ISA for these ops
    struct spec scalar = {
        .op = TOY_OP_SIGMOID,
        .isa = TOY_ISA_SCALAR,
        .m = 8,
        .n = 1,
        .k = 1,
    };
    uint8_t scratch[TOY_EMIT_CAP];
    CHECK(emit(&scalar, scratch, sizeof(scratch)) == 0);
    CHECK(!spec_supported(&scalar));

    static const uint32_t unary_counts[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 63, 64, 1023 };
    for (size_t i = 0; i < sizeof(unary_counts) / sizeof(unary_counts[0]); i++) {
        const uint32_t count = unary_counts[i];
        test_unary(TOY_OP_SIGMOID, "sigmoid", count, true);
        test_unary(TOY_OP_SOFTPLUS, "softplus", count, true);
        test_unary(TOY_OP_SIGMOID, "sigmoid unaligned", count, false);
        test_unary(TOY_OP_SOFTPLUS, "softplus unaligned", count, false);
    }

    static const uint32_t norm_shapes[][2] = {
        { 1, 1 }, { 1, 7 }, { 2, 8 }, { 3, 9 }, { 3, 10 }, { 4, 8 }, { 4, 1 }, { 7, 16 },
        { 8, 1 }, { 5, 1 }, { 16, 128 }, { 16, 127 }, { 1, 1023 },
    };
    static const uint32_t norm_ops[] = { TOY_OP_L2NORM, TOY_OP_RMSNORM_ROWS };
    for (size_t o = 0; o < sizeof(norm_ops) / sizeof(norm_ops[0]); o++) {
        for (size_t i = 0; i < sizeof(norm_shapes) / sizeof(norm_shapes[0]); i++) {
            test_row_norm(norm_ops[o], norm_shapes[i][0], norm_shapes[i][1], 1e-6f, true);
            test_row_norm(norm_ops[o], norm_shapes[i][0], norm_shapes[i][1], 1e-6f, false);
        }
        test_row_norm(norm_ops[o], 16, 128, 0.0f, true);
        test_row_norm(norm_ops[o], 16, 128, 1e-3f, true);
        test_row_norm_slice(norm_ops[o], 16, 128, 0, 16);
        test_row_norm_slice(norm_ops[o], 16, 128, 4, 4);
        test_row_norm_slice(norm_ops[o], 16, 128, 5, 5);
        test_row_norm_slice(norm_ops[o], 9, 7, 1, 7);
    }

    test_conv1d(16, 4, 0, 16, true);
    test_conv1d(16, 4, 3, 8, true);
    test_conv1d(16, 4, 0, 16, false);
    test_conv1d(8, 1, 0, 8, true);
    test_conv1d(8, 1, 5, 3, false);
    test_conv1d(17, 3, 0, 17, true);
    test_conv1d(17, 3, 9, 8, true);
    test_conv1d(40, 2, 37, 3, true);
    test_conv1d(6144, 4, 2048, 2048, true);

    if (failures != 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("ops_test: all checks passed\n");
    return 0;
}

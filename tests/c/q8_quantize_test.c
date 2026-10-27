/*
 * Standalone C test for the f32 to q8_0 activation quantization kernel
 * This file does not link against PHP
 *
 * The reference here is written from the ggml definition only, so it shares
 * no code with codegen/src/q8_0.c. Only the f16 scale uses the shipped
 * converter. The generated AVX2 kernel and the scalar quantizer must match the
 * reference bit for bit
 *
 * Build and run:
 *   cc -std=gnu17 -Wall -Wextra -Werror -O2 -I. -Icodegen/include \
 *      tests/c/q8_quantize_test.c codegen/src/emit.c codegen/src/asm.c \
 *      codegen/src/isa.c codegen/src/q8_0.c -lm -o q8_quantize_test
 *   ./q8_quantize_test
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "toy/codegen.h"
#include "toy/q8_0.h"

static int failures;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            failures++; \
        } \
    } while (0)

static uint64_t rng_state = 0x0F1E2D3C4B5A6978ull;

static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/*
 * d = amax / 127 and q = roundf(x / d), with the scale stored as f16. A
 * non-finite product quantizes to zero, and the clamp runs before the int
 * conversion
 */
static void reference_block(const float *x, struct q8_0_block *out)
{
    float amax = 0.0f;
    for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
        const float v = fabsf(x[i]);
        if (v > amax) {
            amax = v;
        }
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    out->d = f32_to_f16(d);
    for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
        const float t = x[i] * id;
        int q = 0;
        if (fabsf(t) <= FLT_MAX) {
            const float rounded = roundf(t);
            if (rounded > 127.0f) {
                q = 127;
            } else if (rounded < -127.0f) {
                q = -127;
            } else {
                q = (int) rounded;
            }
        }
        out->qs[i] = (int8_t) q;
    }
}

/* Values that stress the scale, the rounding boundary and the int8 range */
static float edge_value(uint32_t pick)
{
    static const float values[] = {
        0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f,
        127.0f, -127.0f, 126.5f, -126.5f, 1e-45f, -1e-45f,
        1.17549435e-38f, -1.17549435e-38f, 3.40282347e38f, -3.40282347e38f,
        INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 63.5f, -63.5f,
    };
    return values[pick % (sizeof(values) / sizeof(values[0]))];
}

static void fill_values(float *x, uint32_t blocks, bool edges)
{
    for (uint32_t b = 0; b < blocks; b++) {
        float *block = x + (size_t)b * TOY_Q8_0_BLOCK;
        const bool edge_block = edges && (rng_next() % 4 == 0);
        // A block scale of 127 gives exact products, the other blocks stay in a
        // narrow range around zero
        const bool exact_scale = edges && (rng_next() % 4 == 0);
        for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
            if (edge_block && rng_next() % 3 == 0) {
                block[i] = edge_value(rng_next());
            } else if (exact_scale) {
                block[i] = (float)((int32_t)(rng_next() % 255) - 127);
            } else {
                block[i] = (float)(int32_t)(rng_next() % 65536) / 512.0f - 64.0f;
            }
        }
        if (exact_scale) {
            block[rng_next() % TOY_Q8_0_BLOCK] = 127.0f;
        }
    }
}

static void *alloc_executable(size_t bytes)
{
    void *code = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return code == MAP_FAILED ? NULL : code;
}

static bool seal_executable(void *code, size_t bytes)
{
    return mprotect(code, bytes, PROT_READ | PROT_EXEC) == 0;
}

static void test_blocks(uint32_t blocks, bool edges)
{
    const size_t values = (size_t)blocks * TOY_Q8_0_BLOCK;
    const size_t bytes = (size_t)blocks * TOY_Q8_0_BLOCK_BYTES;
    const size_t guarded = bytes + 64;

    float *x = malloc(values * sizeof(float));
    uint8_t *expected = malloc(guarded);
    uint8_t *got = malloc(guarded);
    struct q8_0_block *reference = malloc((size_t)blocks * sizeof(struct q8_0_block));

    CHECK(x != NULL && expected != NULL && got != NULL && reference != NULL);
    if (x == NULL || expected == NULL || got == NULL || reference == NULL) {
        free(x);
        free(expected);
        free(got);
        free(reference);
        return;
    }

    fill_values(x, blocks, edges);
    for (uint32_t b = 0; b < blocks; b++) {
        reference_block(x + (size_t)b * TOY_Q8_0_BLOCK, &reference[b]);
    }

    q8_0_quantize_blocks(blocks, x, expected);
    CHECK(memcmp(expected, reference, bytes) == 0);

    memset(got, 0xA5, guarded);
    q8_0_quantize_rows(1, blocks * TOY_Q8_0_BLOCK, x, got);
    CHECK(memcmp(got, expected, bytes) == 0);
    // The quantizer writes exactly one 34-byte block per 32 values
    for (size_t i = bytes; i < guarded; i++) {
        if (got[i] != 0xA5) {
            fprintf(stderr, "FAIL blocks=%u: wrote past %zu bytes\n", blocks, bytes);
            failures++;
            break;
        }
    }

    struct spec spec = {
        .op = TOY_OP_Q8_0_QUANTIZE,
        .isa = TOY_ISA_AVX2,
        .m = blocks,
        .n = 1,
        .k = 1,
    };
    if (spec_supported(&spec)) {
        void *code = alloc_executable(TOY_EMIT_CAP);
        CHECK(code != NULL);
        if (code != NULL) {
            const size_t length = emit(&spec, code, TOY_EMIT_CAP);
            CHECK(length != 0);
            if (length != 0 && seal_executable(code, TOY_EMIT_CAP)) {
                kernel_fn kernel;
                memcpy(&kernel, &code, sizeof(kernel));
                memset(got, 0xA5, guarded);
                kernel(x, NULL, got);
                if (memcmp(got, expected, bytes) != 0) {
                    for (uint32_t b = 0; b < blocks; b++) {
                        if (memcmp(got + (size_t)b * TOY_Q8_0_BLOCK_BYTES,
                                   expected + (size_t)b * TOY_Q8_0_BLOCK_BYTES,
                                   TOY_Q8_0_BLOCK_BYTES) != 0) {
                            fprintf(stderr, "FAIL avx2 blocks=%u edges=%d: block %u differs\n",
                                    blocks, (int) edges, b);
                            break;
                        }
                    }
                    failures++;
                }
                for (size_t i = bytes; i < guarded; i++) {
                    if (got[i] != 0xA5) {
                        fprintf(stderr, "FAIL avx2 blocks=%u: wrote past %zu bytes\n", blocks,
                                bytes);
                        failures++;
                        break;
                    }
                }
            }
            munmap(code, TOY_EMIT_CAP);
        }
    }

    free(x);
    free(expected);
    free(got);
    free(reference);
}

static void run_kernel(uint32_t blocks, const float *x, uint8_t *out, const char *label)
{
    struct spec spec = {
        .op = TOY_OP_Q8_0_QUANTIZE,
        .isa = TOY_ISA_AVX2,
        .m = blocks,
        .n = 1,
        .k = 1,
    };
    if (!spec_supported(&spec)) {
        return;
    }
    void *code = alloc_executable(TOY_EMIT_CAP);
    CHECK(code != NULL);
    if (code == NULL) {
        return;
    }
    const size_t length = emit(&spec, code, TOY_EMIT_CAP);
    CHECK(length != 0);
    if (length != 0 && seal_executable(code, TOY_EMIT_CAP)) {
        kernel_fn kernel;
        memcpy(&kernel, &code, sizeof(kernel));
        kernel(x, NULL, out);
    } else {
        fprintf(stderr, "FAIL %s: the kernel did not run\n", label);
        failures++;
    }
    munmap(code, TOY_EMIT_CAP);
}

/*
 * Fixed blocks that pin the edge rules: exact halves round away from zero, a
 * non-finite value becomes zero, and the scale covers zero, subnormal, tiny,
 * huge and infinite magnitudes
 */
static void test_fixed(void)
{
    enum { blocks = 12 };
    static const float zeros[TOY_Q8_0_BLOCK];
    static const float nan_block[TOY_Q8_0_BLOCK] = {
        [0 ... TOY_Q8_0_BLOCK - 1] = NAN,
    };
    float x[blocks][TOY_Q8_0_BLOCK];
    struct q8_0_block expected[blocks];
    uint8_t got[blocks * TOY_Q8_0_BLOCK_BYTES];

    memcpy(x[0], zeros, sizeof(zeros));
    memcpy(x[1], nan_block, sizeof(nan_block));
    for (int i = 0; i < TOY_Q8_0_BLOCK; i++) {
        // amax = 127 gives d = 1, so every half is an exact tie
        x[2][i] = (float)(int32_t)(i - 16) + 0.5f;
        x[2][16] = 127.0f;
        // amax = 254 gives d = 2, so an odd value becomes a tie again
        x[3][i] = (float)(2 * (int32_t)(i - 16) + 1);
        x[3][0] = 254.0f;
        x[4][i] = INFINITY;
        x[5][i] = i % 2 == 0 ? INFINITY : -INFINITY;
        // An infinite value sets an infinite scale, the finite values become zero
        x[6][i] = i == 7 ? INFINITY : (float)(int32_t)(i - 16);
        x[7][i] = NAN;
        x[7][0] = 1.0f;
        x[8][i] = 1.40129846e-45f;
        x[9][i] = 1.17549435e-38f;
        x[10][i] = 3.40282347e38f;
        x[11][i] = (float)(int32_t)(i - 16) / 1024.0f;
        x[11][3] = 127.0f / 1024.0f;
    }

    const size_t bytes = sizeof(expected);
    for (uint32_t b = 0; b < blocks; b++) {
        reference_block(x[b], &expected[b]);
    }
    q8_0_quantize_blocks(blocks, &x[0][0], got);
    CHECK(memcmp(got, expected, bytes) == 0);

    memset(got, 0xA5, sizeof(got));
    run_kernel(blocks, &x[0][0], got, "fixed");
    for (uint32_t b = 0; b < blocks; b++) {
        if (memcmp(got + (size_t)b * TOY_Q8_0_BLOCK_BYTES, &expected[b],
                   TOY_Q8_0_BLOCK_BYTES) != 0) {
            fprintf(stderr, "FAIL fixed: block %u differs (scale %04x vs %04x)\n", b,
                    ((const struct q8_0_block *)(got + (size_t)b * TOY_Q8_0_BLOCK_BYTES))->d,
                    expected[b].d);
            failures++;
        }
    }
}

int main(void)
{
    // The generator has no scalar kernel for this op
    uint8_t *stub = malloc(TOY_EMIT_CAP);
    const struct spec scalar = {
        .op = TOY_OP_Q8_0_QUANTIZE,
        .isa = TOY_ISA_SCALAR,
        .m = 8,
        .n = 1,
        .k = 1,
    };
    CHECK(!spec_supported(&scalar));
    CHECK(emit(&scalar, stub, TOY_EMIT_CAP) == 0);

    test_fixed();

    // A block count of one has no loop, a larger count runs the loop
    const uint32_t counts[] = { 1, 2, 3, 5, 7, 16, 17, 64, 257 };
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        test_blocks(counts[i], false);
        test_blocks(counts[i], true);
    }

    // A shape above the codegen limit falls back to the reference, the reference
    // itself still covers it
    struct spec huge = {
        .op = TOY_OP_Q8_0_QUANTIZE,
        .isa = TOY_ISA_AVX2,
        .m = (uint32_t)(INT32_MAX / TOY_Q8_0_BLOCK_BYTES) + 1,
        .n = 1,
        .k = 1,
    };
    CHECK(emit(&huge, stub, TOY_EMIT_CAP) == 0);
    free(stub);

    if (failures == 0) {
        printf("q8_0 quantize: all cases match the independent scalar reference\n");
        return 0;
    }
    fprintf(stderr, "q8_0 quantize: %d failures\n", failures);
    return 1;
}

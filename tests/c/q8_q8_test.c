/*
 * Standalone C tests for the q8_0 x q8_0 kernels
 * This file does not link against PHP
 *
 * The scalar reference here is written from the ggml definition only, so it
 * shares no code with codegen/src/ref.c. Both the generated AVX2 kernel and the
 * reference kernels must match it bit for bit
 *
 * Build and run:
 *   cc -std=gnu17 -Wall -Wextra -Werror -O2 -I. -Icodegen/include \
 *      tests/c/q8_q8_test.c codegen/src/emit.c codegen/src/asm.c \
 *      codegen/src/isa.c codegen/src/q8_0.c codegen/src/ref.c -lm -o q8_q8_test
 *   ./q8_q8_test
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "toy/codegen.h"
#include "toy/q8_0.h"
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

// A scale above the half subnormal range, so the f16 round trip stays finite
static uint16_t rng_scale(void)
{
    // Exponent 1..14, random mantissa: 2^-14 .. roughly 2
    const uint16_t exponent = (uint16_t)(1 + rng_next() % 14);
    return (uint16_t)((exponent << 10) | (rng_next() & 0x3FF));
}

static void fill_random(void *data, size_t bytes, bool edges)
{
    uint8_t *bytes_out = data;
    for (size_t i = 0; i < bytes; i++) {
        const uint32_t pick = rng_next();
        if (edges && pick % 7 == 0) {
            // -128, 127 and 0 stress the sign and the saturation margins
            bytes_out[i] = (uint8_t[]){ 0x80, 0x7F, 0x00 }[pick % 3];
        } else {
            bytes_out[i] = (uint8_t)(pick >> 8);
        }
    }
}

static void fill_blocks(struct q8_0_block *blocks, uint32_t count, bool edges)
{
    for (uint32_t b = 0; b < count; b++) {
        blocks[b].d = rng_scale();
        fill_random(blocks[b].qs, TOY_Q8_0_BLOCK, edges);
    }
}

/*
 * The byte pair as vpsignb and vpmaddubsw compute it: the weight byte gives the
 * unsigned magnitude and flips the sign of the activation byte, and the int8
 * negation wraps
 */
static int32_t pair_product(int8_t weight, int8_t value)
{
    const uint32_t magnitude = (uint32_t)(uint8_t)(weight < 0 ? -weight : weight);
    const int8_t signed_value = weight < 0 ? (int8_t)-value : value;

    return (int32_t)magnitude * (int32_t)signed_value;
}

/*
 * Independent scalar reference
 *
 * lane_of[p / 4] += the pair product as an exact integer, then one fma per lane
 * with the combined scale, then the ggml hsum_float_8 tree
 */
static void scalar_reference(uint32_t rows, uint32_t cols, uint32_t blocks,
                             const struct q8_0_block *w, const struct q8_0_block *x,
                             float *out)
{
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            const struct q8_0_block *wrow = w + (size_t)r * blocks;
            const struct q8_0_block *xcol = x + (size_t)c * blocks;
            float acc[8];

            for (int lane = 0; lane < 8; lane++) {
                acc[lane] = 0.0f;
            }
            for (uint32_t b = 0; b < blocks; b++) {
                int32_t lane_of[8];
                for (int lane = 0; lane < 8; lane++) {
                    lane_of[lane] = 0;
                }
                for (int p = 0; p < TOY_Q8_0_BLOCK; p++) {
                    lane_of[p / 4] += pair_product(wrow[b].qs[p], xcol[b].qs[p]);
                }
                const float scale = f16_to_f32(wrow[b].d) * f16_to_f32(xcol[b].d);
                for (int lane = 0; lane < 8; lane++) {
                    acc[lane] = fmaf(scale, (float)lane_of[lane], acc[lane]);
                }
            }
            // The tree shape matters: a flat sum gives a different result
            out[(size_t)r * cols + c] = ((acc[0] + acc[4]) + (acc[2] + acc[6]))
                                      + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
        }
    }
}

static bool same_bits(float left, float right)
{
    return memcmp(&left, &right, sizeof(float)) == 0;
}

static void *alloc_executable(size_t bytes)
{
    void *code = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return code == MAP_FAILED ? NULL : code;
}

static bool seal_executable(void *code, size_t bytes)
{
    return mprotect(code, bytes, PROT_READ | PROT_EXEC) == 0;
}

static void compare(const char *label, uint32_t rows, uint32_t cols, const float *got,
                    const float *expected)
{
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            const size_t index = (size_t)r * cols + c;
            if (!same_bits(got[index], expected[index])) {
                fprintf(stderr, "FAIL %s at [%u, %u]: got %.9g (0x%08x), expected %.9g (0x%08x)\n",
                        label, r, c, (double)got[index], *(const uint32_t *)&got[index],
                        (double)expected[index], *(const uint32_t *)&expected[index]);
                failures++;
                return;
            }
        }
    }
}

// Runs one generated kernel and both references against the scalar reference
static void test_shape(uint32_t m, uint32_t n, uint32_t k, bool edges)
{
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const size_t weightBlocks = (size_t)m * blocks;
    const size_t actBlocks = (size_t)n * blocks;

    struct q8_0_block *w = malloc(weightBlocks * sizeof(struct q8_0_block));
    struct q8_0_block *x = malloc(actBlocks * sizeof(struct q8_0_block));
    float *expected = malloc((size_t)m * n * sizeof(float));
    float *got = malloc((size_t)m * n * sizeof(float));
    float *vector = malloc((size_t)m * sizeof(float));
    float *vectorGot = malloc((size_t)m * sizeof(float));

    fill_blocks(w, (uint32_t)weightBlocks, edges);
    fill_blocks(x, (uint32_t)actBlocks, edges);
    scalar_reference(m, n, blocks, w, x, expected);

    ref_q8_0_q8_0_matmul(m, n, k, w, x, got);
    compare("reference matmul", m, n, got, expected);

    // The matvec reference only sees the first activation column
    scalar_reference(m, 1, blocks, w, x, vector);
    ref_q8_0_q8_0_matvec(m, k, w, x, vectorGot);
    compare("reference matvec", m, 1, vectorGot, vector);

    if (spec_supported(&(struct spec){
            .op = TOY_OP_Q8_0_Q8_0_MATMUL, .isa = TOY_ISA_AVX2, .m = m, .n = n, .k = k })) {
        struct spec spec = {
            .op = TOY_OP_Q8_0_Q8_0_MATMUL,
            .isa = TOY_ISA_AVX2,
            .m = m,
            .n = n,
            .k = k,
        };
        void *code = alloc_executable(TOY_EMIT_CAP);
        CHECK(code != NULL);
        if (code != NULL) {
            const size_t length = emit(&spec, code, TOY_EMIT_CAP);
            CHECK(length != 0);
            if (length != 0 && seal_executable(code, TOY_EMIT_CAP)) {
                kernel_fn kernel;
                memcpy(&kernel, &code, sizeof(kernel));
                memset(got, 0, (size_t)m * n * sizeof(float));
                kernel(w, x, got);
                compare("avx2 matmul", m, n, got, expected);
            }
            munmap(code, TOY_EMIT_CAP);
        }

        spec.op = TOY_OP_Q8_0_Q8_0_MATVEC;
        spec.n = 1;
        code = alloc_executable(TOY_EMIT_CAP);
        CHECK(code != NULL);
        if (code != NULL) {
            const size_t length = emit(&spec, code, TOY_EMIT_CAP);
            CHECK(length != 0);
            if (length != 0 && seal_executable(code, TOY_EMIT_CAP)) {
                kernel_fn kernel;
                memcpy(&kernel, &code, sizeof(kernel));
                memset(vectorGot, 0, (size_t)m * sizeof(float));
                kernel(w, x, vectorGot);
                compare("avx2 matvec", m, 1, vectorGot, vector);
            }
            munmap(code, TOY_EMIT_CAP);
        }
    }

    free(w);
    free(x);
    free(expected);
    free(got);
    free(vector);
    free(vectorGot);
}

int main(void)
{
    // The generator has no scalar kernel for these ops
    uint8_t *stub = malloc(TOY_EMIT_CAP);
    const struct spec scalar = {
        .op = TOY_OP_Q8_0_Q8_0_MATVEC,
        .isa = TOY_ISA_SCALAR,
        .m = 8,
        .n = 1,
        .k = 32,
    };
    CHECK(!spec_supported(&scalar));
    CHECK(emit(&scalar, stub, TOY_EMIT_CAP) == 0);
    free(stub);

    const uint32_t shapes[][2] = {
        { 1, 32 }, { 2, 32 }, { 3, 64 }, { 7, 64 }, { 8, 32 }, { 9, 96 },
        { 16, 32 }, { 17, 64 }, { 23, 128 }, { 64, 1024 }, { 65, 32 },
    };
    const uint32_t columns[] = { 1, 2, 32, 64 };

    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        const uint32_t m = shapes[s][0];
        const uint32_t k = shapes[s][1];
        for (size_t c = 0; c < sizeof(columns) / sizeof(columns[0]); c++) {
            test_shape(m, columns[c], k, false);
            test_shape(m, columns[c], k, true);
        }
    }

    if (failures == 0) {
        printf("q8_0 x q8_0: all cases match the independent scalar reference\n");
        return 0;
    }
    fprintf(stderr, "q8_0 x q8_0: %d failures\n", failures);
    return 1;
}

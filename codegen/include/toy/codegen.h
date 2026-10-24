#ifndef TOY_CODEGEN_H
#define TOY_CODEGEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum op {
    TOY_OP_GEMM = 1,
    TOY_OP_MATVEC = 2,
    TOY_OP_DOT = 3,
    TOY_OP_ADD = 4,
    TOY_OP_SCALE = 5,
    TOY_OP_RMSNORM = 6,
    // q8_0 weights with an f32 input: a is q8_0 (m x k), b is f32, c is f32
    TOY_OP_Q8_0_DOT = 7,
    TOY_OP_Q8_0_MATVEC = 8,
    TOY_OP_Q8_0_MATMUL = 9,
    TOY_OP_MUL = 10,
};

enum isa {
    TOY_ISA_SCALAR = 1,
    TOY_ISA_AVX2 = 2,
};

/*
 * The struct holds the shape and the parameter, so a kernel has three
 * pointer arguments only
 *
 * GEMM:   m, n, k
 * MATVEC: the shape is m x k, n must be 1
 * DOT:    k is the length, m and n must be 1
 *
 * ADD:     m is the element count, n and k must be 1, param_bits is unused
 * MUL:     m is the element count, n and k must be 1, param_bits is unused
 * SCALE:   m is the element count, param_bits holds the f32 factor bits
 * RMSNORM: m is the element count, param_bits holds the f32 eps bits
 *
 * Q8_0_DOT:    k is the length, m and n must be 1
 * Q8_0_MATVEC: the shape is m x k, n must be 1
 * Q8_0_MATMUL: m, n, k
 *
 * For q8_0, k must be a multiple of 32. A weight row is a series of 34-byte
 * blocks
 */
struct spec {
    uint32_t op;
    uint32_t isa;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t param_bits;
};

typedef void (*kernel_fn)(const void *a, const void *b, void *c);

/*
 * Writes the machine code into buf and returns the byte count. Returns 0 on
 * failure
 *
 * buf must hold at least 16 KiB, see TOY_EMIT_CAP
 *
 * The generator writes code only and does not allocate executable memory.
 * The caller provides the writable block and seals it
 */
#define TOY_EMIT_CAP (16u * 1024u)

size_t emit(const struct spec *spec, uint8_t *buf, size_t cap);

// Reports if the host can run a kernel for this ISA. Both ISAs need FMA
bool isa_available(uint32_t isa);

// Reports if the ISA and the op extensions are available. q8_0 also needs F16C
bool spec_supported(const struct spec *spec);

// Reports F16C availability for the diagnostic API
bool f16c_available(void);

const char *op_name(uint32_t op);
const char *isa_name(uint32_t isa);

// Returns the floating-point operation count of the kernel, 0 for an unknown op
double spec_flops(const struct spec *spec);

#ifdef __cplusplus
}
#endif

#endif

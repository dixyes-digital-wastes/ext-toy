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
    // q8_0 weights with q8_0 activations: a is q8_0 (m x k), b is q8_0, c is f32
    TOY_OP_Q8_0_Q8_0_MATVEC = 11,
    TOY_OP_Q8_0_Q8_0_MATMUL = 12,
    // f32 to q8_0 activation quantization: a is f32, c is q8_0
    TOY_OP_Q8_0_QUANTIZE = 13,
    // Unary f32 math, AVX2 only: c[i] = f(a[i]) for m elements
    TOY_OP_SIGMOID = 14,
    TOY_OP_SOFTPLUS = 15,
    // Depthwise conv with SiLU: a is w (m x taps), b is x (taps x k), c is m
    TOY_OP_CONV1D_SILU = 16,
    // Row l2 norm: a is x (m x k), b is unused, c is m x k
    TOY_OP_L2NORM = 17,
    // Row rms norm against one weight row: a is x (m x k), b is w (k), c is m x k
    TOY_OP_RMSNORM_ROWS = 18,
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
 * Q8_0_Q8_0_MATVEC: the shape is m x k, n must be 1
 * Q8_0_Q8_0_MATMUL: m, n, k
 *
 * Q8_0_QUANTIZE: m is the number of 32 value blocks, n and k must be 1. The
 * source is m blocks of 32 f32 values and the destination is m 34-byte blocks,
 * so a row range is just a source pointer offset
 *
 * For q8_0, k must be a multiple of 32. A weight row is a series of 34-byte
 * blocks, and a q8_0 activation is one such series of blocks per column
 *
 * SIGMOID, SOFTPLUS: m is the element count, n and k must be 1
 * CONV1D_SILU: m is the channel count, k is the stride between two taps of x,
 *              n must be 1, param_bits holds the tap count
 * L2NORM: m is the row count, k is the row width, n must be 1, param_bits
 *         holds the f32 eps bits
 * RMSNORM_ROWS: the same shape as L2NORM, b points at the shared weight row
 *
 * The four transcendental ops have no scalar ISA kernel; a scalar host uses
 * the reference directly
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

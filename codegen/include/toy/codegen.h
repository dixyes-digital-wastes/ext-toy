#ifndef TOY_CODEGEN_H
#define TOY_CODEGEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum toy_op {
    TOY_OP_GEMM = 1,
    TOY_OP_MATVEC = 2,
    TOY_OP_DOT = 3,
    TOY_OP_ADD = 4,
    TOY_OP_SCALE = 5,
    TOY_OP_RMSNORM = 6,
    TOY_OP_Q8_0_DOT = 7,
    TOY_OP_Q8_0_MATVEC = 8,
    TOY_OP_Q8_0_MATMUL = 9,
};

enum toy_isa {
    TOY_ISA_SCALAR = 1,
    TOY_ISA_AVX2 = 2,
};

struct toy_spec {
    uint32_t op;
    uint32_t isa;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t param_bits;
};

typedef void (*toy_kernel_fn)(const void *a, const void *b, void *c);

#define TOY_EMIT_CAP (16u * 1024u)

size_t toy_emit(const struct toy_spec *spec, uint8_t *buf, size_t cap);

bool toy_isa_available(uint32_t isa);

bool toy_spec_supported(const struct toy_spec *spec);

bool toy_f16c_available(void);

const char *toy_op_name(uint32_t op);
const char *toy_isa_name(uint32_t isa);

double toy_spec_flops(const struct toy_spec *spec);

#ifdef __cplusplus
}
#endif

#endif

#ifndef KERNEL_H
#define KERNEL_H

#include "tensor.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Generated kernels live in the kernel arena; the spec cache keys them by
 * kind, ISA, dtype, shape and param bits. Element-wise kernels are also
 * memoized in the tensor, keyed by kind, dtype, element count and param bits
 */

enum {
    TOY_KERNEL_KEY_SIZE = 32,
};

static_assert(TOY_KERNEL_KEY_SIZE == 32, "the spec key size is part of the cache ABI");

typedef enum kernel_kind {
    TOY_KERNEL_ADD = 1,
    TOY_KERNEL_SCALE = 2,
    TOY_KERNEL_RMSNORM = 3,
    TOY_KERNEL_MATMUL = 4,
    TOY_KERNEL_MATVEC = 5,
    TOY_KERNEL_MATMUL_Q8_0 = 6,
    TOY_KERNEL_MATVEC_Q8_0 = 7,
    TOY_KERNEL_MUL = 8,
    TOY_KERNEL_MATMUL_Q8_0_Q8_0 = 9,
    TOY_KERNEL_MATVEC_Q8_0_Q8_0 = 10,
    TOY_KERNEL_Q8_0_QUANTIZE = 11,
    TOY_KERNEL_SIGMOID = 12,
    TOY_KERNEL_SOFTPLUS = 13,
    TOY_KERNEL_CONV1D_SILU = 14,
    TOY_KERNEL_L2NORM = 15,
    TOY_KERNEL_RMSNORM_ROWS = 16,
} kernel_kind;

typedef struct kernel_stats {
    bool enabled;
    bool avx2_available;
    bool f16c_available;
    const char *isa;
    uint64_t flf_calls;
    uint64_t zif_calls;
    uint64_t memo_hits;
    uint64_t memo_misses;
    uint64_t arena_hits;
    uint64_t arena_misses;
    uint64_t generated;
    uint64_t generation_failures;
    uint64_t kernel_calls;
    uint64_t scalar_calls;
} kernel_stats;

void kernel_startup(void);
void kernel_shutdown(void);

/* True when the host runs AVX2 and FMA code */
bool kernel_avx2_available(void);

void kernel_count_flf(void);
void kernel_count_zif(void);
void kernel_count_kernel_call(void);
void kernel_count_scalar_call(void);
void kernel_stats_read(kernel_stats *stats);
void kernel_stats_reset(void);

/*
 * Disables generation and lookups globally; calls then use the scalar
 * reference. Returns the previous setting.
 */
bool kernel_set_enabled(bool enabled);
bool kernel_enabled(void);

/* Selects the codegen ISA; fails when the host cannot run it */
bool kernel_set_isa(uint32_t isa);
/* TOY_ISA_* of the selected codegen ISA, 0 when no ISA is usable */
uint32_t kernel_isa(void);

/*
 * Resolves an element-wise kernel for count elements of the tensor dtype.
 * The tensor memo is consulted first, then the arena spec cache. NULL means
 * the caller must use the scalar reference.
 */
kernel_entry kernel_for_element(tensor *tensor_value, uint8_t kind, uint32_t param_bits,
                                uint32_t count);

/* Resolves a shape-keyed kernel. It does not touch any tensor memo. */
kernel_entry kernel_for_shape(uint8_t kind, uint8_t dtype, uint32_t m, uint32_t n,
                              uint32_t k);

/*
 * Resolves a shape-keyed kernel with a parameter, for the kernels whose shape
 * alone does not identify them: the eps of l2norm, the tap count of conv1d
 */
kernel_entry kernel_for_shape_param(uint8_t kind, uint8_t dtype, uint32_t m, uint32_t n,
                                    uint32_t k, uint32_t param_bits);

#endif /* KERNEL_H */

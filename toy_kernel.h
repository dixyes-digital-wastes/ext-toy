#ifndef TOY_KERNEL_H
#define TOY_KERNEL_H

#include "toy_tensor.h"

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

typedef enum toy_kernel_kind {
    TOY_KERNEL_ADD = 1,
    TOY_KERNEL_SCALE = 2,
    TOY_KERNEL_RMSNORM = 3,
    TOY_KERNEL_MATMUL = 4,
    TOY_KERNEL_MATVEC = 5,
    TOY_KERNEL_MATMUL_Q8_0 = 6,
    TOY_KERNEL_MATVEC_Q8_0 = 7,
} toy_kernel_kind;

typedef struct toy_kernel_stats {
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
} toy_kernel_stats;

void toy_kernel_startup(void);
void toy_kernel_shutdown(void);

/* True when the host runs AVX2 and FMA code */
bool toy_kernel_avx2_available(void);

void toy_kernel_count_flf(void);
void toy_kernel_count_zif(void);
void toy_kernel_count_kernel_call(void);
void toy_kernel_count_scalar_call(void);
void toy_kernel_stats_read(toy_kernel_stats *stats);
void toy_kernel_stats_reset(void);

/*
 * Disables generation and lookups globally; calls then use the scalar
 * reference. Returns the previous setting.
 */
bool toy_kernel_set_enabled(bool enabled);
bool toy_kernel_enabled(void);

/* Selects the codegen ISA; fails when the host cannot run it */
bool toy_kernel_set_isa(uint32_t isa);
/* TOY_ISA_* of the selected codegen ISA, 0 when no ISA is usable */
uint32_t toy_kernel_isa(void);

/*
 * Resolves an element-wise kernel for count elements of the tensor dtype.
 * The tensor memo is consulted first, then the arena spec cache. NULL means
 * the caller must use the scalar reference.
 */
toy_kernel_entry toy_kernel_for_element(toy_tensor *tensor, uint8_t kind, uint32_t param_bits,
                                        uint32_t count);

/* Resolves a shape-keyed kernel. It does not touch any tensor memo. */
toy_kernel_entry toy_kernel_for_shape(uint8_t kind, uint8_t dtype, uint32_t m, uint32_t n,
                                      uint32_t k);

#endif /* TOY_KERNEL_H */

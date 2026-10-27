#include "kernel.h"

#include "arena_config.h"

#include <assert.h>
#include <string.h>

#include "toy/codegen.h"

static_assert(sizeof(void *) == sizeof(kernel_entry),
              "a code pointer and a kernel entry have the same size");

/*
 * The 32-byte spec key identifies one generated kernel
 * Zero padding keeps the hash stable because the whole key is compared
 */
typedef struct kernel_key {
    uint8_t kind;
    uint8_t isa;
    uint16_t reserved;
    uint32_t param_bits;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t dtype;
    uint32_t version;
    uint32_t reserved2;
} kernel_key;

static_assert(sizeof(kernel_key) == TOY_KERNEL_KEY_SIZE,
              "the kernel key is exactly one cache key");

typedef struct kernel_state {
    bool enabled;
    uint32_t isa;
} kernel_state;

static kernel_state kernel_current = { .enabled = true };
static kernel_stats kernel_counters;

static bool kernel_cpu_ready;
static bool kernel_cpu_avx2;

static bool kernel_fill(const uint8_t *key, void *context, uint8_t *code, size_t capacity)
{
    (void) key;

    const struct spec *spec = context;
    return emit(spec, code, capacity) != 0;
}

static void kernel_detect_cpu(void)
{
    if (kernel_cpu_ready) {
        return;
    }

    kernel_cpu_avx2 = isa_available(TOY_ISA_AVX2);
    kernel_cpu_ready = true;
}

static uint8_t kernel_op(uint8_t kind)
{
    switch (kind) {
        case TOY_KERNEL_ADD:
            return TOY_OP_ADD;
        case TOY_KERNEL_MUL:
            return TOY_OP_MUL;
        case TOY_KERNEL_SCALE:
            return TOY_OP_SCALE;
        case TOY_KERNEL_RMSNORM:
            return TOY_OP_RMSNORM;
        case TOY_KERNEL_MATMUL:
            return TOY_OP_GEMM;
        case TOY_KERNEL_MATVEC:
            return TOY_OP_MATVEC;
        case TOY_KERNEL_MATMUL_Q8_0:
            return TOY_OP_Q8_0_MATMUL;
        case TOY_KERNEL_MATVEC_Q8_0:
            return TOY_OP_Q8_0_MATVEC;
        case TOY_KERNEL_MATMUL_Q8_0_Q8_0:
            return TOY_OP_Q8_0_Q8_0_MATMUL;
        case TOY_KERNEL_MATVEC_Q8_0_Q8_0:
            return TOY_OP_Q8_0_Q8_0_MATVEC;
        case TOY_KERNEL_Q8_0_QUANTIZE:
            return TOY_OP_Q8_0_QUANTIZE;
        case TOY_KERNEL_SIGMOID:
            return TOY_OP_SIGMOID;
        case TOY_KERNEL_SOFTPLUS:
            return TOY_OP_SOFTPLUS;
        case TOY_KERNEL_CONV1D_SILU:
            return TOY_OP_CONV1D_SILU;
        case TOY_KERNEL_L2NORM:
            return TOY_OP_L2NORM;
        case TOY_KERNEL_RMSNORM_ROWS:
            return TOY_OP_RMSNORM_ROWS;
        default:
            return 0;
    }
}

static kernel_entry kernel_entry_from_code(void *code)
{
    kernel_entry entry;
    memcpy(&entry, &code, sizeof(entry));
    return entry;
}

static kernel_entry kernel_resolve(uint8_t kind, uint8_t dtype, uint32_t m,
                                           uint32_t n, uint32_t k, uint32_t param_bits)
{
    if (!kernel_current.enabled || kernel_op(kind) == 0) {
        return NULL;
    }

    uint32_t isa = kernel_isa();
    if (isa == 0) {
        return NULL;
    }

    struct spec spec = {
        .op = kernel_op(kind),
        .isa = isa,
        .m = m,
        .n = n,
        .k = k,
        .param_bits = param_bits,
    };
    /* Unsupported ISA or missing F16C for q8_0: the caller uses the scalar path */
    if (!spec_supported(&spec)) {
        return NULL;
    }

    spec_cache *cache = kernel_arena_cache();
    if (cache == NULL) {
        return NULL;
    }

    kernel_key key = { 0 };
    key.kind = kind;
    key.isa = (uint8_t) isa;
    key.param_bits = param_bits;
    key.m = m;
    key.n = n;
    key.k = k;
    key.dtype = dtype;
    key.version = 1;

    uint8_t bytes[TOY_KERNEL_KEY_SIZE];
    memcpy(bytes, &key, sizeof(key));

    arena_block block;
    if (spec_cache_lookup(cache, bytes, &block)) {
        kernel_counters.arena_hits++;
        return kernel_entry_from_code(block.code);
    }
    kernel_counters.arena_misses++;

    uint8_t *code = spec_cache_get_code(cache, bytes, TOY_EMIT_CAP, kernel_fill, &spec);
    if (code == NULL) {
        kernel_counters.generation_failures++;
        return NULL;
    }

    kernel_counters.generated++;
    return kernel_entry_from_code(code);
}

/* kind and dtype sit in the top bytes, the ISA in the next byte, the count below */
static uint64_t kernel_element_key(uint8_t kind, uint8_t dtype, uint32_t isa,
                                       uint32_t count)
{
    return ((uint64_t) kind << 60) | ((uint64_t) dtype << 56) | ((uint64_t) isa << 48)
        | (uint64_t) count;
}

void kernel_startup(void)
{
    kernel_detect_cpu();
    kernel_stats_reset();
}

void kernel_shutdown(void)
{
}

bool kernel_avx2_available(void)
{
    kernel_detect_cpu();
    return kernel_cpu_avx2;
}

void kernel_count_flf(void)
{
    kernel_counters.flf_calls++;
}

void kernel_count_zif(void)
{
    kernel_counters.zif_calls++;
}

void kernel_count_kernel_call(void)
{
    kernel_counters.kernel_calls++;
}

void kernel_count_scalar_call(void)
{
    kernel_counters.scalar_calls++;
}

void kernel_stats_read(kernel_stats *stats)
{
    *stats = kernel_counters;
    stats->enabled = kernel_current.enabled;
    stats->avx2_available = kernel_avx2_available();
    stats->f16c_available = f16c_available();
    stats->isa = isa_name(kernel_isa());
}

void kernel_stats_reset(void)
{
    memset(&kernel_counters, 0, sizeof(kernel_counters));
}

bool kernel_set_enabled(bool enabled)
{
    bool previous = kernel_current.enabled;
    kernel_current.enabled = enabled;
    return previous;
}

bool kernel_enabled(void)
{
    return kernel_current.enabled;
}

bool kernel_set_isa(uint32_t isa)
{
    kernel_detect_cpu();

    if (isa != TOY_ISA_SCALAR && isa != TOY_ISA_AVX2) {
        return false;
    }
    if (!isa_available(isa)) {
        return false;
    }

    kernel_current.isa = isa;
    return true;
}

uint32_t kernel_isa(void)
{
    kernel_detect_cpu();

    if (kernel_current.isa == 0) {
        /* The scalar codegen also needs FMA, so no FMA means no kernels */
        if (kernel_cpu_avx2) {
            kernel_current.isa = TOY_ISA_AVX2;
        } else if (isa_available(TOY_ISA_SCALAR)) {
            kernel_current.isa = TOY_ISA_SCALAR;
        }
    }
    return kernel_current.isa;
}

kernel_entry kernel_for_element(tensor *tensor_value, uint8_t kind, uint32_t param_bits,
                                uint32_t count)
{
    if (!kernel_current.enabled || tensor_value == NULL || count == 0) {
        return NULL;
    }

    uint32_t isa_value = kernel_isa();
    if (isa_value == 0) {
        return NULL;
    }

    const uint64_t key = kernel_element_key(kind, tensor_value->dtype, isa_value, count);
    tensor_memo *memo = &tensor_value->memo;

    if (memo->kernel != NULL && memo->key == key && memo->param_bits == param_bits) {
        kernel_counters.memo_hits++;
        return memo->kernel;
    }
    kernel_counters.memo_misses++;

    kernel_entry entry = kernel_resolve(kind, tensor_value->dtype, count, 1, 1, param_bits);
    if (entry != NULL) {
        memo->kernel = entry;
        memo->key = key;
        memo->param_bits = param_bits;
    }
    return entry;
}

kernel_entry kernel_for_shape(uint8_t kind, uint8_t dtype, uint32_t m, uint32_t n,
                                      uint32_t k)
{
    return kernel_resolve(kind, dtype, m, n, k, 0);
}

kernel_entry kernel_for_shape_param(uint8_t kind, uint8_t dtype, uint32_t m, uint32_t n,
                                    uint32_t k, uint32_t param_bits)
{
    return kernel_resolve(kind, dtype, m, n, k, param_bits);
}

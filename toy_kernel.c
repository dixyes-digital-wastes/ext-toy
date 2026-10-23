#include "toy_kernel.h"

#include "toy_kernel_arena.h"

#include <string.h>

#include "toy/codegen.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
# define TOY_KERNEL_ARCH_X86 1
#endif

static_assert(sizeof(void *) == sizeof(toy_kernel_entry),
              "a code pointer and a kernel entry have the same size");

/*
 * The 32-byte spec key identifies one generated kernel
 * Zero padding keeps the hash stable because the whole key is compared
 */
typedef struct toy_kernel_key {
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
} toy_kernel_key;

static_assert(sizeof(toy_kernel_key) == TOY_KERNEL_KEY_SIZE,
              "the kernel key is exactly one cache key");

typedef struct toy_kernel_state {
    bool enabled;
    uint32_t isa;
} toy_kernel_state;

static toy_kernel_state toy_kernel_current = { .enabled = true };
static toy_kernel_stats toy_kernel_counters;

static bool toy_kernel_cpu_ready;
static bool toy_kernel_cpu_avx2;
static bool toy_kernel_cpu_fma;

static bool toy_kernel_fill(const uint8_t *key, void *context, uint8_t *code, size_t capacity)
{
    (void) key;

    const struct toy_spec *spec = context;
    return toy_emit(spec, code, capacity) != 0;
}

static void toy_kernel_detect_cpu(void)
{
    if (toy_kernel_cpu_ready) {
        return;
    }

#if defined(TOY_KERNEL_ARCH_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    toy_kernel_cpu_fma = __builtin_cpu_supports("fma");
    toy_kernel_cpu_avx2 = toy_kernel_cpu_fma && __builtin_cpu_supports("avx2");
#endif

    toy_kernel_cpu_ready = true;
}

static uint8_t toy_kernel_op(uint8_t kind)
{
    switch (kind) {
        case TOY_KERNEL_ADD:
            return TOY_OP_ADD;
        case TOY_KERNEL_SCALE:
            return TOY_OP_SCALE;
        case TOY_KERNEL_RMSNORM:
            return TOY_OP_RMSNORM;
        case TOY_KERNEL_MATMUL:
            return TOY_OP_GEMM;
        case TOY_KERNEL_MATVEC:
            return TOY_OP_MATVEC;
        default:
            return 0;
    }
}

static toy_kernel_entry toy_kernel_entry_from_code(void *code)
{
    toy_kernel_entry entry;
    memcpy(&entry, &code, sizeof(entry));
    return entry;
}

static toy_kernel_entry toy_kernel_resolve(uint8_t kind, uint8_t dtype, uint32_t m,
                                           uint32_t n, uint32_t k, uint32_t param_bits)
{
    if (!toy_kernel_current.enabled || toy_kernel_op(kind) == 0) {
        return NULL;
    }

    uint32_t isa = toy_kernel_isa();
    if (isa == 0) {
        return NULL;
    }

    toy_spec_cache *cache = toy_kernel_arena_cache();
    if (cache == NULL) {
        return NULL;
    }

    toy_kernel_key key = { 0 };
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

    toy_arena_block block;
    if (toy_spec_cache_lookup(cache, bytes, &block)) {
        toy_kernel_counters.arena_hits++;
        return toy_kernel_entry_from_code(block.code);
    }
    toy_kernel_counters.arena_misses++;

    struct toy_spec spec = {
        .op = toy_kernel_op(kind),
        .isa = isa,
        .m = m,
        .n = n,
        .k = k,
        .param_bits = param_bits,
    };
    uint8_t *code = toy_spec_cache_get_code(cache, bytes, TOY_EMIT_CAP, toy_kernel_fill, &spec);
    if (code == NULL) {
        toy_kernel_counters.generation_failures++;
        return NULL;
    }

    toy_kernel_counters.generated++;
    return toy_kernel_entry_from_code(code);
}

/* kind and dtype sit in the top bytes, the ISA in the next byte, the count below */
static uint64_t toy_kernel_element_key(uint8_t kind, uint8_t dtype, uint32_t isa,
                                       uint32_t count)
{
    return ((uint64_t) kind << 60) | ((uint64_t) dtype << 56) | ((uint64_t) isa << 48)
        | (uint64_t) count;
}

void toy_kernel_startup(void)
{
    toy_kernel_detect_cpu();
    toy_kernel_stats_reset();
}

void toy_kernel_shutdown(void)
{
}

bool toy_kernel_avx2_available(void)
{
    toy_kernel_detect_cpu();
    return toy_kernel_cpu_avx2;
}

void toy_kernel_count_flf(void)
{
    toy_kernel_counters.flf_calls++;
}

void toy_kernel_count_zif(void)
{
    toy_kernel_counters.zif_calls++;
}

void toy_kernel_count_kernel_call(void)
{
    toy_kernel_counters.kernel_calls++;
}

void toy_kernel_count_scalar_call(void)
{
    toy_kernel_counters.scalar_calls++;
}

void toy_kernel_stats_read(toy_kernel_stats *stats)
{
    *stats = toy_kernel_counters;
    stats->enabled = toy_kernel_current.enabled;
    stats->avx2_available = toy_kernel_avx2_available();
    stats->isa = toy_isa_name(toy_kernel_isa());
}

void toy_kernel_stats_reset(void)
{
    memset(&toy_kernel_counters, 0, sizeof(toy_kernel_counters));
}

bool toy_kernel_set_enabled(bool enabled)
{
    bool previous = toy_kernel_current.enabled;
    toy_kernel_current.enabled = enabled;
    return previous;
}

bool toy_kernel_enabled(void)
{
    return toy_kernel_current.enabled;
}

bool toy_kernel_set_isa(uint32_t isa)
{
    toy_kernel_detect_cpu();

    if (isa != TOY_ISA_SCALAR && isa != TOY_ISA_AVX2) {
        return false;
    }
    if (isa == TOY_ISA_AVX2 && !toy_kernel_cpu_avx2) {
        return false;
    }
    if (isa == TOY_ISA_SCALAR && !toy_kernel_cpu_fma) {
        return false;
    }

    toy_kernel_current.isa = isa;
    return true;
}

uint32_t toy_kernel_isa(void)
{
    toy_kernel_detect_cpu();

    if (toy_kernel_current.isa == 0) {
        /* The scalar codegen also needs FMA, so no FMA means no kernels */
        if (toy_kernel_cpu_avx2) {
            toy_kernel_current.isa = TOY_ISA_AVX2;
        } else if (toy_kernel_cpu_fma) {
            toy_kernel_current.isa = TOY_ISA_SCALAR;
        }
    }
    return toy_kernel_current.isa;
}

toy_kernel_entry toy_kernel_for_element(toy_tensor *tensor, uint8_t kind, uint32_t param_bits,
                                        uint32_t count)
{
    if (!toy_kernel_current.enabled || tensor == NULL || count == 0) {
        return NULL;
    }

    uint32_t isa = toy_kernel_isa();
    if (isa == 0) {
        return NULL;
    }

    const uint64_t key = toy_kernel_element_key(kind, tensor->dtype, isa, count);
    toy_tensor_memo *memo = &tensor->memo;

    if (memo->kernel != NULL && memo->key == key && memo->param_bits == param_bits) {
        toy_kernel_counters.memo_hits++;
        return memo->kernel;
    }
    toy_kernel_counters.memo_misses++;

    toy_kernel_entry entry = toy_kernel_resolve(kind, tensor->dtype, count, 1, 1, param_bits);
    if (entry != NULL) {
        memo->kernel = entry;
        memo->key = key;
        memo->param_bits = param_bits;
    }
    return entry;
}

toy_kernel_entry toy_kernel_for_shape(uint8_t kind, uint32_t m, uint32_t n, uint32_t k)
{
    return toy_kernel_resolve(kind, TOY_DTYPE_F32, m, n, k, 0);
}

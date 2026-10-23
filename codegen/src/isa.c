#include "toy/codegen.h"

/*
 * CPU feature detection
 *
 * The standalone generator keeps this code in exec.c, together with its mmap
 * executable memory. This snapshot only needs the check, so the code lives in
 * its own file
 *
 * Both generated ISAs use FMA instructions, so scalar also needs FMA
 * The q8_0 kernels use vcvtph2ps, so they also need F16C
 */

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
# define TOY_ISA_ARCH_X86 1
#endif

#if defined(_MSC_VER) && defined(_M_X64)
# include <intrin.h>
# include <immintrin.h>
# define TOY_ISA_MSVC_X64 1
#endif

static bool toy_isa_ready;
static bool toy_isa_fma;
static bool toy_isa_avx2;
static bool toy_isa_f16c;

#if defined(TOY_ISA_MSVC_X64)
/*
 * MSVC has no __builtin_cpu_supports, so read CPUID and XCR0 directly
 * FMA and F16C are VEX-encoded instructions, so the OS must save the YMM state
 * before they are available
 */
static void toy_isa_init_msvc(void) {
    int regs[4] = { 0, 0, 0, 0 };
    __cpuid(regs, 0);
    const int max_leaf = regs[0];

    __cpuid(regs, 1);
    const bool fma = (regs[2] & (1 << 12)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool f16c = (regs[2] & (1 << 29)) != 0;

    unsigned long long xcr0 = 0;
    if (osxsave && avx) {
        xcr0 = _xgetbv(0);
    }
    const bool ymm = (xcr0 & 0x6u) == 0x6u;

    bool avx2 = false;
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        avx2 = (regs[1] & (1 << 5)) != 0;
    }

    toy_isa_fma = fma && ymm;
    toy_isa_f16c = f16c && ymm;
    toy_isa_avx2 = avx2 && ymm;
}
#endif

static void toy_isa_init(void) {
    if (toy_isa_ready) {
        return;
    }

#if defined(TOY_ISA_MSVC_X64)
    toy_isa_init_msvc();
#elif defined(TOY_ISA_ARCH_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    toy_isa_fma = __builtin_cpu_supports("fma") != 0;
    toy_isa_avx2 = toy_isa_fma && __builtin_cpu_supports("avx2") != 0;
    toy_isa_f16c = __builtin_cpu_supports("f16c") != 0;
#endif

    toy_isa_ready = true;
}

bool toy_isa_available(uint32_t isa) {
    toy_isa_init();

    switch (isa) {
        case TOY_ISA_SCALAR:
            return toy_isa_fma;
        case TOY_ISA_AVX2:
            return toy_isa_avx2;
        default:
            return false;
    }
}

bool toy_spec_supported(const struct toy_spec *spec) {
    if (spec == NULL || !toy_isa_available(spec->isa)) {
        return false;
    }

    if (spec->op >= TOY_OP_Q8_0_DOT && spec->op <= TOY_OP_Q8_0_MATMUL) {
        return toy_isa_f16c;
    }
    return true;
}

bool toy_f16c_available(void) {
    toy_isa_init();
    return toy_isa_f16c;
}

#include "toy/codegen.h"

/*
 * CPU feature detection
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

static bool isa_ready;
static bool isa_fma;
static bool isa_avx2;
static bool isa_f16c;

#if defined(TOY_ISA_MSVC_X64)
/*
 * MSVC has no __builtin_cpu_supports, so read CPUID and XCR0 directly
 * FMA and F16C are VEX-encoded instructions, so the OS must save the YMM state
 * before they are available
 */
static void isa_init_msvc(void) {
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

    isa_fma = fma && ymm;
    isa_f16c = f16c && ymm;
    isa_avx2 = avx2 && ymm;
}
#endif

static void isa_init(void) {
    if (isa_ready) {
        return;
    }

#if defined(TOY_ISA_MSVC_X64)
    isa_init_msvc();
#elif defined(TOY_ISA_ARCH_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    isa_fma = __builtin_cpu_supports("fma") != 0;
    isa_avx2 = isa_fma && __builtin_cpu_supports("avx2") != 0;
    isa_f16c = __builtin_cpu_supports("f16c") != 0;
#endif

    isa_ready = true;
}

bool isa_available(uint32_t isa) {
    isa_init();

    switch (isa) {
        case TOY_ISA_SCALAR:
            return isa_fma;
        case TOY_ISA_AVX2:
            return isa_avx2;
        default:
            return false;
    }
}

bool spec_supported(const struct spec *spec) {
    if (spec == NULL || !isa_available(spec->isa)) {
        return false;
    }

    if (spec->op == TOY_OP_Q8_0_Q8_0_MATVEC || spec->op == TOY_OP_Q8_0_Q8_0_MATMUL) {
        /*
         * The bit-exact lane order needs the byte and word integer products,
         * which only the AVX2 kernel has. A scalar host uses the reference
         */
        return spec->isa == TOY_ISA_AVX2 && isa_f16c;
    }
    if (spec->op >= TOY_OP_Q8_0_DOT && spec->op <= TOY_OP_Q8_0_MATMUL) {
        return isa_f16c;
    }
    return true;
}

bool f16c_available(void) {
    isa_init();
    return isa_f16c;
}

#include "toy/codegen.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "toy/asm.h"

enum {
    LABEL_I = 0,
    LABEL_J = 1,
    LABEL_P = 2,
    MATVEC_P = 1,
    MATVEC_TAIL_I = 2,
    MATVEC_TAIL_J = 3,
    MATVEC_TAIL_P = 4,
    DOT_P = 0,
};

static void emit_reduce_ymm_to_xmm(struct toy_asm *as, uint8_t ymm) {
    toy_asm_vextractf128(as, TOY_XMM9, ymm, 1);
    toy_asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
    toy_asm_vshufps(as, TOY_XMM9, ymm, ymm, 0x4E);
    toy_asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
    toy_asm_vshufps(as, TOY_XMM9, ymm, ymm, 0xB1);
    toy_asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
}

static void emit_scalar_block8(struct toy_asm *as, uint32_t blocks, uint32_t cols, uint32_t k,
                               uint32_t bStride, uint32_t cStride,
                               uint8_t labelI, uint8_t labelJ, uint8_t labelP) {
    if (blocks == 0 || cols == 0) {
        return;
    }
    const int32_t bStep = (int32_t)(bStride * 4);
    const int32_t aBlockBytes = (int32_t)(8 * k * 4);
    const int32_t cBlockBytes = (int32_t)(8 * cStride * 4);

    toy_asm_mov_rr(as, TOY_R8, TOY_RDI);
    toy_asm_mov_rr(as, TOY_R9, TOY_RDX);
    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    toy_asm_label(as, labelI);
    toy_asm_xor_rr(as, TOY_RCX, TOY_RCX);
    toy_asm_label(as, labelJ);
    for (uint8_t r = 0; r < 8; r++) {
        toy_asm_vxorps(as, false, r, r, r);
    }
    toy_asm_lea(as, TOY_R10, toy_mem_index(TOY_RSI, TOY_RCX, 4, 0));
    toy_asm_xor_rr(as, TOY_R11, TOY_R11);
    if (k > 1) {
        toy_asm_label(as, labelP);
    }
    toy_asm_vmovss_load(as, TOY_XMM8, toy_mem_base(TOY_R10, 0));
    for (uint8_t r = 0; r < 8; r++) {
        toy_asm_vfmadd231ss_mem(as, r, TOY_XMM8,
                                toy_mem_index(TOY_R8, TOY_R11, 4, (int32_t)(r * k * 4)));
    }
    if (k > 1) {
        toy_asm_add_imm(as, TOY_R10, bStep);
        toy_asm_add_imm(as, TOY_R11, 1);
        toy_asm_cmp_imm(as, TOY_R11, (int32_t)k);
        toy_asm_jcc(as, TOY_CC_NE, labelP);
    }
    for (uint8_t r = 0; r < 8; r++) {
        toy_asm_vmovss_store(as, toy_mem_index(TOY_R9, TOY_RCX, 4, (int32_t)(r * cStride * 4)), r);
    }
    if (cols > 1) {
        toy_asm_add_imm(as, TOY_RCX, 1);
        toy_asm_cmp_imm(as, TOY_RCX, (int32_t)cols);
        toy_asm_jcc(as, TOY_CC_NE, labelJ);
    }
    if (blocks > 1) {
        toy_asm_add_imm(as, TOY_R8, aBlockBytes);
        toy_asm_add_imm(as, TOY_R9, cBlockBytes);
        toy_asm_add_imm(as, TOY_RAX, 1);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t)blocks);
        toy_asm_jcc(as, TOY_CC_NE, labelI);
    }
}

static void emit_scalar_region(struct toy_asm *as, uint32_t rows, uint32_t cols, uint32_t k,
                               uint32_t startRow, uint32_t startCol,
                               uint32_t bStride, uint32_t cStride,
                               uint8_t labelI, uint8_t labelJ, uint8_t labelP) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int32_t aRowBytes = (int32_t)(k * 4);
    const int32_t cRowBytes = (int32_t)(cStride * 4);
    const int32_t bStep = (int32_t)(bStride * 4);
    const int32_t cStep = (int32_t)((cStride - cols) * 4);

    // r8 = a + startRow * k
    toy_asm_mov_imm32(as, TOY_RAX, startRow);
    toy_asm_imul_imm(as, TOY_RAX, TOY_RAX, aRowBytes);
    toy_asm_lea(as, TOY_R8, toy_mem_index(TOY_RDI, TOY_RAX, 1, 0));
    // r10 = c + startRow * cStride + startCol
    toy_asm_mov_imm32(as, TOY_RAX, startRow);
    toy_asm_imul_imm(as, TOY_RAX, TOY_RAX, cRowBytes);
    toy_asm_lea(as, TOY_R10, toy_mem_index(TOY_RDX, TOY_RAX, 1, (int32_t)(startCol * 4)));

    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    toy_asm_label(as, labelI);
    toy_asm_xor_rr(as, TOY_RCX, TOY_RCX);
    toy_asm_label(as, labelJ);
    toy_asm_lea(as, TOY_R9, toy_mem_index(TOY_RSI, TOY_RCX, 4, (int32_t)(startCol * 4)));
    toy_asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    toy_asm_xor_rr(as, TOY_R11, TOY_R11);
    if (k > 1) {
        toy_asm_label(as, labelP);
    }
    toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_R8, TOY_R11, 4, 0));
    toy_asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1, toy_mem_base(TOY_R9, 0));
    if (k > 1) {
        toy_asm_add_imm(as, TOY_R9, bStep);
        toy_asm_add_imm(as, TOY_R11, 1);
        toy_asm_cmp_imm(as, TOY_R11, (int32_t)k);
        toy_asm_jcc(as, TOY_CC_NE, labelP);
    }
    toy_asm_vmovss_store(as, toy_mem_base(TOY_R10, 0), TOY_XMM0);
    toy_asm_add_imm(as, TOY_R10, 4);
    if (cols > 1) {
        toy_asm_add_imm(as, TOY_RCX, 1);
        toy_asm_cmp_imm(as, TOY_RCX, (int32_t)cols);
        toy_asm_jcc(as, TOY_CC_NE, labelJ);
    }
    if (rows > 1) {
        toy_asm_add_imm(as, TOY_R8, aRowBytes);
        if (cStep != 0) {
            toy_asm_add_imm(as, TOY_R10, cStep);
        }
        toy_asm_add_imm(as, TOY_RAX, 1);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t)rows);
        toy_asm_jcc(as, TOY_CC_NE, labelI);
    }
}

static void emit_avx2_gemm(struct toy_asm *as, uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t n4 = n & ~3u;
    const uint32_t m8 = m & ~7u;

    if (m8 != 0 && n4 != 0) {
        const uint32_t k4 = k & ~3u;
        const uint32_t kt = k & 3u;
        const uint32_t blocks = m8 / 8;

        toy_asm_mov_rr(as, TOY_R8, TOY_RDI);
        toy_asm_mov_rr(as, TOY_R9, TOY_RDX);
        toy_asm_xor_rr(as, TOY_R11, TOY_R11);
        toy_asm_label(as, LABEL_I);
        toy_asm_xor_rr(as, TOY_RCX, TOY_RCX);
        toy_asm_label(as, LABEL_J);
        for (uint8_t r = 0; r < 8; r++) {
            toy_asm_vxorps(as, true, r, r, r);
        }
        toy_asm_mov_rr(as, TOY_R10, TOY_RCX);
        toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
        if (k4 != 0) {
            toy_asm_label(as, LABEL_P);
            for (uint8_t q = 0; q < 4; q++) {
                toy_asm_vmovups_load_xmm(as, (uint8_t)(TOY_XMM8 + q),
                                         toy_mem_index(TOY_RSI, TOY_R10, 4, (int32_t)(q * n * 4)));
            }
            for (uint8_t q = 0; q < 4; q++) {
                for (uint8_t r = 0; r < 8; r++) {
                    toy_asm_vbroadcastss(as, TOY_YMM12,
                                         toy_mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4 + q * 4)));
                    toy_asm_vfmadd231ps(as, r, TOY_YMM12, (uint8_t)(TOY_YMM8 + q));
                }
            }
            toy_asm_add_imm(as, TOY_RAX, 4);
            toy_asm_add_imm(as, TOY_R10, (int32_t)(4 * n));
            toy_asm_cmp_imm(as, TOY_RAX, (int32_t)k4);
            toy_asm_jcc(as, TOY_CC_NE, LABEL_P);
        }
        for (uint32_t q = 0; q < kt; q++) {
            toy_asm_vmovups_load_xmm(as, TOY_XMM8,
                                     toy_mem_index(TOY_RSI, TOY_R10, 4, (int32_t)(q * n * 4)));
            for (uint8_t r = 0; r < 8; r++) {
                toy_asm_vbroadcastss(as, TOY_YMM12,
                                     toy_mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4 + q * 4)));
                toy_asm_vfmadd231ps(as, r, TOY_YMM12, TOY_YMM8);
            }
        }
        for (uint8_t r = 0; r < 8; r++) {
            toy_asm_vmovups_store_xmm(as, toy_mem_index(TOY_R9, TOY_RCX, 4, (int32_t)(r * n * 4)), r);
        }
        if (n4 > 4) {
            toy_asm_add_imm(as, TOY_RCX, 4);
            toy_asm_cmp_imm(as, TOY_RCX, (int32_t)n4);
            toy_asm_jcc(as, TOY_CC_NE, LABEL_J);
        }
        if (blocks > 1) {
            toy_asm_add_imm(as, TOY_R8, (int32_t)(8 * k * 4));
            toy_asm_add_imm(as, TOY_R9, (int32_t)(8 * n * 4));
            toy_asm_add_imm(as, TOY_R11, 1);
            toy_asm_cmp_imm(as, TOY_R11, (int32_t)blocks);
            toy_asm_jcc(as, TOY_CC_NE, LABEL_I);
        }
    }

    emit_scalar_region(as, m8, n - n4, k, 0, n4, n, n, 3, 4, 5);
    emit_scalar_region(as, m - m8, n, k, m8, 0, n, n, 6, 7, 8);
}

static void emit_avx2_matvec(struct toy_asm *as, uint32_t m, uint32_t k) {
    const uint32_t m8 = m & ~7u;
    const uint32_t k8 = k & ~7u;
    const uint32_t kt = k & 7u;

    if (m8 != 0) {
        const uint32_t blocks = m8 / 8;
        toy_asm_mov_rr(as, TOY_R8, TOY_RDI);
        toy_asm_mov_rr(as, TOY_R9, TOY_RDX);
        toy_asm_xor_rr(as, TOY_R10, TOY_R10);
        toy_asm_label(as, LABEL_I);
        for (uint8_t r = 0; r < 8; r++) {
            toy_asm_vxorps(as, true, r, r, r);
        }
        toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
        if (k8 != 0) {
            toy_asm_label(as, MATVEC_P);
            toy_asm_vmovups_load_ymm(as, TOY_YMM8, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
            for (uint8_t r = 0; r < 8; r++) {
                toy_asm_vfmadd231ps_mem(as, r, TOY_YMM8,
                                        toy_mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4)));
            }
            toy_asm_add_imm(as, TOY_RAX, 8);
            toy_asm_cmp_imm(as, TOY_RAX, (int32_t)k8);
            toy_asm_jcc(as, TOY_CC_NE, MATVEC_P);
        }
        for (uint8_t r = 0; r < 8; r++) {
            emit_reduce_ymm_to_xmm(as, r);
            for (uint32_t q = 0; q < kt; q++) {
                toy_asm_vmovss_load(as, TOY_XMM9, toy_mem_base(TOY_RSI, (int32_t)((k8 + q) * 4)));
                toy_asm_vfmadd231ss_mem(as, r, TOY_XMM9,
                                        toy_mem_base(TOY_R8, (int32_t)(r * k * 4 + (k8 + q) * 4)));
            }
            toy_asm_vmovss_store(as, toy_mem_base(TOY_R9, (int32_t)(r * 4)), r);
        }
        if (blocks > 1) {
            toy_asm_add_imm(as, TOY_R8, (int32_t)(8 * k * 4));
            toy_asm_add_imm(as, TOY_R9, 8 * 4);
            toy_asm_add_imm(as, TOY_R10, 1);
            toy_asm_cmp_imm(as, TOY_R10, (int32_t)blocks);
            toy_asm_jcc(as, TOY_CC_NE, LABEL_I);
        }
    }

    emit_scalar_region(as, m - m8, 1, k, m8, 0, 1, 1, MATVEC_TAIL_I, MATVEC_TAIL_J, MATVEC_TAIL_P);
}

static void emit_avx2_dot(struct toy_asm *as, uint32_t k) {
    const uint32_t k8 = k & ~7u;
    const uint32_t kt = k & 7u;

    toy_asm_vxorps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM0);
    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (k8 != 0) {
        toy_asm_label(as, DOT_P);
        toy_asm_vmovups_load_ymm(as, TOY_YMM8, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
        toy_asm_vfmadd231ps_mem(as, TOY_YMM0, TOY_YMM8, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
        toy_asm_add_imm(as, TOY_RAX, 8);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t)k8);
        toy_asm_jcc(as, TOY_CC_NE, DOT_P);
    }
    emit_reduce_ymm_to_xmm(as, TOY_XMM0);
    for (uint32_t q = 0; q < kt; q++) {
        toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_base(TOY_RDI, (int32_t)((k8 + q) * 4)));
        toy_asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1,
                                toy_mem_base(TOY_RSI, (int32_t)((k8 + q) * 4)));
    }
    toy_asm_vmovss_store(as, toy_mem_base(TOY_RDX, 0), TOY_XMM0);
}

static void emit_const_xmm(struct toy_asm *as, uint8_t xmm, uint32_t bits) {
    toy_asm_mov_imm32(as, TOY_RAX, bits);
    toy_asm_vmovd_gpr(as, xmm, TOY_RAX);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void emit_element_binary(struct toy_asm *as, uint32_t op, uint32_t count, bool wide,
                                uint32_t param_bits) {
    if (count == 0) {
        return;
    }
    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = count - (count % step);
    const bool scale = op == TOY_OP_SCALE;

    if (scale) {
        emit_const_xmm(as, TOY_XMM7, param_bits);
        if (wide) {
            toy_asm_vshufps(as, TOY_XMM7, TOY_XMM7, TOY_XMM7, 0x00);
            toy_asm_vinsertf128(as, TOY_YMM7, TOY_YMM7, TOY_XMM7, 1);
        }
    }

    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        toy_asm_label(as, 0);
        if (scale) {
            if (wide) {
                toy_asm_vmovups_load_ymm(as, TOY_YMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
                toy_asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM7);
                toy_asm_vmovups_store_ymm(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
            } else {
                toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
                toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
                toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
            }
        } else {
            if (wide) {
                toy_asm_vmovups_load_ymm(as, TOY_YMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
                toy_asm_vmovups_load_ymm(as, TOY_YMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
                toy_asm_vaddps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
                toy_asm_vmovups_store_ymm(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
            } else {
                toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
                toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
                toy_asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
                toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
            }
        }
        toy_asm_add_imm(as, TOY_RAX, (int32_t) step);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        toy_asm_jcc(as, TOY_CC_NE, 0);
    }
    if (main < count) {
        toy_asm_label(as, 1);
        if (scale) {
            toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
            toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
            toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        } else {
            toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
            toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
            toy_asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        }
        toy_asm_add_imm(as, TOY_RAX, 1);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t) count);
        toy_asm_jcc(as, TOY_CC_NE, 1);
    }
}

static void emit_rmsnorm(struct toy_asm *as, uint32_t n, bool wide, uint32_t eps_bits) {
    if (n == 0) {
        return;
    }

    toy_asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    toy_asm_label(as, 0);
    toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
    toy_asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
    toy_asm_add_imm(as, TOY_RAX, 1);
    toy_asm_cmp_imm(as, TOY_RAX, (int32_t) n);
    toy_asm_jcc(as, TOY_CC_NE, 0);

    // scale = 1 / sqrt(sumsq / n + eps)
    emit_const_xmm(as, TOY_XMM1, float_bits((float) n));
    toy_asm_vdivps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
    emit_const_xmm(as, TOY_XMM1, eps_bits);
    toy_asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
    toy_asm_vsqrtps(as, false, TOY_XMM0, TOY_XMM0);
    emit_const_xmm(as, TOY_XMM1, float_bits(1.0f));
    toy_asm_vdivps(as, false, TOY_XMM0, TOY_XMM1, TOY_XMM0);

    toy_asm_vshufps(as, TOY_XMM7, TOY_XMM0, TOY_XMM0, 0x00);
    if (wide) {
        toy_asm_vinsertf128(as, TOY_YMM7, TOY_YMM7, TOY_XMM7, 1);
    }

    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = n - (n % step);
    toy_asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        toy_asm_label(as, 1);
        if (wide) {
            toy_asm_vmovups_load_ymm(as, TOY_YMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
            toy_asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM7);
            toy_asm_vmovups_load_ymm(as, TOY_YMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
            toy_asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
            toy_asm_vmovups_store_ymm(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
        } else {
            toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
            toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
            toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
            toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        }
        toy_asm_add_imm(as, TOY_RAX, (int32_t) step);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        toy_asm_jcc(as, TOY_CC_NE, 1);
    }
    if (main < n) {
        toy_asm_label(as, 2);
        toy_asm_vmovss_load(as, TOY_XMM0, toy_mem_index(TOY_RDI, TOY_RAX, 4, 0));
        toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
        toy_asm_vmovss_load(as, TOY_XMM1, toy_mem_index(TOY_RSI, TOY_RAX, 4, 0));
        toy_asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
        toy_asm_vmovss_store(as, toy_mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        toy_asm_add_imm(as, TOY_RAX, 1);
        toy_asm_cmp_imm(as, TOY_RAX, (int32_t) n);
        toy_asm_jcc(as, TOY_CC_NE, 2);
    }
}

static bool spec_valid(const struct toy_spec *spec) {
    if (spec == NULL) {
        return false;
    }
    if (spec->op < TOY_OP_GEMM || spec->op > TOY_OP_RMSNORM) {
        return false;
    }
    if (spec->isa != TOY_ISA_SCALAR && spec->isa != TOY_ISA_AVX2) {
        return false;
    }
    if (spec->m == 0 || spec->n == 0 || spec->k == 0) {
        return false;
    }
    if (spec->op <= TOY_OP_DOT) {
        if (spec->m > 65535 || spec->n > 65535 || spec->k > 65535) {
            return false;
        }
        if (spec->op == TOY_OP_MATVEC && spec->n != 1) {
            return false;
        }
        if (spec->op == TOY_OP_DOT && (spec->m != 1 || spec->n != 1)) {
            return false;
        }
        const uint64_t m = spec->m, n = spec->n, k = spec->k;
        if (m * k * 4 > (uint64_t)INT32_MAX || m * n * 4 > (uint64_t)INT32_MAX ||
            n * k * 4 > (uint64_t)INT32_MAX || 8 * k * 4 > (uint64_t)INT32_MAX ||
            8 * n * 4 > (uint64_t)INT32_MAX) {
            return false;
        }
        return true;
    }
    if (spec->n != 1 || spec->k != 1) {
        return false;
    }
    if (spec->m > (uint32_t)(INT32_MAX / 4)) {
        return false;
    }
    return true;
}

size_t toy_emit(const struct toy_spec *spec, uint8_t *buf, size_t cap) {
    if (!spec_valid(spec) || buf == NULL || cap == 0) {
        return 0;
    }
    struct toy_asm as;
    toy_asm_init(&as, buf, cap);

    const bool wide = spec->isa == TOY_ISA_AVX2;
    switch (spec->op) {
        case TOY_OP_GEMM:
            if (wide) {
                emit_avx2_gemm(&as, spec->m, spec->n, spec->k);
            } else {
                emit_scalar_block8(&as, spec->m / 8, spec->n, spec->k, spec->n, spec->n, 0, 1, 2);
                emit_scalar_region(&as, spec->m % 8, spec->n, spec->k, (spec->m / 8) * 8, 0,
                                   spec->n, spec->n, 3, 4, 5);
            }
            break;
        case TOY_OP_MATVEC:
            if (wide) {
                emit_avx2_matvec(&as, spec->m, spec->k);
            } else {
                emit_scalar_block8(&as, spec->m / 8, 1, spec->k, 1, 1, 0, 1, 2);
                emit_scalar_region(&as, spec->m % 8, 1, spec->k, (spec->m / 8) * 8, 0,
                                   1, 1, 3, 4, 5);
            }
            break;
        case TOY_OP_DOT:
            if (wide) {
                emit_avx2_dot(&as, spec->k);
            } else {
                emit_scalar_region(&as, 1, 1, spec->k, 0, 0, 1, 1, 0, 1, 2);
            }
            break;
        case TOY_OP_ADD:
        case TOY_OP_SCALE:
            emit_element_binary(&as, spec->op, spec->m, wide, spec->param_bits);
            break;
        case TOY_OP_RMSNORM:
            emit_rmsnorm(&as, spec->m, wide, spec->param_bits);
            break;
    }
    toy_asm_vzeroupper(&as);
    toy_asm_ret(&as);
    if (!toy_asm_finish(&as)) {
        return 0;
    }
    return as.len;
}

const char *toy_op_name(uint32_t op) {
    switch (op) {
        case TOY_OP_GEMM: return "gemm";
        case TOY_OP_MATVEC: return "matvec";
        case TOY_OP_DOT: return "dot";
        case TOY_OP_ADD: return "add";
        case TOY_OP_SCALE: return "scale";
        case TOY_OP_RMSNORM: return "rmsnorm";
        default: return "unknown";
    }
}

const char *toy_isa_name(uint32_t isa) {
    switch (isa) {
        case TOY_ISA_SCALAR: return "scalar";
        case TOY_ISA_AVX2: return "avx2";
        default: return "unknown";
    }
}

double toy_spec_flops(const struct toy_spec *spec) {
    if (spec == NULL) {
        return 0.0;
    }
    switch (spec->op) {
        case TOY_OP_GEMM: return 2.0 * spec->m * spec->n * spec->k;
        case TOY_OP_MATVEC: return 2.0 * spec->m * spec->k;
        case TOY_OP_DOT: return 2.0 * spec->k;
        case TOY_OP_ADD: return 2.0 * spec->m;
        case TOY_OP_SCALE: return 1.0 * spec->m;
        case TOY_OP_RMSNORM: return 4.0 * spec->m;
        default: return 0.0;
    }
}

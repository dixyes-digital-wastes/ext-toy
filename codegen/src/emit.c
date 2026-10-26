#include "toy/codegen.h"

#include <float.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "toy/asm.h"
#include "toy/q8_0.h"

/*
 * The Windows x64 ABI differs from SysV: integer arguments arrive in rcx, rdx
 * and r8, and rdi, rsi and xmm6 to xmm15 are callee-saved. The generator keeps
 * the SysV register plan, so on this target a prologue moves the arguments and
 * saves those registers, and an epilogue restores them
 */
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
# define TOY_WIN64_ABI 1
#endif

#if defined(TOY_WIN64_ABI)
enum {
    TOY_WIN64_XMM_FIRST = 6,
    TOY_WIN64_XMM_LAST = 15,
    TOY_WIN64_STACK_BYTES = 160,
};

static void emit_win64_prologue(struct assembler *as) {
    asm_push_r64(as, TOY_RDI);
    asm_push_r64(as, TOY_RSI);
    asm_sub_imm(as, TOY_RSP, TOY_WIN64_STACK_BYTES);
    for (uint8_t r = TOY_WIN64_XMM_FIRST; r <= TOY_WIN64_XMM_LAST; r++) {
        const int32_t offset = (int32_t)((r - TOY_WIN64_XMM_FIRST) * 16);
        asm_vmovups_store_xmm(as, mem_base(TOY_RSP, offset), r);
    }
    asm_mov_rr(as, TOY_RDI, TOY_RCX);
    asm_mov_rr(as, TOY_RSI, TOY_RDX);
    asm_mov_rr(as, TOY_RDX, TOY_R8);
}

static void emit_win64_epilogue(struct assembler *as) {
    for (uint8_t r = TOY_WIN64_XMM_FIRST; r <= TOY_WIN64_XMM_LAST; r++) {
        const int32_t offset = (int32_t)((r - TOY_WIN64_XMM_FIRST) * 16);
        asm_vmovups_load_xmm(as, r, mem_base(TOY_RSP, offset));
    }
    asm_add_imm(as, TOY_RSP, TOY_WIN64_STACK_BYTES);
    asm_pop_r64(as, TOY_RSI);
    asm_pop_r64(as, TOY_RDI);
}
#endif

// A kernel uses at most 9 labels
enum {
    LABEL_I = 0,
    LABEL_J = 1,
    LABEL_P = 2,
    // The matvec main loop has a row label and a k label, the row tail has one k loop
    MATVEC_P = 1,
    MATVEC_TAIL_P = 2,
    // dot has one k loop
    DOT_P = 0,
};

static void emit_reduce_ymm_to_xmm(struct assembler *as, uint8_t ymm) {
    asm_vextractf128(as, TOY_XMM9, ymm, 1);
    asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
    asm_vshufps(as, TOY_XMM9, ymm, ymm, 0x4E);
    asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
    asm_vshufps(as, TOY_XMM9, ymm, ymm, 0xB1);
    asm_vaddps(as, false, ymm, ymm, TOY_XMM9);
}

/*
 * Scalar block of 8 rows: one k loop per column, xmm0-7 hold the accumulators
 * of the 8 rows
 *
 * The order inside a row is the same as the elementwise code, but 8 separate
 * FMA chains hide the latency
 */
static void emit_scalar_block8(struct assembler *as, uint32_t blocks, uint32_t cols, uint32_t k,
                               uint32_t bStride, uint32_t cStride,
                               uint8_t labelI, uint8_t labelJ, uint8_t labelP) {
    if (blocks == 0 || cols == 0) {
        return;
    }
    const int32_t bStep = (int32_t)(bStride * 4);
    const int32_t aBlockBytes = (int32_t)(8 * k * 4);
    const int32_t cBlockBytes = (int32_t)(8 * cStride * 4);

    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_mov_rr(as, TOY_R9, TOY_RDX);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    asm_label(as, labelI);
    asm_xor_rr(as, TOY_RCX, TOY_RCX);
    asm_label(as, labelJ);
    for (uint8_t r = 0; r < 8; r++) {
        asm_vxorps(as, false, r, r, r);
    }
    asm_lea(as, TOY_R10, mem_index(TOY_RSI, TOY_RCX, 4, 0));
    asm_xor_rr(as, TOY_R11, TOY_R11);
    if (k > 1) {
        asm_label(as, labelP);
    }
    asm_vmovss_load(as, TOY_XMM8, mem_base(TOY_R10, 0));
    for (uint8_t r = 0; r < 8; r++) {
        asm_vfmadd231ss_mem(as, r, TOY_XMM8,
                                mem_index(TOY_R8, TOY_R11, 4, (int32_t)(r * k * 4)));
    }
    if (k > 1) {
        asm_add_imm(as, TOY_R10, bStep);
        asm_add_imm(as, TOY_R11, 1);
        asm_cmp_imm(as, TOY_R11, (int32_t)k);
        asm_jcc(as, TOY_CC_NE, labelP);
    }
    for (uint8_t r = 0; r < 8; r++) {
        asm_vmovss_store(as, mem_index(TOY_R9, TOY_RCX, 4, (int32_t)(r * cStride * 4)), r);
    }
    if (cols > 1) {
        asm_add_imm(as, TOY_RCX, 1);
        asm_cmp_imm(as, TOY_RCX, (int32_t)cols);
        asm_jcc(as, TOY_CC_NE, labelJ);
    }
    if (blocks > 1) {
        asm_add_imm(as, TOY_R8, aBlockBytes);
        asm_add_imm(as, TOY_R9, cBlockBytes);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t)blocks);
        asm_jcc(as, TOY_CC_NE, labelI);
    }
}

/*
 * Scalar region: sequential accumulation, each output sums over k in
 * increasing order
 *
 * The input is three pointers, the output goes to
 * C + (startRow + i) * cStride + startCol + j
 * bStride is the row stride of B along k, cStride is the row stride of C
 * All counts are generation-time constants, the loop bounds are immediates
 */
static void emit_scalar_region(struct assembler *as, uint32_t rows, uint32_t cols, uint32_t k,
                               uint32_t startRow, uint32_t startCol,
                               uint32_t bStride, uint32_t cStride,
                               uint8_t labelI, uint8_t labelJ, uint8_t labelP) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int32_t aRowBytes = (int32_t)(k * 4);
    const int32_t cRowBytes = (int32_t)(cStride * 4);
    const int32_t bStep = (int32_t)(bStride * 4);
    // When the j loop ends r10 points at startCol + cols, so step to the same column of the next row
    const int32_t cStep = (int32_t)((cStride - cols) * 4);

    // r8 = a + startRow * k
    asm_mov_imm32(as, TOY_RAX, startRow);
    asm_imul_imm(as, TOY_RAX, TOY_RAX, aRowBytes);
    asm_lea(as, TOY_R8, mem_index(TOY_RDI, TOY_RAX, 1, 0));
    // r10 = c + startRow * cStride + startCol
    asm_mov_imm32(as, TOY_RAX, startRow);
    asm_imul_imm(as, TOY_RAX, TOY_RAX, cRowBytes);
    asm_lea(as, TOY_R10, mem_index(TOY_RDX, TOY_RAX, 1, (int32_t)(startCol * 4)));

    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    asm_label(as, labelI);
    asm_xor_rr(as, TOY_RCX, TOY_RCX);
    asm_label(as, labelJ);
    asm_lea(as, TOY_R9, mem_index(TOY_RSI, TOY_RCX, 4, (int32_t)(startCol * 4)));
    asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    asm_xor_rr(as, TOY_R11, TOY_R11);
    if (k > 1) {
        asm_label(as, labelP);
    }
    asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_R8, TOY_R11, 4, 0));
    asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1, mem_base(TOY_R9, 0));
    if (k > 1) {
        asm_add_imm(as, TOY_R9, bStep);
        asm_add_imm(as, TOY_R11, 1);
        asm_cmp_imm(as, TOY_R11, (int32_t)k);
        asm_jcc(as, TOY_CC_NE, labelP);
    }
    asm_vmovss_store(as, mem_base(TOY_R10, 0), TOY_XMM0);
    asm_add_imm(as, TOY_R10, 4);
    if (cols > 1) {
        asm_add_imm(as, TOY_RCX, 1);
        asm_cmp_imm(as, TOY_RCX, (int32_t)cols);
        asm_jcc(as, TOY_CC_NE, labelJ);
    }
    if (rows > 1) {
        asm_add_imm(as, TOY_R8, aRowBytes);
        if (cStep != 0) {
            asm_add_imm(as, TOY_R10, cStep);
        }
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t)rows);
        asm_jcc(as, TOY_CC_NE, labelI);
    }
}

/*
 * AVX2 GEMM: 8 rows x 4 columns x depth 4
 *
 * ymm0-7 hold the accumulators of the 8 rows, each uses the low 4 lanes only
 * ymm8-11 hold 4 column vectors of B, ymm12 broadcasts one element of A
 * Each output accumulates over k in increasing order, so it matches the scalar
 * reference bit for bit
 * The row tail and the column tail are known shapes, the scalar region handles
 * them
 */
static void emit_avx2_gemm(struct assembler *as, uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t n4 = n & ~3u;
    const uint32_t m8 = m & ~7u;

    if (m8 != 0 && n4 != 0) {
        const uint32_t k4 = k & ~3u;
        const uint32_t kt = k & 3u;
        const uint32_t blocks = m8 / 8;

        // i is the outer loop: the A panel of 8 rows (8 x k) stays in L2 and serves all j blocks
        // j is the inner loop, B streams through in column blocks
        asm_mov_rr(as, TOY_R8, TOY_RDI);
        asm_mov_rr(as, TOY_R9, TOY_RDX);
        asm_xor_rr(as, TOY_R11, TOY_R11);
        asm_label(as, LABEL_I);
        asm_xor_rr(as, TOY_RCX, TOY_RCX);
        asm_label(as, LABEL_J);
        for (uint8_t r = 0; r < 8; r++) {
            asm_vxorps(as, true, r, r, r);
        }
        // r10 = j + p * n, in elements
        asm_mov_rr(as, TOY_R10, TOY_RCX);
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        if (k4 != 0) {
            asm_label(as, LABEL_P);
            for (uint8_t q = 0; q < 4; q++) {
                asm_vmovups_load_xmm(as, (uint8_t)(TOY_XMM8 + q),
                                         mem_index(TOY_RSI, TOY_R10, 4, (int32_t)(q * n * 4)));
            }
            // q is the outer loop, so 7 independent instructions separate two FMAs on one accumulator
            for (uint8_t q = 0; q < 4; q++) {
                for (uint8_t r = 0; r < 8; r++) {
                    asm_vbroadcastss(as, TOY_YMM12,
                                         mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4 + q * 4)));
                    asm_vfmadd231ps(as, r, TOY_YMM12, (uint8_t)(TOY_YMM8 + q));
                }
            }
            asm_add_imm(as, TOY_RAX, 4);
            asm_add_imm(as, TOY_R10, (int32_t)(4 * n));
            asm_cmp_imm(as, TOY_RAX, (int32_t)k4);
            asm_jcc(as, TOY_CC_NE, LABEL_P);
        }
        for (uint32_t q = 0; q < kt; q++) {
            asm_vmovups_load_xmm(as, TOY_XMM8,
                                     mem_index(TOY_RSI, TOY_R10, 4, (int32_t)(q * n * 4)));
            for (uint8_t r = 0; r < 8; r++) {
                asm_vbroadcastss(as, TOY_YMM12,
                                     mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4 + q * 4)));
                asm_vfmadd231ps(as, r, TOY_YMM12, TOY_YMM8);
            }
        }
        for (uint8_t r = 0; r < 8; r++) {
            asm_vmovups_store_xmm(as, mem_index(TOY_R9, TOY_RCX, 4, (int32_t)(r * n * 4)), r);
        }
        if (n4 > 4) {
            asm_add_imm(as, TOY_RCX, 4);
            asm_cmp_imm(as, TOY_RCX, (int32_t)n4);
            asm_jcc(as, TOY_CC_NE, LABEL_J);
        }
        if (blocks > 1) {
            asm_add_imm(as, TOY_R8, (int32_t)(8 * k * 4));
            asm_add_imm(as, TOY_R9, (int32_t)(8 * n * 4));
            asm_add_imm(as, TOY_R11, 1);
            asm_cmp_imm(as, TOY_R11, (int32_t)blocks);
            asm_jcc(as, TOY_CC_NE, LABEL_I);
        }
    }

    // Column tail: the part of the first m8 rows where j >= n4
    emit_scalar_region(as, m8, n - n4, k, 0, n4, n, n, 3, 4, 5);
    // Row tail: rows where every j falls after m8
    emit_scalar_region(as, m - m8, n, k, m8, 0, n, n, 6, 7, 8);
}

/*
 * AVX2 matvec row tail: at most 7 rows, one ymm accumulator per row
 *
 * The lane mapping, the horizontal sum and the k tail are the same as in the
 * 8-row group, so a row accumulates in one order for every m and every row
 * offset
 */
static void emit_avx2_matvec_tail(struct assembler *as, uint32_t rows, uint32_t k,
                                  uint32_t startRow) {
    if (rows == 0) {
        return;
    }
    const uint32_t k8 = k & ~7u;
    const uint32_t kt = k & 7u;
    const int32_t rowBytes = (int32_t)(k * 4);

    // r8 = a + startRow * k, r9 = c + startRow
    asm_mov_imm32(as, TOY_RAX, startRow);
    asm_imul_imm(as, TOY_RAX, TOY_RAX, rowBytes);
    asm_lea(as, TOY_R8, mem_index(TOY_RDI, TOY_RAX, 1, 0));
    asm_lea(as, TOY_R9, mem_base(TOY_RDX, (int32_t)(startRow * 4)));

    for (uint8_t r = 0; r < rows; r++) {
        asm_vxorps(as, true, r, r, r);
    }
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (k8 != 0) {
        asm_label(as, MATVEC_TAIL_P);
        asm_vmovups_load_ymm(as, TOY_YMM8, mem_index(TOY_RSI, TOY_RAX, 4, 0));
        for (uint8_t r = 0; r < rows; r++) {
            asm_vfmadd231ps_mem(as, r, TOY_YMM8,
                                    mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * rowBytes)));
        }
        asm_add_imm(as, TOY_RAX, 8);
        asm_cmp_imm(as, TOY_RAX, (int32_t)k8);
        asm_jcc(as, TOY_CC_NE, MATVEC_TAIL_P);
    }
    for (uint8_t r = 0; r < rows; r++) {
        emit_reduce_ymm_to_xmm(as, r);
        for (uint32_t q = 0; q < kt; q++) {
            asm_vmovss_load(as, TOY_XMM9, mem_base(TOY_RSI, (int32_t)((k8 + q) * 4)));
            asm_vfmadd231ss_mem(as, r, TOY_XMM9,
                                    mem_base(TOY_R8, (int32_t)(r * rowBytes + (k8 + q) * 4)));
        }
        asm_vmovss_store(as, mem_base(TOY_R9, (int32_t)(r * 4)), r);
    }
}

/*
 * AVX2 matvec: 8 rows per group, one ymm accumulator per row, 8 steps along k
 * A lane holds a subsequence of k with step 8, a horizontal sum at the end
 * handles the k tail
 *
 * The row tail uses the same scheme instead of the scalar region, so the
 * result of a row does not depend on m or on its offset inside the matrix
 */
static void emit_avx2_matvec(struct assembler *as, uint32_t m, uint32_t k) {
    const uint32_t m8 = m & ~7u;
    const uint32_t k8 = k & ~7u;
    const uint32_t kt = k & 7u;

    if (m8 != 0) {
        const uint32_t blocks = m8 / 8;
        asm_mov_rr(as, TOY_R8, TOY_RDI);
        asm_mov_rr(as, TOY_R9, TOY_RDX);
        asm_xor_rr(as, TOY_R10, TOY_R10);
        asm_label(as, LABEL_I);
        for (uint8_t r = 0; r < 8; r++) {
            asm_vxorps(as, true, r, r, r);
        }
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        if (k8 != 0) {
            asm_label(as, MATVEC_P);
            asm_vmovups_load_ymm(as, TOY_YMM8, mem_index(TOY_RSI, TOY_RAX, 4, 0));
            for (uint8_t r = 0; r < 8; r++) {
                asm_vfmadd231ps_mem(as, r, TOY_YMM8,
                                        mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * k * 4)));
            }
            asm_add_imm(as, TOY_RAX, 8);
            asm_cmp_imm(as, TOY_RAX, (int32_t)k8);
            asm_jcc(as, TOY_CC_NE, MATVEC_P);
        }
        for (uint8_t r = 0; r < 8; r++) {
            emit_reduce_ymm_to_xmm(as, r);
            for (uint32_t q = 0; q < kt; q++) {
                asm_vmovss_load(as, TOY_XMM9, mem_base(TOY_RSI, (int32_t)((k8 + q) * 4)));
                asm_vfmadd231ss_mem(as, r, TOY_XMM9,
                                        mem_base(TOY_R8, (int32_t)(r * k * 4 + (k8 + q) * 4)));
            }
            asm_vmovss_store(as, mem_base(TOY_R9, (int32_t)(r * 4)), r);
        }
        if (blocks > 1) {
            asm_add_imm(as, TOY_R8, (int32_t)(8 * k * 4));
            asm_add_imm(as, TOY_R9, 8 * 4);
            asm_add_imm(as, TOY_R10, 1);
            asm_cmp_imm(as, TOY_R10, (int32_t)blocks);
            asm_jcc(as, TOY_CC_NE, LABEL_I);
        }
    }

    emit_avx2_matvec_tail(as, m - m8, k, m8);
}

// AVX2 dot: one ymm accumulator, 8 steps along k, horizontal sum at the end
static void emit_avx2_dot(struct assembler *as, uint32_t k) {
    const uint32_t k8 = k & ~7u;
    const uint32_t kt = k & 7u;

    asm_vxorps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM0);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (k8 != 0) {
        asm_label(as, DOT_P);
        asm_vmovups_load_ymm(as, TOY_YMM8, mem_index(TOY_RDI, TOY_RAX, 4, 0));
        asm_vfmadd231ps_mem(as, TOY_YMM0, TOY_YMM8, mem_index(TOY_RSI, TOY_RAX, 4, 0));
        asm_add_imm(as, TOY_RAX, 8);
        asm_cmp_imm(as, TOY_RAX, (int32_t)k8);
        asm_jcc(as, TOY_CC_NE, DOT_P);
    }
    emit_reduce_ymm_to_xmm(as, TOY_XMM0);
    for (uint32_t q = 0; q < kt; q++) {
        asm_vmovss_load(as, TOY_XMM1, mem_base(TOY_RDI, (int32_t)((k8 + q) * 4)));
        asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1,
                                mem_base(TOY_RSI, (int32_t)((k8 + q) * 4)));
    }
    asm_vmovss_store(as, mem_base(TOY_RDX, 0), TOY_XMM0);
}

// Writes a generation-time constant into the low lane of xmm
static void emit_const_xmm(struct assembler *as, uint8_t xmm, uint32_t bits) {
    asm_mov_imm32(as, TOY_RAX, bits);
    asm_vmovd_gpr(as, xmm, TOY_RAX);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/*
 * ADD:   c[i] = a[i] + b[i]
 * MUL:   c[i] = a[i] * b[i]
 * SCALE: c[i] = a[i] * param
 *
 * wide selects 8 lanes of AVX2, otherwise the code works on one element. The
 * tail count is a generation-time constant
 */
static void emit_element_binary(struct assembler *as, uint32_t op, uint32_t count, bool wide,
                                uint32_t param_bits) {
    if (count == 0) {
        return;
    }
    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = count - (count % step);
    const bool scale = op == TOY_OP_SCALE;
    const bool mul = op == TOY_OP_MUL;

    if (scale) {
        // Broadcast the factor into xmm7 or ymm7
        emit_const_xmm(as, TOY_XMM7, param_bits);
        if (wide) {
            asm_vshufps(as, TOY_XMM7, TOY_XMM7, TOY_XMM7, 0x00);
            asm_vinsertf128(as, TOY_YMM7, TOY_YMM7, TOY_XMM7, 1);
        }
    }

    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        asm_label(as, 0);
        if (scale) {
            if (wide) {
                asm_vmovups_load_ymm(as, TOY_YMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
                asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM7);
                asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
            } else {
                asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
                asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
                asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
            }
        } else {
            if (wide) {
                asm_vmovups_load_ymm(as, TOY_YMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
                asm_vmovups_load_ymm(as, TOY_YMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
                if (mul) {
                    asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
                } else {
                    asm_vaddps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
                }
                asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
            } else {
                asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
                asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
                if (mul) {
                    asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
                } else {
                    asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
                }
                asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
            }
        }
        asm_add_imm(as, TOY_RAX, (int32_t) step);
        asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        asm_jcc(as, TOY_CC_NE, 0);
    }
    if (main < count) {
        asm_label(as, 1);
        if (scale) {
            asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
            asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
            asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        } else {
            asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
            asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
            if (mul) {
                asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            } else {
                asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            }
            asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        }
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) count);
        asm_jcc(as, TOY_CC_NE, 1);
    }
}

/*
 * RMSNORM: out[i] = (x[i] * scale) * w[i], scale = 1 / sqrt(sum(x^2) / n + eps)
 *
 * The sum of squares is one chain in element order, so it matches the scalar
 * reference bit for bit
 * wide applies to the output loop only
 */
static void emit_rmsnorm(struct assembler *as, uint32_t n, bool wide, uint32_t eps_bits) {
    if (n == 0) {
        return;
    }

    asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    asm_label(as, 0);
    asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RDI, TOY_RAX, 4, 0));
    asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1, mem_index(TOY_RDI, TOY_RAX, 4, 0));
    asm_add_imm(as, TOY_RAX, 1);
    asm_cmp_imm(as, TOY_RAX, (int32_t) n);
    asm_jcc(as, TOY_CC_NE, 0);

    // scale = 1 / sqrt(sumsq / n + eps)
    emit_const_xmm(as, TOY_XMM1, float_bits((float) n));
    asm_vdivps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
    emit_const_xmm(as, TOY_XMM1, eps_bits);
    asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
    asm_vsqrtps(as, false, TOY_XMM0, TOY_XMM0);
    emit_const_xmm(as, TOY_XMM1, float_bits(1.0f));
    asm_vdivps(as, false, TOY_XMM0, TOY_XMM1, TOY_XMM0);

    asm_vshufps(as, TOY_XMM7, TOY_XMM0, TOY_XMM0, 0x00);
    if (wide) {
        asm_vinsertf128(as, TOY_YMM7, TOY_YMM7, TOY_XMM7, 1);
    }

    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = n - (n % step);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        asm_label(as, 1);
        if (wide) {
            asm_vmovups_load_ymm(as, TOY_YMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
            asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM7);
            asm_vmovups_load_ymm(as, TOY_YMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
            asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
            asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_YMM0);
        } else {
            asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
            asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
            asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
            asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        }
        asm_add_imm(as, TOY_RAX, (int32_t) step);
        asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        asm_jcc(as, TOY_CC_NE, 1);
    }
    if (main < n) {
        asm_label(as, 2);
        asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_RDI, TOY_RAX, 4, 0));
        asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM7);
        asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
        asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
        asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) n);
        asm_jcc(as, TOY_CC_NE, 2);
    }
}

/* ---------- element-wise transcendentals ---------- */

/*
 * x86 has no exp or log instruction, so the kernels below build both from a
 * polynomial. exp takes the integer part of x * log2(e) out through the float
 * exponent bits and log takes the exponent of its argument out the same way.
 * One register holds the constants, which are rebuilt from an immediate
 * wherever they are used, so every kernel below keeps a fixed register plan
 */
enum {
    MATH_X = 0,
    MATH_R = 1,
    MATH_N = 2,
    MATH_P = 3,
    MATH_C = 4,
    MATH_S = 5,
};

static void emit_const_vec_bits(struct assembler *as, bool wide, uint8_t reg, uint32_t bits) {
    asm_mov_imm32(as, TOY_R11, bits);
    asm_vmovd_gpr(as, reg, TOY_R11);
    if (wide) {
        asm_vbroadcastss_reg(as, reg, reg);
    }
}

static void emit_const_vec(struct assembler *as, bool wide, uint8_t reg, float value) {
    emit_const_vec_bits(as, wide, reg, float_bits(value));
}

/*
 * exp(x) with x in MATH_X, the result in MATH_N, MATH_R, MATH_P and MATH_C as
 * scratch
 *
 * The input itself is clamped, so the exponent keeps the power of two a
 * normal f32 and the reduced argument stays small: a very negative input
 * underflows to zero and a very positive one saturates instead of turning
 * into a float with another sign. r = x - n * ln2 stays within ln2 / 2 and
 * the degree 7 Taylor series there is below one f32 ulp. MATH_X holds the
 * clamped input afterwards
 */
static void emit_exp_vec(struct assembler *as, bool wide) {
    emit_const_vec(as, wide, MATH_C, -88.0f);
    asm_vmaxps(as, wide, MATH_X, MATH_X, MATH_C);
    emit_const_vec(as, wide, MATH_C, 88.0f);
    asm_vminps(as, wide, MATH_X, MATH_X, MATH_C);

    emit_const_vec(as, wide, MATH_C, 1.4426950408889634f);
    asm_vmulps(as, wide, MATH_R, MATH_X, MATH_C);

    // n = round(t) through the float mantissa
    emit_const_vec(as, wide, MATH_C, 12582912.0f);
    asm_vaddps(as, wide, MATH_N, MATH_R, MATH_C);
    asm_vsubps(as, wide, MATH_N, MATH_N, MATH_C);

    // r = x - n * ln2 in two steps, each product is exact
    emit_const_vec(as, wide, MATH_C, 0.693145751953125f);
    asm_vmulps(as, wide, MATH_P, MATH_N, MATH_C);
    asm_vsubps(as, wide, MATH_P, MATH_X, MATH_P);
    emit_const_vec(as, wide, MATH_C, 1.428606765330187045e-06f);
    asm_vmulps(as, wide, MATH_R, MATH_N, MATH_C);
    asm_vsubps(as, wide, MATH_P, MATH_P, MATH_R);

    // 2^n from the exponent bits
    asm_vcvttps2dq(as, wide, MATH_R, MATH_N);
    asm_vpslld(as, wide, MATH_R, MATH_R, 23);
    emit_const_vec_bits(as, wide, MATH_C, 0x3F800000u);
    asm_vpaddd(as, wide, MATH_R, MATH_R, MATH_C);

    // Horner from c7 down to c0, with the coefficient as the FMA addend
    asm_vxorps(as, wide, MATH_N, MATH_N, MATH_N);
    emit_const_vec(as, wide, MATH_C, 1.0f / 5040.0f);
    asm_vaddps(as, wide, MATH_N, MATH_N, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 720.0f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 120.0f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 24.0f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 6.0f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    emit_const_vec(as, wide, MATH_C, 0.5f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);
    asm_vfmadd213ps(as, wide, MATH_N, MATH_P, MATH_C);

    asm_vmulps(as, wide, MATH_N, MATH_N, MATH_R);
}

/*
 * sigmoid(x) = e / (1 + e) with e = exp(x), which keeps every intermediate
 * finite and needs no second exponential. The result lands in MATH_X
 */
static void emit_sigmoid_vec(struct assembler *as, bool wide) {
    emit_exp_vec(as, wide);
    emit_const_vec(as, wide, MATH_C, 1.0f);
    asm_vaddps(as, wide, MATH_P, MATH_N, MATH_C);
    asm_vdivps(as, wide, MATH_X, MATH_N, MATH_P);
}

/* silu(x) = x * sigmoid(x), the result lands in MATH_X */
static void emit_silu_vec(struct assembler *as, bool wide) {
    // exp clamps its input, so keep x for the final multiply
    asm_vxorps(as, wide, MATH_S, MATH_S, MATH_S);
    asm_vaddps(as, wide, MATH_S, MATH_S, MATH_X);
    emit_exp_vec(as, wide);
    emit_const_vec(as, wide, MATH_C, 1.0f);
    asm_vaddps(as, wide, MATH_P, MATH_N, MATH_C);
    asm_vdivps(as, wide, MATH_N, MATH_N, MATH_P);
    asm_vmulps(as, wide, MATH_X, MATH_N, MATH_S);
}

/*
 * softplus(x) = log1p(exp(min(x, 20))), then the larger of that and x
 *
 * Above 20 the exponential would overflow, so the clamp saturates the
 * logarithm near 20 instead, and the maximum restores softplus(x) == x, which
 * holds in f32 from 20 on. The logarithm splits u = 1 + e into its exponent
 * and its mantissa and takes the atanh series of the mantissa. The result
 * lands in MATH_X
 */
static void emit_softplus_vec(struct assembler *as, bool wide) {
    asm_vxorps(as, wide, MATH_S, MATH_S, MATH_S);
    asm_vaddps(as, wide, MATH_S, MATH_S, MATH_X);
    emit_const_vec(as, wide, MATH_C, 20.0f);
    asm_vminps(as, wide, MATH_X, MATH_X, MATH_C);
    emit_exp_vec(as, wide);

    // u = 1 + e, then the exponent and the mantissa bits of u. The integer
    // lane operations read the float bits, so u needs no conversion
    emit_const_vec(as, wide, MATH_C, 1.0f);
    asm_vaddps(as, wide, MATH_P, MATH_N, MATH_C);
    asm_vpsrld(as, wide, MATH_N, MATH_P, 23);
    emit_const_vec_bits(as, wide, MATH_C, 127u);
    asm_vpsubd(as, wide, MATH_N, MATH_N, MATH_C);
    emit_const_vec_bits(as, wide, MATH_C, 0x007FFFFFu);
    asm_vpand(as, wide, MATH_R, MATH_P, MATH_C);
    emit_const_vec_bits(as, wide, MATH_C, 0x3F800000u);
    asm_vpor(as, wide, MATH_R, MATH_R, MATH_C);

    // z = (m - 1) / (m + 1), log(m) = 2 * z * (1 + z^2 * q)
    asm_vsubps(as, wide, MATH_P, MATH_R, MATH_C);
    asm_vaddps(as, wide, MATH_R, MATH_R, MATH_C);
    asm_vdivps(as, wide, MATH_P, MATH_P, MATH_R);
    asm_vmulps(as, wide, MATH_R, MATH_P, MATH_P);

    asm_vxorps(as, wide, MATH_X, MATH_X, MATH_X);
    emit_const_vec(as, wide, MATH_C, 1.0f / 11.0f);
    asm_vaddps(as, wide, MATH_X, MATH_X, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 9.0f);
    asm_vfmadd213ps(as, wide, MATH_X, MATH_R, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 7.0f);
    asm_vfmadd213ps(as, wide, MATH_X, MATH_R, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 5.0f);
    asm_vfmadd213ps(as, wide, MATH_X, MATH_R, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f / 3.0f);
    asm_vfmadd213ps(as, wide, MATH_X, MATH_R, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.0f);
    asm_vfmadd213ps(as, wide, MATH_X, MATH_R, MATH_C);
    asm_vmulps(as, wide, MATH_X, MATH_X, MATH_P);
    asm_vaddps(as, wide, MATH_X, MATH_X, MATH_X);

    // log(u) = e * ln2 + log(m), then the larger of that and x
    asm_vcvtdq2ps(as, wide, MATH_N, MATH_N);
    emit_const_vec(as, wide, MATH_C, 0.693145751953125f);
    asm_vmulps(as, wide, MATH_R, MATH_N, MATH_C);
    emit_const_vec(as, wide, MATH_C, 1.428606765330187045e-06f);
    asm_vmulps(as, wide, MATH_P, MATH_N, MATH_C);
    asm_vaddps(as, wide, MATH_R, MATH_R, MATH_P);
    asm_vaddps(as, wide, MATH_X, MATH_X, MATH_R);
    asm_vmaxps(as, wide, MATH_X, MATH_X, MATH_S);
}

/*
 * Unary math over m elements, eight at a time and then one at a time for the
 * tail. The count is a generation-time constant
 */
static void emit_unary_math(struct assembler *as, uint32_t op, uint32_t count, bool wide) {
    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = count - (count % step);

    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        asm_label(as, 0);
        if (wide) {
            asm_vmovups_load_ymm(as, MATH_X, mem_index(TOY_RDI, TOY_RAX, 4, 0));
        } else {
            asm_vmovss_load(as, MATH_X, mem_index(TOY_RDI, TOY_RAX, 4, 0));
        }
        if (op == TOY_OP_SIGMOID) {
            emit_sigmoid_vec(as, wide);
        } else {
            emit_softplus_vec(as, wide);
        }
        if (wide) {
            asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        } else {
            asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        }
        asm_add_imm(as, TOY_RAX, (int32_t) step);
        asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        asm_jcc(as, TOY_CC_NE, 0);
    }
    if (main < count) {
        // The tail always handles one element
        asm_label(as, 1);
        asm_vmovss_load(as, MATH_X, mem_index(TOY_RDI, TOY_RAX, 4, 0));
        if (op == TOY_OP_SIGMOID) {
            emit_sigmoid_vec(as, false);
        } else {
            emit_softplus_vec(as, false);
        }
        asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) count);
        asm_jcc(as, TOY_CC_NE, 1);
    }
}

/*
 * The row norms below share one kernel body. The sum of squares stays one
 * scalar fma chain per row, in element order, so the kernel matches the
 * reference bit for bit. Four rows run at the same time, which hides the FMA
 * latency without changing the order of any single sum. The scale of a row
 * comes from its own sum alone, and the output loop is vectorized
 *
 * L2NORM divides by sqrt(sum + eps), RMSNORM_ROWS by sqrt(sum / k + eps) and
 * then multiplies every row by the same weight row. m is the row count and k
 * the row width
 */
enum {
    ROW_NORM_ROWS = 4,
    ROW_NORM_GROUP = 0,
    ROW_NORM_SUM = 1,
    ROW_NORM_SCALE = 8,
    ROW_NORM_TAIL_LOOP = 10,
    ROW_NORM_TAIL_SUM = 11,
    ROW_NORM_TAIL_MAIN = 12,
    ROW_NORM_TAIL_TAIL = 13,
};

/* scale = 1 / sqrt(sum [ / width] + eps), copied into every lane of dst */
static void emit_row_scale(struct assembler *as, uint8_t sum, uint8_t dst, bool wide,
                           uint32_t width, uint32_t eps_bits, bool mean) {
    if (mean) {
        emit_const_xmm(as, TOY_XMM4, float_bits((float) width));
        asm_vdivps(as, false, sum, sum, TOY_XMM4);
    }
    emit_const_xmm(as, TOY_XMM4, eps_bits);
    asm_vaddps(as, false, sum, sum, TOY_XMM4);
    asm_vsqrtps(as, false, sum, sum);
    emit_const_xmm(as, TOY_XMM4, float_bits(1.0f));
    asm_vdivps(as, false, sum, TOY_XMM4, sum);
    asm_vshufps(as, dst, sum, sum, 0x00);
    if (wide) {
        asm_vinsertf128(as, dst, dst, dst, 1);
    }
}

/*
 * The output loop of one row: out = x * scale, then times the weight row in
 * the rmsnorm case. The weight pointer is b, in rsi
 */
static void emit_row_norm_out(struct assembler *as, uint8_t scale, bool wide, uint32_t width,
                              int32_t row_bytes, uint8_t main_label, uint8_t tail_label,
                              bool weighted) {
    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = width - (width % step);

    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        asm_label(as, main_label);
        if (wide) {
            asm_vmovups_load_ymm(as, TOY_YMM0, mem_index(TOY_R8, TOY_RAX, 4, row_bytes));
            asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, scale);
            if (weighted) {
                asm_vmovups_load_ymm(as, TOY_YMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
                asm_vmulps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
            }
            asm_vmovups_store_ymm(as, mem_index(TOY_R9, TOY_RAX, 4, row_bytes), TOY_YMM0);
        } else {
            asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_R8, TOY_RAX, 4, row_bytes));
            asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, scale);
            if (weighted) {
                asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
                asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
            }
            asm_vmovss_store(as, mem_index(TOY_R9, TOY_RAX, 4, row_bytes), TOY_XMM0);
        }
        asm_add_imm(as, TOY_RAX, (int32_t) step);
        asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        asm_jcc(as, TOY_CC_NE, main_label);
    }
    if (main < width) {
        asm_label(as, tail_label);
        asm_vmovss_load(as, TOY_XMM0, mem_index(TOY_R8, TOY_RAX, 4, row_bytes));
        asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, scale);
        if (weighted) {
            asm_vmovss_load(as, TOY_XMM1, mem_index(TOY_RSI, TOY_RAX, 4, 0));
            asm_vmulps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM1);
        }
        asm_vmovss_store(as, mem_index(TOY_R9, TOY_RAX, 4, row_bytes), TOY_XMM0);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) width);
        asm_jcc(as, TOY_CC_NE, tail_label);
    }
}

static void emit_row_norm(struct assembler *as, uint32_t rows, uint32_t width, bool wide,
                          uint32_t eps_bits, bool mean, bool weighted) {
    const uint32_t groups = rows / ROW_NORM_ROWS;
    const uint32_t tail = rows - groups * ROW_NORM_ROWS;
    const int32_t row_bytes = (int32_t)(width * 4);
    const int32_t group_bytes = (int32_t)(ROW_NORM_ROWS * width * 4);

    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_mov_rr(as, TOY_R9, TOY_RDX);
    asm_xor_rr(as, TOY_R11, TOY_R11);

    if (groups != 0) {
        asm_label(as, ROW_NORM_GROUP);
        for (uint8_t r = 0; r < ROW_NORM_ROWS; r++) {
            asm_vxorps(as, false, r, r, r);
        }
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        asm_label(as, ROW_NORM_SUM);
        for (uint8_t r = 0; r < ROW_NORM_ROWS; r++) {
            asm_vmovss_load(as, TOY_XMM4,
                            mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * row_bytes)));
            asm_vfmadd231ss_mem(as, r, TOY_XMM4,
                                mem_index(TOY_R8, TOY_RAX, 4, (int32_t)(r * row_bytes)));
        }
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) width);
        asm_jcc(as, TOY_CC_NE, ROW_NORM_SUM);

        for (uint8_t r = 0; r < ROW_NORM_ROWS; r++) {
            emit_row_scale(as, r, ROW_NORM_SCALE + r, wide, width, eps_bits, mean);
        }
        for (uint8_t r = 0; r < ROW_NORM_ROWS; r++) {
            emit_row_norm_out(as, ROW_NORM_SCALE + r, wide, width, (int32_t)(r * row_bytes),
                              2 + r * 2, 3 + r * 2, weighted);
        }

        asm_add_imm(as, TOY_R8, group_bytes);
        asm_add_imm(as, TOY_R9, group_bytes);
        asm_add_imm(as, TOY_R11, ROW_NORM_ROWS);
        asm_cmp_imm(as, TOY_R11, (int32_t)(groups * ROW_NORM_ROWS));
        asm_jcc(as, TOY_CC_NE, ROW_NORM_GROUP);
    }

    if (tail != 0) {
        asm_label(as, ROW_NORM_TAIL_LOOP);
        asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        asm_label(as, ROW_NORM_TAIL_SUM);
        asm_vmovss_load(as, TOY_XMM4, mem_index(TOY_R8, TOY_RAX, 4, 0));
        asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM4, mem_index(TOY_R8, TOY_RAX, 4, 0));
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) width);
        asm_jcc(as, TOY_CC_NE, ROW_NORM_TAIL_SUM);

        emit_row_scale(as, TOY_XMM0, TOY_XMM7, wide, width, eps_bits, mean);
        emit_row_norm_out(as, TOY_YMM7, wide, width, 0, ROW_NORM_TAIL_MAIN,
                          ROW_NORM_TAIL_TAIL, weighted);

        asm_add_imm(as, TOY_R8, row_bytes);
        asm_add_imm(as, TOY_R9, row_bytes);
        asm_add_imm(as, TOY_R11, 1);
        asm_cmp_imm(as, TOY_R11, (int32_t) rows);
        asm_jcc(as, TOY_CC_NE, ROW_NORM_TAIL_LOOP);
    }
}

/* L2NORM: out[r][i] = x[r][i] / sqrt(sum(x[r][i]^2) + eps) */
static void emit_l2norm(struct assembler *as, uint32_t rows, uint32_t width, bool wide,
                        uint32_t eps_bits) {
    emit_row_norm(as, rows, width, wide, eps_bits, false, false);
}

/* RMSNORM_ROWS: out[r][i] = (x[r][i] / sqrt(sum(x[r][i]^2) / k + eps)) * w[i] */
static void emit_rmsnorm_rows(struct assembler *as, uint32_t rows, uint32_t width, bool wide,
                              uint32_t eps_bits) {
    emit_row_norm(as, rows, width, wide, eps_bits, true, true);
}

/*
 * CONV1D_SILU: out[c] = silu(sum over j of w[c * taps + j] * x[j * stride + c])
 *
 * Pass one keeps one scalar fma chain per channel in increasing tap order, so
 * the sums match the reference bit for bit. Pass two then runs the activation
 * over the output, which lets one AVX2 loop cover eight channels. m is the
 * channel count, k the stride between two taps of x and param_bits the tap
 * count
 */
static void emit_conv1d_silu(struct assembler *as, uint32_t count, uint32_t stride,
                             uint32_t taps, bool wide) {
    // The dot pass: r10 walks the weight rows, rax the channel
    asm_xor_rr(as, TOY_R10, TOY_R10);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    asm_label(as, 0);
    asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    for (uint32_t j = 0; j < taps; j++) {
        asm_vmovss_load(as, TOY_XMM1,
                        mem_index(TOY_RSI, TOY_RAX, 4, (int32_t)(j * stride * 4)));
        asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM1,
                            mem_index(TOY_RDI, TOY_R10, 1, (int32_t)(j * 4)));
    }
    asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), TOY_XMM0);
    asm_add_imm(as, TOY_R10, (int32_t)(taps * 4));
    asm_add_imm(as, TOY_RAX, 1);
    asm_cmp_imm(as, TOY_RAX, (int32_t) count);
    asm_jcc(as, TOY_CC_NE, 0);

    // The activation pass
    const uint32_t step = wide ? 8u : 1u;
    const uint32_t main = count - (count % step);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (main != 0) {
        asm_label(as, 1);
        if (wide) {
            asm_vmovups_load_ymm(as, MATH_X, mem_index(TOY_RDX, TOY_RAX, 4, 0));
        } else {
            asm_vmovss_load(as, MATH_X, mem_index(TOY_RDX, TOY_RAX, 4, 0));
        }
        emit_silu_vec(as, wide);
        if (wide) {
            asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        } else {
            asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        }
        asm_add_imm(as, TOY_RAX, (int32_t) step);
        asm_cmp_imm(as, TOY_RAX, (int32_t) main);
        asm_jcc(as, TOY_CC_NE, 1);
    }
    if (main < count) {
        // The tail always handles one element
        asm_label(as, 2);
        asm_vmovss_load(as, MATH_X, mem_index(TOY_RDX, TOY_RAX, 4, 0));
        emit_silu_vec(as, false);
        asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RAX, 4, 0), MATH_X);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t) count);
        asm_jcc(as, TOY_CC_NE, 2);
    }
}


/* ---------- q8_0 ---------- */

static bool op_is_q8_0(uint32_t op) {
    return (op >= TOY_OP_Q8_0_DOT && op <= TOY_OP_Q8_0_MATMUL)
        || op == TOY_OP_Q8_0_Q8_0_MATVEC || op == TOY_OP_Q8_0_Q8_0_MATMUL;
}

/*
 * q8_0 scalar region: one element at a time
 *
 * A weight row holds blocks 34-byte blocks. Each block runs 32 scalar FMAs into
 * sum, then the f16 scale folds into the output. The caller splits the row tail
 * and the column tail with generation-time constants
 */
static void emit_q8_scalar(struct assembler *as, uint32_t rows, uint32_t cols, uint32_t n,
                           uint32_t k, uint32_t startCol,
                           uint8_t labelRow, uint8_t labelCol, uint8_t labelBlock) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const int32_t rowBytes = (int32_t)(blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t rowTotal = (int32_t)(rows * blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t xBlockStep = (int32_t)(TOY_Q8_0_BLOCK * n * 4);
    const int32_t startColBytes = (int32_t)(startCol * 4);

    asm_xor_rr(as, TOY_RCX, TOY_RCX);
    asm_mov_rr(as, TOY_R9, TOY_RDX);
    if (startColBytes != 0) {
        asm_add_imm(as, TOY_R9, startColBytes);
    }
    asm_label(as, labelRow);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    asm_label(as, labelCol);
    asm_vxorps(as, false, TOY_XMM4, TOY_XMM4, TOY_XMM4);
    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_lea(as, TOY_R10, mem_index(TOY_RSI, TOY_RAX, 4, startColBytes));
    asm_xor_rr(as, TOY_R11, TOY_R11);
    if (blocks > 1) {
        asm_label(as, labelBlock);
    }
    asm_vxorps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    for (int g = 0; g < 8; g++) {
        asm_vpmovsxbd(as, false, TOY_XMM1, mem_base(TOY_R8, (int32_t)(2 + 4 * g)));
        asm_vcvtdq2ps(as, false, TOY_XMM1, TOY_XMM1);
        for (int l = 0; l < 4; l++) {
            const int p = 4 * g + l;
            asm_vpermilps(as, TOY_XMM2, TOY_XMM1, (uint8_t)(l * 0x55));
            asm_vfmadd231ss_mem(as, TOY_XMM0, TOY_XMM2,
                                    mem_base(TOY_R10, (int32_t)(p * n * 4)));
        }
    }
    asm_vcvtph2ps(as, TOY_XMM3, mem_base(TOY_R8, 0));
    asm_vfmadd231ss(as, TOY_XMM4, TOY_XMM3, TOY_XMM0);
    if (blocks > 1) {
        asm_add_imm(as, TOY_R8, TOY_Q8_0_BLOCK_BYTES);
        asm_add_imm(as, TOY_R10, xBlockStep);
        asm_add_imm(as, TOY_R11, 1);
        asm_cmp_imm(as, TOY_R11, (int32_t)blocks);
        asm_jcc(as, TOY_CC_NE, labelBlock);
    }
    asm_vmovss_store(as, mem_index(TOY_R9, TOY_RAX, 4, 0), TOY_XMM4);
    if (cols > 1) {
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t)cols);
        asm_jcc(as, TOY_CC_NE, labelCol);
    }
    asm_add_imm(as, TOY_RDI, rowBytes);
    asm_add_imm(as, TOY_R9, (int32_t)(n * 4));
    asm_add_imm(as, TOY_RCX, 1);
    if (rows > 1) {
        asm_cmp_imm(as, TOY_RCX, (int32_t)rows);
        asm_jcc(as, TOY_CC_NE, labelRow);
    }
    // Restore rdi to the base of w for the later regions of the same kernel
    asm_sub_imm(as, TOY_RDI, rowTotal);
}

/*
 * AVX2 kernel for q8_0 dot and matvec: four ymm accumulators per row, each
 * block covers 4 groups of 8 elements
 * A horizontal sum closes the block, then the f16 scale folds into the scalar
 * output
 */
static void emit_q8_avx2_dotmatvec(struct assembler *as, uint32_t rows, uint32_t k) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const int32_t rowBytes = (int32_t)(blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t rowTotal = (int32_t)(rows * blocks * TOY_Q8_0_BLOCK_BYTES);

    asm_xor_rr(as, TOY_RCX, TOY_RCX);
    asm_label(as, 0);
    asm_vxorps(as, false, TOY_XMM7, TOY_XMM7, TOY_XMM7);
    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_mov_rr(as, TOY_R10, TOY_RSI);
    asm_xor_rr(as, TOY_RAX, TOY_RAX);
    if (blocks > 1) {
        asm_label(as, 1);
    }
    for (uint8_t g = 0; g < 4; g++) {
        asm_vxorps(as, true, g, g, g);
    }
    for (uint8_t g = 0; g < 4; g++) {
        asm_vmovups_load_ymm(as, TOY_YMM4, mem_base(TOY_R10, (int32_t)(g * 32)));
        asm_vpmovsxbd(as, true, TOY_YMM5, mem_base(TOY_R8, (int32_t)(2 + g * 8)));
        asm_vcvtdq2ps(as, true, TOY_YMM5, TOY_YMM5);
        asm_vfmadd231ps(as, g, TOY_YMM5, TOY_YMM4);
    }
    asm_vaddps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM1);
    asm_vaddps(as, true, TOY_YMM2, TOY_YMM2, TOY_YMM3);
    asm_vaddps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM2);
    asm_vextractf128(as, TOY_XMM4, TOY_YMM0, 1);
    asm_vaddps(as, false, TOY_XMM0, TOY_XMM0, TOY_XMM4);
    asm_vhaddps(as, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    asm_vhaddps(as, TOY_XMM0, TOY_XMM0, TOY_XMM0);
    asm_vcvtph2ps(as, TOY_XMM6, mem_base(TOY_R8, 0));
    asm_vfmadd231ss(as, TOY_XMM7, TOY_XMM6, TOY_XMM0);
    if (blocks > 1) {
        asm_add_imm(as, TOY_R8, TOY_Q8_0_BLOCK_BYTES);
        asm_add_imm(as, TOY_R10, TOY_Q8_0_BLOCK * 4);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t)blocks);
        asm_jcc(as, TOY_CC_NE, 1);
    }
    asm_vmovss_store(as, mem_index(TOY_RDX, TOY_RCX, 4, 0), TOY_XMM7);
    asm_add_imm(as, TOY_RDI, rowBytes);
    asm_add_imm(as, TOY_RCX, 1);
    if (rows > 1) {
        asm_cmp_imm(as, TOY_RCX, (int32_t)rows);
        asm_jcc(as, TOY_CC_NE, 0);
    }
    asm_sub_imm(as, TOY_RDI, rowTotal);
}

/*
 * Main body for one row of 16 columns: 32 p values unrolled, two ymm registers
 * carry the columns
 *
 * On entry rdi is the weight base of the current row, rdx is the output address
 * of the current row, rsi is the base of x, r10 is the column offset
 */
static void emit_q8_matmul_body(struct assembler *as, uint32_t n, uint32_t blocks,
                                int32_t xBlockStep, uint8_t labelBlock) {
    asm_vxorps(as, true, TOY_YMM0, TOY_YMM0, TOY_YMM0);
    asm_vxorps(as, true, TOY_YMM1, TOY_YMM1, TOY_YMM1);
    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_lea(as, TOY_RCX, mem_index(TOY_RSI, TOY_R10, 4, 0));
    asm_xor_rr(as, TOY_R11, TOY_R11);
    if (blocks > 1) {
        asm_label(as, labelBlock);
    }
    asm_vxorps(as, true, TOY_YMM2, TOY_YMM2, TOY_YMM2);
    asm_vxorps(as, true, TOY_YMM3, TOY_YMM3, TOY_YMM3);
    for (int g = 0; g < 8; g++) {
        asm_vpmovsxbd(as, false, TOY_XMM4, mem_base(TOY_R8, (int32_t)(2 + 4 * g)));
        asm_vcvtdq2ps(as, false, TOY_XMM4, TOY_XMM4);
        for (int l = 0; l < 4; l++) {
            const int p = 4 * g + l;
            asm_vpermilps(as, TOY_XMM5, TOY_XMM4, (uint8_t)(l * 0x55));
            asm_vbroadcastss_reg(as, TOY_YMM5, TOY_XMM5);
            asm_vmovups_load_ymm(as, TOY_YMM6, mem_base(TOY_RCX, (int32_t)(p * n * 4)));
            asm_vmovups_load_ymm(as, TOY_YMM7,
                                     mem_base(TOY_RCX, (int32_t)(p * n * 4 + 32)));
            asm_vfmadd231ps(as, TOY_YMM2, TOY_YMM5, TOY_YMM6);
            asm_vfmadd231ps(as, TOY_YMM3, TOY_YMM5, TOY_YMM7);
        }
    }
    asm_vcvtph2ps(as, TOY_XMM4, mem_base(TOY_R8, 0));
    asm_vbroadcastss_reg(as, TOY_YMM4, TOY_XMM4);
    asm_vfmadd231ps(as, TOY_YMM0, TOY_YMM4, TOY_YMM2);
    asm_vfmadd231ps(as, TOY_YMM1, TOY_YMM4, TOY_YMM3);
    if (blocks > 1) {
        asm_add_imm(as, TOY_R8, TOY_Q8_0_BLOCK_BYTES);
        asm_add_imm(as, TOY_RCX, xBlockStep);
        asm_add_imm(as, TOY_R11, 1);
        asm_cmp_imm(as, TOY_R11, (int32_t)blocks);
        asm_jcc(as, TOY_CC_NE, labelBlock);
    }
    asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_R10, 4, 0), TOY_YMM0);
    asm_vmovups_store_ymm(as, mem_index(TOY_RDX, TOY_R10, 4, 32), TOY_YMM1);
}

static void emit_q8_avx2_matmul(struct assembler *as, uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const uint32_t n16 = n & ~15u;
    const int32_t rowBytes = (int32_t)(blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t rowTotal = (int32_t)(m * blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t xBlockStep = (int32_t)(TOY_Q8_0_BLOCK * n * 4);
    const int32_t cRowStep = (int32_t)(n * 4);
    const int32_t cTotal = (int32_t)(m * n * 4);
    const uint32_t tileRows = m > 64 ? 64 : m;
    const uint32_t tileMain = (m / tileRows) * tileRows;
    const uint32_t tileTail = m - tileMain;

    if (n16 != 0) {
        /*
         * The column block is the outer loop, so all output rows reuse the k x 16
         * slice of x. The rows then split into tiles of tileRows, so the weights and
         * the x slice of one tile stay in L2 and the weight stream does not push x
         * out of the cache
         */
        asm_xor_rr(as, TOY_R10, TOY_R10);
        asm_label(as, 1);
        asm_xor_rr(as, TOY_R9, TOY_R9);
        asm_label(as, 3);
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        asm_label(as, 0);
        emit_q8_matmul_body(as, n, blocks, xBlockStep, 2);
        asm_add_imm(as, TOY_RDI, rowBytes);
        asm_add_imm(as, TOY_RDX, cRowStep);
        asm_add_imm(as, TOY_RAX, 1);
        asm_cmp_imm(as, TOY_RAX, (int32_t)tileRows);
        asm_jcc(as, TOY_CC_NE, 0);
        asm_add_imm(as, TOY_R9, (int32_t)tileRows);
        asm_cmp_imm(as, TOY_R9, (int32_t)tileMain);
        asm_jcc(as, TOY_CC_NE, 3);
        if (tileTail != 0) {
            asm_xor_rr(as, TOY_RAX, TOY_RAX);
            asm_label(as, 4);
            emit_q8_matmul_body(as, n, blocks, xBlockStep, 5);
            asm_add_imm(as, TOY_RDI, rowBytes);
            asm_add_imm(as, TOY_RDX, cRowStep);
            asm_add_imm(as, TOY_RAX, 1);
            asm_cmp_imm(as, TOY_RAX, (int32_t)tileTail);
            asm_jcc(as, TOY_CC_NE, 4);
        }
        asm_sub_imm(as, TOY_RDI, rowTotal);
        asm_sub_imm(as, TOY_RDX, cTotal);
        if (n16 > 16) {
            asm_add_imm(as, TOY_R10, 16);
            asm_cmp_imm(as, TOY_R10, (int32_t)n16);
            asm_jcc(as, TOY_CC_NE, 1);
        }
    }

    emit_q8_scalar(as, m, n - n16, n, k, n16, 6, 7, 8);
}

/* ---------- q8_0 weights against q8_0 activations ---------- */

/*
 * The kernel reproduces the ggml lane order exactly: the 32 int8 products of a
 * block split into 8 int32 lanes of 4 adjacent values, the combined f16 scale
 * accumulates into every lane with one fma, and hsum_float_8 reduces the lanes
 * once at the end
 *
 * One activation column is a series of 34-byte blocks, the weights are m such
 * series. A row group of 8 rows shares one activation block, so Q8Q8_ACT and
 * Q8Q8_XD hold it for the whole group
 */
enum {
    // Accumulators hold 8 rows. The ones vector lives across groups, so it
    // avoids xmm9, the scratch register of the horizontal sum
    Q8Q8_ROWS = 8,
    Q8Q8_ACT = 8,
    Q8Q8_D = 10,
    Q8Q8_W = 11,
    Q8Q8_T = 12,
    Q8Q8_S = 13,
    Q8Q8_ONES = 14,
    Q8Q8_XD = 15,
    // Labels of the outer loops and of one block loop per row group
    Q8Q8_LABEL_GROUP = 0,
    Q8Q8_LABEL_COLUMN = 1,
    Q8Q8_LABEL_TAIL_COLUMN = 2,
    Q8Q8_LABEL_BLOCK = 3,
    Q8Q8_LABEL_TAIL_BLOCK = 4,
};

/*
 * One weight row of the current group: r8 + r10 is the block of the row, so
 * rowDisp is r * rowBytes
 */
static void emit_q8_q8_row(struct assembler *as, uint8_t acc, int32_t rowDisp) {
    asm_vcvtph2ps(as, Q8Q8_D, mem_index(TOY_R8, TOY_R10, 1, rowDisp));
    asm_vmulss(as, Q8Q8_D, Q8Q8_D, Q8Q8_XD);
    asm_vbroadcastss_reg(as, Q8Q8_D, Q8Q8_D);
    asm_vmovups_load_ymm(as, Q8Q8_W, mem_index(TOY_R8, TOY_R10, 1, rowDisp + 2));
    asm_vpsignb(as, true, Q8Q8_T, Q8Q8_W, Q8Q8_W);
    asm_vpsignb(as, true, Q8Q8_S, Q8Q8_ACT, Q8Q8_W);
    asm_vpmaddubsw(as, true, Q8Q8_T, Q8Q8_T, Q8Q8_S);
    asm_vpmaddwd(as, true, Q8Q8_T, Q8Q8_T, Q8Q8_ONES);
    asm_vcvtdq2ps(as, true, Q8Q8_T, Q8Q8_T);
    asm_vfmadd231ps(as, acc, Q8Q8_D, Q8Q8_T);
}

/*
 * Zero rows accumulators, run the block loop and store one row per accumulator
 * r8 is the weight base of the group, r11 the activation column, r9 the output
 * base of the group and rcx the column, so the store is c + row * outStride + column
 */
static void emit_q8_q8_group(struct assembler *as, uint32_t rows, uint32_t blocks,
                             int32_t rowBytes, int32_t outStride, uint8_t blockLabel) {
    for (uint8_t r = 0; r < rows; r++) {
        asm_vxorps(as, true, r, r, r);
    }
    asm_xor_rr(as, TOY_R10, TOY_R10);
    if (blocks > 1) {
        asm_label(as, blockLabel);
    }
    asm_vcvtph2ps(as, Q8Q8_XD, mem_index(TOY_R11, TOY_R10, 1, 0));
    asm_vmovups_load_ymm(as, Q8Q8_ACT, mem_index(TOY_R11, TOY_R10, 1, 2));
    for (uint8_t r = 0; r < rows; r++) {
        emit_q8_q8_row(as, r, (int32_t)(r * rowBytes));
    }
    if (blocks > 1) {
        asm_add_imm(as, TOY_R10, TOY_Q8_0_BLOCK_BYTES);
        // One row and one activation column have the same byte length
        asm_cmp_imm(as, TOY_R10, rowBytes);
        asm_jcc(as, TOY_CC_NE, blockLabel);
    }
    for (uint8_t r = 0; r < rows; r++) {
        emit_reduce_ymm_to_xmm(as, r);
        asm_vmovss_store(as, mem_index(TOY_R9, TOY_RCX, 4, (int32_t)(r * outStride)), r);
    }
}

/*
 * AVX2 q8_0 x q8_0 for m rows, n columns and k values
 *
 * Rows run in groups of 8 and the column loop sits inside, so the weight block
 * of a row group is decoded again from L1 for every column and the weight
 * stream touches memory once
 */
static void emit_q8_q8_avx2(struct assembler *as, uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t blocks = k / TOY_Q8_0_BLOCK;
    const uint32_t groups = m / Q8Q8_ROWS;
    const uint32_t tail = m - groups * Q8Q8_ROWS;
    const int32_t rowBytes = (int32_t)(blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t outStride = (int32_t)(n * 4);
    const int32_t groupBytes = (int32_t)(Q8Q8_ROWS * blocks * TOY_Q8_0_BLOCK_BYTES);
    const int32_t groupOut = (int32_t)(Q8Q8_ROWS * n * 4);

    // int16 ones turn the pair sums into int32 lanes
    asm_mov_imm32(as, TOY_RAX, 0x0001'0001);
    asm_vmovd_gpr(as, Q8Q8_ONES, TOY_RAX);
    asm_vbroadcastss_reg(as, Q8Q8_ONES, Q8Q8_ONES);

    if (groups != 0) {
        asm_mov_rr(as, TOY_R8, TOY_RDI);
        asm_mov_rr(as, TOY_R9, TOY_RDX);
        asm_xor_rr(as, TOY_RAX, TOY_RAX);
        asm_label(as, Q8Q8_LABEL_GROUP);
        asm_mov_rr(as, TOY_R11, TOY_RSI);
        asm_xor_rr(as, TOY_RCX, TOY_RCX);
        if (n > 1) {
            asm_label(as, Q8Q8_LABEL_COLUMN);
        }
        emit_q8_q8_group(as, Q8Q8_ROWS, blocks, rowBytes, outStride, Q8Q8_LABEL_BLOCK);
        if (n > 1) {
            asm_add_imm(as, TOY_R11, rowBytes);
            asm_add_imm(as, TOY_RCX, 1);
            asm_cmp_imm(as, TOY_RCX, (int32_t)n);
            asm_jcc(as, TOY_CC_NE, Q8Q8_LABEL_COLUMN);
        }
        asm_add_imm(as, TOY_R8, groupBytes);
        asm_add_imm(as, TOY_R9, groupOut);
        asm_add_imm(as, TOY_RAX, 1);
        if (groups > 1) {
            asm_cmp_imm(as, TOY_RAX, (int32_t)groups);
            asm_jcc(as, TOY_CC_NE, Q8Q8_LABEL_GROUP);
        }
    }

    if (tail != 0) {
        // The tail starts after the whole groups
        asm_lea(as, TOY_R8, mem_base(TOY_RDI, (int32_t)(groups * Q8Q8_ROWS) * rowBytes));
        asm_lea(as, TOY_R9, mem_base(TOY_RDX, (int32_t)(groups * Q8Q8_ROWS) * outStride));
        asm_mov_rr(as, TOY_R11, TOY_RSI);
        asm_xor_rr(as, TOY_RCX, TOY_RCX);
        if (n > 1) {
            asm_label(as, Q8Q8_LABEL_TAIL_COLUMN);
        }
        emit_q8_q8_group(as, tail, blocks, rowBytes, outStride, Q8Q8_LABEL_TAIL_BLOCK);
        if (n > 1) {
            asm_add_imm(as, TOY_R11, rowBytes);
            asm_add_imm(as, TOY_RCX, 1);
            asm_cmp_imm(as, TOY_RCX, (int32_t)n);
            asm_jcc(as, TOY_CC_NE, Q8Q8_LABEL_TAIL_COLUMN);
        }
    }
}

/* ---------- f32 to q8_0 activation quantization ---------- */

/*
 * Register plan of the quantizer: xmm0 to xmm3 hold constants that survive
 * the loop, xmm4 and xmm5 the scale and the reciprocal, xmm6 to xmm9 the
 * values, xmm10 to xmm13 the int32 lanes and xmm14 and xmm15 the scratch
 */
enum {
    Q8Q_ABS_MASK = 0,
    Q8Q_HALF = 1,
    Q8Q_NEG_HALF = 2,
    Q8Q_FLT_MAX = 3,
    Q8Q_SCALE = 4,
    Q8Q_ID = 5,
    Q8Q_VALUE = 6,
    Q8Q_LANES = 10,
    Q8Q_SCRATCH = 14,
    Q8Q_SCRATCH_HIGH = 15,
};

// Writes a generation-time constant into all lanes of xmm, rax is scratch
static void emit_const_xmm_broadcast(struct assembler *as, uint8_t xmm, uint32_t bits) {
    emit_const_xmm(as, xmm, bits);
    asm_vshufps(as, xmm, xmm, xmm, 0x00);
}

/*
 * AVX2 f32 to q8_0, one 32 value block per iteration
 *
 * The sequence follows q8_0_quantize_block: the largest magnitude over 127
 * gives the scale, the reciprocal multiplies every value, and the product
 * rounds half away from zero and saturates to int8. The magnitude comparison
 * feeds the rounding, because it also keeps an infinite product at zero
 *
 * The comparisons use the ordered predicates, so a NaN value never enters the
 * scale and quantizes to zero
 */
static void emit_q8_quantize_avx2(struct assembler *as, uint32_t blocks) {
    if (blocks == 0) {
        return;
    }

    emit_const_xmm_broadcast(as, TOY_XMM0, 0x7FFFFFFFu);
    asm_vbroadcastss_reg(as, Q8Q_ABS_MASK, TOY_XMM0);
    emit_const_xmm_broadcast(as, TOY_XMM1, float_bits(0.5f));
    asm_vbroadcastss_reg(as, Q8Q_HALF, TOY_XMM1);
    emit_const_xmm_broadcast(as, TOY_XMM2, float_bits(-0.5f));
    asm_vbroadcastss_reg(as, Q8Q_NEG_HALF, TOY_XMM2);
    emit_const_xmm_broadcast(as, TOY_XMM3, float_bits(FLT_MAX));
    asm_vbroadcastss_reg(as, Q8Q_FLT_MAX, TOY_XMM3);

    asm_mov_rr(as, TOY_R8, TOY_RDI);
    asm_mov_rr(as, TOY_R9, TOY_RDX);
    asm_xor_rr(as, TOY_R10, TOY_R10);
    if (blocks > 1) {
        asm_label(as, 0);
    }

    for (uint8_t v = 0; v < 4; v++) {
        asm_vmovups_load_ymm(as, (uint8_t)(Q8Q_VALUE + v),
                             mem_base(TOY_R8, (int32_t)(v * 32)));
        asm_vandps(as, true, (uint8_t)(Q8Q_LANES + v), (uint8_t)(Q8Q_VALUE + v), Q8Q_ABS_MASK);
    }

    // The maximum ignores NaN: maxps returns the second source, and that source is
    // the running maximum when the value is the first source
    asm_vxorps(as, true, Q8Q_SCALE, Q8Q_SCALE, Q8Q_SCALE);
    for (uint8_t v = 0; v < 4; v++) {
        asm_vmaxps(as, true, Q8Q_SCALE, (uint8_t)(Q8Q_LANES + v), Q8Q_SCALE);
    }
    // The high half comes out first: a 128-bit vmaxps zeroes the upper lanes
    asm_vextractf128(as, Q8Q_ID, Q8Q_SCALE, 1);
    asm_vmaxps(as, false, Q8Q_SCALE, Q8Q_SCALE, Q8Q_ID);
    asm_vpermilps(as, Q8Q_ID, Q8Q_SCALE, 0xB1);
    asm_vmaxps(as, false, Q8Q_SCALE, Q8Q_SCALE, Q8Q_ID);
    asm_vpermilps(as, Q8Q_ID, Q8Q_SCALE, 0x4E);
    asm_vmaxps(as, false, Q8Q_SCALE, Q8Q_SCALE, Q8Q_ID);

    // scale = amax / 127, stored as f16 in the first two bytes of the block
    emit_const_xmm_broadcast(as, Q8Q_ID, float_bits(127.0f));
    asm_vdivps(as, false, Q8Q_SCALE, Q8Q_SCALE, Q8Q_ID);
    asm_vcvtps2ph(as, Q8Q_ID, Q8Q_SCALE, 0);
    asm_vmovd_xmm_gpr(as, TOY_RAX, Q8Q_ID);
    asm_mov_store16(as, mem_base(TOY_R9, 0), TOY_RAX);

    // id = 1 / scale, zero for a zero scale
    emit_const_xmm_broadcast(as, Q8Q_ID, float_bits(1.0f));
    asm_vdivps(as, false, Q8Q_ID, Q8Q_ID, Q8Q_SCALE);
    asm_vxorps(as, false, Q8Q_SCRATCH_HIGH, Q8Q_SCRATCH_HIGH, Q8Q_SCRATCH_HIGH);
    asm_vcmpps(as, false, Q8Q_SCALE, Q8Q_SCALE, Q8Q_SCRATCH_HIGH, 4 /* NEQ */);
    asm_vandps(as, false, Q8Q_ID, Q8Q_ID, Q8Q_SCALE);
    asm_vbroadcastss_reg(as, Q8Q_ID, Q8Q_ID);

    for (uint8_t v = 0; v < 4; v++) {
        const uint8_t value = (uint8_t)(Q8Q_VALUE + v);
        const uint8_t q = (uint8_t)(Q8Q_LANES + v);

        asm_vmulps(as, true, value, value, Q8Q_ID);
        // q = roundf(value): truncate, then step away from zero at a half
        asm_vcvttps2dq(as, true, q, value);
        asm_vcvtdq2ps(as, true, Q8Q_SCRATCH, q);
        asm_vsubps(as, true, Q8Q_SCRATCH, value, Q8Q_SCRATCH);
        asm_vcmpps(as, true, Q8Q_SCRATCH_HIGH, Q8Q_SCRATCH, Q8Q_HALF, 13 /* GE */);
        asm_vpsubd(as, true, q, q, Q8Q_SCRATCH_HIGH);
        asm_vcmpps(as, true, Q8Q_SCRATCH, Q8Q_SCRATCH, Q8Q_NEG_HALF, 2 /* LE */);
        asm_vpaddd(as, true, q, q, Q8Q_SCRATCH);
        // A non-finite product quantizes to zero
        asm_vandps(as, true, Q8Q_SCRATCH_HIGH, value, Q8Q_ABS_MASK);
        asm_vcmpps(as, true, Q8Q_SCRATCH_HIGH, Q8Q_SCRATCH_HIGH, Q8Q_FLT_MAX, 2 /* LE */);
        asm_vpand(as, true, q, q, Q8Q_SCRATCH_HIGH);
    }

    // pack the four int32 vectors into 32 int8 values in order
    for (uint8_t v = 0; v < 4; v++) {
        asm_vextractf128(as, Q8Q_SCRATCH, (uint8_t)(Q8Q_LANES + v), 1);
        asm_vpackssdw(as, (uint8_t)(Q8Q_LANES + v), (uint8_t)(Q8Q_LANES + v), Q8Q_SCRATCH);
    }
    asm_vpacksswb(as, Q8Q_SCRATCH, (uint8_t)(Q8Q_LANES + 0), (uint8_t)(Q8Q_LANES + 1));
    asm_vpacksswb(as, Q8Q_SCRATCH_HIGH, (uint8_t)(Q8Q_LANES + 2), (uint8_t)(Q8Q_LANES + 3));
    asm_vmovups_store_xmm(as, mem_base(TOY_R9, 2), Q8Q_SCRATCH);
    asm_vmovups_store_xmm(as, mem_base(TOY_R9, 18), Q8Q_SCRATCH_HIGH);

    asm_add_imm(as, TOY_R8, 4 * TOY_Q8_0_BLOCK);
    asm_add_imm(as, TOY_R9, TOY_Q8_0_BLOCK_BYTES);
    if (blocks > 1) {
        asm_add_imm(as, TOY_R10, 1);
        asm_cmp_imm(as, TOY_R10, (int32_t) blocks);
        asm_jcc(as, TOY_CC_NE, 0);
    }
}

static bool spec_valid(const struct spec *spec) {
    if (spec == NULL) {
        return false;
    }
    if (spec->op < TOY_OP_GEMM || spec->op > TOY_OP_RMSNORM_ROWS) {
        return false;
    }
    if (spec->isa != TOY_ISA_SCALAR && spec->isa != TOY_ISA_AVX2) {
        return false;
    }
    if (spec->m == 0 || spec->n == 0 || spec->k == 0) {
        return false;
    }
    if (spec->op == TOY_OP_Q8_0_QUANTIZE) {
        // m is the block count, the source and the destination are flat streams
        if (spec->n != 1 || spec->k != 1) {
            return false;
        }
        return spec->m <= (uint32_t)(INT32_MAX / TOY_Q8_0_BLOCK_BYTES);
    }
    if (op_is_q8_0(spec->op)) {
        if (spec->m > 65535 || spec->n > 65535 || spec->k > 65535) {
            return false;
        }
        const uint64_t m = spec->m, n = spec->n, k = spec->k;
        if (k % TOY_Q8_0_BLOCK != 0) {
            return false;
        }
        if (spec->op == TOY_OP_Q8_0_DOT && (m != 1 || n != 1)) {
            return false;
        }
        if ((spec->op == TOY_OP_Q8_0_MATVEC || spec->op == TOY_OP_Q8_0_Q8_0_MATVEC) && n != 1) {
            return false;
        }
        const uint64_t rowBytes = (k / TOY_Q8_0_BLOCK) * TOY_Q8_0_BLOCK_BYTES;
        if (m * rowBytes > (uint64_t)INT32_MAX || m * n * 4 > (uint64_t)INT32_MAX) {
            return false;
        }
        return true;
    }

    if (spec->op == TOY_OP_SIGMOID || spec->op == TOY_OP_SOFTPLUS) {
        // The unary kernels are AVX2 only; the reference covers the rest
        if (spec->isa != TOY_ISA_AVX2 || spec->n != 1 || spec->k != 1) {
            return false;
        }
        return spec->m <= (uint32_t)(INT32_MAX / 4);
    }
    if (spec->op == TOY_OP_CONV1D_SILU) {
        const uint64_t taps = spec->param_bits;
        if (spec->isa != TOY_ISA_AVX2 || spec->n != 1 || taps == 0 || taps > 64) {
            return false;
        }
        // The tap displacement and the output offset are int32
        const uint64_t reach = (taps - 1) * spec->k * 4 + (uint64_t)spec->m * 4;
        return spec->m <= (uint32_t)(INT32_MAX / 4) && reach <= (uint64_t)INT32_MAX;
    }
    if (spec->op == TOY_OP_L2NORM || spec->op == TOY_OP_RMSNORM_ROWS) {
        if (spec->isa != TOY_ISA_AVX2 || spec->n != 1) {
            return false;
        }
        const uint64_t bytes = (uint64_t)spec->m * spec->k * 4;
        return spec->m <= (uint32_t)(INT32_MAX / 4) && bytes <= (uint64_t)INT32_MAX;
    }

    if (spec->op >= TOY_OP_ADD) {
        // Elementwise: m is the element count, the index increments as a 32-bit integer
        if (spec->n != 1 || spec->k != 1) {
            return false;
        }
        if (spec->m > (uint32_t)(INT32_MAX / 4)) {
            return false;
        }
        return true;
    }

    if (spec->m > 65535 || spec->n > 65535 || spec->k > 65535) {
        return false;
    }
    // Displacements and pointer increments are int32, so they cannot overflow
    const uint64_t m = spec->m, n = spec->n, k = spec->k;

    if (spec->op == TOY_OP_MATVEC && spec->n != 1) {
        return false;
    }
    if (spec->op == TOY_OP_DOT && (spec->m != 1 || spec->n != 1)) {
        return false;
    }
    if (m * k * 4 > (uint64_t)INT32_MAX || m * n * 4 > (uint64_t)INT32_MAX ||
        n * k * 4 > (uint64_t)INT32_MAX || 8 * k * 4 > (uint64_t)INT32_MAX ||
        8 * n * 4 > (uint64_t)INT32_MAX) {
        return false;
    }
    return true;
}

size_t emit(const struct spec *spec, uint8_t *buf, size_t cap) {
    if (!spec_valid(spec) || buf == NULL || cap == 0) {
        return 0;
    }
    struct assembler as;
    asm_init(&as, buf, cap);

#if defined(TOY_WIN64_ABI)
    emit_win64_prologue(&as);
#endif

    if (spec->op == TOY_OP_Q8_0_QUANTIZE) {
        // Only the AVX2 kernel exists, see spec_supported
        if (spec->isa != TOY_ISA_AVX2) {
            return 0;
        }
        emit_q8_quantize_avx2(&as, spec->m);
#if defined(TOY_WIN64_ABI)
        emit_win64_epilogue(&as);
#endif
        asm_vzeroupper(&as);
        asm_ret(&as);
        if (!asm_finish(&as)) {
            return 0;
        }
        return as.len;
    }

    if (op_is_q8_0(spec->op)) {
        switch (spec->isa) {
            case TOY_ISA_SCALAR:
                if (spec->op == TOY_OP_Q8_0_DOT) {
                    emit_q8_scalar(&as, 1, 1, 1, spec->k, 0, 0, 1, 2);
                } else if (spec->op == TOY_OP_Q8_0_MATVEC) {
                    emit_q8_scalar(&as, spec->m, 1, 1, spec->k, 0, 0, 1, 2);
                } else if (spec->op == TOY_OP_Q8_0_MATMUL) {
                    emit_q8_scalar(&as, spec->m, spec->n, spec->n, spec->k, 0, 0, 1, 2);
                } else {
                    // No scalar kernel for q8_0 x q8_0, see spec_supported
                    return 0;
                }
                break;
            case TOY_ISA_AVX2:
                if (spec->op == TOY_OP_Q8_0_DOT) {
                    emit_q8_avx2_dotmatvec(&as, 1, spec->k);
                } else if (spec->op == TOY_OP_Q8_0_MATVEC) {
                    emit_q8_avx2_dotmatvec(&as, spec->m, spec->k);
                } else if (spec->op == TOY_OP_Q8_0_MATMUL) {
                    emit_q8_avx2_matmul(&as, spec->m, spec->n, spec->k);
                } else if (spec->op == TOY_OP_Q8_0_Q8_0_MATVEC) {
                    emit_q8_q8_avx2(&as, spec->m, 1, spec->k);
                } else {
                    emit_q8_q8_avx2(&as, spec->m, spec->n, spec->k);
                }
                break;
        }
#if defined(TOY_WIN64_ABI)
        emit_win64_epilogue(&as);
#endif
        asm_vzeroupper(&as);
        asm_ret(&as);
        if (!asm_finish(&as)) {
            return 0;
        }
        return as.len;
    }

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
        case TOY_OP_MUL:
        case TOY_OP_SCALE:
            emit_element_binary(&as, spec->op, spec->m, wide, spec->param_bits);
            break;
        case TOY_OP_RMSNORM:
            emit_rmsnorm(&as, spec->m, wide, spec->param_bits);
            break;
        case TOY_OP_SIGMOID:
        case TOY_OP_SOFTPLUS:
            if (!wide) {
                return 0;
            }
            emit_unary_math(&as, spec->op, spec->m, wide);
            break;
        case TOY_OP_CONV1D_SILU:
            if (!wide) {
                return 0;
            }
            emit_conv1d_silu(&as, spec->m, spec->k, spec->param_bits, wide);
            break;
        case TOY_OP_L2NORM:
            if (!wide) {
                return 0;
            }
            emit_l2norm(&as, spec->m, spec->k, wide, spec->param_bits);
            break;
        case TOY_OP_RMSNORM_ROWS:
            if (!wide) {
                return 0;
            }
            emit_rmsnorm_rows(&as, spec->m, spec->k, wide, spec->param_bits);
            break;
    }
#if defined(TOY_WIN64_ABI)
    emit_win64_epilogue(&as);
#endif
    asm_vzeroupper(&as);
    asm_ret(&as);
    if (!asm_finish(&as)) {
        return 0;
    }
    return as.len;
}

const char *op_name(uint32_t op) {
    switch (op) {
        case TOY_OP_GEMM: return "gemm";
        case TOY_OP_MATVEC: return "matvec";
        case TOY_OP_DOT: return "dot";
        case TOY_OP_ADD: return "add";
        case TOY_OP_MUL: return "mul";
        case TOY_OP_SCALE: return "scale";
        case TOY_OP_RMSNORM: return "rmsnorm";
        case TOY_OP_Q8_0_DOT: return "q8_0_dot";
        case TOY_OP_Q8_0_MATVEC: return "q8_0_matvec";
        case TOY_OP_Q8_0_MATMUL: return "q8_0_matmul";
        case TOY_OP_Q8_0_Q8_0_MATVEC: return "q8_0_q8_0_matvec";
        case TOY_OP_Q8_0_Q8_0_MATMUL: return "q8_0_q8_0_matmul";
        case TOY_OP_Q8_0_QUANTIZE: return "q8_0_quantize";
        case TOY_OP_SIGMOID: return "sigmoid";
        case TOY_OP_SOFTPLUS: return "softplus";
        case TOY_OP_CONV1D_SILU: return "conv1d_silu";
        case TOY_OP_L2NORM: return "l2norm";
        case TOY_OP_RMSNORM_ROWS: return "rmsnorm_rows";
        default: return "unknown";
    }
}

const char *isa_name(uint32_t isa) {
    switch (isa) {
        case TOY_ISA_SCALAR: return "scalar";
        case TOY_ISA_AVX2: return "avx2";
        default: return "unknown";
    }
}

double spec_flops(const struct spec *spec) {
    if (spec == NULL) {
        return 0.0;
    }
    switch (spec->op) {
        case TOY_OP_GEMM:
        case TOY_OP_Q8_0_MATMUL:
        case TOY_OP_Q8_0_Q8_0_MATMUL:
            return 2.0 * spec->m * spec->n * spec->k;
        case TOY_OP_MATVEC:
        case TOY_OP_Q8_0_MATVEC:
        case TOY_OP_Q8_0_Q8_0_MATVEC:
            return 2.0 * spec->m * spec->k;
        case TOY_OP_DOT:
        case TOY_OP_Q8_0_DOT:
            return 2.0 * spec->k;
        case TOY_OP_ADD: return 2.0 * spec->m;
        case TOY_OP_MUL: return 1.0 * spec->m;
        case TOY_OP_SCALE: return 1.0 * spec->m;
        case TOY_OP_RMSNORM: return 4.0 * spec->m;
        // One magnitude test and one multiply per value, then a round
        case TOY_OP_Q8_0_QUANTIZE: return 3.0 * TOY_Q8_0_BLOCK * spec->m;
        case TOY_OP_SIGMOID: return 4.0 * spec->m;
        case TOY_OP_SOFTPLUS: return 6.0 * spec->m;
        case TOY_OP_CONV1D_SILU: return 2.0 * spec->m * spec->param_bits + 4.0 * spec->m;
        case TOY_OP_L2NORM: return 4.0 * spec->m * spec->k;
        case TOY_OP_RMSNORM_ROWS: return 5.0 * spec->m * spec->k;
        default: return 0.0;
    }
}

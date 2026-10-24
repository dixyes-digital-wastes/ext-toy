#include "toy/codegen.h"

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

static bool spec_valid(const struct spec *spec) {
    if (spec == NULL) {
        return false;
    }
    if (spec->op < TOY_OP_GEMM || spec->op > TOY_OP_Q8_0_Q8_0_MATMUL) {
        return false;
    }
    if (spec->isa != TOY_ISA_SCALAR && spec->isa != TOY_ISA_AVX2) {
        return false;
    }
    if (spec->m == 0 || spec->n == 0 || spec->k == 0) {
        return false;
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
        default: return 0.0;
    }
}

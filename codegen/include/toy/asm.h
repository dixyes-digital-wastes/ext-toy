#ifndef TOY_ASM_H
#define TOY_ASM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * x86-64 instruction encoder
 *
 * Covers only the instructions that the generator needs. All branches use
 * rel32, so one fixup pass corrects them. A memory operand supports
 * base + index * scale + disp32, but not RIP-relative or absolute addresses
 */

enum {
    TOY_RAX = 0,
    TOY_RCX = 1,
    TOY_RDX = 2,
    TOY_RBX = 3,
    TOY_RSP = 4,
    TOY_RBP = 5,
    TOY_RSI = 6,
    TOY_RDI = 7,
    TOY_R8 = 8,
    TOY_R9 = 9,
    TOY_R10 = 10,
    TOY_R11 = 11,
    TOY_R12 = 12,
    TOY_R13 = 13,
    TOY_R14 = 14,
    TOY_R15 = 15,
};

enum {
    TOY_XMM0 = 0,
    TOY_XMM1 = 1,
    TOY_XMM2 = 2,
    TOY_XMM3 = 3,
    TOY_XMM4 = 4,
    TOY_XMM5 = 5,
    TOY_XMM6 = 6,
    TOY_XMM7 = 7,
    TOY_XMM8 = 8,
    TOY_XMM9 = 9,
    TOY_XMM10 = 10,
    TOY_XMM11 = 11,
    TOY_XMM12 = 12,
    TOY_XMM13 = 13,
    TOY_XMM14 = 14,
    TOY_XMM15 = 15,
};

// Shares the XMM numbers, only to tell the operand width apart
enum {
    TOY_YMM0 = 0,
    TOY_YMM1 = 1,
    TOY_YMM2 = 2,
    TOY_YMM3 = 3,
    TOY_YMM4 = 4,
    TOY_YMM5 = 5,
    TOY_YMM6 = 6,
    TOY_YMM7 = 7,
    TOY_YMM8 = 8,
    TOY_YMM9 = 9,
    TOY_YMM10 = 10,
    TOY_YMM11 = 11,
    TOY_YMM12 = 12,
    TOY_YMM13 = 13,
    TOY_YMM14 = 14,
    TOY_YMM15 = 15,
};

enum cc {
    TOY_CC_B = 2,
    TOY_CC_AE = 3,
    TOY_CC_E = 4,
    TOY_CC_NE = 5,
    TOY_CC_BE = 6,
    TOY_CC_A = 7,
    TOY_CC_L = 12,
    TOY_CC_GE = 13,
    TOY_CC_LE = 14,
    TOY_CC_G = 15,
};

struct mem {
    uint8_t base;
    bool has_index;
    uint8_t index;
    uint8_t scale;
    int32_t disp;
};

static inline struct mem mem_base(uint8_t base, int32_t disp) {
    return (struct mem){ .base = base, .scale = 1, .disp = disp };
}

static inline struct mem mem_index(uint8_t base, uint8_t index, uint8_t scale, int32_t disp) {
    return (struct mem){
        .base = base,
        .has_index = true,
        .index = index,
        .scale = scale,
        .disp = disp,
    };
}

enum {
    TOY_ASM_MAX_LABELS = 64,
    TOY_ASM_MAX_FIXUPS = 256,
};

struct fixup {
    uint8_t label;
    size_t at;
};

struct assembler {
    uint8_t *buf;
    size_t cap;
    size_t len;
    bool overflow;
    size_t labels[TOY_ASM_MAX_LABELS];
    bool labels_set[TOY_ASM_MAX_LABELS];
    struct fixup fixups[TOY_ASM_MAX_FIXUPS];
    size_t fixup_count;
};

void asm_init(struct assembler *as, uint8_t *buf, size_t cap);
bool asm_finish(struct assembler *as);

void asm_label(struct assembler *as, uint8_t label);
void asm_jmp(struct assembler *as, uint8_t label);
void asm_jcc(struct assembler *as, enum cc cc, uint8_t label);
void asm_ret(struct assembler *as);
void asm_vzeroupper(struct assembler *as);

void asm_mov_imm32(struct assembler *as, uint8_t dst, uint32_t imm);
void asm_mov_imm64(struct assembler *as, uint8_t dst, uint64_t imm);
void asm_mov_rr(struct assembler *as, uint8_t dst, uint8_t src);
void asm_mov_load(struct assembler *as, uint8_t dst, struct mem mem);
void asm_mov_store(struct assembler *as, struct mem mem, uint8_t src);
// A 16-bit store, the low half of src
void asm_mov_store16(struct assembler *as, struct mem mem, uint8_t src);
void asm_push_r64(struct assembler *as, uint8_t reg);
void asm_pop_r64(struct assembler *as, uint8_t reg);

void asm_lea(struct assembler *as, uint8_t dst, struct mem mem);

void asm_add_imm(struct assembler *as, uint8_t dst, int32_t imm);
void asm_sub_imm(struct assembler *as, uint8_t dst, int32_t imm);
void asm_add_rr(struct assembler *as, uint8_t dst, uint8_t src);
void asm_sub_rr(struct assembler *as, uint8_t dst, uint8_t src);
void asm_imul_imm(struct assembler *as, uint8_t dst, uint8_t src, int32_t imm);

void asm_cmp_imm(struct assembler *as, uint8_t reg, int32_t imm);
void asm_cmp_rr(struct assembler *as, uint8_t lhs, uint8_t rhs);
void asm_xor_rr(struct assembler *as, uint8_t dst, uint8_t src);

void asm_movsx_byte(struct assembler *as, uint8_t dst, struct mem mem);
void asm_vcvtsi2ss(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t gpr);

void asm_vmovups_load_ymm(struct assembler *as, uint8_t dst, struct mem mem);
void asm_vmovups_store_ymm(struct assembler *as, struct mem mem, uint8_t src);
void asm_vmovups_load_xmm(struct assembler *as, uint8_t dst, struct mem mem);
void asm_vmovups_store_xmm(struct assembler *as, struct mem mem, uint8_t src);
void asm_vmovss_load(struct assembler *as, uint8_t dst, struct mem mem);
void asm_vmovss_store(struct assembler *as, struct mem mem, uint8_t src);
void asm_vmulss(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t src2);
void asm_vbroadcastss(struct assembler *as, uint8_t dst, struct mem mem);

void asm_vfmadd231ps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vfmadd231ps_mem(struct assembler *as, uint8_t dst, uint8_t src2, struct mem mem);
void asm_vfmadd231ss(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vfmadd231ss_mem(struct assembler *as, uint8_t dst, uint8_t src2, struct mem mem);
// dst = src2 * dst + src3, for a Horner step where the addend is a constant
void asm_vfmadd213ps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);

void asm_vaddps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vsubps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vmulps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vxorps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vmaxps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vminps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);

void asm_vpand(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
void asm_vpor(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
void asm_vpaddd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
void asm_vpsubd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
// Shift the 32-bit lanes left or right by an immediate count
void asm_vpslld(struct assembler *as, bool wide, uint8_t dst, uint8_t src, uint8_t imm);
void asm_vpsrld(struct assembler *as, bool wide, uint8_t dst, uint8_t src, uint8_t imm);
// Truncates toward zero, so an out-of-range input yields the integer indefinite
void asm_vcvttps2dq(struct assembler *as, bool wide, uint8_t dst, uint8_t src);

void asm_vpmovsxbd(struct assembler *as, bool wide, uint8_t dst, struct mem mem);
void asm_vcvtdq2ps(struct assembler *as, bool wide, uint8_t dst, uint8_t src);
// dst = src1 with the sign of src2, a zero in src2 clears the element
void asm_vpsignb(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
// Unsigned src1 bytes times signed src2 bytes, adjacent pairs add with saturation
void asm_vpmaddubsw(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
// Signed 16-bit products, adjacent pairs add into int32 lanes
void asm_vpmaddwd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2);
void asm_vcvtph2ps(struct assembler *as, uint8_t dst, struct mem mem);
void asm_vhaddps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vpermilps(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm);
void asm_vbroadcastss_reg(struct assembler *as, uint8_t dst, uint8_t src);

void asm_vextractf128(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm);
void asm_vshufps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3, uint8_t imm);

void asm_vdivps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
// vsqrtps has no second source register
void asm_vsqrtps(struct assembler *as, bool wide, uint8_t dst, uint8_t src);
// Moves a 32-bit integer into the low xmm lane, for a constant known at generation time
void asm_vmovd_gpr(struct assembler *as, uint8_t dst, uint8_t gpr);
// Moves the low xmm lane into a 32-bit integer register
void asm_vmovd_xmm_gpr(struct assembler *as, uint8_t gpr, uint8_t src);
void asm_vinsertf128(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm);

void asm_vandps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
// dst = compare(src2, src3) per lane, the imm selects the predicate
void asm_vcmpps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3, uint8_t imm);
// Signed saturation of the int32 lanes, 128-bit sources
void asm_vpackssdw(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3);
void asm_vpacksswb(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3);
// F16C round to nearest even; dst may be the same register as src
void asm_vcvtps2ph(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm);

#endif

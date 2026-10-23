#ifndef TOY_ASM_H
#define TOY_ASM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

enum toy_cc {
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

struct toy_mem {
    uint8_t base;
    bool has_index;
    uint8_t index;
    uint8_t scale;
    int32_t disp;
};

static inline struct toy_mem toy_mem_base(uint8_t base, int32_t disp) {
    return (struct toy_mem){ .base = base, .scale = 1, .disp = disp };
}

static inline struct toy_mem toy_mem_index(uint8_t base, uint8_t index, uint8_t scale, int32_t disp) {
    return (struct toy_mem){
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

struct toy_fixup {
    uint8_t label;
    size_t at;
};

struct toy_asm {
    uint8_t *buf;
    size_t cap;
    size_t len;
    bool overflow;
    size_t labels[TOY_ASM_MAX_LABELS];
    bool labels_set[TOY_ASM_MAX_LABELS];
    struct toy_fixup fixups[TOY_ASM_MAX_FIXUPS];
    size_t fixup_count;
};

void toy_asm_init(struct toy_asm *as, uint8_t *buf, size_t cap);
bool toy_asm_finish(struct toy_asm *as);

void toy_asm_label(struct toy_asm *as, uint8_t label);
void toy_asm_jmp(struct toy_asm *as, uint8_t label);
void toy_asm_jcc(struct toy_asm *as, enum toy_cc cc, uint8_t label);
void toy_asm_ret(struct toy_asm *as);
void toy_asm_vzeroupper(struct toy_asm *as);

void toy_asm_mov_imm32(struct toy_asm *as, uint8_t dst, uint32_t imm);
void toy_asm_mov_imm64(struct toy_asm *as, uint8_t dst, uint64_t imm);
void toy_asm_mov_rr(struct toy_asm *as, uint8_t dst, uint8_t src);
void toy_asm_mov_load(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_mov_store(struct toy_asm *as, struct toy_mem mem, uint8_t src);
void toy_asm_push_r64(struct toy_asm *as, uint8_t reg);
void toy_asm_pop_r64(struct toy_asm *as, uint8_t reg);

void toy_asm_lea(struct toy_asm *as, uint8_t dst, struct toy_mem mem);

void toy_asm_add_imm(struct toy_asm *as, uint8_t dst, int32_t imm);
void toy_asm_sub_imm(struct toy_asm *as, uint8_t dst, int32_t imm);
void toy_asm_add_rr(struct toy_asm *as, uint8_t dst, uint8_t src);
void toy_asm_sub_rr(struct toy_asm *as, uint8_t dst, uint8_t src);
void toy_asm_imul_imm(struct toy_asm *as, uint8_t dst, uint8_t src, int32_t imm);

void toy_asm_cmp_imm(struct toy_asm *as, uint8_t reg, int32_t imm);
void toy_asm_cmp_rr(struct toy_asm *as, uint8_t lhs, uint8_t rhs);
void toy_asm_xor_rr(struct toy_asm *as, uint8_t dst, uint8_t src);

void toy_asm_movsx_byte(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_vcvtsi2ss(struct toy_asm *as, uint8_t dst, uint8_t src1, uint8_t gpr);

void toy_asm_vmovups_load_ymm(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_vmovups_store_ymm(struct toy_asm *as, struct toy_mem mem, uint8_t src);
void toy_asm_vmovups_load_xmm(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_vmovups_store_xmm(struct toy_asm *as, struct toy_mem mem, uint8_t src);
void toy_asm_vmovss_load(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_vmovss_store(struct toy_asm *as, struct toy_mem mem, uint8_t src);
void toy_asm_vbroadcastss(struct toy_asm *as, uint8_t dst, struct toy_mem mem);

void toy_asm_vfmadd231ps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vfmadd231ps_mem(struct toy_asm *as, uint8_t dst, uint8_t src2, struct toy_mem mem);
void toy_asm_vfmadd231ss(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vfmadd231ss_mem(struct toy_asm *as, uint8_t dst, uint8_t src2, struct toy_mem mem);

void toy_asm_vaddps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vmulps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vxorps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);

void toy_asm_vpmovsxbd(struct toy_asm *as, bool wide, uint8_t dst, struct toy_mem mem);
void toy_asm_vcvtdq2ps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src);
void toy_asm_vcvtph2ps(struct toy_asm *as, uint8_t dst, struct toy_mem mem);
void toy_asm_vhaddps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vpermilps(struct toy_asm *as, uint8_t dst, uint8_t src, uint8_t imm);
void toy_asm_vbroadcastss_reg(struct toy_asm *as, uint8_t dst, uint8_t src);

void toy_asm_vextractf128(struct toy_asm *as, uint8_t dst, uint8_t src, uint8_t imm);
void toy_asm_vshufps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3, uint8_t imm);

void toy_asm_vdivps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3);
void toy_asm_vsqrtps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src);
void toy_asm_vmovd_gpr(struct toy_asm *as, uint8_t dst, uint8_t gpr);
void toy_asm_vinsertf128(struct toy_asm *as, uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm);

#endif

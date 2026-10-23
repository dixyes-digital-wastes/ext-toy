#include "toy/asm.h"

#include <limits.h>

static void emit_u8(struct toy_asm *as, uint8_t v) {
    if (as->overflow) {
        return;
    }
    if (as->len >= as->cap) {
        as->overflow = true;
        return;
    }
    as->buf[as->len++] = v;
}

static void emit_u32(struct toy_asm *as, uint32_t v) {
    emit_u8(as, (uint8_t)(v & 0xFF));
    emit_u8(as, (uint8_t)((v >> 8) & 0xFF));
    emit_u8(as, (uint8_t)((v >> 16) & 0xFF));
    emit_u8(as, (uint8_t)((v >> 24) & 0xFF));
}

static void emit_u64(struct toy_asm *as, uint64_t v) {
    emit_u32(as, (uint32_t)(v & 0xFFFFFFFF));
    emit_u32(as, (uint32_t)(v >> 32));
}

static void emit_rex(struct toy_asm *as, bool w, uint8_t reg, uint8_t index, uint8_t base) {
    uint8_t rex = (uint8_t)(0x40 | (w ? 0x08 : 0) | (((reg >> 3) & 1) << 2) |
                            (((index >> 3) & 1) << 1) | ((base >> 3) & 1));
    if (rex != 0x40) {
        emit_u8(as, rex);
    }
}

static uint8_t scale_bits(uint8_t scale) {
    switch (scale) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        default:
            return 0xFF;
    }
}

static void emit_modrm_mem(struct toy_asm *as, uint8_t reg, const struct toy_mem *mem) {
    uint8_t ss = scale_bits(mem->scale);
    if (ss == 0xFF || mem->base > 15 || (mem->has_index && mem->index > 15)) {
        as->overflow = true;
        return;
    }

    bool need_sib = mem->has_index || (mem->base & 7) == 4;
    bool zero_disp_ok = (mem->base & 7) != 5;
    uint8_t mod;
    if (mem->disp == 0 && zero_disp_ok) {
        mod = 0;
    } else if (mem->disp >= -128 && mem->disp <= 127) {
        mod = 1;
    } else {
        mod = 2;
    }
    uint8_t rm = need_sib ? 4 : (uint8_t)(mem->base & 7);

    emit_u8(as, (uint8_t)((mod << 6) | ((reg & 7) << 3) | rm));
    if (need_sib) {
        uint8_t index = mem->has_index ? (uint8_t)(mem->index & 7) : 4;
        emit_u8(as, (uint8_t)((ss << 6) | (index << 3) | (mem->base & 7)));
    }
    if (mod == 1) {
        emit_u8(as, (uint8_t)(int8_t)mem->disp);
    } else if (mod == 2) {
        emit_u32(as, (uint32_t)mem->disp);
    }
}

static void emit_modrm_reg(struct toy_asm *as, uint8_t reg, uint8_t rm) {
    emit_u8(as, (uint8_t)(0xC0 | ((reg & 7) << 3) | (rm & 7)));
}

static void emit_rel32(struct toy_asm *as, uint8_t label) {
    if (label >= TOY_ASM_MAX_LABELS || as->fixup_count >= TOY_ASM_MAX_FIXUPS) {
        as->overflow = true;
        return;
    }
    as->fixups[as->fixup_count++] = (struct toy_fixup){ .label = label, .at = as->len };
    emit_u32(as, 0);
}

static void emit_gpr_mem(struct toy_asm *as, bool w, uint8_t reg, struct toy_mem mem, uint8_t opcode) {
    emit_rex(as, w, reg, mem.has_index ? mem.index : 0, mem.base);
    emit_u8(as, opcode);
    emit_modrm_mem(as, reg, &mem);
}

static void emit_gpr_imm(struct toy_asm *as, uint8_t ext, uint8_t rm, int32_t imm) {
    emit_rex(as, true, 0, 0, rm);
    if (imm >= -128 && imm <= 127) {
        emit_u8(as, 0x83);
        emit_modrm_reg(as, ext, rm);
        emit_u8(as, (uint8_t)(int8_t)imm);
    } else {
        emit_u8(as, 0x81);
        emit_modrm_reg(as, ext, rm);
        emit_u32(as, (uint32_t)imm);
    }
}

static void emit_vex(struct toy_asm *as, uint8_t mmmmm, uint8_t pp, bool l, bool w, uint8_t vvvv,
                     uint8_t reg, bool has_mem, const struct toy_mem *mem, uint8_t rm, uint8_t opcode) {
    uint8_t r = (uint8_t)((reg >> 3) & 1);
    uint8_t x = (has_mem && mem->has_index) ? (uint8_t)((mem->index >> 3) & 1) : 0;
    uint8_t b = has_mem ? (uint8_t)((mem->base >> 3) & 1) : (uint8_t)((rm >> 3) & 1);

    emit_u8(as, 0xC4);
    emit_u8(as, (uint8_t)(((r ^ 1) << 7) | ((x ^ 1) << 6) | ((b ^ 1) << 5) | (mmmmm & 0x1F)));
    emit_u8(as, (uint8_t)((w ? 0x80 : 0) | ((vvvv ^ 0xF) << 3) | (l ? 0x04 : 0) | (pp & 0x03)));
    emit_u8(as, opcode);
    if (has_mem) {
        emit_modrm_mem(as, reg, mem);
    } else {
        emit_modrm_reg(as, reg, rm);
    }
}

void toy_asm_init(struct toy_asm *as, uint8_t *buf, size_t cap) {
    *as = (struct toy_asm){ .buf = buf, .cap = cap };
}

bool toy_asm_finish(struct toy_asm *as) {
    if (as->overflow) {
        return false;
    }
    for (size_t i = 0; i < as->fixup_count; i++) {
        const struct toy_fixup *fx = &as->fixups[i];
        if (!as->labels_set[fx->label]) {
            return false;
        }
        int64_t rel = (int64_t)as->labels[fx->label] - (int64_t)(fx->at + 4);
        if (rel < INT32_MIN || rel > INT32_MAX) {
            return false;
        }
        uint32_t v = (uint32_t)(int32_t)rel;
        for (size_t k = 0; k < 4; k++) {
            as->buf[fx->at + k] = (uint8_t)(v >> (8 * k));
        }
    }
    return true;
}

void toy_asm_label(struct toy_asm *as, uint8_t label) {
    if (label >= TOY_ASM_MAX_LABELS || as->labels_set[label]) {
        as->overflow = true;
        return;
    }
    as->labels[label] = as->len;
    as->labels_set[label] = true;
}

void toy_asm_jmp(struct toy_asm *as, uint8_t label) {
    emit_u8(as, 0xE9);
    emit_rel32(as, label);
}

void toy_asm_jcc(struct toy_asm *as, enum toy_cc cc, uint8_t label) {
    emit_u8(as, 0x0F);
    emit_u8(as, (uint8_t)(0x80 | (cc & 0x0F)));
    emit_rel32(as, label);
}

void toy_asm_ret(struct toy_asm *as) {
    emit_u8(as, 0xC3);
}

void toy_asm_vzeroupper(struct toy_asm *as) {
    emit_u8(as, 0xC5);
    emit_u8(as, 0xF8);
    emit_u8(as, 0x77);
}

void toy_asm_mov_imm32(struct toy_asm *as, uint8_t dst, uint32_t imm) {
    emit_rex(as, false, 0, 0, dst);
    emit_u8(as, (uint8_t)(0xB8 + (dst & 7)));
    emit_u32(as, imm);
}

void toy_asm_mov_imm64(struct toy_asm *as, uint8_t dst, uint64_t imm) {
    emit_rex(as, true, 0, 0, dst);
    emit_u8(as, (uint8_t)(0xB8 + (dst & 7)));
    emit_u64(as, imm);
}

void toy_asm_mov_rr(struct toy_asm *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x89);
    emit_modrm_reg(as, src, dst);
}

void toy_asm_push_r64(struct toy_asm *as, uint8_t reg) {
    if (reg >= 8) {
        emit_u8(as, 0x41);
    }
    emit_u8(as, (uint8_t)(0x50 + (reg & 7)));
}

void toy_asm_pop_r64(struct toy_asm *as, uint8_t reg) {
    if (reg >= 8) {
        emit_u8(as, 0x41);
    }
    emit_u8(as, (uint8_t)(0x58 + (reg & 7)));
}

void toy_asm_mov_load(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_gpr_mem(as, true, dst, mem, 0x8B);
}

void toy_asm_mov_store(struct toy_asm *as, struct toy_mem mem, uint8_t src) {
    emit_gpr_mem(as, true, src, mem, 0x89);
}

void toy_asm_lea(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_gpr_mem(as, true, dst, mem, 0x8D);
}

void toy_asm_add_imm(struct toy_asm *as, uint8_t dst, int32_t imm) {
    emit_gpr_imm(as, 0, dst, imm);
}

void toy_asm_sub_imm(struct toy_asm *as, uint8_t dst, int32_t imm) {
    emit_gpr_imm(as, 5, dst, imm);
}

void toy_asm_add_rr(struct toy_asm *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x01);
    emit_modrm_reg(as, src, dst);
}

void toy_asm_sub_rr(struct toy_asm *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x29);
    emit_modrm_reg(as, src, dst);
}

void toy_asm_imul_imm(struct toy_asm *as, uint8_t dst, uint8_t src, int32_t imm) {
    emit_rex(as, true, dst, 0, src);
    emit_u8(as, 0x69);
    emit_modrm_reg(as, dst, src);
    emit_u32(as, (uint32_t)imm);
}

void toy_asm_cmp_imm(struct toy_asm *as, uint8_t reg, int32_t imm) {
    emit_gpr_imm(as, 7, reg, imm);
}

void toy_asm_cmp_rr(struct toy_asm *as, uint8_t lhs, uint8_t rhs) {
    emit_rex(as, true, rhs, 0, lhs);
    emit_u8(as, 0x39);
    emit_modrm_reg(as, rhs, lhs);
}

void toy_asm_xor_rr(struct toy_asm *as, uint8_t dst, uint8_t src) {
    emit_rex(as, false, src, 0, dst);
    emit_u8(as, 0x31);
    emit_modrm_reg(as, src, dst);
}

void toy_asm_movsx_byte(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_rex(as, false, dst, mem.has_index ? mem.index : 0, mem.base);
    emit_u8(as, 0x0F);
    emit_u8(as, 0xBE);
    emit_modrm_mem(as, dst, &mem);
}

void toy_asm_vcvtsi2ss(struct toy_asm *as, uint8_t dst, uint8_t src1, uint8_t gpr) {
    emit_vex(as, 1, 2, false, false, src1, dst, false, NULL, gpr, 0x2A);
}

void toy_asm_vmovups_load_ymm(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 1, 0, true, false, 0, dst, true, &mem, 0, 0x10);
}

void toy_asm_vmovups_store_ymm(struct toy_asm *as, struct toy_mem mem, uint8_t src) {
    emit_vex(as, 1, 0, true, false, 0, src, true, &mem, 0, 0x11);
}

void toy_asm_vmovups_load_xmm(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 1, 0, false, false, 0, dst, true, &mem, 0, 0x10);
}

void toy_asm_vmovups_store_xmm(struct toy_asm *as, struct toy_mem mem, uint8_t src) {
    emit_vex(as, 1, 0, false, false, 0, src, true, &mem, 0, 0x11);
}

void toy_asm_vmovss_load(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 1, 2, false, false, 0, dst, true, &mem, 0, 0x10);
}

void toy_asm_vmovss_store(struct toy_asm *as, struct toy_mem mem, uint8_t src) {
    emit_vex(as, 1, 2, false, false, 0, src, true, &mem, 0, 0x11);
}

void toy_asm_vbroadcastss(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 2, 1, true, false, 0, dst, true, &mem, 0, 0x18);
}

void toy_asm_vfmadd231ps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 2, 1, true, false, src2, dst, false, NULL, src3, 0xB8);
}

void toy_asm_vfmadd231ps_mem(struct toy_asm *as, uint8_t dst, uint8_t src2, struct toy_mem mem) {
    emit_vex(as, 2, 1, true, false, src2, dst, true, &mem, 0, 0xB8);
}

void toy_asm_vfmadd231ss(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 2, 1, false, false, src2, dst, false, NULL, src3, 0xB9);
}

void toy_asm_vfmadd231ss_mem(struct toy_asm *as, uint8_t dst, uint8_t src2, struct toy_mem mem) {
    emit_vex(as, 2, 1, false, false, src2, dst, true, &mem, 0, 0xB9);
}

void toy_asm_vaddps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x58);
}

void toy_asm_vmulps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x59);
}

void toy_asm_vxorps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x57);
}

void toy_asm_vpmovsxbd(struct toy_asm *as, bool wide, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 2, 1, wide, false, 0, dst, true, &mem, 0, 0x21);
}

void toy_asm_vcvtdq2ps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src) {
    emit_vex(as, 1, 0, wide, false, 0, dst, false, NULL, src, 0x5B);
}

void toy_asm_vcvtph2ps(struct toy_asm *as, uint8_t dst, struct toy_mem mem) {
    emit_vex(as, 2, 1, false, false, 0, dst, true, &mem, 0, 0x13);
}

void toy_asm_vhaddps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 3, false, false, src2, dst, false, NULL, src3, 0x7C);
}

void toy_asm_vpermilps(struct toy_asm *as, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 3, 1, false, false, 0, dst, false, NULL, src, 0x04);
    emit_u8(as, imm);
}

void toy_asm_vbroadcastss_reg(struct toy_asm *as, uint8_t dst, uint8_t src) {
    emit_vex(as, 2, 1, true, false, 0, dst, false, NULL, src, 0x18);
}

void toy_asm_vextractf128(struct toy_asm *as, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 3, 1, true, false, 0, src, false, NULL, dst, 0x19);
    emit_u8(as, imm);
}

void toy_asm_vshufps(struct toy_asm *as, uint8_t dst, uint8_t src2, uint8_t src3, uint8_t imm) {
    emit_vex(as, 1, 0, false, false, src2, dst, false, NULL, src3, 0xC6);
    emit_u8(as, imm);
}

void toy_asm_vdivps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x5E);
}

void toy_asm_vsqrtps(struct toy_asm *as, bool wide, uint8_t dst, uint8_t src) {
    emit_vex(as, 1, 0, wide, false, 0, dst, false, NULL, src, 0x51);
}

void toy_asm_vmovd_gpr(struct toy_asm *as, uint8_t dst, uint8_t gpr) {
    emit_vex(as, 1, 1, false, false, 0, dst, false, NULL, gpr, 0x6E);
}

void toy_asm_vinsertf128(struct toy_asm *as, uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm) {
    emit_vex(as, 3, 1, true, false, src1, dst, false, NULL, src2, 0x38);
    emit_u8(as, imm);
}

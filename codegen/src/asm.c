#include "toy/asm.h"

#include <limits.h>

static void emit_u8(struct assembler *as, uint8_t v) {
    if (as->overflow) {
        return;
    }
    if (as->len >= as->cap) {
        as->overflow = true;
        return;
    }
    as->buf[as->len++] = v;
}

static void emit_u32(struct assembler *as, uint32_t v) {
    emit_u8(as, (uint8_t)(v & 0xFF));
    emit_u8(as, (uint8_t)((v >> 8) & 0xFF));
    emit_u8(as, (uint8_t)((v >> 16) & 0xFF));
    emit_u8(as, (uint8_t)((v >> 24) & 0xFF));
}

static void emit_u64(struct assembler *as, uint64_t v) {
    emit_u32(as, (uint32_t)(v & 0xFFFFFFFF));
    emit_u32(as, (uint32_t)(v >> 32));
}

static void emit_rex(struct assembler *as, bool w, uint8_t reg, uint8_t index, uint8_t base) {
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

static void emit_modrm_mem(struct assembler *as, uint8_t reg, const struct mem *mem) {
    uint8_t ss = scale_bits(mem->scale);
    if (ss == 0xFF || mem->base > 15 || (mem->has_index && mem->index > 15)) {
        as->overflow = true;
        return;
    }

    // rsp and r12 cannot be the rm operand directly, they need a SIB byte
    bool need_sib = mem->has_index || (mem->base & 7) == 4;
    // mod=00 with rbp or r13 means RIP-relative, so encode a zero displacement too
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

static void emit_modrm_reg(struct assembler *as, uint8_t reg, uint8_t rm) {
    emit_u8(as, (uint8_t)(0xC0 | ((reg & 7) << 3) | (rm & 7)));
}

static void emit_rel32(struct assembler *as, uint8_t label) {
    if (label >= TOY_ASM_MAX_LABELS || as->fixup_count >= TOY_ASM_MAX_FIXUPS) {
        as->overflow = true;
        return;
    }
    as->fixups[as->fixup_count++] = (struct fixup){ .label = label, .at = as->len };
    emit_u32(as, 0);
}

static void emit_gpr_mem(struct assembler *as, bool w, uint8_t reg, struct mem mem, uint8_t opcode) {
    emit_rex(as, w, reg, mem.has_index ? mem.index : 0, mem.base);
    emit_u8(as, opcode);
    emit_modrm_mem(as, reg, &mem);
}

static void emit_gpr_imm(struct assembler *as, uint8_t ext, uint8_t rm, int32_t imm) {
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

// VEX prefix and opcode, then ModRM and an optional SIB byte and displacement
// vvvv is the real register number, the encoding inverts it. Reserved bits take 0,
// so the encoded bits read 1111
static void emit_vex(struct assembler *as, uint8_t mmmmm, uint8_t pp, bool l, bool w, uint8_t vvvv,
                     uint8_t reg, bool has_mem, const struct mem *mem, uint8_t rm, uint8_t opcode) {
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

void asm_init(struct assembler *as, uint8_t *buf, size_t cap) {
    *as = (struct assembler){ .buf = buf, .cap = cap };
}

bool asm_finish(struct assembler *as) {
    if (as->overflow) {
        return false;
    }
    for (size_t i = 0; i < as->fixup_count; i++) {
        const struct fixup *fx = &as->fixups[i];
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

void asm_label(struct assembler *as, uint8_t label) {
    if (label >= TOY_ASM_MAX_LABELS || as->labels_set[label]) {
        as->overflow = true;
        return;
    }
    as->labels[label] = as->len;
    as->labels_set[label] = true;
}

void asm_jmp(struct assembler *as, uint8_t label) {
    emit_u8(as, 0xE9);
    emit_rel32(as, label);
}

void asm_jcc(struct assembler *as, enum cc cc, uint8_t label) {
    emit_u8(as, 0x0F);
    emit_u8(as, (uint8_t)(0x80 | (cc & 0x0F)));
    emit_rel32(as, label);
}

void asm_ret(struct assembler *as) {
    emit_u8(as, 0xC3);
}

void asm_vzeroupper(struct assembler *as) {
    emit_u8(as, 0xC5);
    emit_u8(as, 0xF8);
    emit_u8(as, 0x77);
}

void asm_mov_imm32(struct assembler *as, uint8_t dst, uint32_t imm) {
    emit_rex(as, false, 0, 0, dst);
    emit_u8(as, (uint8_t)(0xB8 + (dst & 7)));
    emit_u32(as, imm);
}

void asm_mov_imm64(struct assembler *as, uint8_t dst, uint64_t imm) {
    emit_rex(as, true, 0, 0, dst);
    emit_u8(as, (uint8_t)(0xB8 + (dst & 7)));
    emit_u64(as, imm);
}

void asm_mov_rr(struct assembler *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x89);
    emit_modrm_reg(as, src, dst);
}

void asm_push_r64(struct assembler *as, uint8_t reg) {
    if (reg >= 8) {
        emit_u8(as, 0x41);
    }
    emit_u8(as, (uint8_t)(0x50 + (reg & 7)));
}

void asm_pop_r64(struct assembler *as, uint8_t reg) {
    if (reg >= 8) {
        emit_u8(as, 0x41);
    }
    emit_u8(as, (uint8_t)(0x58 + (reg & 7)));
}

void asm_mov_load(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_gpr_mem(as, true, dst, mem, 0x8B);
}

void asm_mov_store(struct assembler *as, struct mem mem, uint8_t src) {
    emit_gpr_mem(as, true, src, mem, 0x89);
}

void asm_mov_store16(struct assembler *as, struct mem mem, uint8_t src) {
    emit_u8(as, 0x66);
    emit_rex(as, false, src, mem.has_index ? mem.index : 0, mem.base);
    emit_u8(as, 0x89);
    emit_modrm_mem(as, src, &mem);
}

void asm_lea(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_gpr_mem(as, true, dst, mem, 0x8D);
}

void asm_add_imm(struct assembler *as, uint8_t dst, int32_t imm) {
    emit_gpr_imm(as, 0, dst, imm);
}

void asm_sub_imm(struct assembler *as, uint8_t dst, int32_t imm) {
    emit_gpr_imm(as, 5, dst, imm);
}

void asm_add_rr(struct assembler *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x01);
    emit_modrm_reg(as, src, dst);
}

void asm_sub_rr(struct assembler *as, uint8_t dst, uint8_t src) {
    emit_rex(as, true, src, 0, dst);
    emit_u8(as, 0x29);
    emit_modrm_reg(as, src, dst);
}

void asm_imul_imm(struct assembler *as, uint8_t dst, uint8_t src, int32_t imm) {
    emit_rex(as, true, dst, 0, src);
    emit_u8(as, 0x69);
    emit_modrm_reg(as, dst, src);
    emit_u32(as, (uint32_t)imm);
}

void asm_cmp_imm(struct assembler *as, uint8_t reg, int32_t imm) {
    emit_gpr_imm(as, 7, reg, imm);
}

void asm_cmp_rr(struct assembler *as, uint8_t lhs, uint8_t rhs) {
    emit_rex(as, true, rhs, 0, lhs);
    emit_u8(as, 0x39);
    emit_modrm_reg(as, rhs, lhs);
}

void asm_xor_rr(struct assembler *as, uint8_t dst, uint8_t src) {
    emit_rex(as, false, src, 0, dst);
    emit_u8(as, 0x31);
    emit_modrm_reg(as, src, dst);
}

void asm_movsx_byte(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_rex(as, false, dst, mem.has_index ? mem.index : 0, mem.base);
    emit_u8(as, 0x0F);
    emit_u8(as, 0xBE);
    emit_modrm_mem(as, dst, &mem);
}

void asm_vcvtsi2ss(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t gpr) {
    emit_vex(as, 1, 2, false, false, src1, dst, false, NULL, gpr, 0x2A);
}

void asm_vmovups_load_ymm(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_vex(as, 1, 0, true, false, 0, dst, true, &mem, 0, 0x10);
}

void asm_vmovups_store_ymm(struct assembler *as, struct mem mem, uint8_t src) {
    emit_vex(as, 1, 0, true, false, 0, src, true, &mem, 0, 0x11);
}

void asm_vmovups_load_xmm(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_vex(as, 1, 0, false, false, 0, dst, true, &mem, 0, 0x10);
}

void asm_vmovups_store_xmm(struct assembler *as, struct mem mem, uint8_t src) {
    emit_vex(as, 1, 0, false, false, 0, src, true, &mem, 0, 0x11);
}

void asm_vmovss_load(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_vex(as, 1, 2, false, false, 0, dst, true, &mem, 0, 0x10);
}

void asm_vmovss_store(struct assembler *as, struct mem mem, uint8_t src) {
    emit_vex(as, 1, 2, false, false, 0, src, true, &mem, 0, 0x11);
}

void asm_vmulss(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 2, false, false, src1, dst, false, NULL, src2, 0x59);
}

void asm_vbroadcastss(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_vex(as, 2, 1, true, false, 0, dst, true, &mem, 0, 0x18);
}

void asm_vfmadd231ps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 2, 1, true, false, src2, dst, false, NULL, src3, 0xB8);
}

void asm_vfmadd213ps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 2, 1, wide, false, src2, dst, false, NULL, src3, 0xA8);
}

void asm_vfmadd231ps_mem(struct assembler *as, uint8_t dst, uint8_t src2, struct mem mem) {
    emit_vex(as, 2, 1, true, false, src2, dst, true, &mem, 0, 0xB8);
}

void asm_vfmadd231ss(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 2, 1, false, false, src2, dst, false, NULL, src3, 0xB9);
}

void asm_vfmadd231ss_mem(struct assembler *as, uint8_t dst, uint8_t src2, struct mem mem) {
    emit_vex(as, 2, 1, false, false, src2, dst, true, &mem, 0, 0xB9);
}

void asm_vaddps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x58);
}

void asm_vsubps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x5C);
}

void asm_vmulps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x59);
}

void asm_vmaxps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x5F);
}

void asm_vminps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x5D);
}

void asm_vpand(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 1, wide, false, src1, dst, false, NULL, src2, 0xDB);
}

void asm_vpor(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 1, wide, false, src1, dst, false, NULL, src2, 0xEB);
}

void asm_vpaddd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 1, wide, false, src1, dst, false, NULL, src2, 0xFE);
}

void asm_vpsubd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 1, wide, false, src1, dst, false, NULL, src2, 0xFA);
}

// The shift count is an immediate, so VEX.vvvv carries the destination
void asm_vpslld(struct assembler *as, bool wide, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 1, 1, wide, false, dst, 6, false, NULL, src, 0x72);
    emit_u8(as, imm);
}

void asm_vpsrld(struct assembler *as, bool wide, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 1, 1, wide, false, dst, 2, false, NULL, src, 0x72);
    emit_u8(as, imm);
}

void asm_vcvttps2dq(struct assembler *as, bool wide, uint8_t dst, uint8_t src) {
    emit_vex(as, 1, 2, wide, false, 0, dst, false, NULL, src, 0x5B);
}

void asm_vxorps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x57);
}

// The destination can be m128, so the destination goes in rm and the source ymm in reg
void asm_vpmovsxbd(struct assembler *as, bool wide, uint8_t dst, struct mem mem) {
    emit_vex(as, 2, 1, wide, false, 0, dst, true, &mem, 0, 0x21);
}

void asm_vcvtdq2ps(struct assembler *as, bool wide, uint8_t dst, uint8_t src) {
    emit_vex(as, 1, 0, wide, false, 0, dst, false, NULL, src, 0x5B);
}

void asm_vpsignb(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 2, 1, wide, false, src1, dst, false, NULL, src2, 0x08);
}

void asm_vpmaddubsw(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 2, 1, wide, false, src1, dst, false, NULL, src2, 0x04);
}

void asm_vpmaddwd(struct assembler *as, bool wide, uint8_t dst, uint8_t src1, uint8_t src2) {
    emit_vex(as, 1, 1, wide, false, src1, dst, false, NULL, src2, 0xF5);
}

void asm_vcvtph2ps(struct assembler *as, uint8_t dst, struct mem mem) {
    emit_vex(as, 2, 1, false, false, 0, dst, true, &mem, 0, 0x13);
}

void asm_vhaddps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 3, false, false, src2, dst, false, NULL, src3, 0x7C);
}

void asm_vpermilps(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 3, 1, false, false, 0, dst, false, NULL, src, 0x04);
    emit_u8(as, imm);
}

void asm_vbroadcastss_reg(struct assembler *as, uint8_t dst, uint8_t src) {
    emit_vex(as, 2, 1, true, false, 0, dst, false, NULL, src, 0x18);
}

void asm_vextractf128(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 3, 1, true, false, 0, src, false, NULL, dst, 0x19);
    emit_u8(as, imm);
}

void asm_vshufps(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3, uint8_t imm) {
    emit_vex(as, 1, 0, false, false, src2, dst, false, NULL, src3, 0xC6);
    emit_u8(as, imm);
}

void asm_vdivps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x5E);
}

// vsqrtps has no second source register, so VEX.vvvv stays 1111
void asm_vsqrtps(struct assembler *as, bool wide, uint8_t dst, uint8_t src) {
    emit_vex(as, 1, 0, wide, false, 0, dst, false, NULL, src, 0x51);
}

void asm_vmovd_gpr(struct assembler *as, uint8_t dst, uint8_t gpr) {
    emit_vex(as, 1, 1, false, false, 0, dst, false, NULL, gpr, 0x6E);
}

void asm_vinsertf128(struct assembler *as, uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm) {
    emit_vex(as, 3, 1, true, false, src1, dst, false, NULL, src2, 0x38);
    emit_u8(as, imm);
}

void asm_vmovd_xmm_gpr(struct assembler *as, uint8_t gpr, uint8_t src) {
    emit_vex(as, 1, 1, false, false, 0, src, false, NULL, gpr, 0x7E);
}

void asm_vandps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0x54);
}

void asm_vcmpps(struct assembler *as, bool wide, uint8_t dst, uint8_t src2, uint8_t src3,
                uint8_t imm) {
    emit_vex(as, 1, 0, wide, false, src2, dst, false, NULL, src3, 0xC2);
    emit_u8(as, imm);
}

void asm_vpackssdw(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 1, false, false, src2, dst, false, NULL, src3, 0x6B);
}

void asm_vpacksswb(struct assembler *as, uint8_t dst, uint8_t src2, uint8_t src3) {
    emit_vex(as, 1, 1, false, false, src2, dst, false, NULL, src3, 0x63);
}

// The destination is the rm operand for the register form, the same as the
// memory form, so reg holds the source
void asm_vcvtps2ph(struct assembler *as, uint8_t dst, uint8_t src, uint8_t imm) {
    emit_vex(as, 3, 1, false, false, 0, src, false, NULL, dst, 0x1D);
    emit_u8(as, imm);
}

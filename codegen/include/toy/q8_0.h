#ifndef TOY_Q8_0_H
#define TOY_Q8_0_H

#include <stddef.h>
#include <stdint.h>

enum {
    TOY_Q8_0_BLOCK = 32,
    TOY_Q8_0_BLOCK_BYTES = 34,
};

struct toy_q8_0_block {
    uint16_t d;
    int8_t qs[TOY_Q8_0_BLOCK];
};

static_assert(sizeof(struct toy_q8_0_block) == TOY_Q8_0_BLOCK_BYTES,
              "the q8_0 block is 34 bytes");
static_assert(offsetof(struct toy_q8_0_block, qs) == 2,
              "the f16 scale comes first");

float toy_f16_to_f32(uint16_t h);
uint16_t toy_f32_to_f16(float f);

size_t toy_q8_0_row_bytes(uint32_t k);

void toy_q8_0_quantize_rows(uint32_t rows, uint32_t k, const float *src, void *dst);
void toy_q8_0_dequantize_rows(uint32_t rows, uint32_t k, const void *src, float *dst);

#endif

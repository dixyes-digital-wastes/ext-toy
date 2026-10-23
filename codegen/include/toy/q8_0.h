#ifndef TOY_Q8_0_H
#define TOY_Q8_0_H

#include <stddef.h>
#include <stdint.h>

/*
 * The q8_0 block layout is the same as block_q8_0 in llama.cpp/ggml:
 * an f16 scale followed by 32 int8 values, 34 bytes in total. The weights
 * are a series of blocks
 */
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

// Portable half to float conversion, does not use F16C
float toy_f16_to_f32(uint16_t h);
uint16_t toy_f32_to_f16(float f);

size_t toy_q8_0_row_bytes(uint32_t k);

// k must be a multiple of 32. The quantization rule matches ggml: d = amax / 127, q = roundf(x / d)
void toy_q8_0_quantize_rows(uint32_t rows, uint32_t k, const float *src, void *dst);
void toy_q8_0_dequantize_rows(uint32_t rows, uint32_t k, const void *src, float *dst);

#endif

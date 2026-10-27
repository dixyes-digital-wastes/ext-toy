#ifndef TENSOR_H
#define TENSOR_H

#include "attributes.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Marks the exported C function. The extension build also defines this macro
 * in php_toy.h. This copy keeps a standalone build of the tensor core complete
 */
#ifndef TOY_API
# ifdef _WIN32
#  define TOY_API __declspec(dllexport)
# elif defined(__GNUC__) || defined(__clang__)
#  define TOY_API __attribute__((visibility("default")))
# else
#  define TOY_API
# endif
#endif

enum {
    TOY_TENSOR_MAX_DIMS = 4,
    TOY_TENSOR_ALIGNMENT = 64,
    TOY_TENSOR_Q8_0_BLOCK_VALUES = 32,
    TOY_TENSOR_Q8_0_BLOCK_BYTES = 34,
};

static_assert(TOY_TENSOR_MAX_DIMS == 4, "tensor rank limit is part of the ABI");
static_assert(TOY_TENSOR_ALIGNMENT == 64, "tensor buffers are 64-byte aligned");
static_assert(TOY_TENSOR_Q8_0_BLOCK_VALUES == 32, "q8_0 block size matches ggml");
static_assert(TOY_TENSOR_Q8_0_BLOCK_BYTES == 34, "q8_0 block layout matches ggml");
static_assert(sizeof(float) == 4, "f32 requires a 32-bit float");

typedef enum dtype {
    TOY_DTYPE_INVALID = 0,
    TOY_DTYPE_F32 = 1,
    TOY_DTYPE_Q8_0 = 2,
    TOY_DTYPE_F16 = 3,
    TOY_DTYPE_BF16 = 4,
    TOY_DTYPE_I8 = 5,
} dtype;

static_assert(TOY_DTYPE_INVALID == 0, "dtype 0 is invalid");
static_assert(TOY_DTYPE_F32 == 1 && TOY_DTYPE_Q8_0 == 2 && TOY_DTYPE_F16 == 3
    && TOY_DTYPE_BF16 == 4 && TOY_DTYPE_I8 == 5,
    "dtype values match the spec ABI");

typedef enum layout {
    TOY_LAYOUT_INVALID = 0,
    TOY_LAYOUT_ROW_MAJOR = 1,
} layout;

typedef enum tensor_flags {
    TOY_TENSOR_FLAG_OWNS_DATA = 0x1u,
    TOY_TENSOR_FLAG_ALIAS = 0x2u,
} tensor_flags;

typedef enum tensor_status {
    TOY_OK = 0,
    TOY_ERROR_DTYPE,
    TOY_ERROR_SHAPE,
    TOY_ERROR_BYTES,
    TOY_ERROR_OVERFLOW,
    TOY_ERROR_MEMORY,
} tensor_status;

typedef struct buffer {
    uint32_t refcount;
    size_t bytes;
    void *data;
} buffer;

typedef void (*kernel_entry)(const void *a, const void *b, void *c);

/*
 * Element-wise kernel memo
 * The key packs kind and dtype with the element count. It stays valid across
 * reshape, which preserves the element count, and a cast allocates a new
 * tensor with an empty memo
 */
typedef struct tensor_memo {
    kernel_entry kernel;
    uint64_t key;
    uint32_t param_bits;
} tensor_memo;

typedef struct tensor {
    uint32_t refcount;
    uint8_t dtype;
    uint8_t rank;
    uint8_t layout;
    uint8_t flags;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    size_t bytes;
    void *data;
    buffer *buffer;
    tensor_memo memo;
} tensor;

/* Returns NULL for an unknown dtype */
const char *dtype_to_name(dtype dtype_value);

TOY_NODISCARD bool dtype_from_name(const char *name, size_t length, dtype *out_dtype);

/* Computes the byte count of a shape; q8_0 needs a last dim divisible by 32 */
TOY_NODISCARD tensor_status tensor_shape_bytes(dtype dtype_value, uint8_t rank,
                                               const uint32_t *dims, size_t *bytes);

/* Allocates a zero-filled tensor with its own 64-byte aligned buffer */
TOY_NODISCARD tensor *tensor_alloc(dtype dtype_value, uint8_t rank,
                                   const uint32_t *dims, tensor_status *status);

/* Wraps external memory; the tensor does not own the memory. This is the only
 * exported C function */
TOY_NODISCARD TOY_API tensor *toy_tensor_alias(void *data, size_t bytes,
                                               dtype dtype_value, uint8_t rank,
                                               const uint32_t dims[TOY_TENSOR_MAX_DIMS]);

tensor *tensor_retain(tensor *self);
void tensor_release(tensor *self);

/* Reshapes in place; the byte count must not change */
TOY_NODISCARD tensor_status tensor_reshape(tensor *self, uint8_t rank,
                                           const uint32_t *dims);

/* Copies into a newly allocated tensor */
TOY_NODISCARD tensor *tensor_clone(const tensor *self, tensor_status *status);

/* Fills every element with one scalar; f32 and q8_0 only */
TOY_NODISCARD tensor_status tensor_fill(tensor *self, float value);

/*
 * Converts into a newly allocated tensor of another dtype; f32 and q8_0 only
 * Same-dtype conversion copies the bytes
 */
TOY_NODISCARD tensor *tensor_cast(const tensor *self, dtype dtype_value,
                                  tensor_status *status);

size_t tensor_numel(const tensor *self);

static inline void *tensor_data(const tensor *self)
{
    return self->data;
}

#endif /* TENSOR_H */

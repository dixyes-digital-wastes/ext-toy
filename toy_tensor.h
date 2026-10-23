#ifndef TOY_TENSOR_H
#define TOY_TENSOR_H

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

typedef enum toy_dtype {
    TOY_DTYPE_INVALID = 0,
    TOY_DTYPE_F32 = 1,
    TOY_DTYPE_Q8_0 = 2,
    TOY_DTYPE_F16 = 3,
    TOY_DTYPE_BF16 = 4,
    TOY_DTYPE_I8 = 5,
} toy_dtype;

static_assert(TOY_DTYPE_INVALID == 0, "dtype 0 is invalid");
static_assert(TOY_DTYPE_F32 == 1 && TOY_DTYPE_Q8_0 == 2 && TOY_DTYPE_F16 == 3
    && TOY_DTYPE_BF16 == 4 && TOY_DTYPE_I8 == 5,
    "dtype values match the toy_spec ABI");

typedef enum toy_layout {
    TOY_LAYOUT_INVALID = 0,
    TOY_LAYOUT_ROW_MAJOR = 1,
} toy_layout;

typedef enum toy_tensor_flags {
    TOY_TENSOR_FLAG_OWNS_DATA = 0b01u,
    TOY_TENSOR_FLAG_ALIAS = 0b10u,
} toy_tensor_flags;

typedef enum toy_status {
    TOY_OK = 0,
    TOY_ERROR_DTYPE,
    TOY_ERROR_SHAPE,
    TOY_ERROR_BYTES,
    TOY_ERROR_OVERFLOW,
    TOY_ERROR_MEMORY,
} toy_status;

typedef struct toy_buffer {
    uint32_t refcount;
    size_t bytes;
    void *data;
} toy_buffer;

typedef void (*toy_kernel_entry)(const void *a, const void *b, void *c);

/*
 * Element-wise kernel memo
 * The key packs kind and dtype with the element count. It stays valid across
 * reshape, which preserves the element count, and a cast allocates a new
 * tensor with an empty memo
 */
typedef struct toy_tensor_memo {
    toy_kernel_entry kernel;
    uint64_t key;
    uint32_t param_bits;
} toy_tensor_memo;

typedef struct toy_tensor {
    uint32_t refcount;
    uint8_t dtype;
    uint8_t rank;
    uint8_t layout;
    uint8_t flags;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    size_t bytes;
    void *data;
    toy_buffer *buffer;
    toy_tensor_memo memo;
} toy_tensor;

/* Returns NULL for an unknown dtype */
const char *toy_dtype_name(toy_dtype dtype);

[[nodiscard]] bool toy_dtype_from_name(const char *name, size_t length, toy_dtype *dtype);

/* Computes the byte count of a shape; q8_0 needs a last dim divisible by 32 */
[[nodiscard]] toy_status toy_tensor_shape_bytes(toy_dtype dtype, uint8_t rank,
                                                const uint32_t *dims, size_t *bytes);

/* Allocates a zero-filled tensor with its own 64-byte aligned buffer */
[[nodiscard]] toy_tensor *toy_tensor_alloc(toy_dtype dtype, uint8_t rank,
                                           const uint32_t *dims, toy_status *status);

/* Wraps external memory; the tensor does not own the memory. This is the only
 * exported C function */
[[nodiscard]] TOY_API toy_tensor *toy_tensor_alias(void *data, size_t bytes, toy_dtype dtype,
                                                   uint8_t rank,
                                                   const uint32_t dims[TOY_TENSOR_MAX_DIMS]);

toy_tensor *toy_tensor_retain(toy_tensor *tensor);
void toy_tensor_release(toy_tensor *tensor);

/* Reshapes in place; the byte count must not change */
[[nodiscard]] toy_status toy_tensor_reshape(toy_tensor *tensor, uint8_t rank,
                                            const uint32_t *dims);

/* Copies into a newly allocated tensor */
[[nodiscard]] toy_tensor *toy_tensor_clone(const toy_tensor *tensor, toy_status *status);

/* Fills every element with one scalar; f32 and q8_0 only */
[[nodiscard]] toy_status toy_tensor_fill(toy_tensor *tensor, float value);

/*
 * Converts into a newly allocated tensor of another dtype; f32 and q8_0 only
 * Same-dtype conversion copies the bytes
 */
[[nodiscard]] toy_tensor *toy_tensor_cast(const toy_tensor *tensor, toy_dtype dtype,
                                          toy_status *status);

size_t toy_tensor_numel(const toy_tensor *tensor);

static inline void *toy_tensor_data(const toy_tensor *tensor)
{
    return tensor->data;
}

#endif /* TOY_TENSOR_H */

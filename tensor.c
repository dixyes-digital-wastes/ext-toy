#include "tensor.h"

#include <stdlib.h>
#include <string.h>

#include "toy/q8_0.h"

#if defined(_MSC_VER)
# include <malloc.h>
#endif

enum {
    alignment = TOY_TENSOR_ALIGNMENT,
    q8_0_block_values = TOY_TENSOR_Q8_0_BLOCK_VALUES,
    q8_0_block_bytes = TOY_TENSOR_Q8_0_BLOCK_BYTES,
};

static void *alloc_aligned(size_t bytes)
{
    size_t rounded = (bytes + (alignment - 1)) & ~(alignment - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(rounded, alignment);
#else
    return aligned_alloc(alignment, rounded);
#endif
}

static void free_aligned(void *data)
{
#if defined(_MSC_VER)
    _aligned_free(data);
#else
    free(data);
#endif
}

static bool dims_valid(uint8_t rank, const uint32_t *dims)
{
    if (rank < 1 || rank > TOY_TENSOR_MAX_DIMS) {
        return false;
    }
    for (uint8_t i = 0; i < rank; i++) {
        if (dims[i] == 0) {
            return false;
        }
    }
    return true;
}

const char *dtype_to_name(dtype dtype_value)
{
    switch (dtype_value) {
        case TOY_DTYPE_F32:
            return "f32";
        case TOY_DTYPE_Q8_0:
            return "q8_0";
        case TOY_DTYPE_F16:
            return "f16";
        case TOY_DTYPE_BF16:
            return "bf16";
        case TOY_DTYPE_I8:
            return "i8";
        default:
            return "invalid";
    }
}

bool dtype_from_name(const char *name, size_t length, dtype *out_dtype)
{
    static const struct dtype_name {
        const char *name;
        size_t length;
        dtype dtype;
    } dtype_names[] = {
        { .name = "f32", .length = sizeof("f32") - 1, .dtype = TOY_DTYPE_F32 },
        { .name = "q8_0", .length = sizeof("q8_0") - 1, .dtype = TOY_DTYPE_Q8_0 },
        { .name = "f16", .length = sizeof("f16") - 1, .dtype = TOY_DTYPE_F16 },
        { .name = "bf16", .length = sizeof("bf16") - 1, .dtype = TOY_DTYPE_BF16 },
        { .name = "i8", .length = sizeof("i8") - 1, .dtype = TOY_DTYPE_I8 },
    };
    enum {
        dtype_count_entries = sizeof(dtype_names) / sizeof(dtype_names[0]),
    };

    for (size_t i = 0; i < dtype_count_entries; i++) {
        if (length == dtype_names[i].length
            && memcmp(name, dtype_names[i].name, length) == 0) {
            *out_dtype = dtype_names[i].dtype;
            return true;
        }
    }
    return false;
}

tensor_status tensor_shape_bytes(dtype dtype_value, uint8_t rank, const uint32_t *dims,
                                 size_t *bytes)
{
    if (dtype_value < TOY_DTYPE_F32 || dtype_value > TOY_DTYPE_I8) {
        return TOY_ERROR_DTYPE;
    }
    if (!dims_valid(rank, dims)) {
        return TOY_ERROR_SHAPE;
    }

    size_t numel = 1;
    for (uint8_t i = 0; i < rank; i++) {
        if (dims[i] > SIZE_MAX / numel) {
            return TOY_ERROR_OVERFLOW;
        }
        numel *= dims[i];
    }

    size_t total;
    switch (dtype_value) {
        case TOY_DTYPE_F32:
            if (numel > SIZE_MAX / sizeof(float)) {
                return TOY_ERROR_OVERFLOW;
            }
            total = numel * sizeof(float);
            break;
        case TOY_DTYPE_F16:
        case TOY_DTYPE_BF16:
            if (numel > SIZE_MAX / sizeof(uint16_t)) {
                return TOY_ERROR_OVERFLOW;
            }
            total = numel * sizeof(uint16_t);
            break;
        case TOY_DTYPE_I8:
            total = numel;
            break;
        case TOY_DTYPE_Q8_0: {
            if (dims[rank - 1] % q8_0_block_values != 0) {
                return TOY_ERROR_SHAPE;
            }
            size_t blocks = numel / q8_0_block_values;
            if (blocks > SIZE_MAX / q8_0_block_bytes) {
                return TOY_ERROR_OVERFLOW;
            }
            total = blocks * q8_0_block_bytes;
            break;
        }
        default:
            return TOY_ERROR_DTYPE;
    }

    *bytes = total;
    return TOY_OK;
}

static void buffer_release(buffer *buf)
{
    if (--buf->refcount == 0) {
        free_aligned(buf->data);
        free(buf);
    }
}

tensor *tensor_alloc(dtype dtype_value, uint8_t rank, const uint32_t *dims,
                     tensor_status *status)
{
    size_t bytes;
    tensor_status result = tensor_shape_bytes(dtype_value, rank, dims, &bytes);
    if (result != TOY_OK) {
        if (status) {
            *status = result;
        }
        return NULL;
    }

    if (bytes > SIZE_MAX - (alignment - 1)) {
        if (status) {
            *status = TOY_ERROR_OVERFLOW;
        }
        return NULL;
    }

    tensor *self = calloc(1, sizeof(*self));
    buffer *buf = calloc(1, sizeof(*buf));
    void *data = alloc_aligned(bytes);
    if (!self || !buf || !data) {
        free(self);
        free(buf);
        free_aligned(data);
        if (status) {
            *status = TOY_ERROR_MEMORY;
        }
        return NULL;
    }

    buf->refcount = 1;
    buf->bytes = bytes;
    buf->data = data;
    memset(data, 0, bytes);

    self->refcount = 1;
    self->dtype = (uint8_t) dtype_value;
    self->rank = rank;
    self->layout = TOY_LAYOUT_ROW_MAJOR;
    self->flags = TOY_TENSOR_FLAG_OWNS_DATA;
    for (uint8_t i = 0; i < rank; i++) {
        self->dims[i] = dims[i];
    }
    self->bytes = bytes;
    self->data = data;
    self->buffer = buf;

    if (status) {
        *status = TOY_OK;
    }
    return self;
}

tensor *toy_tensor_alias(void *data, size_t bytes, dtype dtype_value, uint8_t rank,
                         const uint32_t dims[TOY_TENSOR_MAX_DIMS])
{
    size_t expected;

    if (data == NULL) {
        return NULL;
    }
    if (tensor_shape_bytes(dtype_value, rank, dims, &expected) != TOY_OK
        || expected != bytes) {
        return NULL;
    }

    tensor *self = calloc(1, sizeof(*self));
    if (!self) {
        return NULL;
    }

    self->refcount = 1;
    self->dtype = (uint8_t) dtype_value;
    self->rank = rank;
    self->layout = TOY_LAYOUT_ROW_MAJOR;
    self->flags = TOY_TENSOR_FLAG_ALIAS;
    for (uint8_t i = 0; i < rank; i++) {
        self->dims[i] = dims[i];
    }
    self->bytes = bytes;
    self->data = data;
    self->buffer = NULL;

    return self;
}

tensor *tensor_retain(tensor *self)
{
    if (self) {
        self->refcount++;
    }
    return self;
}

void tensor_release(tensor *self)
{
    if (!self || --self->refcount > 0) {
        return;
    }
    if (self->buffer) {
        buffer_release(self->buffer);
    }
    free(self);
}

tensor_status tensor_reshape(tensor *self, uint8_t rank, const uint32_t *dims)
{
    size_t bytes;
    tensor_status status = tensor_shape_bytes((dtype) self->dtype, rank, dims, &bytes);

    if (status != TOY_OK) {
        return status;
    }
    if (bytes != self->bytes) {
        return TOY_ERROR_BYTES;
    }

    self->rank = rank;
    for (uint8_t i = 0; i < rank; i++) {
        self->dims[i] = dims[i];
    }
    return TOY_OK;
}

tensor *tensor_clone(const tensor *self, tensor_status *status)
{
    tensor *copy = tensor_alloc((dtype) self->dtype, self->rank, self->dims, status);
    if (!copy) {
        return NULL;
    }
    memcpy(copy->data, self->data, self->bytes);
    return copy;
}

tensor_status tensor_fill(tensor *self, float value)
{
    const size_t numel = tensor_numel(self);

    switch ((dtype) self->dtype) {
        case TOY_DTYPE_F32: {
            float *data = tensor_data(self);
            for (size_t i = 0; i < numel; i++) {
                data[i] = value;
            }
            return TOY_OK;
        }
        case TOY_DTYPE_Q8_0: {
            /* Every block holds the same 32 values, so quantize one block */
            float values[TOY_TENSOR_Q8_0_BLOCK_VALUES];
            for (uint32_t i = 0; i < TOY_TENSOR_Q8_0_BLOCK_VALUES; i++) {
                values[i] = value;
            }
            struct q8_0_block block;
            q8_0_quantize_rows(1, TOY_TENSOR_Q8_0_BLOCK_VALUES, values, &block);

            const size_t blocks = self->bytes / TOY_TENSOR_Q8_0_BLOCK_BYTES;
            uint8_t *data = tensor_data(self);
            for (size_t b = 0; b < blocks; b++) {
                memcpy(data + b * TOY_TENSOR_Q8_0_BLOCK_BYTES, &block,
                       TOY_TENSOR_Q8_0_BLOCK_BYTES);
            }
            return TOY_OK;
        }
        default:
            return TOY_ERROR_DTYPE;
    }
}

tensor *tensor_cast(const tensor *self, dtype dtype_value, tensor_status *status)
{
    if (self->dtype == (uint8_t) dtype_value) {
        return tensor_clone(self, status);
    }
    if (dtype_value < TOY_DTYPE_F32 || dtype_value > TOY_DTYPE_I8) {
        if (status) {
            *status = TOY_ERROR_DTYPE;
        }
        return NULL;
    }

    tensor *converted = tensor_alloc(dtype_value, self->rank, self->dims, status);
    if (!converted) {
        return NULL;
    }

    const uint32_t last = self->dims[self->rank - 1];
    const size_t rows = tensor_numel(self) / last;
    if (rows > UINT32_MAX) {
        tensor_release(converted);
        if (status) {
            *status = TOY_ERROR_OVERFLOW;
        }
        return NULL;
    }

    switch ((dtype) self->dtype) {
        case TOY_DTYPE_F32:
            if (dtype_value == TOY_DTYPE_Q8_0) {
                q8_0_quantize_rows((uint32_t) rows, last, tensor_data(self),
                                   tensor_data(converted));
                return converted;
            }
            break;
        case TOY_DTYPE_Q8_0:
            if (dtype_value == TOY_DTYPE_F32) {
                q8_0_dequantize_rows((uint32_t) rows, last, tensor_data(self),
                                     tensor_data(converted));
                return converted;
            }
            break;
        default:
            break;
    }

    tensor_release(converted);
    if (status) {
        *status = TOY_ERROR_DTYPE;
    }
    return NULL;
}

size_t tensor_numel(const tensor *self)
{
    size_t numel = 1;
    for (uint8_t i = 0; i < self->rank; i++) {
        numel *= self->dims[i];
    }
    return numel;
}

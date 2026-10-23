#include "toy_tensor.h"

#include <stdlib.h>
#include <string.h>

#include "toy/q8_0.h"

#if defined(_MSC_VER)
# include <malloc.h>
#endif

enum {
    toy_alignment = TOY_TENSOR_ALIGNMENT,
    toy_q8_0_block_values = TOY_TENSOR_Q8_0_BLOCK_VALUES,
    toy_q8_0_block_bytes = TOY_TENSOR_Q8_0_BLOCK_BYTES,
};

static void *toy_alloc_aligned(size_t bytes)
{
    size_t rounded = (bytes + (toy_alignment - 1)) & ~(toy_alignment - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(rounded, toy_alignment);
#else
    return aligned_alloc(toy_alignment, rounded);
#endif
}

static void toy_free_aligned(void *data)
{
#if defined(_MSC_VER)
    _aligned_free(data);
#else
    free(data);
#endif
}

static bool toy_dims_valid(uint8_t rank, const uint32_t *dims)
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

const char *toy_dtype_name(toy_dtype dtype)
{
    switch (dtype) {
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

bool toy_dtype_from_name(const char *name, size_t length, toy_dtype *dtype)
{
    static const struct toy_dtype_name {
        const char *name;
        size_t length;
        toy_dtype dtype;
    } toy_dtype_names[] = {
        { .name = "f32", .length = sizeof("f32") - 1, .dtype = TOY_DTYPE_F32 },
        { .name = "q8_0", .length = sizeof("q8_0") - 1, .dtype = TOY_DTYPE_Q8_0 },
        { .name = "f16", .length = sizeof("f16") - 1, .dtype = TOY_DTYPE_F16 },
        { .name = "bf16", .length = sizeof("bf16") - 1, .dtype = TOY_DTYPE_BF16 },
        { .name = "i8", .length = sizeof("i8") - 1, .dtype = TOY_DTYPE_I8 },
    };
    enum {
        toy_dtype_name_count = sizeof(toy_dtype_names) / sizeof(toy_dtype_names[0]),
    };

    for (size_t i = 0; i < toy_dtype_name_count; i++) {
        if (length == toy_dtype_names[i].length
            && memcmp(name, toy_dtype_names[i].name, length) == 0) {
            *dtype = toy_dtype_names[i].dtype;
            return true;
        }
    }
    return false;
}

toy_status toy_tensor_shape_bytes(toy_dtype dtype, uint8_t rank, const uint32_t *dims,
                                  size_t *bytes)
{
    if (dtype < TOY_DTYPE_F32 || dtype > TOY_DTYPE_I8) {
        return TOY_ERROR_DTYPE;
    }
    if (!toy_dims_valid(rank, dims)) {
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
    switch (dtype) {
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
            if (dims[rank - 1] % toy_q8_0_block_values != 0) {
                return TOY_ERROR_SHAPE;
            }
            size_t blocks = numel / toy_q8_0_block_values;
            if (blocks > SIZE_MAX / toy_q8_0_block_bytes) {
                return TOY_ERROR_OVERFLOW;
            }
            total = blocks * toy_q8_0_block_bytes;
            break;
        }
        default:
            return TOY_ERROR_DTYPE;
    }

    *bytes = total;
    return TOY_OK;
}

static void toy_buffer_release(toy_buffer *buffer)
{
    if (--buffer->refcount == 0) {
        toy_free_aligned(buffer->data);
        free(buffer);
    }
}

toy_tensor *toy_tensor_alloc(toy_dtype dtype, uint8_t rank, const uint32_t *dims,
                             toy_status *status)
{
    size_t bytes;
    toy_status result = toy_tensor_shape_bytes(dtype, rank, dims, &bytes);
    if (result != TOY_OK) {
        if (status) {
            *status = result;
        }
        return NULL;
    }

    if (bytes > SIZE_MAX - (toy_alignment - 1)) {
        if (status) {
            *status = TOY_ERROR_OVERFLOW;
        }
        return NULL;
    }

    toy_tensor *tensor = calloc(1, sizeof(*tensor));
    toy_buffer *buffer = calloc(1, sizeof(*buffer));
    void *data = toy_alloc_aligned(bytes);
    if (!tensor || !buffer || !data) {
        free(tensor);
        free(buffer);
        toy_free_aligned(data);
        if (status) {
            *status = TOY_ERROR_MEMORY;
        }
        return NULL;
    }

    buffer->refcount = 1;
    buffer->bytes = bytes;
    buffer->data = data;
    memset(data, 0, bytes);

    tensor->refcount = 1;
    tensor->dtype = (uint8_t) dtype;
    tensor->rank = rank;
    tensor->layout = TOY_LAYOUT_ROW_MAJOR;
    tensor->flags = TOY_TENSOR_FLAG_OWNS_DATA;
    for (uint8_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
    }
    tensor->bytes = bytes;
    tensor->data = data;
    tensor->buffer = buffer;

    if (status) {
        *status = TOY_OK;
    }
    return tensor;
}

toy_tensor *toy_tensor_alias(void *data, size_t bytes, toy_dtype dtype, uint8_t rank,
                             const uint32_t dims[TOY_TENSOR_MAX_DIMS])
{
    size_t expected;

    if (data == NULL) {
        return NULL;
    }
    if (toy_tensor_shape_bytes(dtype, rank, dims, &expected) != TOY_OK
        || expected != bytes) {
        return NULL;
    }

    toy_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        return NULL;
    }

    tensor->refcount = 1;
    tensor->dtype = (uint8_t) dtype;
    tensor->rank = rank;
    tensor->layout = TOY_LAYOUT_ROW_MAJOR;
    tensor->flags = TOY_TENSOR_FLAG_ALIAS;
    for (uint8_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
    }
    tensor->bytes = bytes;
    tensor->data = data;
    tensor->buffer = NULL;

    return tensor;
}

toy_tensor *toy_tensor_retain(toy_tensor *tensor)
{
    if (tensor) {
        tensor->refcount++;
    }
    return tensor;
}

void toy_tensor_release(toy_tensor *tensor)
{
    if (!tensor || --tensor->refcount > 0) {
        return;
    }
    if (tensor->buffer) {
        toy_buffer_release(tensor->buffer);
    }
    free(tensor);
}

toy_status toy_tensor_reshape(toy_tensor *tensor, uint8_t rank, const uint32_t *dims)
{
    size_t bytes;
    toy_status status = toy_tensor_shape_bytes((toy_dtype) tensor->dtype, rank, dims, &bytes);

    if (status != TOY_OK) {
        return status;
    }
    if (bytes != tensor->bytes) {
        return TOY_ERROR_BYTES;
    }

    tensor->rank = rank;
    for (uint8_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
    }
    return TOY_OK;
}

toy_tensor *toy_tensor_clone(const toy_tensor *tensor, toy_status *status)
{
    toy_tensor *copy = toy_tensor_alloc((toy_dtype) tensor->dtype, tensor->rank,
                                        tensor->dims, status);
    if (!copy) {
        return NULL;
    }
    memcpy(copy->data, tensor->data, tensor->bytes);
    return copy;
}

toy_status toy_tensor_fill(toy_tensor *tensor, float value)
{
    const size_t numel = toy_tensor_numel(tensor);

    switch ((toy_dtype) tensor->dtype) {
        case TOY_DTYPE_F32: {
            float *data = toy_tensor_data(tensor);
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
            struct toy_q8_0_block block;
            toy_q8_0_quantize_rows(1, TOY_TENSOR_Q8_0_BLOCK_VALUES, values, &block);

            const size_t blocks = tensor->bytes / TOY_TENSOR_Q8_0_BLOCK_BYTES;
            uint8_t *data = toy_tensor_data(tensor);
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

toy_tensor *toy_tensor_cast(const toy_tensor *tensor, toy_dtype dtype, toy_status *status)
{
    if (tensor->dtype == (uint8_t) dtype) {
        return toy_tensor_clone(tensor, status);
    }
    if (dtype < TOY_DTYPE_F32 || dtype > TOY_DTYPE_I8) {
        if (status) {
            *status = TOY_ERROR_DTYPE;
        }
        return NULL;
    }

    toy_tensor *converted = toy_tensor_alloc(dtype, tensor->rank, tensor->dims, status);
    if (!converted) {
        return NULL;
    }

    const uint32_t last = tensor->dims[tensor->rank - 1];
    const size_t rows = toy_tensor_numel(tensor) / last;
    if (rows > UINT32_MAX) {
        toy_tensor_release(converted);
        if (status) {
            *status = TOY_ERROR_OVERFLOW;
        }
        return NULL;
    }

    switch ((toy_dtype) tensor->dtype) {
        case TOY_DTYPE_F32:
            if (dtype == TOY_DTYPE_Q8_0) {
                toy_q8_0_quantize_rows((uint32_t) rows, last, toy_tensor_data(tensor),
                                       toy_tensor_data(converted));
                return converted;
            }
            break;
        case TOY_DTYPE_Q8_0:
            if (dtype == TOY_DTYPE_F32) {
                toy_q8_0_dequantize_rows((uint32_t) rows, last, toy_tensor_data(tensor),
                                         toy_tensor_data(converted));
                return converted;
            }
            break;
        default:
            break;
    }

    toy_tensor_release(converted);
    if (status) {
        *status = TOY_ERROR_DTYPE;
    }
    return NULL;
}

size_t toy_tensor_numel(const toy_tensor *tensor)
{
    size_t numel = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        numel *= tensor->dims[i];
    }
    return numel;
}

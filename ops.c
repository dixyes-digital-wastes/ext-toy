#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"

#include "php_toy.h"
#include "attributes.h"
#include "gdn.h"
#include "kernel.h"

#include <math.h>
#include <string.h>

#include "toy/codegen.h"
#include "toy/q8_0.h"
#include "toy/ref.h"

static const double rmsnorm_default_eps = 1e-6;

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matmul_f32, 0, 5, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matvec_f32, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matmul_q8_0_f32, 0, 5, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matvec_q8_0_f32, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matmul_q8_0_q8_0, 0, 5, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_matvec_q8_0_q8_0, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_quantize_q8_0, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rowBegin, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rowEnd, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_add, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_mul, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_scale, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, factor, IS_DOUBLE, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_silu, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_sigmoid, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_softplus, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_l2norm_rows, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, eps, IS_DOUBLE, 0, "1e-6")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, start, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, count, IS_LONG, 1, "null")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_rmsnorm_rows, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, eps, IS_DOUBLE, 0, "1e-6")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, start, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, count, IS_LONG, 1, "null")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_shift_rows, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, row, Toy\\Tensor, 1, "null")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_conv1d_silu, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, rows, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, kernel, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, start, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, count, IS_LONG, 1, "null")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_rmsnorm, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, eps, IS_DOUBLE, 0, "1e-6")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_rmsnorm_into, 0, 3, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, out, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, eps, IS_DOUBLE, 0, "1e-6")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_gdn_recurrence, 0, 10, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, q, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, k, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, v, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, decay, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, beta, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, state, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, headBegin, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, headEnd, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, colBegin, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, colEnd, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, scale, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, out, Toy\\Tensor, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_kernel_stats, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, reset, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_kernel_enable, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, enable, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_kernel_isa, 0, 0, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, isa, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()

ZEND_FRAMELESS_FUNCTION(tensor_add_flf, 3);
ZEND_FRAMELESS_FUNCTION(tensor_mul_flf, 3);
ZEND_FRAMELESS_FUNCTION(tensor_scale_flf, 3);
ZEND_FRAMELESS_FUNCTION(tensor_silu_flf, 2);
ZEND_FRAMELESS_FUNCTION(tensor_sigmoid_flf, 2);
ZEND_FRAMELESS_FUNCTION(tensor_softplus_flf, 2);
ZEND_FRAMELESS_FUNCTION(tensor_rmsnorm_flf, 3);

/*
 * add, mul and scale take an optional out, so a two-argument call uses the
 * three-argument handler with NULL as the third argument
 */
static const zend_frameless_function_info frameless_function_infos_tensor_add[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_add_flf, 3), 3 },
    { 0 },
};

static const zend_frameless_function_info frameless_function_infos_tensor_mul[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_mul_flf, 3), 3 },
    { 0 },
};

static const zend_frameless_function_info frameless_function_infos_tensor_scale[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_scale_flf, 3), 3 },
    { 0 },
};

/*
 * silu has an optional out as its second argument, so the two-argument
 * handler serves the one-argument and the two-argument call
 */
static const zend_frameless_function_info frameless_function_infos_tensor_silu[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_silu_flf, 2), 2 },
    { 0 },
};

/*
 * sigmoid and softplus take an optional out as their second argument, like
 * silu
 */
static const zend_frameless_function_info frameless_function_infos_tensor_sigmoid[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_sigmoid_flf, 2), 2 },
    { 0 },
};

static const zend_frameless_function_info frameless_function_infos_tensor_softplus[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_softplus_flf, 2), 2 },
    { 0 },
};

/*
 * rmsnorm has a default eps, so one three-argument handler serves the
 * two-argument and the three-argument call
 */
static const zend_frameless_function_info frameless_function_infos_tensor_rmsnorm[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(tensor_rmsnorm_flf, 3), 3 },
    { 0 },
};

/* ZEND_FRAMELESS_FE has no namespace variant, so reuse the entry layout */
#define TOY_NS_FRAMELESS_FE(ns, name, handler, arg_info, infos) \
    ZEND_RAW_FENTRY(ZEND_NS_NAME(ns, #name), handler, arg_info, 0, infos, NULL)

const zend_function_entry module_functions[] = {
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_f32, zif_tensor_matmul_f32,
                     arginfo_tensor_matmul_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_f32, zif_tensor_matvec_f32,
                     arginfo_tensor_matvec_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_q8_0_f32, zif_tensor_matmul_q8_0_f32,
                     arginfo_tensor_matmul_q8_0_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_q8_0_f32, zif_tensor_matvec_q8_0_f32,
                     arginfo_tensor_matvec_q8_0_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_q8_0_q8_0, zif_tensor_matmul_q8_0_q8_0,
                     arginfo_tensor_matmul_q8_0_q8_0)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_q8_0_q8_0, zif_tensor_matvec_q8_0_q8_0,
                     arginfo_tensor_matvec_q8_0_q8_0)
    ZEND_NS_NAMED_FE("Toy\\Tensor", quantize_q8_0, zif_tensor_quantize_q8_0,
                     arginfo_tensor_quantize_q8_0)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", add, zif_tensor_add,
                        arginfo_tensor_add, frameless_function_infos_tensor_add)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", mul, zif_tensor_mul,
                        arginfo_tensor_mul, frameless_function_infos_tensor_mul)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", scale, zif_tensor_scale,
                        arginfo_tensor_scale, frameless_function_infos_tensor_scale)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", silu, zif_tensor_silu,
                        arginfo_tensor_silu, frameless_function_infos_tensor_silu)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", sigmoid, zif_tensor_sigmoid,
                        arginfo_tensor_sigmoid, frameless_function_infos_tensor_sigmoid)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", softplus, zif_tensor_softplus,
                        arginfo_tensor_softplus, frameless_function_infos_tensor_softplus)
    ZEND_NS_NAMED_FE("Toy\\Tensor", l2norm_rows, zif_tensor_l2norm_rows,
                     arginfo_tensor_l2norm_rows)
    ZEND_NS_NAMED_FE("Toy\\Tensor", rmsnorm_rows, zif_tensor_rmsnorm_rows,
                     arginfo_tensor_rmsnorm_rows)
    ZEND_NS_NAMED_FE("Toy\\Tensor", shift_rows, zif_tensor_shift_rows,
                     arginfo_tensor_shift_rows)
    ZEND_NS_NAMED_FE("Toy\\Tensor", conv1d_silu, zif_tensor_conv1d_silu,
                     arginfo_tensor_conv1d_silu)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", rmsnorm, zif_tensor_rmsnorm,
                        arginfo_tensor_rmsnorm, frameless_function_infos_tensor_rmsnorm)
    ZEND_NS_NAMED_FE("Toy\\Tensor", rmsnorm_into, zif_tensor_rmsnorm_into,
                     arginfo_tensor_rmsnorm_into)
    ZEND_NS_NAMED_FE("Toy\\Tensor", gdn_recurrence, zif_tensor_gdn_recurrence,
                     arginfo_tensor_gdn_recurrence)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_stats, zif_tensor_kernel_stats,
                     arginfo_tensor_kernel_stats)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_enable, zif_tensor_kernel_enable,
                     arginfo_tensor_kernel_enable)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_isa, zif_tensor_kernel_isa,
                     arginfo_tensor_kernel_isa)
    ZEND_FE_END
};

TOY_NODISCARD static tensor *php_tensor_arg(zval *value)
{
    tensor *self = php_tensor_fetch(value);

    if (!self) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
    }
    return self;
}

/* A frameless handler does not use ZPP, so check the types here */
TOY_NODISCARD static bool php_tensor_object(zval *value, tensor **out,
                                            uint32_t arg_num)
{
    if (UNEXPECTED(Z_TYPE_P(value) != IS_OBJECT)
        || UNEXPECTED(!instanceof_function(Z_OBJCE_P(value), tensor_ce))) {
        zend_argument_type_error(arg_num, "must be of type Toy\\Tensor, %s given",
                                 zend_zval_type_name(value));
        return false;
    }

    tensor *fetched = php_tensor_fetch(value);
    if (UNEXPECTED(fetched == NULL)) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
        return false;
    }
    *out = fetched;
    return true;
}

TOY_NODISCARD static bool php_tensor_require_f32(const tensor *self,
                                                 uint32_t arg_num)
{
    if (self->dtype != TOY_DTYPE_F32) {
        zend_argument_value_error(arg_num, "must be an f32 tensor");
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_require_q8_0(const tensor *self,
                                                  uint32_t arg_num)
{
    if (self->dtype != TOY_DTYPE_Q8_0) {
        zend_argument_value_error(arg_num, "must be a q8_0 tensor");
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_dim(zend_long value, uint32_t *dim,
                                         uint32_t arg_num)
{
    if (value < 1 || (zend_ulong) value > UINT32_MAX) {
        zend_argument_value_error(arg_num, "must be between 1 and %u",
                                  (unsigned) UINT32_MAX);
        return false;
    }
    *dim = (uint32_t) value;
    return true;
}

TOY_NODISCARD static bool php_tensor_expect_rank2(const tensor *self,
                                                  uint32_t arg_num, uint32_t dim0,
                                                  uint32_t dim1)
{
    if (self->rank != 2) {
        zend_argument_value_error(arg_num, "must be a 2-D tensor");
        return false;
    }
    if (self->dims[0] != dim0 || self->dims[1] != dim1) {
        zend_argument_value_error(arg_num, "must have shape [%u, %u], got [%u, %u]",
                                  dim0, dim1, self->dims[0], self->dims[1]);
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_expect_rank1(const tensor *self,
                                                  uint32_t arg_num, uint32_t length)
{
    if (self->rank != 1) {
        zend_argument_value_error(arg_num, "must be a 1-D tensor");
        return false;
    }
    if (self->dims[0] != length) {
        zend_argument_value_error(arg_num, "must have shape [%u], got [%u]",
                                  length, self->dims[0]);
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_expect_rank3(const tensor *self,
                                                  uint32_t arg_num, uint32_t dim0,
                                                  uint32_t dim1, uint32_t dim2)
{
    if (self->rank != 3) {
        zend_argument_value_error(arg_num, "must be a 3-D tensor");
        return false;
    }
    if (self->dims[0] != dim0 || self->dims[1] != dim1 || self->dims[2] != dim2) {
        zend_argument_value_error(arg_num,
                                  "must have shape [%u, %u, %u], got [%u, %u, %u]",
                                  dim0, dim1, dim2,
                                  self->dims[0], self->dims[1], self->dims[2]);
        return false;
    }
    return true;
}

/* A slice bound, from zero up to the dimension length */
TOY_NODISCARD static bool php_tensor_bound(zend_long value, uint32_t limit,
                                           uint32_t *out, uint32_t arg_num)
{
    if (value < 0 || (zend_ulong) value > limit) {
        zend_argument_value_error(arg_num, "must be between 0 and %u", limit);
        return false;
    }
    *out = (uint32_t) value;
    return true;
}

TOY_NODISCARD static tensor *php_tensor_alloc_result_dtype(dtype dtype_value, uint8_t rank,
                                                          const uint32_t *dims)
{
    tensor_status status;
    tensor *self = tensor_alloc(dtype_value, rank, dims, &status);

    if (!self) {
        if (status == TOY_ERROR_MEMORY) {
            zend_throw_error(NULL, "Tensor allocation failed");
        } else {
            zend_value_error("Result tensor is too large");
        }
    }
    return self;
}

TOY_NODISCARD static tensor *php_tensor_alloc_result(uint8_t rank,
                                                     const uint32_t *dims)
{
    return php_tensor_alloc_result_dtype(TOY_DTYPE_F32, rank, dims);
}

static uint32_t php_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* An element count above 32 bits gets no kernel, the scalar path still works */
static uint32_t php_element_count(size_t numel)
{
    return numel <= UINT32_MAX ? (uint32_t) numel : 0;
}

TOY_NODISCARD static bool php_tensor_optional_out(zval *value, uint32_t arg_num,
                                                  tensor **out)
{
    if (value == NULL || Z_TYPE_P(value) == IS_NULL) {
        *out = NULL;
        return true;
    }
    if (Z_TYPE_P(value) != IS_OBJECT
        || !instanceof_function(Z_OBJCE_P(value), tensor_ce)) {
        zend_argument_type_error(arg_num, "must be of type ?Toy\\Tensor, %s given",
                                 zend_zval_type_name(value));
        return false;
    }
    tensor *fetched = php_tensor_fetch(value);
    if (UNEXPECTED(fetched == NULL)) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
        return false;
    }
    *out = fetched;
    return true;
}

TOY_NODISCARD static bool php_tensor_out_matches(const tensor *out, uint8_t rank,
                                                     const uint32_t *dims, uint32_t arg_num)
{
    if (out->dtype != TOY_DTYPE_F32) {
        zend_argument_value_error(arg_num, "must be an f32 tensor");
        return false;
    }
    if (out->rank != rank) {
        zend_argument_value_error(arg_num, "must have the same shape as the result");
        return false;
    }
    for (uint8_t i = 0; i < rank; i++) {
        if (out->dims[i] != dims[i]) {
            zend_argument_value_error(arg_num, "must have the same shape as the result");
            return false;
        }
    }
    return true;
}

/* Same data range, to tell if out is one of the inputs */
TOY_NODISCARD static bool php_tensor_same_range(const tensor *a, const tensor *b)
{
    return a->bytes == b->bytes
        && (uintptr_t) tensor_data(a) == (uintptr_t) tensor_data(b);
}

TOY_NODISCARD static bool php_tensor_ranges_overlap(const tensor *a,
                                                        const tensor *b)
{
    const uintptr_t a_begin = (uintptr_t) tensor_data(a);
    const uintptr_t b_begin = (uintptr_t) tensor_data(b);

    return a_begin < b_begin + b->bytes && b_begin < a_begin + a->bytes;
}

/*
 * add and scale accept an out that matches one input exactly. A partial
 * overlap breaks the elementwise read and write
 */
TOY_NODISCARD static bool php_tensor_out_alias_check(const tensor *out,
                                                         const tensor *a,
                                                         const tensor *b,
                                                         uint32_t arg_num)
{
    if (php_tensor_same_range(out, a) || php_tensor_same_range(out, b)) {
        return true;
    }
    if (php_tensor_ranges_overlap(out, a) || php_tensor_ranges_overlap(out, b)) {
        zend_argument_value_error(arg_num, "must not partially overlap the inputs");
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_out_disjoint_check(const tensor *out,
                                                            const tensor *a,
                                                            const tensor *b,
                                                            uint32_t out_arg,
                                                            uint32_t a_arg, uint32_t b_arg)
{
    if (php_tensor_ranges_overlap(out, a)) {
        zend_argument_value_error(out_arg, "must not overlap argument #%u", a_arg);
        return false;
    }
    if (php_tensor_ranges_overlap(out, b)) {
        zend_argument_value_error(out_arg, "must not overlap argument #%u", b_arg);
        return false;
    }
    return true;
}

/* rmsnorm_into accepts an out that matches x exactly, but not one that touches w */
TOY_NODISCARD static bool php_tensor_out_rmsnorm_check(const tensor *out,
                                                           const tensor *x,
                                                           const tensor *w,
                                                           uint32_t arg_num)
{
    if (!php_tensor_same_range(out, x) && php_tensor_ranges_overlap(out, x)) {
        zend_argument_value_error(arg_num, "must not partially overlap argument #1 ($x)");
        return false;
    }
    if (php_tensor_ranges_overlap(out, w)) {
        zend_argument_value_error(arg_num, "must not overlap argument #2 ($w)");
        return false;
    }
    return true;
}

static bool php_tensor_add_impl(tensor *a, tensor *b, zval *out_value,
                                    zval *return_value)
{
    if (!php_tensor_require_f32(a, 1) || !php_tensor_require_f32(b, 2)) {
        return false;
    }
    if (a->rank != b->rank) {
        zend_argument_value_error(2,
            "must have the same number of dimensions as argument #1 ($a)");
        return false;
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) {
            zend_argument_value_error(2, "must have the same shape as argument #1 ($a)");
            return false;
        }
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 3, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, a->rank, a->dims, 3)
            || !php_tensor_out_alias_check(out, a, b, 3))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(a->rank, a->dims);
        if (result == NULL) {
            return false;
        }
    }

    size_t numel = tensor_numel(a);
    uint32_t count = php_element_count(numel);
    kernel_entry kernel = count != 0
        ? kernel_for_element(a, TOY_KERNEL_ADD, 0, count)
        : NULL;
    if (kernel != NULL) {
        kernel(tensor_data(a), tensor_data(b), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_add_f32(tensor_data(a), tensor_data(b), tensor_data(result), numel);
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_mul_impl(tensor *a, tensor *b, zval *out_value,
                                    zval *return_value)
{
    if (!php_tensor_require_f32(a, 1) || !php_tensor_require_f32(b, 2)) {
        return false;
    }
    if (a->rank != b->rank) {
        zend_argument_value_error(2,
            "must have the same number of dimensions as argument #1 ($a)");
        return false;
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) {
            zend_argument_value_error(2, "must have the same shape as argument #1 ($a)");
            return false;
        }
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 3, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, a->rank, a->dims, 3)
            || !php_tensor_out_alias_check(out, a, b, 3))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(a->rank, a->dims);
        if (result == NULL) {
            return false;
        }
    }

    size_t numel = tensor_numel(a);
    uint32_t count = php_element_count(numel);
    kernel_entry kernel = count != 0
        ? kernel_for_element(a, TOY_KERNEL_MUL, 0, count)
        : NULL;
    if (kernel != NULL) {
        kernel(tensor_data(a), tensor_data(b), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_mul_f32(tensor_data(a), tensor_data(b), tensor_data(result), numel);
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_scale_impl(tensor *a, double factor, zval *out_value,
                                      zval *return_value)
{
    if (!php_tensor_require_f32(a, 1)) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 3, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, a->rank, a->dims, 3)
            || !php_tensor_out_alias_check(out, a, a, 3))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(a->rank, a->dims);
        if (result == NULL) {
            return false;
        }
    }

    const float value = (float) factor;
    size_t numel = tensor_numel(a);
    uint32_t count = php_element_count(numel);
    kernel_entry kernel = count != 0
        ? kernel_for_element(a, TOY_KERNEL_SCALE, php_float_bits(value), count)
        : NULL;
    if (kernel != NULL) {
        kernel(tensor_data(a), NULL, tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_scale_f32(tensor_data(a), value, tensor_data(result), numel);
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_silu_impl(tensor *a, zval *out_value, zval *return_value)
{
    if (!php_tensor_require_f32(a, 1)) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 2, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, a->rank, a->dims, 2)
            || !php_tensor_out_alias_check(out, a, a, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(a->rank, a->dims);
        if (result == NULL) {
            return false;
        }
    }

    ref_silu_f32(tensor_data(a), tensor_data(result), tensor_numel(a));
    kernel_count_scalar_call();

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * sigmoid and softplus share the element-wise rules of silu: any rank up to
 * four, and an out that may be the input itself
 */
static bool php_tensor_activation_impl(tensor *a, zval *out_value, zval *return_value,
                                       uint8_t kernel_kind)
{
    if (!php_tensor_require_f32(a, 1)) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 2, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, a->rank, a->dims, 2)
            || !php_tensor_out_alias_check(out, a, a, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(a->rank, a->dims);
        if (result == NULL) {
            return false;
        }
    }

    size_t numel = tensor_numel(a);
    uint32_t count = php_element_count(numel);
    kernel_entry kernel = count != 0
        ? kernel_for_element(a, kernel_kind, 0, count)
        : NULL;
    if (kernel != NULL) {
        kernel(tensor_data(a), NULL, tensor_data(result));
        kernel_count_kernel_call();
    } else {
        if (kernel_kind == TOY_KERNEL_SIGMOID) {
            ref_sigmoid_f32(tensor_data(a), tensor_data(result), numel);
        } else {
            ref_softplus_f32(tensor_data(a), tensor_data(result), numel);
        }
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * Resolves the optional row range of l2norm_rows and conv1d_silu. A missing
 * count means every row from start on
 */
TOY_NODISCARD static bool php_tensor_row_range(uint32_t rows, zend_long start_arg,
                                               zend_long count_arg, bool count_is_null,
                                               uint32_t *start, uint32_t *count,
                                               uint32_t start_arg_num, uint32_t count_arg_num)
{
    if (start_arg < 0 || (zend_ulong) start_arg > rows) {
        zend_argument_value_error(start_arg_num, "must be between 0 and %u", rows);
        return false;
    }

    const uint32_t first = (uint32_t) start_arg;
    const uint32_t available = rows - first;
    uint32_t length = available;
    if (!count_is_null) {
        if (count_arg < 0 || (zend_ulong) count_arg > available) {
            zend_argument_value_error(count_arg_num, "must be between 0 and %u",
                                      available);
            return false;
        }
        length = (uint32_t) count_arg;
    }

    *start = first;
    *count = length;
    return true;
}

static bool php_tensor_l2norm_rows_impl(tensor *x, double eps, zend_long start_arg,
                                        zend_long count_arg, bool count_is_null,
                                        zval *out_value, zval *return_value)
{
    if (!php_tensor_require_f32(x, 1)) {
        return false;
    }
    if (x->rank != 2) {
        zend_argument_value_error(1, "must be a 2-D tensor");
        return false;
    }

    const uint32_t rows = x->dims[0];
    const uint32_t width = x->dims[1];
    uint32_t start, count;
    if (!php_tensor_row_range(rows, start_arg, count_arg, count_is_null, &start, &count,
                              3, 4)) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 5, &out)) {
        return false;
    }
    if (out != NULL && !php_tensor_out_matches(out, x->rank, x->dims, 5)) {
        return false;
    }
    if (out != NULL && !php_tensor_same_range(out, x)
        && php_tensor_ranges_overlap(out, x)) {
        zend_argument_value_error(5, "must not partially overlap argument #1 ($x)");
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(x->rank, x->dims);
        if (result == NULL) {
            return false;
        }
    }

    const float eps32 = (float) eps;
    const size_t offset = (size_t) start * width;
    const float *source = (const float *) tensor_data(x) + offset;
    float *target = (float *) tensor_data(result) + offset;
    if (count != 0) {
        kernel_entry kernel = kernel_for_shape_param(TOY_KERNEL_L2NORM, TOY_DTYPE_F32,
                                                     count, 1, width,
                                                     php_float_bits(eps32));
        if (kernel != NULL) {
            kernel(source, NULL, target);
            kernel_count_kernel_call();
        } else {
            ref_l2norm_rows_f32(count, width, source, target, eps32);
            kernel_count_scalar_call();
        }
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * rmsnorm_rows scales every row like rmsnorm, against one weight row that all
 * rows share. The out may be the input itself, but not overlap the weight
 */
static bool php_tensor_rmsnorm_rows_impl(tensor *x, tensor *w, double eps,
                                         zend_long start_arg, zend_long count_arg,
                                         bool count_is_null, zval *out_value,
                                         zval *return_value)
{
    if (!php_tensor_require_f32(x, 1) || !php_tensor_require_f32(w, 2)) {
        return false;
    }
    if (x->rank != 2) {
        zend_argument_value_error(1, "must be a 2-D tensor");
        return false;
    }

    const uint32_t rows = x->dims[0];
    const uint32_t width = x->dims[1];
    if (!php_tensor_expect_rank1(w, 2, width)) {
        return false;
    }
    uint32_t start, count;
    if (!php_tensor_row_range(rows, start_arg, count_arg, count_is_null, &start, &count,
                              4, 5)) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 6, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, x->rank, x->dims, 6)
            || !php_tensor_out_rmsnorm_check(out, x, w, 6))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(x->rank, x->dims);
        if (result == NULL) {
            return false;
        }
    }

    const float eps32 = (float) eps;
    const size_t offset = (size_t) start * width;
    const float *source = (const float *) tensor_data(x) + offset;
    float *target = (float *) tensor_data(result) + offset;
    if (count != 0) {
        kernel_entry kernel = kernel_for_shape_param(TOY_KERNEL_RMSNORM_ROWS, TOY_DTYPE_F32,
                                                     count, 1, width,
                                                     php_float_bits(eps32));
        if (kernel != NULL) {
            kernel(source, tensor_data(w), target);
            kernel_count_kernel_call();
        } else {
            ref_rmsnorm_rows_f32(count, width, source, tensor_data(w), target, eps32);
            kernel_count_scalar_call();
        }
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * shift_rows moves every row of x up by one and writes row into the last one,
 * which is the state update of a causal convolution. Without row the last row
 * keeps its value. A byte move has no arithmetic to generate, so it runs in C
 * and works in place
 */
static bool php_tensor_shift_rows_impl(tensor *x, tensor *row, zval *out_value,
                                       zval *return_value)
{
    if (!php_tensor_require_f32(x, 1)) {
        return false;
    }
    if (x->rank != 2) {
        zend_argument_value_error(1, "must be a 2-D tensor");
        return false;
    }
    if (row != NULL
        && (!php_tensor_require_f32(row, 2)
            || !php_tensor_expect_rank1(row, 2, x->dims[1]))) {
        return false;
    }

    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 3, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, x->rank, x->dims, 3)
            || !php_tensor_out_alias_check(out, x, x, 3))) {
        return false;
    }
    if (row != NULL
        && (php_tensor_ranges_overlap(row, x)
            || (out != NULL && php_tensor_ranges_overlap(row, out)))) {
        zend_argument_value_error(2, "must not overlap argument #1 ($x) or #3 ($out)");
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(x->rank, x->dims);
        if (result == NULL) {
            return false;
        }
    }

    const size_t row_bytes = (size_t) x->dims[1] * 4;
    const size_t rows = x->dims[0];
    const uint8_t *source = (const uint8_t *) tensor_data(x);
    uint8_t *target = (uint8_t *) tensor_data(result);
    memmove(target, source + row_bytes, (rows - 1) * row_bytes);
    if (row != NULL) {
        memcpy(target + (rows - 1) * row_bytes, tensor_data(row), row_bytes);
    } else {
        memmove(target + (rows - 1) * row_bytes, source + (rows - 1) * row_bytes, row_bytes);
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * The weights hold one kernel per row and x one row per tap, so x[j * rows + c]
 * is tap j of channel c. The activation runs after the convolution, which
 * means out must not share memory with either input
 */
static bool php_tensor_conv1d_silu_impl(tensor *w, tensor *x, uint32_t rows,
                                        uint32_t taps, zend_long start_arg,
                                        zend_long count_arg, bool count_is_null,
                                        zval *out_value, zval *return_value)
{
    if (!php_tensor_require_f32(w, 1) || !php_tensor_require_f32(x, 2)
        || !php_tensor_expect_rank2(w, 1, rows, taps)
        || !php_tensor_expect_rank2(x, 2, taps, rows)) {
        return false;
    }

    uint32_t start, count;
    if (!php_tensor_row_range(rows, start_arg, count_arg, count_is_null, &start, &count,
                              5, 6)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { rows, 0, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 7, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 1, dims, 7)
            || !php_tensor_out_disjoint_check(out, w, x, 7, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(1, dims);
        if (result == NULL) {
            return false;
        }
    }

    const float *weights = (const float *) tensor_data(w) + (size_t) start * taps;
    const float *inputs = (const float *) tensor_data(x) + start;
    float *target = (float *) tensor_data(result) + start;
    if (count != 0) {
        kernel_entry kernel = kernel_for_shape_param(TOY_KERNEL_CONV1D_SILU, TOY_DTYPE_F32,
                                                     count, 1, rows, taps);
        if (kernel != NULL) {
            kernel(weights, inputs, target);
            kernel_count_kernel_call();
        } else {
            ref_conv1d_silu_f32(count, rows, taps, weights, inputs, target);
            kernel_count_scalar_call();
        }
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static void php_tensor_rmsnorm_impl(tensor *x, tensor *w, double eps,
                                        zval *return_value)
{
    if (!php_tensor_require_f32(x, 1) || !php_tensor_require_f32(w, 2)) {
        return;
    }
    if (x->rank != 1) {
        zend_argument_value_error(1, "must be a 1-D tensor");
        return;
    }
    if (!php_tensor_expect_rank1(w, 2, x->dims[0])) {
        return;
    }

    uint32_t n = x->dims[0];
    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { n, 0, 0, 0 };
    tensor *result = php_tensor_alloc_result(1, dims);
    if (!result) {
        return;
    }

    const float eps32 = (float) eps;
    kernel_entry kernel = kernel_for_element(x, TOY_KERNEL_RMSNORM,
                                                     php_float_bits(eps32), n);
    if (kernel != NULL) {
        kernel(tensor_data(x), tensor_data(w), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_rmsnorm_f32(tensor_data(x), tensor_data(w), tensor_data(result),
                        n, eps32);
        kernel_count_scalar_call();
    }

    if (php_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static bool php_tensor_matmul_impl(tensor *a, tensor *b, uint32_t m, uint32_t n,
                                       uint32_t k, zval *out_value, zval *return_value)
{
    if (!php_tensor_require_f32(a, 1) || !php_tensor_require_f32(b, 2)
        || !php_tensor_expect_rank2(a, 1, m, k)
        || !php_tensor_expect_rank2(b, 2, k, n)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 6, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 2, dims, 6)
            || !php_tensor_out_disjoint_check(out, a, b, 6, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(2, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATMUL, TOY_DTYPE_F32, m, n, k);
    if (kernel != NULL) {
        kernel(tensor_data(a), tensor_data(b), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_gemm(m, n, k, tensor_data(a), tensor_data(b),
                     tensor_data(result));
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_matvec_impl(tensor *w, tensor *x, uint32_t m, uint32_t k,
                                       zval *out_value, zval *return_value)
{
    if (!php_tensor_require_f32(w, 1) || !php_tensor_require_f32(x, 2)
        || !php_tensor_expect_rank2(w, 1, m, k)
        || !php_tensor_expect_rank1(x, 2, k)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 5, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 1, dims, 5)
            || !php_tensor_out_disjoint_check(out, w, x, 5, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(1, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATVEC, TOY_DTYPE_F32, m, 1, k);
    if (kernel != NULL) {
        kernel(tensor_data(w), tensor_data(x), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        /*
         * The AVX2 kernel sums across k lanes, so it has a different order
         * from the portable scalar reference. With kernels enabled, a shape
         * that the generator rejects keeps the AVX2 order, otherwise slicing
         * the same row into a smaller m would change its bytes
         */
        if (kernel_enabled() && kernel_isa() == TOY_ISA_AVX2) {
            ref_matvec_avx2_order(m, k, tensor_data(w), tensor_data(x),
                                  tensor_data(result));
        } else {
            ref_matvec(m, k, tensor_data(w), tensor_data(x),
                           tensor_data(result));
        }
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_matmul_q8_0_impl(tensor *w, tensor *x, uint32_t m,
                                            uint32_t n, uint32_t k, zval *out_value,
                                            zval *return_value)
{
    if (!php_tensor_require_q8_0(w, 1) || !php_tensor_require_f32(x, 2)
        || !php_tensor_expect_rank2(w, 1, m, k)
        || !php_tensor_expect_rank2(x, 2, k, n)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 6, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 2, dims, 6)
            || !php_tensor_out_disjoint_check(out, w, x, 6, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(2, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATMUL_Q8_0, TOY_DTYPE_Q8_0,
                                                   m, n, k);
    if (kernel != NULL) {
        kernel(tensor_data(w), tensor_data(x), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_q8_0_matmul(m, n, k, tensor_data(w), tensor_data(x),
                            tensor_data(result));
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_matvec_q8_0_impl(tensor *w, tensor *x, uint32_t m,
                                            uint32_t k, zval *out_value, zval *return_value)
{
    if (!php_tensor_require_q8_0(w, 1) || !php_tensor_require_f32(x, 2)
        || !php_tensor_expect_rank2(w, 1, m, k)
        || !php_tensor_expect_rank1(x, 2, k)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 5, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 1, dims, 5)
            || !php_tensor_out_disjoint_check(out, w, x, 5, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(1, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATVEC_Q8_0, TOY_DTYPE_Q8_0,
                                                   m, 1, k);
    if (kernel != NULL) {
        kernel(tensor_data(w), tensor_data(x), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_q8_0_matvec(m, k, tensor_data(w), tensor_data(x),
                            tensor_data(result));
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * A q8_0 activation holds one vector of k values per column, so k must be a
 * whole number of value blocks
 */
static bool php_tensor_require_block_multiple(uint32_t k, uint32_t arg_num)
{
    if (k % TOY_Q8_0_BLOCK == 0) {
        return true;
    }
    zend_argument_value_error(arg_num, "must be a multiple of %d",
                              (int) TOY_Q8_0_BLOCK);
    return false;
}

static bool php_tensor_matmul_q8_0_q8_0_impl(tensor *w, tensor *x, uint32_t m,
                                             uint32_t n, uint32_t k, zval *out_value,
                                             zval *return_value)
{
    if (!php_tensor_require_q8_0(w, 1) || !php_tensor_require_q8_0(x, 2)
        || !php_tensor_require_block_multiple(k, 5)
        || !php_tensor_expect_rank2(w, 1, m, k)
        || !php_tensor_expect_rank2(x, 2, k, n)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 6, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 2, dims, 6)
            || !php_tensor_out_disjoint_check(out, w, x, 6, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(2, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATMUL_Q8_0_Q8_0, TOY_DTYPE_Q8_0,
                                           m, n, k);
    if (kernel != NULL) {
        kernel(tensor_data(w), tensor_data(x), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_q8_0_q8_0_matmul(m, n, k, tensor_data(w), tensor_data(x),
                             tensor_data(result));
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

static bool php_tensor_matvec_q8_0_q8_0_impl(tensor *w, tensor *x, uint32_t m,
                                             uint32_t k, zval *out_value,
                                             zval *return_value)
{
    if (!php_tensor_require_q8_0(w, 1) || !php_tensor_require_q8_0(x, 2)
        || !php_tensor_require_block_multiple(k, 4)
        || !php_tensor_expect_rank2(w, 1, m, k)
        || !php_tensor_expect_rank1(x, 2, k)) {
        return false;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 5, &out)) {
        return false;
    }
    if (out != NULL
        && (!php_tensor_out_matches(out, 1, dims, 5)
            || !php_tensor_out_disjoint_check(out, w, x, 5, 1, 2))) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(1, dims);
        if (result == NULL) {
            return false;
        }
    }

    kernel_entry kernel = kernel_for_shape(TOY_KERNEL_MATVEC_Q8_0_Q8_0, TOY_DTYPE_Q8_0,
                                           m, 1, k);
    if (kernel != NULL) {
        kernel(tensor_data(w), tensor_data(x), tensor_data(result));
        kernel_count_kernel_call();
    } else {
        ref_q8_0_q8_0_matvec(m, k, tensor_data(w), tensor_data(x),
                             tensor_data(result));
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * The state is the only in/out tensor, so it may not share bytes with any
 * input, and an explicit out may not share bytes with the state or an input
 */
TOY_NODISCARD static bool php_tensor_gdn_aliasing(const tensor *state,
                                                  const tensor *out,
                                                  const tensor *const sources[5])
{
    for (uint32_t i = 0; i < 5; i++) {
        if (php_tensor_ranges_overlap(state, sources[i])) {
            zend_argument_value_error(6, "must not overlap argument #%u", i + 1);
            return false;
        }
    }
    if (out == NULL) {
        return true;
    }
    if (php_tensor_ranges_overlap(out, state)) {
        zend_argument_value_error(12, "must not overlap argument #6 ($state)");
        return false;
    }
    for (uint32_t i = 0; i < 5; i++) {
        if (php_tensor_ranges_overlap(out, sources[i])) {
            zend_argument_value_error(12, "must not overlap argument #%u", i + 1);
            return false;
        }
    }
    return true;
}

/*
 * The gated delta net recurrence of one token. The state is
 * [heads, value_dim, key_dim], one key row per value column, and each local
 * column updates its own row in place
 */
static bool php_tensor_gdn_recurrence_impl(tensor *q, tensor *k, tensor *v,
                                           tensor *decay, tensor *beta, tensor *state,
                                           zend_long head_begin_arg, zend_long head_end_arg,
                                           zend_long col_begin_arg, zend_long col_end_arg,
                                           double scale, zval *out_value,
                                           zval *return_value)
{
    if (!php_tensor_require_f32(q, 1) || !php_tensor_require_f32(k, 2)
        || !php_tensor_require_f32(v, 3) || !php_tensor_require_f32(decay, 4)
        || !php_tensor_require_f32(beta, 5) || !php_tensor_require_f32(state, 6)) {
        return false;
    }
    if (q->rank != 2) {
        zend_argument_value_error(1, "must be a 2-D tensor");
        return false;
    }

    const uint32_t heads = q->dims[0];
    const uint32_t key_dim = q->dims[1];
    if (!php_tensor_expect_rank2(k, 2, heads, key_dim)) {
        return false;
    }
    if (v->rank != 2) {
        zend_argument_value_error(3, "must be a 2-D tensor");
        return false;
    }
    if (v->dims[0] != heads) {
        zend_argument_value_error(3, "must have the same number of heads as argument #1 ($q)");
        return false;
    }
    const uint32_t value_dim = v->dims[1];
    if (!php_tensor_expect_rank1(decay, 4, heads)
        || !php_tensor_expect_rank1(beta, 5, heads)
        /* The state is one key_dim row per value column, the ggml layout */
        || !php_tensor_expect_rank3(state, 6, heads, value_dim, key_dim)) {
        return false;
    }

    uint32_t head_begin, head_end, col_begin, col_end;
    if (!php_tensor_bound(head_begin_arg, heads, &head_begin, 7)
        || !php_tensor_bound(head_end_arg, heads, &head_end, 8)
        || !php_tensor_bound(col_begin_arg, value_dim, &col_begin, 9)
        || !php_tensor_bound(col_end_arg, value_dim, &col_end, 10)) {
        return false;
    }
    if (head_begin > head_end) {
        zend_argument_value_error(7, "must not be greater than argument #8 ($headEnd)");
        return false;
    }
    if (col_begin > col_end) {
        zend_argument_value_error(9, "must not be greater than argument #10 ($colEnd)");
        return false;
    }

    uint32_t out_dims[TOY_TENSOR_MAX_DIMS] = { heads, value_dim, 0, 0 };
    tensor *out = NULL;
    if (out_value != NULL && !php_tensor_optional_out(out_value, 12, &out)) {
        return false;
    }
    if (out != NULL && !php_tensor_out_matches(out, 2, out_dims, 12)) {
        return false;
    }

    const tensor *sources[5] = { q, k, v, decay, beta };
    if (!php_tensor_gdn_aliasing(state, out, sources)) {
        return false;
    }

    tensor *result = out;
    if (result == NULL) {
        result = php_tensor_alloc_result(2, out_dims);
        if (result == NULL) {
            return false;
        }
    }

    const gdn_problem problem = {
        .key_dim = key_dim,
        .value_dim = value_dim,
        .scale = (float) scale,
        .q = tensor_data(q),
        .k = tensor_data(k),
        .v = tensor_data(v),
        .decay = tensor_data(decay),
        .beta = tensor_data(beta),
        .state = tensor_data(state),
        .out = tensor_data(result),
    };
    const gdn_range range = {
        .head_begin = head_begin,
        .head_end = head_end,
        .col_begin = col_begin,
        .col_end = col_end,
    };

    bool accelerated = false;
    if (kernel_enabled() && kernel_isa() == TOY_ISA_AVX2) {
        accelerated = gdn_recurrence_avx2(&problem, &range);
    }
    if (accelerated) {
        kernel_count_kernel_call();
    } else {
        gdn_recurrence_scalar(&problem, &range);
        kernel_count_scalar_call();
    }

    if (out != NULL) {
        ZVAL_OBJ_COPY(return_value, Z_OBJ_P(out_value));
        return true;
    }
    return php_tensor_wrap(return_value, result) == SUCCESS;
}

/*
 * Quantizes rows [rowBegin, rowEnd) of an f32 tensor into q8_0 blocks
 *
 * A 1-D tensor is one row and the result keeps its rank. A 2-D tensor is rows
 * of k values, and the result holds only the selected rows, so a caller never
 * copies the source range
 */
static bool php_tensor_quantize_q8_0_impl(tensor *x, zend_long row_begin,
                                          zend_long row_end_arg, bool row_end_is_null,
                                          zval *return_value)
{
    if (!php_tensor_require_f32(x, 1)) {
        return false;
    }
    if (x->rank != 1 && x->rank != 2) {
        zend_argument_value_error(1, "must be a 1-D or 2-D tensor");
        return false;
    }

    const uint32_t k = x->dims[x->rank - 1];
    if (k % TOY_Q8_0_BLOCK != 0) {
        zend_argument_value_error(1, "must have a last dimension that is a multiple of %d",
                                  (int) TOY_Q8_0_BLOCK);
        return false;
    }

    const uint32_t rows = x->rank == 1 ? 1 : x->dims[0];
    if (row_begin < 0 || (zend_ulong) row_begin >= rows) {
        zend_argument_value_error(2, "must be between 0 and %u", rows - 1);
        return false;
    }
    const zend_long row_end = row_end_is_null ? (zend_long) rows : row_end_arg;
    if ((zend_ulong) row_end > rows) {
        zend_argument_value_error(3, "must be between 0 and %u", rows);
        return false;
    }
    if (row_end <= row_begin) {
        zend_argument_value_error(3, "must be greater than argument #2 ($rowBegin)");
        return false;
    }

    const uint32_t row_count = (uint32_t) (row_end - row_begin);
    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { 0 };
    uint8_t rank = 1;
    dims[0] = k;
    if (x->rank == 2) {
        rank = 2;
        dims[0] = row_count;
        dims[1] = k;
    }

    tensor *result = php_tensor_alloc_result_dtype(TOY_DTYPE_Q8_0, rank, dims);
    if (result == NULL) {
        return false;
    }

    const uint64_t blocks = (uint64_t) row_count * (k / TOY_Q8_0_BLOCK);
    const float *src = (const float *) tensor_data(x) + (size_t) row_begin * k;
    void *dst = tensor_data(result);
    kernel_entry kernel = blocks <= UINT32_MAX
        ? kernel_for_shape(TOY_KERNEL_Q8_0_QUANTIZE, TOY_DTYPE_F32, (uint32_t) blocks, 1, 1)
        : NULL;
    if (kernel != NULL) {
        kernel(src, NULL, dst);
        kernel_count_kernel_call();
    } else {
        q8_0_quantize_rows(row_count, k, src, dst);
        kernel_count_scalar_call();
    }

    return php_tensor_wrap(return_value, result) == SUCCESS;
}

PHP_FUNCTION(tensor_matmul_f32)
{
    zval *a_value, *b_value, *out_value = NULL;
    zend_long m_arg, n_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(5, 6)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(n_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    tensor *b = php_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }

    uint32_t m, n, k;
    if (!php_tensor_dim(m_arg, &m, 3)
        || !php_tensor_dim(n_arg, &n, 4)
        || !php_tensor_dim(k_arg, &k, 5)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matmul_impl(a, b, m, n, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_matvec_f32)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long m_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 5)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, k;
    if (!php_tensor_dim(m_arg, &m, 3) || !php_tensor_dim(k_arg, &k, 4)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matvec_impl(w, x, m, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_matmul_q8_0_f32)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long m_arg, n_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(5, 6)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(n_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, n, k;
    if (!php_tensor_dim(m_arg, &m, 3)
        || !php_tensor_dim(n_arg, &n, 4)
        || !php_tensor_dim(k_arg, &k, 5)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matmul_q8_0_impl(w, x, m, n, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_matvec_q8_0_f32)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long m_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 5)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, k;
    if (!php_tensor_dim(m_arg, &m, 3) || !php_tensor_dim(k_arg, &k, 4)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matvec_q8_0_impl(w, x, m, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_matmul_q8_0_q8_0)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long m_arg, n_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(5, 6)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(n_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, n, k;
    if (!php_tensor_dim(m_arg, &m, 3)
        || !php_tensor_dim(n_arg, &n, 4)
        || !php_tensor_dim(k_arg, &k, 5)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matmul_q8_0_q8_0_impl(w, x, m, n, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_matvec_q8_0_q8_0)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long m_arg, k_arg;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 5)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(k_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, k;
    if (!php_tensor_dim(m_arg, &m, 3) || !php_tensor_dim(k_arg, &k, 4)) {
        RETURN_THROWS();
    }
    if (!php_tensor_matvec_q8_0_q8_0_impl(w, x, m, k, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_quantize_q8_0)
{
    zval *x_value;
    zend_long row_begin = 0;
    zend_long row_end_arg = 0;
    bool row_end_is_null = true;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(row_begin)
        Z_PARAM_LONG_OR_NULL(row_end_arg, row_end_is_null)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    if (!x) {
        RETURN_THROWS();
    }
    if (!php_tensor_quantize_q8_0_impl(x, row_begin, row_end_arg, row_end_is_null,
                                       return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_add)
{
    zval *a_value, *b_value, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    tensor *b = php_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }
    if (!php_tensor_add_impl(a, b, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_mul)
{
    zval *a_value, *b_value, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    tensor *b = php_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }
    if (!php_tensor_mul_impl(a, b, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_scale)
{
    zval *a_value, *out_value = NULL;
    double factor;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_DOUBLE(factor)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    if (!php_tensor_scale_impl(a, factor, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_silu)
{
    zval *a_value, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    if (!php_tensor_silu_impl(a, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_sigmoid)
{
    zval *a_value, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    if (!php_tensor_activation_impl(a, out_value, return_value, TOY_KERNEL_SIGMOID)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_softplus)
{
    zval *a_value, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *a = php_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    if (!php_tensor_activation_impl(a, out_value, return_value, TOY_KERNEL_SOFTPLUS)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_l2norm_rows)
{
    zval *x_value, *out_value = NULL;
    double eps = rmsnorm_default_eps;
    zend_long start_arg = 0;
    zend_long count_arg = 0;
    bool count_is_null = true;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
        Z_PARAM_LONG(start_arg)
        Z_PARAM_LONG_OR_NULL(count_arg, count_is_null)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    if (!x) {
        RETURN_THROWS();
    }
    if (!php_tensor_l2norm_rows_impl(x, eps, start_arg, count_arg, count_is_null,
                                     out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_rmsnorm_rows)
{
    zval *x_value, *w_value, *out_value = NULL;
    double eps = rmsnorm_default_eps;
    zend_long start_arg = 0;
    zend_long count_arg = 0;
    bool count_is_null = true;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 6)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
        Z_PARAM_LONG(start_arg)
        Z_PARAM_LONG_OR_NULL(count_arg, count_is_null)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    tensor *w = php_tensor_arg(w_value);
    if (!x || !w) {
        RETURN_THROWS();
    }
    if (!php_tensor_rmsnorm_rows_impl(x, w, eps, start_arg, count_arg, count_is_null,
                                      out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_shift_rows)
{
    zval *x_value, *row_value = NULL, *out_value = NULL;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(row_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    if (!x) {
        RETURN_THROWS();
    }
    tensor *row = NULL;
    if (!php_tensor_optional_out(row_value, 2, &row)) {
        RETURN_THROWS();
    }
    if (!php_tensor_shift_rows_impl(x, row, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_conv1d_silu)
{
    zval *w_value, *x_value, *out_value = NULL;
    zend_long rows_arg, kernel_arg;
    zend_long start_arg = 0;
    zend_long count_arg = 0;
    bool count_is_null = true;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 7)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_LONG(rows_arg)
        Z_PARAM_LONG(kernel_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(start_arg)
        Z_PARAM_LONG_OR_NULL(count_arg, count_is_null)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *w = php_tensor_arg(w_value);
    tensor *x = php_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t rows, taps;
    if (!php_tensor_dim(rows_arg, &rows, 3)
        || !php_tensor_dim(kernel_arg, &taps, 4)) {
        RETURN_THROWS();
    }
    if (!php_tensor_conv1d_silu_impl(w, x, rows, taps, start_arg, count_arg,
                                     count_is_null, out_value, return_value)) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_rmsnorm)
{
    zval *x_value, *w_value;
    double eps = rmsnorm_default_eps;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    tensor *w = php_tensor_arg(w_value);
    if (!x || !w) {
        RETURN_THROWS();
    }
    php_tensor_rmsnorm_impl(x, w, eps, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(tensor_rmsnorm_into)
{
    zval *x_value, *w_value, *out_value;
    double eps = rmsnorm_default_eps;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(3, 4)
        Z_PARAM_OBJECT_OF_CLASS(x_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(w_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(out_value, tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
    ZEND_PARSE_PARAMETERS_END();

    tensor *x = php_tensor_arg(x_value);
    tensor *w = php_tensor_arg(w_value);
    tensor *out = php_tensor_arg(out_value);
    if (!x || !w || !out) {
        RETURN_THROWS();
    }
    if (!php_tensor_require_f32(x, 1) || !php_tensor_require_f32(w, 2)) {
        RETURN_THROWS();
    }
    if (x->rank != 1) {
        zend_argument_value_error(1, "must be a 1-D tensor");
        RETURN_THROWS();
    }
    if (!php_tensor_expect_rank1(w, 2, x->dims[0])) {
        RETURN_THROWS();
    }

    uint32_t n = x->dims[0];
    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { n, 0, 0, 0 };
    if (!php_tensor_out_matches(out, 1, dims, 3)
        || !php_tensor_out_rmsnorm_check(out, x, w, 3)) {
        RETURN_THROWS();
    }

    const float eps32 = (float) eps;
    kernel_entry kernel = kernel_for_element(x, TOY_KERNEL_RMSNORM,
                                                     php_float_bits(eps32), n);
    if (kernel != NULL) {
        kernel(tensor_data(x), tensor_data(w), tensor_data(out));
        kernel_count_kernel_call();
    } else {
        ref_rmsnorm_f32(tensor_data(x), tensor_data(w), tensor_data(out),
                        n, eps32);
        kernel_count_scalar_call();
    }

    RETURN_OBJ_COPY(Z_OBJ_P(out_value));
}

PHP_FUNCTION(tensor_gdn_recurrence)
{
    zval *q_value, *k_value, *v_value, *decay_value, *beta_value, *state_value;
    zval *out_value = NULL;
    zend_long head_begin_arg, head_end_arg, col_begin_arg, col_end_arg;
    double scale = 1.0;

    kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(10, 12)
        Z_PARAM_OBJECT_OF_CLASS(q_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(k_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(v_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(decay_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(beta_value, tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(state_value, tensor_ce)
        Z_PARAM_LONG(head_begin_arg)
        Z_PARAM_LONG(head_end_arg)
        Z_PARAM_LONG(col_begin_arg)
        Z_PARAM_LONG(col_end_arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(out_value, tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    tensor *q = php_tensor_arg(q_value);
    tensor *k = php_tensor_arg(k_value);
    tensor *v = php_tensor_arg(v_value);
    tensor *decay = php_tensor_arg(decay_value);
    tensor *beta = php_tensor_arg(beta_value);
    tensor *state = php_tensor_arg(state_value);
    if (!q || !k || !v || !decay || !beta || !state) {
        RETURN_THROWS();
    }

    if (!php_tensor_gdn_recurrence_impl(q, k, v, decay, beta, state,
                                        head_begin_arg, head_end_arg,
                                        col_begin_arg, col_end_arg, scale,
                                        out_value, return_value)) {
        RETURN_THROWS();
    }
}

ZEND_FRAMELESS_FUNCTION(tensor_add_flf, 3)
{
    kernel_count_flf();

    tensor *a, *b;
    if (!php_tensor_object(arg1, &a, 1) || !php_tensor_object(arg2, &b, 2)) {
        return;
    }

    zval *out_value = NULL;
    if (arg3 != NULL && Z_TYPE_P(arg3) != IS_NULL && Z_TYPE_P(arg3) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg3, &out, 3)) {
            return;
        }
        out_value = arg3;
    }
    php_tensor_add_impl(a, b, out_value, return_value);
}

ZEND_FRAMELESS_FUNCTION(tensor_mul_flf, 3)
{
    kernel_count_flf();

    tensor *a, *b;
    if (!php_tensor_object(arg1, &a, 1) || !php_tensor_object(arg2, &b, 2)) {
        return;
    }

    zval *out_value = NULL;
    if (arg3 != NULL && Z_TYPE_P(arg3) != IS_NULL && Z_TYPE_P(arg3) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg3, &out, 3)) {
            return;
        }
        out_value = arg3;
    }
    php_tensor_mul_impl(a, b, out_value, return_value);
}

ZEND_FRAMELESS_FUNCTION(tensor_scale_flf, 3)
{
    kernel_count_flf();

    tensor *a;
    if (!php_tensor_object(arg1, &a, 1)) {
        return;
    }

    double factor;
    bool is_null = false;
    if (!zend_parse_arg_double(arg2, &factor, &is_null, false, 2)) {
        zend_wrong_parameter_type_error(2, Z_EXPECTED_DOUBLE, arg2);
        return;
    }

    zval *out_value = NULL;
    if (arg3 != NULL && Z_TYPE_P(arg3) != IS_NULL && Z_TYPE_P(arg3) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg3, &out, 3)) {
            return;
        }
        out_value = arg3;
    }
    php_tensor_scale_impl(a, factor, out_value, return_value);
}

ZEND_FRAMELESS_FUNCTION(tensor_silu_flf, 2)
{
    kernel_count_flf();

    tensor *a;
    if (!php_tensor_object(arg1, &a, 1)) {
        return;
    }

    zval *out_value = NULL;
    if (arg2 != NULL && Z_TYPE_P(arg2) != IS_NULL && Z_TYPE_P(arg2) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg2, &out, 2)) {
            return;
        }
        out_value = arg2;
    }
    php_tensor_silu_impl(a, out_value, return_value);
}

ZEND_FRAMELESS_FUNCTION(tensor_sigmoid_flf, 2)
{
    kernel_count_flf();

    tensor *a;
    if (!php_tensor_object(arg1, &a, 1)) {
        return;
    }

    zval *out_value = NULL;
    if (arg2 != NULL && Z_TYPE_P(arg2) != IS_NULL && Z_TYPE_P(arg2) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg2, &out, 2)) {
            return;
        }
        out_value = arg2;
    }
    php_tensor_activation_impl(a, out_value, return_value, TOY_KERNEL_SIGMOID);
}

ZEND_FRAMELESS_FUNCTION(tensor_softplus_flf, 2)
{
    kernel_count_flf();

    tensor *a;
    if (!php_tensor_object(arg1, &a, 1)) {
        return;
    }

    zval *out_value = NULL;
    if (arg2 != NULL && Z_TYPE_P(arg2) != IS_NULL && Z_TYPE_P(arg2) != IS_UNDEF) {
        tensor *out;
        if (!php_tensor_object(arg2, &out, 2)) {
            return;
        }
        out_value = arg2;
    }
    php_tensor_activation_impl(a, out_value, return_value, TOY_KERNEL_SOFTPLUS);
}

ZEND_FRAMELESS_FUNCTION(tensor_rmsnorm_flf, 3)
{
    kernel_count_flf();

    tensor *x, *w;
    if (!php_tensor_object(arg1, &x, 1) || !php_tensor_object(arg2, &w, 2)) {
        return;
    }

    double eps;
    bool is_null = false;
    if (!zend_parse_arg_double(arg3, &eps, &is_null, false, 3)) {
        zend_wrong_parameter_type_error(3, Z_EXPECTED_DOUBLE, arg3);
        return;
    }
    php_tensor_rmsnorm_impl(x, w, eps, return_value);
}

PHP_FUNCTION(tensor_kernel_stats)
{
    bool reset = false;

    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(reset)
    ZEND_PARSE_PARAMETERS_END();

    kernel_stats stats;
    kernel_stats_read(&stats);

    array_init(return_value);
    add_assoc_bool(return_value, "enabled", stats.enabled);
    add_assoc_bool(return_value, "avx2_available", stats.avx2_available);
    add_assoc_bool(return_value, "f16c_available", stats.f16c_available);
    add_assoc_string(return_value, "isa", (char *) stats.isa);
    add_assoc_long(return_value, "flf_calls", (zend_long) stats.flf_calls);
    add_assoc_long(return_value, "zif_calls", (zend_long) stats.zif_calls);
    add_assoc_long(return_value, "memo_hits", (zend_long) stats.memo_hits);
    add_assoc_long(return_value, "memo_misses", (zend_long) stats.memo_misses);
    add_assoc_long(return_value, "arena_hits", (zend_long) stats.arena_hits);
    add_assoc_long(return_value, "arena_misses", (zend_long) stats.arena_misses);
    add_assoc_long(return_value, "generated", (zend_long) stats.generated);
    add_assoc_long(return_value, "generation_failures", (zend_long) stats.generation_failures);
    add_assoc_long(return_value, "kernel_calls", (zend_long) stats.kernel_calls);
    add_assoc_long(return_value, "scalar_calls", (zend_long) stats.scalar_calls);

    if (reset) {
        kernel_stats_reset();
    }
}

PHP_FUNCTION(tensor_kernel_enable)
{
    bool enable;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_BOOL(enable)
    ZEND_PARSE_PARAMETERS_END();

    RETURN_BOOL(kernel_set_enabled(enable));
}

PHP_FUNCTION(tensor_kernel_isa)
{
    zend_string *name = NULL;

    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(name)
    ZEND_PARSE_PARAMETERS_END();

    if (name != NULL) {
        uint32_t isa;
        if (zend_string_equals_literal(name, "scalar")) {
            isa = TOY_ISA_SCALAR;
        } else if (zend_string_equals_literal(name, "avx2")) {
            isa = TOY_ISA_AVX2;
        } else {
            zend_argument_value_error(1, "must be \"scalar\" or \"avx2\"");
            RETURN_THROWS();
        }
        if (!kernel_set_isa(isa)) {
            zend_argument_value_error(1, "is not supported by this host");
            RETURN_THROWS();
        }
    }

    RETURN_STRING(isa_name(kernel_isa()));
}

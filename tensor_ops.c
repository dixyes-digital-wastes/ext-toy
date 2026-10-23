#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"

#include "php_toy.h"
#include "toy_kernel.h"
#include "toy_tensor_ops.h"

#include <math.h>
#include <string.h>

#include "toy/codegen.h"
#include "toy/ref.h"

static const double toy_rmsnorm_default_eps = 1e-6;

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_matmul_f32, 0, 5, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_matvec_f32, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_matmul_q8_0_f32, 0, 5, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_matvec_q8_0_f32, 0, 4, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_add, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, Toy\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_scale, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, a, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, factor, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_toy_tensor_rmsnorm, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, x, Toy\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, w, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, eps, IS_DOUBLE, 0, "1e-6")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_toy_tensor_kernel_stats, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, reset, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_toy_tensor_kernel_enable, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, enable, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_toy_tensor_kernel_isa, 0, 0, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, isa, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()

ZEND_FRAMELESS_FUNCTION(toy_tensor_add_flf, 2);
ZEND_FRAMELESS_FUNCTION(toy_tensor_scale_flf, 2);
ZEND_FRAMELESS_FUNCTION(toy_tensor_rmsnorm_flf, 3);

static const zend_frameless_function_info frameless_function_infos_toy_tensor_add[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(toy_tensor_add_flf, 2), 2 },
    { 0 },
};

static const zend_frameless_function_info frameless_function_infos_toy_tensor_scale[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(toy_tensor_scale_flf, 2), 2 },
    { 0 },
};

static const zend_frameless_function_info frameless_function_infos_toy_tensor_rmsnorm[] = {
    { ZEND_FRAMELESS_FUNCTION_NAME(toy_tensor_rmsnorm_flf, 3), 3 },
    { 0 },
};

#define TOY_NS_FRAMELESS_FE(ns, name, handler, arg_info, infos) \
    ZEND_RAW_FENTRY(ZEND_NS_NAME(ns, #name), handler, arg_info, 0, infos, NULL)

const zend_function_entry toy_functions[] = {
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_f32, zif_toy_tensor_matmul_f32,
                     arginfo_toy_tensor_matmul_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_f32, zif_toy_tensor_matvec_f32,
                     arginfo_toy_tensor_matvec_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_q8_0_f32, zif_toy_tensor_matmul_q8_0_f32,
                     arginfo_toy_tensor_matmul_q8_0_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_q8_0_f32, zif_toy_tensor_matvec_q8_0_f32,
                     arginfo_toy_tensor_matvec_q8_0_f32)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", add, zif_toy_tensor_add,
                        arginfo_toy_tensor_add, frameless_function_infos_toy_tensor_add)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", scale, zif_toy_tensor_scale,
                        arginfo_toy_tensor_scale, frameless_function_infos_toy_tensor_scale)
    TOY_NS_FRAMELESS_FE("Toy\\Tensor", rmsnorm, zif_toy_tensor_rmsnorm,
                        arginfo_toy_tensor_rmsnorm, frameless_function_infos_toy_tensor_rmsnorm)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_stats, zif_toy_tensor_kernel_stats,
                     arginfo_toy_tensor_kernel_stats)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_enable, zif_toy_tensor_kernel_enable,
                     arginfo_toy_tensor_kernel_enable)
    ZEND_NS_NAMED_FE("Toy\\Tensor", kernel_isa, zif_toy_tensor_kernel_isa,
                     arginfo_toy_tensor_kernel_isa)
    ZEND_FE_END
};

[[nodiscard]] static toy_tensor *php_toy_tensor_arg(zval *value)
{
    toy_tensor *tensor = php_toy_tensor_fetch(value);

    if (!tensor) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
    }
    return tensor;
}

[[nodiscard]] static bool php_toy_tensor_object(zval *value, toy_tensor **tensor,
                                                uint32_t arg_num)
{
    if (UNEXPECTED(Z_TYPE_P(value) != IS_OBJECT)
        || UNEXPECTED(!instanceof_function(Z_OBJCE_P(value), toy_tensor_ce))) {
        zend_argument_type_error(arg_num, "must be of type Toy\\Tensor, %s given",
                                 zend_zval_type_name(value));
        return false;
    }

    toy_tensor *fetched = php_toy_tensor_fetch(value);
    if (UNEXPECTED(fetched == NULL)) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
        return false;
    }
    *tensor = fetched;
    return true;
}

[[nodiscard]] static bool php_toy_tensor_require_f32(const toy_tensor *tensor,
                                                     uint32_t arg_num)
{
    if (tensor->dtype != TOY_DTYPE_F32) {
        zend_argument_value_error(arg_num, "must be an f32 tensor");
        return false;
    }
    return true;
}

[[nodiscard]] static bool php_toy_tensor_require_q8_0(const toy_tensor *tensor,
                                                      uint32_t arg_num)
{
    if (tensor->dtype != TOY_DTYPE_Q8_0) {
        zend_argument_value_error(arg_num, "must be a q8_0 tensor");
        return false;
    }
    return true;
}

[[nodiscard]] static bool php_toy_tensor_dim(zend_long value, uint32_t *dim,
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

[[nodiscard]] static bool php_toy_tensor_expect_rank2(const toy_tensor *tensor,
                                                      uint32_t arg_num, uint32_t dim0,
                                                      uint32_t dim1)
{
    if (tensor->rank != 2) {
        zend_argument_value_error(arg_num, "must be a 2-D tensor");
        return false;
    }
    if (tensor->dims[0] != dim0 || tensor->dims[1] != dim1) {
        zend_argument_value_error(arg_num, "must have shape [%u, %u], got [%u, %u]",
                                  dim0, dim1, tensor->dims[0], tensor->dims[1]);
        return false;
    }
    return true;
}

[[nodiscard]] static bool php_toy_tensor_expect_rank1(const toy_tensor *tensor,
                                                      uint32_t arg_num, uint32_t length)
{
    if (tensor->rank != 1) {
        zend_argument_value_error(arg_num, "must be a 1-D tensor");
        return false;
    }
    if (tensor->dims[0] != length) {
        zend_argument_value_error(arg_num, "must have shape [%u], got [%u]",
                                  length, tensor->dims[0]);
        return false;
    }
    return true;
}

[[nodiscard]] static toy_tensor *php_toy_tensor_alloc_result(uint8_t rank,
                                                            const uint32_t *dims)
{
    toy_status status;
    toy_tensor *tensor = toy_tensor_alloc(TOY_DTYPE_F32, rank, dims, &status);

    if (!tensor) {
        if (status == TOY_ERROR_MEMORY) {
            zend_throw_error(NULL, "Tensor allocation failed");
        } else {
            zend_value_error("Result tensor is too large");
        }
    }
    return tensor;
}

static uint32_t php_toy_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t php_toy_element_count(size_t numel)
{
    return numel <= UINT32_MAX ? (uint32_t) numel : 0;
}

static void php_toy_tensor_add_impl(toy_tensor *a, toy_tensor *b, zval *return_value)
{
    if (!php_toy_tensor_require_f32(a, 1) || !php_toy_tensor_require_f32(b, 2)) {
        return;
    }
    if (a->rank != b->rank) {
        zend_argument_value_error(2,
            "must have the same number of dimensions as argument #1 ($a)");
        return;
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) {
            zend_argument_value_error(2, "must have the same shape as argument #1 ($a)");
            return;
        }
    }

    toy_tensor *result = php_toy_tensor_alloc_result(a->rank, a->dims);
    if (!result) {
        return;
    }

    size_t numel = toy_tensor_numel(a);
    uint32_t count = php_toy_element_count(numel);
    toy_kernel_entry kernel = count != 0
        ? toy_kernel_for_element(a, TOY_KERNEL_ADD, 0, count)
        : NULL;
    if (kernel != NULL) {
        kernel(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_add_f32(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result), numel);
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_scale_impl(toy_tensor *a, double factor, zval *return_value)
{
    if (!php_toy_tensor_require_f32(a, 1)) {
        return;
    }

    toy_tensor *result = php_toy_tensor_alloc_result(a->rank, a->dims);
    if (!result) {
        return;
    }

    const float value = (float) factor;
    size_t numel = toy_tensor_numel(a);
    uint32_t count = php_toy_element_count(numel);
    toy_kernel_entry kernel = count != 0
        ? toy_kernel_for_element(a, TOY_KERNEL_SCALE, php_toy_float_bits(value), count)
        : NULL;
    if (kernel != NULL) {
        kernel(toy_tensor_data(a), NULL, toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_scale_f32(toy_tensor_data(a), value, toy_tensor_data(result), numel);
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_rmsnorm_impl(toy_tensor *x, toy_tensor *w, double eps,
                                        zval *return_value)
{
    if (!php_toy_tensor_require_f32(x, 1) || !php_toy_tensor_require_f32(w, 2)) {
        return;
    }
    if (x->rank != 1) {
        zend_argument_value_error(1, "must be a 1-D tensor");
        return;
    }
    if (!php_toy_tensor_expect_rank1(w, 2, x->dims[0])) {
        return;
    }

    uint32_t n = x->dims[0];
    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { n, 0, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(1, dims);
    if (!result) {
        return;
    }

    const float eps32 = (float) eps;
    toy_kernel_entry kernel = toy_kernel_for_element(x, TOY_KERNEL_RMSNORM,
                                                     php_toy_float_bits(eps32), n);
    if (kernel != NULL) {
        kernel(toy_tensor_data(x), toy_tensor_data(w), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_rmsnorm_f32(toy_tensor_data(x), toy_tensor_data(w), toy_tensor_data(result),
                        n, eps32);
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_matmul_impl(toy_tensor *a, toy_tensor *b, uint32_t m, uint32_t n,
                                       uint32_t k, zval *return_value)
{
    if (!php_toy_tensor_require_f32(a, 1) || !php_toy_tensor_require_f32(b, 2)
        || !php_toy_tensor_expect_rank2(a, 1, m, k)
        || !php_toy_tensor_expect_rank2(b, 2, k, n)) {
        return;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(2, dims);
    if (!result) {
        return;
    }

    toy_kernel_entry kernel = toy_kernel_for_shape(TOY_KERNEL_MATMUL, TOY_DTYPE_F32, m, n, k);
    if (kernel != NULL) {
        kernel(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_matmul_f32(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result),
                       m, n, k);
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_matvec_impl(toy_tensor *w, toy_tensor *x, uint32_t m, uint32_t k,
                                       zval *return_value)
{
    if (!php_toy_tensor_require_f32(w, 1) || !php_toy_tensor_require_f32(x, 2)
        || !php_toy_tensor_expect_rank2(w, 1, m, k)
        || !php_toy_tensor_expect_rank1(x, 2, k)) {
        return;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(1, dims);
    if (!result) {
        return;
    }

    toy_kernel_entry kernel = toy_kernel_for_shape(TOY_KERNEL_MATVEC, TOY_DTYPE_F32, m, 1, k);
    if (kernel != NULL) {
        kernel(toy_tensor_data(w), toy_tensor_data(x), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_matvec_f32(toy_tensor_data(w), toy_tensor_data(x), toy_tensor_data(result), m, k);
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_matmul_q8_0_impl(toy_tensor *w, toy_tensor *x, uint32_t m,
                                            uint32_t n, uint32_t k, zval *return_value)
{
    if (!php_toy_tensor_require_q8_0(w, 1) || !php_toy_tensor_require_f32(x, 2)
        || !php_toy_tensor_expect_rank2(w, 1, m, k)
        || !php_toy_tensor_expect_rank2(x, 2, k, n)) {
        return;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(2, dims);
    if (!result) {
        return;
    }

    toy_kernel_entry kernel = toy_kernel_for_shape(TOY_KERNEL_MATMUL_Q8_0, TOY_DTYPE_Q8_0,
                                                   m, n, k);
    if (kernel != NULL) {
        kernel(toy_tensor_data(w), toy_tensor_data(x), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_ref_q8_0_matmul(m, n, k, toy_tensor_data(w), toy_tensor_data(x),
                            toy_tensor_data(result));
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

static void php_toy_tensor_matvec_q8_0_impl(toy_tensor *w, toy_tensor *x, uint32_t m,
                                            uint32_t k, zval *return_value)
{
    if (!php_toy_tensor_require_q8_0(w, 1) || !php_toy_tensor_require_f32(x, 2)
        || !php_toy_tensor_expect_rank2(w, 1, m, k)
        || !php_toy_tensor_expect_rank1(x, 2, k)) {
        return;
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(1, dims);
    if (!result) {
        return;
    }

    toy_kernel_entry kernel = toy_kernel_for_shape(TOY_KERNEL_MATVEC_Q8_0, TOY_DTYPE_Q8_0,
                                                   m, 1, k);
    if (kernel != NULL) {
        kernel(toy_tensor_data(w), toy_tensor_data(x), toy_tensor_data(result));
        toy_kernel_count_kernel_call();
    } else {
        toy_ref_q8_0_matvec(m, k, toy_tensor_data(w), toy_tensor_data(x),
                            toy_tensor_data(result));
        toy_kernel_count_scalar_call();
    }

    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        return;
    }
}

PHP_FUNCTION(toy_tensor_matmul_f32)
{
    zval *a_value, *b_value;
    zend_long m_arg, n_arg, k_arg;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(5, 5)
        Z_PARAM_OBJECT_OF_CLASS(a_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, toy_tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(n_arg)
        Z_PARAM_LONG(k_arg)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *a = php_toy_tensor_arg(a_value);
    toy_tensor *b = php_toy_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }

    uint32_t m, n, k;
    if (!php_toy_tensor_dim(m_arg, &m, 3)
        || !php_toy_tensor_dim(n_arg, &n, 4)
        || !php_toy_tensor_dim(k_arg, &k, 5)) {
        RETURN_THROWS();
    }
    php_toy_tensor_matmul_impl(a, b, m, n, k, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_matvec_f32)
{
    zval *w_value, *x_value;
    zend_long m_arg, k_arg;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 4)
        Z_PARAM_OBJECT_OF_CLASS(w_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, toy_tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(k_arg)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *w = php_toy_tensor_arg(w_value);
    toy_tensor *x = php_toy_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, k;
    if (!php_toy_tensor_dim(m_arg, &m, 3) || !php_toy_tensor_dim(k_arg, &k, 4)) {
        RETURN_THROWS();
    }
    php_toy_tensor_matvec_impl(w, x, m, k, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_matmul_q8_0_f32)
{
    zval *w_value, *x_value;
    zend_long m_arg, n_arg, k_arg;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(5, 5)
        Z_PARAM_OBJECT_OF_CLASS(w_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, toy_tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(n_arg)
        Z_PARAM_LONG(k_arg)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *w = php_toy_tensor_arg(w_value);
    toy_tensor *x = php_toy_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, n, k;
    if (!php_toy_tensor_dim(m_arg, &m, 3)
        || !php_toy_tensor_dim(n_arg, &n, 4)
        || !php_toy_tensor_dim(k_arg, &k, 5)) {
        RETURN_THROWS();
    }
    php_toy_tensor_matmul_q8_0_impl(w, x, m, n, k, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_matvec_q8_0_f32)
{
    zval *w_value, *x_value;
    zend_long m_arg, k_arg;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(4, 4)
        Z_PARAM_OBJECT_OF_CLASS(w_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(x_value, toy_tensor_ce)
        Z_PARAM_LONG(m_arg)
        Z_PARAM_LONG(k_arg)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *w = php_toy_tensor_arg(w_value);
    toy_tensor *x = php_toy_tensor_arg(x_value);
    if (!w || !x) {
        RETURN_THROWS();
    }

    uint32_t m, k;
    if (!php_toy_tensor_dim(m_arg, &m, 3) || !php_toy_tensor_dim(k_arg, &k, 4)) {
        RETURN_THROWS();
    }
    php_toy_tensor_matvec_q8_0_impl(w, x, m, k, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_add)
{
    zval *a_value, *b_value;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, toy_tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *a = php_toy_tensor_arg(a_value);
    toy_tensor *b = php_toy_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }
    php_toy_tensor_add_impl(a, b, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_scale)
{
    zval *a_value;
    double factor;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, toy_tensor_ce)
        Z_PARAM_DOUBLE(factor)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *a = php_toy_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    php_toy_tensor_scale_impl(a, factor, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_rmsnorm)
{
    zval *x_value, *w_value;
    double eps = toy_rmsnorm_default_eps;

    toy_kernel_count_zif();

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(x_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(w_value, toy_tensor_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *x = php_toy_tensor_arg(x_value);
    toy_tensor *w = php_toy_tensor_arg(w_value);
    if (!x || !w) {
        RETURN_THROWS();
    }
    php_toy_tensor_rmsnorm_impl(x, w, eps, return_value);
    if (UNEXPECTED(EG(exception))) {
        RETURN_THROWS();
    }
}

ZEND_FRAMELESS_FUNCTION(toy_tensor_add_flf, 2)
{
    toy_kernel_count_flf();

    toy_tensor *a, *b;
    if (!php_toy_tensor_object(arg1, &a, 1) || !php_toy_tensor_object(arg2, &b, 2)) {
        return;
    }
    php_toy_tensor_add_impl(a, b, return_value);
}

ZEND_FRAMELESS_FUNCTION(toy_tensor_scale_flf, 2)
{
    toy_kernel_count_flf();

    toy_tensor *a;
    if (!php_toy_tensor_object(arg1, &a, 1)) {
        return;
    }

    double factor;
    bool is_null = false;
    if (!zend_parse_arg_double(arg2, &factor, &is_null, false, 2)) {
        zend_wrong_parameter_type_error(2, Z_EXPECTED_DOUBLE, arg2);
        return;
    }
    php_toy_tensor_scale_impl(a, factor, return_value);
}

ZEND_FRAMELESS_FUNCTION(toy_tensor_rmsnorm_flf, 3)
{
    toy_kernel_count_flf();

    toy_tensor *x, *w;
    if (!php_toy_tensor_object(arg1, &x, 1) || !php_toy_tensor_object(arg2, &w, 2)) {
        return;
    }

    double eps;
    bool is_null = false;
    if (!zend_parse_arg_double(arg3, &eps, &is_null, false, 3)) {
        zend_wrong_parameter_type_error(3, Z_EXPECTED_DOUBLE, arg3);
        return;
    }
    php_toy_tensor_rmsnorm_impl(x, w, eps, return_value);
}

PHP_FUNCTION(toy_tensor_kernel_stats)
{
    bool reset = false;

    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(reset)
    ZEND_PARSE_PARAMETERS_END();

    toy_kernel_stats stats;
    toy_kernel_stats_read(&stats);

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
        toy_kernel_stats_reset();
    }
}

PHP_FUNCTION(toy_tensor_kernel_enable)
{
    bool enable;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_BOOL(enable)
    ZEND_PARSE_PARAMETERS_END();

    RETURN_BOOL(toy_kernel_set_enabled(enable));
}

PHP_FUNCTION(toy_tensor_kernel_isa)
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
        if (!toy_kernel_set_isa(isa)) {
            zend_argument_value_error(1, "is not supported by this host");
            RETURN_THROWS();
        }
    }

    RETURN_STRING(toy_isa_name(toy_kernel_isa()));
}

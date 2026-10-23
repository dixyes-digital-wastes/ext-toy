#include "php.h"

#include "php_tensor.h"
#include "toy_tensor_ops.h"

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

const zend_function_entry tensor_functions[] = {
    ZEND_NS_NAMED_FE("Toy\\Tensor", matmul_f32, zif_toy_tensor_matmul_f32,
                     arginfo_toy_tensor_matmul_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", matvec_f32, zif_toy_tensor_matvec_f32,
                     arginfo_toy_tensor_matvec_f32)
    ZEND_NS_NAMED_FE("Toy\\Tensor", add, zif_toy_tensor_add,
                     arginfo_toy_tensor_add)
    ZEND_NS_NAMED_FE("Toy\\Tensor", scale, zif_toy_tensor_scale,
                     arginfo_toy_tensor_scale)
    ZEND_NS_NAMED_FE("Toy\\Tensor", rmsnorm, zif_toy_tensor_rmsnorm,
                     arginfo_toy_tensor_rmsnorm)
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

[[nodiscard]] static bool php_toy_tensor_require_f32(const toy_tensor *tensor,
                                                     uint32_t arg_num)
{
    if (tensor->dtype != TOY_DTYPE_F32) {
        zend_argument_value_error(arg_num, "must be an f32 tensor");
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

PHP_FUNCTION(toy_tensor_matmul_f32)
{
    zval *a_value, *b_value;
    zend_long m_arg, n_arg, k_arg;

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
    if (!php_toy_tensor_require_f32(a, 1) || !php_toy_tensor_require_f32(b, 2)
        || !php_toy_tensor_expect_rank2(a, 1, m, k)
        || !php_toy_tensor_expect_rank2(b, 2, k, n)) {
        RETURN_THROWS();
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, n, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(2, dims);
    if (!result) {
        RETURN_THROWS();
    }

    toy_matmul_f32(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result), m, n, k);
    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_matvec_f32)
{
    zval *w_value, *x_value;
    zend_long m_arg, k_arg;

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
    if (!php_toy_tensor_require_f32(w, 1) || !php_toy_tensor_require_f32(x, 2)
        || !php_toy_tensor_expect_rank2(w, 1, m, k)
        || !php_toy_tensor_expect_rank1(x, 2, k)) {
        RETURN_THROWS();
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { m, 0, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(1, dims);
    if (!result) {
        RETURN_THROWS();
    }

    toy_matvec_f32(toy_tensor_data(w), toy_tensor_data(x), toy_tensor_data(result), m, k);
    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_add)
{
    zval *a_value, *b_value;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, toy_tensor_ce)
        Z_PARAM_OBJECT_OF_CLASS(b_value, toy_tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *a = php_toy_tensor_arg(a_value);
    toy_tensor *b = php_toy_tensor_arg(b_value);
    if (!a || !b) {
        RETURN_THROWS();
    }
    if (!php_toy_tensor_require_f32(a, 1) || !php_toy_tensor_require_f32(b, 2)) {
        RETURN_THROWS();
    }
    if (a->rank != b->rank) {
        zend_argument_value_error(2,
            "must have the same number of dimensions as argument #1 ($a)");
        RETURN_THROWS();
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) {
            zend_argument_value_error(2, "must have the same shape as argument #1 ($a)");
            RETURN_THROWS();
        }
    }

    toy_tensor *result = php_toy_tensor_alloc_result(a->rank, a->dims);
    if (!result) {
        RETURN_THROWS();
    }

    toy_add_f32(toy_tensor_data(a), toy_tensor_data(b), toy_tensor_data(result),
                toy_tensor_numel(a));
    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_scale)
{
    zval *a_value;
    double factor;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(a_value, toy_tensor_ce)
        Z_PARAM_DOUBLE(factor)
    ZEND_PARSE_PARAMETERS_END();

    toy_tensor *a = php_toy_tensor_arg(a_value);
    if (!a) {
        RETURN_THROWS();
    }
    if (!php_toy_tensor_require_f32(a, 1)) {
        RETURN_THROWS();
    }

    toy_tensor *result = php_toy_tensor_alloc_result(a->rank, a->dims);
    if (!result) {
        RETURN_THROWS();
    }

    toy_scale_f32(toy_tensor_data(a), (float) factor, toy_tensor_data(result),
                  toy_tensor_numel(a));
    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        RETURN_THROWS();
    }
}

PHP_FUNCTION(toy_tensor_rmsnorm)
{
    zval *x_value, *w_value;
    double eps = toy_rmsnorm_default_eps;

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
    if (!php_toy_tensor_require_f32(x, 1) || !php_toy_tensor_require_f32(w, 2)) {
        RETURN_THROWS();
    }
    if (x->rank != 1) {
        zend_argument_value_error(1, "must be a 1-D tensor");
        RETURN_THROWS();
    }
    if (!php_toy_tensor_expect_rank1(w, 2, x->dims[0])) {
        RETURN_THROWS();
    }

    uint32_t n = x->dims[0];
    uint32_t dims[TOY_TENSOR_MAX_DIMS] = { n, 0, 0, 0 };
    toy_tensor *result = php_toy_tensor_alloc_result(1, dims);
    if (!result) {
        RETURN_THROWS();
    }

    toy_rmsnorm_f32(toy_tensor_data(x), toy_tensor_data(w), toy_tensor_data(result),
                    n, (float) eps);
    if (php_toy_tensor_wrap(return_value, result) != SUCCESS) {
        RETURN_THROWS();
    }
}

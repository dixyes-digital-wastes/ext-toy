#ifndef PHP_TOY_H
#define PHP_TOY_H

#include "php.h"

#include "attributes.h"

/*
 * Marks the C symbols that the extension exports
 * The build hides every other symbol
 */
#ifdef _WIN32
# define TOY_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
# define TOY_API __attribute__((visibility("default")))
#else
# define TOY_API
#endif

#include "tensor.h"

extern zend_module_entry toy_module_entry;
#define phpext_toy_ptr &toy_module_entry

#define PHP_TOY_VERSION "0.1.0"

extern zend_class_entry *tensor_ce;

extern const zend_function_entry module_functions[];

typedef struct php_tensor {
    tensor *tensor;
    zend_object std;
} php_tensor;

static inline php_tensor *php_tensor_from_obj(zend_object *object)
{
    return (php_tensor *) ((char *) object - offsetof(php_tensor, std));
}

static inline tensor *php_tensor_fetch(zval *value)
{
    return php_tensor_from_obj(Z_OBJ_P(value))->tensor;
}

/* Creates a Tensor object; on success the object owns tensor_value */
TOY_NODISCARD zend_result php_tensor_wrap(zval *return_value, tensor *tensor_value);

/*
 * Allocates a zero-filled tensor and wraps it in a Tensor object
 * Returns the tensor owned by return_value, or NULL after an exception
 */
TOY_NODISCARD tensor *php_tensor_alloc_object(zval *return_value, dtype dtype_value,
                                              uint8_t rank,
                                              const uint32_t dims[TOY_TENSOR_MAX_DIMS],
                                              uint32_t shape_arg_num);

PHP_FUNCTION(tensor_matmul_f32);
PHP_FUNCTION(tensor_matvec_f32);
PHP_FUNCTION(tensor_matmul_q8_0_f32);
PHP_FUNCTION(tensor_matvec_q8_0_f32);
PHP_FUNCTION(tensor_matmul_q8_0_q8_0);
PHP_FUNCTION(tensor_matvec_q8_0_q8_0);
PHP_FUNCTION(tensor_quantize_q8_0);
PHP_FUNCTION(tensor_add);
PHP_FUNCTION(tensor_mul);
PHP_FUNCTION(tensor_scale);
PHP_FUNCTION(tensor_silu);
PHP_FUNCTION(tensor_sigmoid);
PHP_FUNCTION(tensor_softplus);
PHP_FUNCTION(tensor_l2norm_rows);
PHP_FUNCTION(tensor_rmsnorm_rows);
PHP_FUNCTION(tensor_shift_rows);
PHP_FUNCTION(tensor_conv1d_silu);
PHP_FUNCTION(tensor_rmsnorm);
PHP_FUNCTION(tensor_rmsnorm_into);
PHP_FUNCTION(tensor_gdn_recurrence);

/* Introspection for tests and measurements */
PHP_FUNCTION(tensor_kernel_stats);
PHP_FUNCTION(tensor_kernel_enable);
PHP_FUNCTION(tensor_kernel_isa);

#endif /* PHP_TOY_H */

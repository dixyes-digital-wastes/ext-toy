#ifndef PHP_TENSOR_H
#define PHP_TENSOR_H

#include "php.h"

#include "toy_tensor.h"

extern zend_module_entry tensor_module_entry;
#define phpext_tensor_ptr &tensor_module_entry

#define PHP_TENSOR_VERSION "0.1.0"

extern zend_class_entry *toy_tensor_ce;

extern const zend_function_entry tensor_functions[];

typedef struct php_toy_tensor {
    toy_tensor *tensor;
    zend_object std;
} php_toy_tensor;

static inline php_toy_tensor *php_toy_tensor_from_obj(zend_object *object)
{
    return (php_toy_tensor *) ((char *) object - offsetof(php_toy_tensor, std));
}

static inline toy_tensor *php_toy_tensor_fetch(zval *value)
{
    return php_toy_tensor_from_obj(Z_OBJ_P(value))->tensor;
}

/* Creates a Tensor object; on success the object owns tensor */
[[nodiscard]] zend_result php_toy_tensor_wrap(zval *return_value, toy_tensor *tensor);

/*
 * Allocates a zero-filled tensor and wraps it in a Tensor object
 * Returns the tensor owned by return_value, or NULL after an exception
 */
[[nodiscard]] toy_tensor *php_toy_tensor_alloc_object(zval *return_value, toy_dtype dtype,
                                                      uint8_t rank,
                                                      const uint32_t dims[TOY_TENSOR_MAX_DIMS],
                                                      uint32_t shape_arg_num);

PHP_FUNCTION(toy_tensor_matmul_f32);
PHP_FUNCTION(toy_tensor_matvec_f32);
PHP_FUNCTION(toy_tensor_add);
PHP_FUNCTION(toy_tensor_scale);
PHP_FUNCTION(toy_tensor_rmsnorm);

#endif /* PHP_TENSOR_H */

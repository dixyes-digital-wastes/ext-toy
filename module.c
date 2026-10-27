#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"

#include "php_ini.h"
#include "php_toy.h"
#include "attributes.h"
#include "kernel.h"
#include "arena_config.h"

PHP_INI_BEGIN()
    PHP_INI_ENTRY(TOY_KERNEL_ARENA_INI_NAME, TOY_KERNEL_ARENA_INI_DEFAULT,
                  PHP_INI_SYSTEM, NULL)
PHP_INI_END()

enum {
    dtype_count = TOY_DTYPE_I8 + 1,
};

static zend_object_handlers php_tensor_handlers;
static zend_string *php_dtype_strings[dtype_count];

zend_class_entry *tensor_ce;

static zend_object *php_tensor_create_object(zend_class_entry *ce)
{
    php_tensor *object = zend_object_alloc(sizeof(php_tensor), ce);

    object->tensor = NULL;
    zend_object_std_init(&object->std, ce);
    object_properties_init(&object->std, ce);

    return &object->std;
}

static void php_tensor_free_object(zend_object *object)
{
    php_tensor *wrapper = php_tensor_from_obj(object);

    if (wrapper->tensor) {
        tensor_release(wrapper->tensor);
        wrapper->tensor = NULL;
    }
    zend_object_std_dtor(object);
}

zend_result php_tensor_wrap(zval *return_value, tensor *tensor_value)
{
    zend_string *dtype_name = tensor_value->dtype < dtype_count
        ? php_dtype_strings[tensor_value->dtype]
        : NULL;

    if (UNEXPECTED(!dtype_name)) {
        tensor_release(tensor_value);
        zend_throw_error(NULL, "Tensor has an invalid dtype");
        return FAILURE;
    }

    object_init_ex(return_value, tensor_ce);

    zend_object *object = Z_OBJ_P(return_value);
    php_tensor *wrapper = php_tensor_from_obj(object);

    wrapper->tensor = tensor_value;
    zend_update_property_str(tensor_ce, object, ZEND_STRL("dtype"), dtype_name);

    if (UNEXPECTED(EG(exception))) {
        wrapper->tensor = NULL;
        tensor_release(tensor_value);
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        return FAILURE;
    }
    return SUCCESS;
}

tensor *php_tensor_alloc_object(zval *return_value, dtype dtype_value, uint8_t rank,
                                const uint32_t dims[TOY_TENSOR_MAX_DIMS],
                                uint32_t shape_arg_num)
{
    tensor_status status;
    tensor *tensor_value = tensor_alloc(dtype_value, rank, dims, &status);

    if (!tensor_value) {
        if (status == TOY_ERROR_MEMORY) {
            zend_throw_error(NULL, "Tensor allocation failed");
        } else if (status == TOY_ERROR_OVERFLOW) {
            zend_argument_value_error(shape_arg_num, "is too large");
        } else {
            zend_argument_value_error(shape_arg_num, "is invalid");
        }
        return NULL;
    }
    if (php_tensor_wrap(return_value, tensor_value) != SUCCESS) {
        return NULL;
    }
    return php_tensor_fetch(return_value);
}

TOY_NODISCARD static bool php_tensor_parse_dtype(zend_string *name, dtype *out_dtype,
                                                 uint32_t arg_num)
{
    if (!dtype_from_name(ZSTR_VAL(name), ZSTR_LEN(name), out_dtype)) {
        zend_argument_value_error(arg_num, "must be a supported dtype, \"%s\" given",
                                  ZSTR_VAL(name));
        return false;
    }
    if (*out_dtype != TOY_DTYPE_F32 && *out_dtype != TOY_DTYPE_Q8_0) {
        zend_argument_value_error(arg_num,
                                  "must be \"f32\" or \"q8_0\", dtype \"%s\" is not implemented",
                                  ZSTR_VAL(name));
        return false;
    }
    return true;
}

TOY_NODISCARD static bool php_tensor_parse_shape(zval *value, uint8_t *rank,
                                                 uint32_t dims[TOY_TENSOR_MAX_DIMS],
                                                 uint32_t arg_num)
{
    HashTable *shape = Z_ARRVAL_P(value);
    uint32_t count = zend_hash_num_elements(shape);

    if (count < 1 || count > TOY_TENSOR_MAX_DIMS) {
        zend_argument_value_error(arg_num, "must contain 1 to %d dimensions",
                                  TOY_TENSOR_MAX_DIMS);
        return false;
    }
    if (!zend_array_is_list(shape)) {
        zend_argument_value_error(arg_num, "must be a list of positive integers");
        return false;
    }

    uint32_t index = 0;
    zval *dim;
    ZEND_HASH_FOREACH_VAL(shape, dim) {
        if (Z_TYPE_P(dim) != IS_LONG
            || Z_LVAL_P(dim) < 1
            || (zend_ulong) Z_LVAL_P(dim) > UINT32_MAX) {
            zend_argument_value_error(arg_num,
                                      "must contain a positive integer at index %u", index);
            return false;
        }
        dims[index++] = (uint32_t) Z_LVAL_P(dim);
    } ZEND_HASH_FOREACH_END();

    *rank = (uint8_t) count;
    return true;
}

TOY_NODISCARD static tensor *php_tensor_require_this(zval *this_value)
{
    tensor *tensor_value = php_tensor_fetch(this_value);

    if (!tensor_value) {
        zend_throw_error(NULL, "Toy\\Tensor object is not initialized");
    }
    return tensor_value;
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_tensor_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_zeros, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_ones, 0, 2, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_full, 0, 3, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_from_bytes, 0, 3, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_ndim, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_bytes, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_reshape, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_tensor_cast, 0, 1, Toy\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_to_bytes, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(Toy_Tensor, __construct);
PHP_METHOD(Toy_Tensor, zeros);
PHP_METHOD(Toy_Tensor, ones);
PHP_METHOD(Toy_Tensor, full);
PHP_METHOD(Toy_Tensor, fromBytes);
PHP_METHOD(Toy_Tensor, shape);
PHP_METHOD(Toy_Tensor, ndim);
PHP_METHOD(Toy_Tensor, bytes);
PHP_METHOD(Toy_Tensor, reshape);
PHP_METHOD(Toy_Tensor, cast);
PHP_METHOD(Toy_Tensor, toBytes);

static const zend_function_entry tensor_methods[] = {
    PHP_ME(Toy_Tensor, __construct, arginfo_tensor_construct, ZEND_ACC_PRIVATE)
    PHP_ME(Toy_Tensor, zeros, arginfo_tensor_zeros,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Toy_Tensor, ones, arginfo_tensor_ones,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Toy_Tensor, full, arginfo_tensor_full,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Toy_Tensor, fromBytes, arginfo_tensor_from_bytes,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Toy_Tensor, shape, arginfo_tensor_shape, ZEND_ACC_PUBLIC)
    PHP_ME(Toy_Tensor, ndim, arginfo_tensor_ndim, ZEND_ACC_PUBLIC)
    PHP_ME(Toy_Tensor, bytes, arginfo_tensor_bytes, ZEND_ACC_PUBLIC)
    PHP_ME(Toy_Tensor, reshape, arginfo_tensor_reshape, ZEND_ACC_PUBLIC)
    PHP_ME(Toy_Tensor, cast, arginfo_tensor_cast, ZEND_ACC_PUBLIC)
    PHP_ME(Toy_Tensor, toBytes, arginfo_tensor_to_bytes, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

PHP_METHOD(Toy_Tensor, __construct)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zend_throw_error(NULL, "Toy\\Tensor cannot be constructed directly, use a static factory");
    RETURN_THROWS();
}

PHP_METHOD(Toy_Tensor, zeros)
{
    zend_string *dtype_name;
    zval *shape;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(dtype_name)
        Z_PARAM_ARRAY(shape)
    ZEND_PARSE_PARAMETERS_END();

    dtype dtype_value;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    uint8_t rank;

    if (!php_tensor_parse_dtype(dtype_name, &dtype_value, 1)
        || !php_tensor_parse_shape(shape, &rank, dims, 2)) {
        RETURN_THROWS();
    }
    if (!php_tensor_alloc_object(return_value, dtype_value, rank, dims, 2)) {
        RETURN_THROWS();
    }
}

PHP_METHOD(Toy_Tensor, ones)
{
    zend_string *dtype_name;
    zval *shape;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(dtype_name)
        Z_PARAM_ARRAY(shape)
    ZEND_PARSE_PARAMETERS_END();

    dtype dtype_value;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    uint8_t rank;

    if (!php_tensor_parse_dtype(dtype_name, &dtype_value, 1)
        || !php_tensor_parse_shape(shape, &rank, dims, 2)) {
        RETURN_THROWS();
    }

    tensor *tensor_value = php_tensor_alloc_object(return_value, dtype_value, rank, dims, 2);
    if (!tensor_value) {
        RETURN_THROWS();
    }

    if (tensor_fill(tensor_value, 1.0f) != TOY_OK) {
        zend_throw_error(NULL, "Tensor fill failed");
    }
}

PHP_METHOD(Toy_Tensor, full)
{
    zend_string *dtype_name;
    zval *shape;
    double value;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(dtype_name)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();

    dtype dtype_value;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    uint8_t rank;

    if (!php_tensor_parse_dtype(dtype_name, &dtype_value, 1)
        || !php_tensor_parse_shape(shape, &rank, dims, 2)) {
        RETURN_THROWS();
    }

    tensor *tensor_value = php_tensor_alloc_object(return_value, dtype_value, rank, dims, 2);
    if (!tensor_value) {
        RETURN_THROWS();
    }

    if (tensor_fill(tensor_value, (float) value) != TOY_OK) {
        zend_throw_error(NULL, "Tensor fill failed");
    }
}

PHP_METHOD(Toy_Tensor, fromBytes)
{
    zend_string *dtype_name;
    zval *shape;
    zend_string *bytes;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(dtype_name)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_STR(bytes)
    ZEND_PARSE_PARAMETERS_END();

    dtype dtype_value;
    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    uint8_t rank;

    if (!php_tensor_parse_dtype(dtype_name, &dtype_value, 1)
        || !php_tensor_parse_shape(shape, &rank, dims, 2)) {
        RETURN_THROWS();
    }

    size_t expected;
    tensor_status status = tensor_shape_bytes(dtype_value, rank, dims, &expected);
    if (status == TOY_ERROR_OVERFLOW) {
        zend_argument_value_error(2, "is too large");
        RETURN_THROWS();
    }
    if (status != TOY_OK) {
        zend_argument_value_error(2, "is invalid");
        RETURN_THROWS();
    }
    if (ZSTR_LEN(bytes) != expected) {
        zend_argument_value_error(3, "must be exactly %zu bytes, %zu given",
                                  expected, ZSTR_LEN(bytes));
        RETURN_THROWS();
    }

    tensor *tensor_value = php_tensor_alloc_object(return_value, dtype_value, rank, dims, 2);
    if (!tensor_value) {
        RETURN_THROWS();
    }
    memcpy(tensor_data(tensor_value), ZSTR_VAL(bytes), expected);
}

PHP_METHOD(Toy_Tensor, shape)
{
    ZEND_PARSE_PARAMETERS_NONE();

    tensor *tensor_value = php_tensor_require_this(ZEND_THIS);
    if (!tensor_value) {
        RETURN_THROWS();
    }

    array_init_size(return_value, tensor_value->rank);
    for (uint8_t i = 0; i < tensor_value->rank; i++) {
        add_next_index_long(return_value, (zend_long) tensor_value->dims[i]);
    }
}

PHP_METHOD(Toy_Tensor, ndim)
{
    ZEND_PARSE_PARAMETERS_NONE();

    tensor *tensor_value = php_tensor_require_this(ZEND_THIS);
    if (!tensor_value) {
        RETURN_THROWS();
    }
    RETURN_LONG((zend_long) tensor_value->rank);
}

PHP_METHOD(Toy_Tensor, bytes)
{
    ZEND_PARSE_PARAMETERS_NONE();

    tensor *tensor_value = php_tensor_require_this(ZEND_THIS);
    if (!tensor_value) {
        RETURN_THROWS();
    }
    RETURN_LONG((zend_long) tensor_value->bytes);
}

PHP_METHOD(Toy_Tensor, reshape)
{
    zval *shape;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape)
    ZEND_PARSE_PARAMETERS_END();

    tensor *tensor_value = php_tensor_require_this(ZEND_THIS);
    if (!tensor_value) {
        RETURN_THROWS();
    }

    uint32_t dims[TOY_TENSOR_MAX_DIMS];
    uint8_t rank;
    if (!php_tensor_parse_shape(shape, &rank, dims, 1)) {
        RETURN_THROWS();
    }

    tensor_status status = tensor_reshape(tensor_value, rank, dims);
    if (status == TOY_ERROR_BYTES) {
        zend_argument_value_error(1, "must preserve the byte count of %zu bytes",
                                  tensor_value->bytes);
        RETURN_THROWS();
    }
    if (status == TOY_ERROR_OVERFLOW) {
        zend_argument_value_error(1, "is too large");
        RETURN_THROWS();
    }
    if (status != TOY_OK) {
        zend_argument_value_error(1, "is invalid for dtype \"%s\"",
                                  dtype_to_name((dtype) tensor_value->dtype));
        RETURN_THROWS();
    }

    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(Toy_Tensor, cast)
{
    zend_string *dtype_name;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(dtype_name)
    ZEND_PARSE_PARAMETERS_END();

    tensor *self = php_tensor_require_this(ZEND_THIS);
    if (!self) {
        RETURN_THROWS();
    }

    dtype dtype_value;
    if (!php_tensor_parse_dtype(dtype_name, &dtype_value, 1)) {
        RETURN_THROWS();
    }

    tensor_status status;
    tensor *copy = tensor_cast(self, dtype_value, &status);
    if (!copy) {
        if (status == TOY_ERROR_MEMORY) {
            zend_throw_error(NULL, "Tensor allocation failed");
        } else if (status == TOY_ERROR_SHAPE) {
            zend_argument_value_error(1,
                "cannot cast a f32 tensor whose last dimension is not a multiple of %d to q8_0",
                TOY_TENSOR_Q8_0_BLOCK_VALUES);
        } else {
            zend_argument_value_error(1, "cannot cast a %s tensor to \"%s\"",
                                      dtype_to_name((dtype) self->dtype),
                                      ZSTR_VAL(dtype_name));
        }
        RETURN_THROWS();
    }
    if (php_tensor_wrap(return_value, copy) != SUCCESS) {
        RETURN_THROWS();
    }
}

PHP_METHOD(Toy_Tensor, toBytes)
{
    ZEND_PARSE_PARAMETERS_NONE();

    tensor *tensor_value = php_tensor_require_this(ZEND_THIS);
    if (!tensor_value) {
        RETURN_THROWS();
    }
    RETURN_STRINGL((const char *) tensor_data(tensor_value), tensor_value->bytes);
}

PHP_MINIT_FUNCTION(toy)
{
    zend_class_entry ce;

    if (REGISTER_INI_ENTRIES() != SUCCESS) {
        return FAILURE;
    }

    INIT_CLASS_ENTRY(ce, "Toy\\Tensor", tensor_methods);
    tensor_ce = zend_register_internal_class_with_flags(&ce, NULL,
        ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES | ZEND_ACC_NOT_SERIALIZABLE);
    if (!tensor_ce) {
        UNREGISTER_INI_ENTRIES();
        return FAILURE;
    }

    tensor_ce->create_object = php_tensor_create_object;

    memcpy(&php_tensor_handlers, zend_get_std_object_handlers(),
           sizeof(zend_object_handlers));
    php_tensor_handlers.offset = offsetof(php_tensor, std);
    php_tensor_handlers.free_obj = php_tensor_free_object;
    php_tensor_handlers.clone_obj = NULL;
    tensor_ce->default_object_handlers = &php_tensor_handlers;

    zend_string *dtype_name = zend_string_init("dtype", sizeof("dtype") - 1, 1);
    zval default_value;
    ZVAL_UNDEF(&default_value);
    zend_declare_typed_property(tensor_ce, dtype_name, &default_value,
                                ZEND_ACC_PUBLIC | ZEND_ACC_READONLY, NULL,
                                (zend_type) ZEND_TYPE_INIT_MASK(MAY_BE_STRING));
    zend_string_release(dtype_name);

    php_dtype_strings[TOY_DTYPE_F32] = zend_string_init_interned("f32", sizeof("f32") - 1, 1);
    php_dtype_strings[TOY_DTYPE_Q8_0] = zend_string_init_interned("q8_0", sizeof("q8_0") - 1, 1);
    php_dtype_strings[TOY_DTYPE_F16] = zend_string_init_interned("f16", sizeof("f16") - 1, 1);
    php_dtype_strings[TOY_DTYPE_BF16] = zend_string_init_interned("bf16", sizeof("bf16") - 1, 1);
    php_dtype_strings[TOY_DTYPE_I8] = zend_string_init_interned("i8", sizeof("i8") - 1, 1);

    kernel_arena_startup();
    kernel_startup();

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(toy)
{
    kernel_shutdown();
    kernel_arena_shutdown();
    UNREGISTER_INI_ENTRIES();

    return SUCCESS;
}

PHP_RINIT_FUNCTION(toy)
{
/*
 * phpize builds pass ZEND_COMPILE_DL_EXT instead of COMPILE_DL_TOY, and
 * they do not include the generated config.h. Accept both markers so the
 * module exports its entry point and updates the TLS cache in either build.
 */
#if defined(ZTS) && (defined(COMPILE_DL_TOY) || defined(ZEND_COMPILE_DL_EXT))
    ZEND_TSRMLS_CACHE_UPDATE();
#endif

    return SUCCESS;
}

static void php_info_kernel_arena(void)
{
    arena *arena = kernel_arena();
    spec_cache *cache = kernel_arena_cache();

    if (arena == NULL || cache == NULL) {
        php_info_print_table_row(2, "kernel arena", "disabled");
        return;
    }

    arena_stats arena_stats;
    spec_cache_stats cache_stats;
    arena_get_stats(arena, &arena_stats);
    spec_cache_get_stats(cache, &cache_stats);

    char buffer[192];
    snprintf(buffer, sizeof(buffer), "%zu bytes, %llu of %llu pages used",
             arena_usable_bytes(arena),
             (unsigned long long) arena_stats.used_pages,
             (unsigned long long) arena_stats.total_pages);
    php_info_print_table_row(2, "kernel arena", buffer);

    snprintf(buffer, sizeof(buffer), "%zu entries, %zu used, %zu bytes",
             cache_stats.capacity, cache_stats.used, cache_stats.table_bytes);
    php_info_print_table_row(2, "kernel spec cache", buffer);

    snprintf(buffer, sizeof(buffer),
             "%llu lookups, %llu hits, %llu misses, %llu insert failures",
             (unsigned long long) cache_stats.lookups,
             (unsigned long long) cache_stats.hits,
             (unsigned long long) cache_stats.misses,
             (unsigned long long) cache_stats.insert_failures);
    php_info_print_table_row(2, "kernel spec cache stats", buffer);

    kernel_stats kernel;
    kernel_stats_read(&kernel);
    snprintf(buffer, sizeof(buffer), "%s, avx2 %s, f16c %s, generated %llu",
             kernel.enabled ? "enabled" : "disabled",
             kernel.avx2_available ? "available" : "unavailable",
             kernel.f16c_available ? "available" : "unavailable",
             (unsigned long long) kernel.generated);
    php_info_print_table_row(2, "kernels", buffer);
}

PHP_MINFO_FUNCTION(toy)
{
    php_info_print_table_start();
    php_info_print_table_row(2, "toy support", "enabled");
    php_info_print_table_row(2, "Tensor class", "Toy\\Tensor");
    php_info_print_table_row(2, "dtype", "f32, q8_0");
    php_info_kernel_arena();
    php_info_print_table_end();
    DISPLAY_INI_ENTRIES();
}

zend_module_entry toy_module_entry = {
    STANDARD_MODULE_HEADER,
    "toy",
    module_functions,
    PHP_MINIT(toy),
    PHP_MSHUTDOWN(toy),
    PHP_RINIT(toy),
    NULL,
    PHP_MINFO(toy),
    PHP_TOY_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#if defined(COMPILE_DL_TOY) || defined(ZEND_COMPILE_DL_EXT)
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(toy)
#endif

#include "toy_kernel_arena.h"

#include "php_ini.h"

static toy_arena *toy_kernel_arena_instance;
static toy_spec_cache *toy_kernel_arena_cache_instance;

/* Reads toy.kernel_arena_size. A result of 0 disables the arena. */
static size_t toy_kernel_arena_parse_size(void)
{
    zend_string *name = zend_string_init(TOY_KERNEL_ARENA_INI_NAME,
                                         sizeof(TOY_KERNEL_ARENA_INI_NAME) - 1, 0);
    zend_string *value = zend_ini_get_value(name);
    zend_string_release(name);

    if (value == NULL) {
        return toy_arena_default_config().total_bytes;
    }

    zend_string *error = NULL;
    zend_long parsed = zend_ini_parse_quantity(value, &error);
    if (error != NULL) {
        php_error_docref(NULL, E_WARNING,
                         TOY_KERNEL_ARENA_INI_NAME " value \"%s\" is not a byte count: %s",
                         ZSTR_VAL(value), ZSTR_VAL(error));
        zend_string_release(error);
        return toy_arena_default_config().total_bytes;
    }
    if (parsed <= 0) {
        return 0;
    }
    return (size_t) parsed;
}

void toy_kernel_arena_startup(void)
{
    size_t total_bytes = toy_kernel_arena_parse_size();
    if (total_bytes == 0) {
        return;
    }

    toy_arena_config config = toy_arena_default_config();
    config.total_bytes = total_bytes;
    config.segment_count = TOY_KERNEL_ARENA_SEGMENTS;

    toy_arena *arena = toy_arena_create(&config);
    if (arena == NULL) {
        php_error_docref(NULL, E_WARNING,
                         "cannot map %zu bytes for the kernel arena, "
                         "generated kernels are disabled",
                         total_bytes);
        return;
    }

    toy_spec_cache *cache = toy_spec_cache_create(arena, TOY_SPEC_CACHE_DEFAULT_CAPACITY);
    if (cache == NULL) {
        php_error_docref(NULL, E_WARNING,
                         "cannot allocate the kernel spec cache, "
                         "generated kernels are disabled");
        toy_arena_destroy(arena);
        return;
    }

    toy_kernel_arena_instance = arena;
    toy_kernel_arena_cache_instance = cache;
}

void toy_kernel_arena_shutdown(void)
{
    toy_spec_cache_destroy(toy_kernel_arena_cache_instance);
    toy_kernel_arena_cache_instance = NULL;
    toy_arena_destroy(toy_kernel_arena_instance);
    toy_kernel_arena_instance = NULL;
}

toy_arena *toy_kernel_arena(void)
{
    return toy_kernel_arena_instance;
}

toy_spec_cache *toy_kernel_arena_cache(void)
{
    return toy_kernel_arena_cache_instance;
}

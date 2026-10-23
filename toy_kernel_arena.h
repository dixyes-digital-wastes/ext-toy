#ifndef TOY_KERNEL_ARENA_H
#define TOY_KERNEL_ARENA_H

#include "php.h"

#include "toy_spec_cache.h"

#define TOY_KERNEL_ARENA_INI_NAME "toy.kernel_arena_size"
#define TOY_KERNEL_ARENA_INI_DEFAULT "32M"
#define TOY_KERNEL_ARENA_SEGMENTS 8

/*
 * Creates the process-wide arena and spec cache from toy.kernel_arena_size
 * A zero size disables them. A mapping failure disables them with a warning.
 * The interpreter path stays available in both cases.
 */
void toy_kernel_arena_startup(void);
void toy_kernel_arena_shutdown(void);

/* NULL when the arena is disabled or its creation failed */
toy_arena *toy_kernel_arena(void);
toy_spec_cache *toy_kernel_arena_cache(void);

#endif /* TOY_KERNEL_ARENA_H */

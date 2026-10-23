#ifndef SPEC_CACHE_H
#define SPEC_CACHE_H

#include "arena.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    TOY_SPEC_KEY_SIZE = 32,
    TOY_SPEC_CACHE_DEFAULT_CAPACITY = 16384,
};

static_assert(TOY_SPEC_KEY_SIZE == 32, "the spec key size is part of the cache ABI");
static_assert(TOY_SPEC_CACHE_DEFAULT_CAPACITY == 16384, "the default cache capacity is fixed");

typedef struct spec_cache_stats {
    size_t capacity;
    size_t used;
    size_t table_bytes;
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t inserts;
    uint64_t insert_failures;
    uint64_t fill_failures;
    uint64_t probe_total;
    uint64_t probe_max;
} spec_cache_stats;

/*
 * Fills the code pages of a new block
 * The pages are writable and not executable. capacity_bytes is the block size.
 */
typedef bool (*spec_fill)(const uint8_t *key, void *context, uint8_t *code,
                          size_t capacity_bytes);

typedef struct spec_cache spec_cache;

/* Returns NULL for an invalid capacity or an allocation failure */
[[nodiscard]] spec_cache *spec_cache_create(arena *arena, size_t capacity);
/* The arena owns the code blocks. This call releases the table only. */
void spec_cache_destroy(spec_cache *cache);

/*
 * Returns the cached block, or generates one when the key is absent
 * The fill callback runs outside the cache lock. A full table rejects the
 * insert and returns the new block to the arena.
 */
[[nodiscard]] bool spec_cache_get_or_create(spec_cache *cache,
                                            const uint8_t key[TOY_SPEC_KEY_SIZE],
                                            size_t bytes_hint, spec_fill fill,
                                            void *context, arena_block *block);

/*
 * Same lookup and generation as spec_cache_get_or_create
 * Returns NULL when the arena is full or the fill callback fails.
 */
[[nodiscard]] uint8_t *spec_cache_get_code(spec_cache *cache,
                                           const uint8_t key[TOY_SPEC_KEY_SIZE],
                                           size_t bytes_hint, spec_fill fill,
                                           void *context);

/* Looks up one entry. It never generates code. */
[[nodiscard]] bool spec_cache_lookup(spec_cache *cache,
                                     const uint8_t key[TOY_SPEC_KEY_SIZE],
                                     arena_block *block);

void spec_cache_get_stats(spec_cache *cache, spec_cache_stats *stats);

#endif /* SPEC_CACHE_H */

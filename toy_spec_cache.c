/*
 * Open-addressing cache from a 32-byte spec key to one kernel block
 * The table lives outside the arena. It never evicts an entry.
 */

#include "toy_spec_cache.h"
#include "toy_platform.h"

#include <stdlib.h>
#include <string.h>

typedef struct toy_spec_entry {
    uint64_t hash;
    uint8_t key[TOY_SPEC_KEY_SIZE];
    uint8_t *code;
    uint32_t pages;
    uint32_t filled;
} toy_spec_entry;

struct toy_spec_cache {
    toy_arena *arena;
    toy_spec_entry *entries;
    size_t capacity;
    size_t mask;
    size_t used;
    toy_platform_lock lock;
    toy_spec_cache_stats stats;
};

static bool toy_spec_cache_is_pow2(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static uint64_t toy_spec_cache_hash_key(const uint8_t key[TOY_SPEC_KEY_SIZE])
{
    uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
    for (size_t offset = 0; offset < TOY_SPEC_KEY_SIZE; offset += 8) {
        uint64_t word;
        memcpy(&word, key + offset, sizeof(word));
        hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
        hash *= UINT64_C(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
    }
    hash *= UINT64_C(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    return hash;
}

/*
 * Linear probing. On a hit, slot_out is the matching slot. On a miss, it is
 * the first empty slot or SIZE_MAX when the table is full.
 */
static bool toy_spec_cache_find_locked(toy_spec_cache *cache, const uint8_t *key,
                                       uint64_t hash, size_t *slot_out)
{
    size_t index = (size_t) (hash & cache->mask);
    uint64_t probes = 0;
    for (size_t i = 0; i < cache->capacity; i++) {
        toy_spec_entry *entry = &cache->entries[index];
        probes++;
        if (entry->filled == 0) {
            cache->stats.probe_total += probes;
            if (probes > cache->stats.probe_max) {
                cache->stats.probe_max = probes;
            }
            *slot_out = index;
            return false;
        }
        if (entry->hash == hash &&
            memcmp(entry->key, key, TOY_SPEC_KEY_SIZE) == 0) {
            cache->stats.probe_total += probes;
            if (probes > cache->stats.probe_max) {
                cache->stats.probe_max = probes;
            }
            *slot_out = index;
            return true;
        }
        index = (index + 1) & cache->mask;
    }
    cache->stats.probe_total += probes;
    if (probes > cache->stats.probe_max) {
        cache->stats.probe_max = probes;
    }
    *slot_out = SIZE_MAX;
    return false;
}

static toy_arena_block toy_spec_cache_entry_block(const toy_spec_entry *entry)
{
    toy_arena_block block = {
        .code = entry->code,
        .pages = entry->pages,
        .guard_pages = 0,
    };
    return block;
}

toy_spec_cache *toy_spec_cache_create(toy_arena *arena, size_t capacity)
{
    if (arena == NULL || capacity == 0) {
        return NULL;
    }
    size_t rounded = 4;
    while (rounded < capacity) {
        rounded <<= 1;
    }
    if (!toy_spec_cache_is_pow2(rounded)) {
        return NULL;
    }

    toy_spec_cache *cache = calloc(1, sizeof(*cache));
    if (cache == NULL) {
        return NULL;
    }
    cache->entries = calloc(rounded, sizeof(toy_spec_entry));
    if (cache->entries == NULL || !toy_platform_lock_init(&cache->lock)) {
        free(cache->entries);
        free(cache);
        return NULL;
    }
    cache->arena = arena;
    cache->capacity = rounded;
    cache->mask = rounded - 1;
    cache->stats.capacity = rounded;
    cache->stats.table_bytes = rounded * sizeof(toy_spec_entry);
    return cache;
}

void toy_spec_cache_destroy(toy_spec_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    toy_platform_lock_destroy(&cache->lock);
    free(cache->entries);
    free(cache);
}

bool toy_spec_cache_get_or_create(toy_spec_cache *cache,
                                  const uint8_t key[TOY_SPEC_KEY_SIZE],
                                  size_t bytes_hint, toy_spec_fill fill, void *context,
                                  toy_arena_block *block)
{
    if (cache == NULL || key == NULL || fill == NULL || block == NULL || bytes_hint == 0) {
        return false;
    }
    uint64_t hash = toy_spec_cache_hash_key(key);
    size_t slot = 0;

    toy_platform_lock_acquire(&cache->lock);
    bool found = toy_spec_cache_find_locked(cache, key, hash, &slot);
    cache->stats.lookups++;
    if (found) {
        cache->stats.hits++;
        *block = toy_spec_cache_entry_block(&cache->entries[slot]);
        toy_platform_lock_release(&cache->lock);
        return true;
    }
    cache->stats.misses++;
    toy_platform_lock_release(&cache->lock);

    toy_arena_block created = { 0 };
    if (!toy_arena_alloc_bytes(cache->arena, bytes_hint, &created)) {
        return false;
    }
    size_t capacity_bytes = (size_t) created.pages * toy_arena_page_size(cache->arena);
    if (!fill(key, context, created.code, capacity_bytes)) {
        toy_platform_lock_acquire(&cache->lock);
        cache->stats.fill_failures++;
        toy_platform_lock_release(&cache->lock);
        toy_arena_free(cache->arena, &created);
        return false;
    }
    if (!toy_arena_seal(cache->arena, &created)) {
        toy_arena_free(cache->arena, &created);
        return false;
    }

    /* Another thread can insert the same key while the fill callback runs. */
    toy_platform_lock_acquire(&cache->lock);
    found = toy_spec_cache_find_locked(cache, key, hash, &slot);
    if (found) {
        cache->stats.hits++;
        *block = toy_spec_cache_entry_block(&cache->entries[slot]);
        toy_platform_lock_release(&cache->lock);
        toy_arena_free(cache->arena, &created);
        return true;
    }
    if (cache->used == cache->capacity || slot == SIZE_MAX) {
        cache->stats.insert_failures++;
        toy_platform_lock_release(&cache->lock);
        toy_arena_free(cache->arena, &created);
        return false;
    }
    toy_spec_entry *entry = &cache->entries[slot];
    entry->hash = hash;
    memcpy(entry->key, key, TOY_SPEC_KEY_SIZE);
    entry->code = created.code;
    entry->pages = created.pages;
    entry->filled = 1;
    cache->used++;
    cache->stats.inserts++;
    *block = created;
    toy_platform_lock_release(&cache->lock);
    return true;
}

uint8_t *toy_spec_cache_get_code(toy_spec_cache *cache,
                                 const uint8_t key[TOY_SPEC_KEY_SIZE],
                                 size_t bytes_hint, toy_spec_fill fill, void *context)
{
    toy_arena_block block;
    if (!toy_spec_cache_get_or_create(cache, key, bytes_hint, fill, context, &block)) {
        return NULL;
    }
    return block.code;
}

bool toy_spec_cache_lookup(toy_spec_cache *cache, const uint8_t key[TOY_SPEC_KEY_SIZE],
                           toy_arena_block *block)
{
    if (cache == NULL || key == NULL || block == NULL) {
        return false;
    }
    uint64_t hash = toy_spec_cache_hash_key(key);
    size_t slot = 0;
    toy_platform_lock_acquire(&cache->lock);
    bool found = toy_spec_cache_find_locked(cache, key, hash, &slot);
    cache->stats.lookups++;
    if (found) {
        cache->stats.hits++;
        *block = toy_spec_cache_entry_block(&cache->entries[slot]);
    } else {
        cache->stats.misses++;
    }
    toy_platform_lock_release(&cache->lock);
    return found;
}

void toy_spec_cache_get_stats(toy_spec_cache *cache, toy_spec_cache_stats *stats)
{
    if (cache == NULL || stats == NULL) {
        return;
    }
    toy_platform_lock_acquire(&cache->lock);
    *stats = cache->stats;
    stats->used = cache->used;
    stats->capacity = cache->capacity;
    stats->table_bytes = cache->capacity * sizeof(toy_spec_entry);
    toy_platform_lock_release(&cache->lock);
}

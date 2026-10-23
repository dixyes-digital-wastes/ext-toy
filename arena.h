#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    TOY_ARENA_MAX_BLOCK_PAGES = 8,
    TOY_ARENA_TIER_LIMIT = 21,
};

static_assert(TOY_ARENA_MAX_BLOCK_PAGES == 8, "block size limit is part of the cache ABI");
static_assert(TOY_ARENA_TIER_LIMIT > 8, "the tier limit covers a 32 MiB arena");

typedef enum arena_kind {
    TOY_ARENA_TIERED = 0,
    TOY_ARENA_GUARDED = 1,
} arena_kind;

typedef struct arena_config {
    /* 0 selects the default of 32 MiB */
    size_t total_bytes;
    arena_kind kind;
    /* Must be a power of two. 0 selects one segment. */
    uint32_t segment_count;
    bool coalesce;
    /* Allow merges above the 8-page block limit. Debug only. */
    bool coalesce_unbounded;
} arena_config;

typedef struct arena_block {
    /* Page aligned. NULL means the block is not allocated. */
    uint8_t *code;
    uint32_t pages;
    /* Always 0 except in guarded mode, where the guard page follows code. */
    uint32_t guard_pages;
} arena_block;

typedef struct arena_stats {
    size_t page_size;
    uint64_t total_pages;
    uint64_t used_pages;
    uint64_t free_pages;
    uint64_t alloc_calls;
    uint64_t alloc_failures;
    uint64_t free_calls;
    uint64_t split_count;
    uint64_t merge_count;
    uint64_t mprotect_calls;
    uint64_t tier_allocs[TOY_ARENA_MAX_BLOCK_PAGES];
    uint64_t free_run_counts[TOY_ARENA_TIER_LIMIT];
    uint64_t largest_free_run_pages;
} arena_stats;

typedef struct arena arena;

arena_config arena_default_config(void);

/* Returns NULL when the size is invalid or the mapping fails */
[[nodiscard]] arena *arena_create(const arena_config *config);
void arena_destroy(arena *arena);

size_t arena_page_size(const arena *arena);
size_t arena_usable_bytes(const arena *arena);
arena_kind arena_kind_of(const arena *arena);

/* Allocates 1, 2, 4 or 8 pages. The block is writable and not executable. */
[[nodiscard]] bool arena_alloc(arena *arena, uint32_t pages, arena_block *block);
/* Rounds the byte count up to a block size. Fails above 8 pages. */
[[nodiscard]] bool arena_alloc_bytes(arena *arena, size_t bytes, arena_block *block);

/* The block may be writable or executable before this call. */
void arena_free(arena *arena, arena_block *block);

/* Seal switches the block to executable. Unseal switches it back to writable. */
[[nodiscard]] bool arena_seal(arena *arena, const arena_block *block);
[[nodiscard]] bool arena_unseal(arena *arena, const arena_block *block);

/* Reads one consistent snapshot. The caller may read it on any thread. */
void arena_get_stats(arena *arena, arena_stats *stats);

#endif /* ARENA_H */

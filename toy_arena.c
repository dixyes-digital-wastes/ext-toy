/*
 * Kernel code arena for generated SIMD kernels
 *
 * One anonymous mapping holds all code pages. Pages move between RW and RX
 * and never become writable and executable at the same time. Blocks come in
 * sizes of 1, 2, 4 and 8 pages, so one mprotect call seals one block.
 */

#include "toy_arena.h"

#include <assert.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum {
    TOY_ARENA_DEFAULT_BYTES = 32 << 20,
    /* 2^3 pages is the largest block */
    TOY_ARENA_MAX_ALLOC_TIER = 3,
    TOY_ARENA_PROT_RW = 0,
    TOY_ARENA_PROT_RX = 1,
    TOY_ARENA_PROT_NONE = 2,
};

static_assert((1u << TOY_ARENA_MAX_ALLOC_TIER) == TOY_ARENA_MAX_BLOCK_PAGES,
              "max allocation tier does not match TOY_ARENA_MAX_BLOCK_PAGES");

/* page_state marks a used page or the interior of a free run */
static const int8_t toy_arena_page_used = -1;
static const int8_t toy_arena_page_interior = -2;

typedef struct toy_arena_counters {
    uint64_t alloc_count;
    uint64_t free_count;
    uint64_t split_count;
    uint64_t merge_count;
    uint64_t tier_allocs[TOY_ARENA_MAX_BLOCK_PAGES];
} toy_arena_counters;

/* A segment is one contiguous power-of-two range of pages, split as a buddy. */
typedef struct toy_arena_segment {
    /* Keeps the hot counters of adjacent segments on different cache lines */
    alignas(64) toy_arena_counters counters;
    uint32_t first_page;
    uint32_t page_count;
    uint32_t max_tier;
    int32_t free_head[TOY_ARENA_TIER_LIMIT];
    int32_t *free_next;
    int32_t *free_prev;
    pthread_mutex_t lock;
} toy_arena_segment;

struct toy_arena {
    toy_arena_kind kind;
    bool coalesce;
    bool coalesce_unbounded;
    uint8_t *base;
    size_t map_bytes;
    size_t page_size;
    uint32_t page_count;
    uint32_t segment_count;
    uint32_t pages_per_segment;
    toy_arena_segment *segments;
    int8_t *page_state;
    uint8_t *page_prot;
    /* Protects the cold counters and the home slot counter */
    pthread_mutex_t stat_lock;
    uint64_t alloc_failures;
    uint64_t mprotect_calls;
    uint32_t next_home_slot;

    /* Guarded mode only */
    pthread_mutex_t guard_lock;
    uint8_t *guard_used;
    uint32_t guard_hint;
    uint64_t guarded_alloc_count;
    uint64_t guarded_free_count;
};

/* Remembers the home segment per thread and per arena */
static _Thread_local toy_arena *toy_arena_home_arena;
static _Thread_local uint32_t toy_arena_home_slot;

static bool toy_arena_is_pow2(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static uint32_t toy_arena_floor_pow2(uint32_t value)
{
    uint32_t result = 1;
    while (result <= value / 2) {
        result <<= 1;
    }
    return result;
}

static uint32_t toy_arena_tier_of_pow2(uint32_t pages)
{
    assert(toy_arena_is_pow2(pages));
    uint32_t tier = 0;
    while ((UINT32_C(1) << tier) != pages) {
        tier++;
    }
    return tier;
}

static bool toy_arena_block_pages_valid(uint32_t pages)
{
    return pages >= 1 && pages <= TOY_ARENA_MAX_BLOCK_PAGES && toy_arena_is_pow2(pages);
}

static void toy_arena_bump_mprotect(toy_arena *arena)
{
    pthread_mutex_lock(&arena->stat_lock);
    arena->mprotect_calls++;
    pthread_mutex_unlock(&arena->stat_lock);
}

static void toy_arena_segment_push_run(toy_arena *arena, toy_arena_segment *segment,
                                       uint32_t local, uint32_t tier)
{
    assert(tier <= segment->max_tier);
    uint32_t pages = UINT32_C(1) << tier;
    assert(local + pages <= segment->page_count);

    int8_t *state = arena->page_state + segment->first_page;
    for (uint32_t i = 0; i < pages; i++) {
        state[local + i] = toy_arena_page_interior;
    }
    state[local] = (int8_t) tier;

    int32_t head = segment->free_head[tier];
    segment->free_prev[local] = -1;
    segment->free_next[local] = head;
    if (head >= 0) {
        segment->free_prev[head] = (int32_t) local;
    }
    segment->free_head[tier] = (int32_t) local;
}

static uint32_t toy_arena_segment_pop_run(toy_arena *arena, toy_arena_segment *segment,
                                          uint32_t tier)
{
    int32_t run = segment->free_head[tier];
    if (run < 0) {
        return UINT32_MAX;
    }
    int32_t next = segment->free_next[run];
    segment->free_head[tier] = next;
    if (next >= 0) {
        segment->free_prev[next] = -1;
    }
    segment->free_next[run] = -1;
    segment->free_prev[run] = -1;

    int8_t *state = arena->page_state + segment->first_page;
    uint32_t pages = UINT32_C(1) << tier;
    for (uint32_t i = 0; i < pages; i++) {
        state[run + i] = toy_arena_page_used;
    }
    return (uint32_t) run;
}

static void toy_arena_segment_unlink_run(toy_arena_segment *segment, uint32_t local, uint32_t tier)
{
    int32_t prev = segment->free_prev[local];
    int32_t next = segment->free_next[local];
    if (prev >= 0) {
        segment->free_next[prev] = next;
    } else {
        segment->free_head[tier] = next;
    }
    if (next >= 0) {
        segment->free_prev[next] = prev;
    }
    segment->free_next[local] = -1;
    segment->free_prev[local] = -1;
}

static bool toy_arena_segment_alloc_run(toy_arena *arena, toy_arena_segment *segment,
                                        uint32_t wanted_tier, uint32_t *local_out)
{
    uint32_t tier = wanted_tier;
    while (tier <= segment->max_tier && segment->free_head[tier] < 0) {
        tier++;
    }
    if (tier > segment->max_tier) {
        return false;
    }

    uint32_t run = toy_arena_segment_pop_run(arena, segment, tier);
    assert(run != UINT32_MAX);
    while (tier > wanted_tier) {
        tier--;
        toy_arena_segment_push_run(arena, segment, run + (UINT32_C(1) << tier), tier);
        segment->counters.split_count++;
    }
    segment->counters.tier_allocs[wanted_tier]++;
    segment->counters.alloc_count++;
    *local_out = run;
    return true;
}

static void toy_arena_segment_free_run(toy_arena *arena, toy_arena_segment *segment,
                                       uint32_t local, uint32_t tier)
{
    /*
     * Merge up to the largest allocatable size only. A larger free run gives
     * no larger block, and every later split would rewrite more page states.
     */
    uint32_t merge_limit = segment->max_tier;
    if (!arena->coalesce_unbounded && merge_limit > TOY_ARENA_MAX_ALLOC_TIER) {
        merge_limit = TOY_ARENA_MAX_ALLOC_TIER;
    }
    while (arena->coalesce && tier < merge_limit) {
        uint32_t buddy = local ^ (UINT32_C(1) << tier);
        if (arena->page_state[segment->first_page + buddy] != (int8_t) tier) {
            break;
        }
        toy_arena_segment_unlink_run(segment, buddy, tier);
        if (buddy < local) {
            local = buddy;
        }
        tier++;
        segment->counters.merge_count++;
    }
    segment->counters.free_count++;
    toy_arena_segment_push_run(arena, segment, local, tier);
}

static bool toy_arena_guarded_scan_from(toy_arena *arena, uint32_t from, uint32_t span,
                                        uint32_t *start_out)
{
    uint32_t run = 0;
    for (uint32_t i = from; i < arena->page_count; i++) {
        if (arena->guard_used[i] != 0) {
            run = 0;
            continue;
        }
        run++;
        if (run == span) {
            *start_out = i + 1 - span;
            return true;
        }
    }
    return false;
}

/* A block is contiguous, so a free run cannot cross the end of the arena. */
static bool toy_arena_guarded_find_run(toy_arena *arena, uint32_t span, uint32_t *start_out)
{
    if (arena->guard_hint + span <= arena->page_count &&
        toy_arena_guarded_scan_from(arena, arena->guard_hint, span, start_out)) {
        return true;
    }
    if (arena->guard_hint != 0 && toy_arena_guarded_scan_from(arena, 0, span, start_out)) {
        return true;
    }
    return false;
}

static void toy_arena_guarded_mark(toy_arena *arena, uint32_t start, uint32_t span, uint8_t value)
{
    for (uint32_t i = 0; i < span; i++) {
        arena->guard_used[start + i] = value;
    }
}

static uint32_t toy_arena_pick_home_segment(toy_arena *arena)
{
    if (toy_arena_home_arena != arena) {
        pthread_mutex_lock(&arena->stat_lock);
        uint32_t slot = arena->next_home_slot++;
        pthread_mutex_unlock(&arena->stat_lock);
        toy_arena_home_arena = arena;
        toy_arena_home_slot = slot;
    }
    return toy_arena_home_slot % arena->segment_count;
}

toy_arena_config toy_arena_default_config(void)
{
    toy_arena_config config = {
        .total_bytes = (size_t) TOY_ARENA_DEFAULT_BYTES,
        .kind = TOY_ARENA_TIERED,
        .segment_count = 8,
        .coalesce = true,
        .coalesce_unbounded = false,
    };
    return config;
}

toy_arena *toy_arena_create(const toy_arena_config *config)
{
    toy_arena_config cfg = config != NULL ? *config : toy_arena_default_config();
    if (cfg.total_bytes == 0) {
        cfg.total_bytes = (size_t) TOY_ARENA_DEFAULT_BYTES;
    }

    long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        return NULL;
    }
    size_t page_size = (size_t) page_size_value;
    uint64_t requested_pages = (uint64_t) cfg.total_bytes / (uint64_t) page_size;
    if (requested_pages == 0 ||
        requested_pages > (UINT64_C(1) << (TOY_ARENA_TIER_LIMIT - 1))) {
        return NULL;
    }
    uint32_t page_count = (uint32_t) requested_pages;

    toy_arena_kind kind = cfg.kind;
    uint32_t segment_count = 0;
    uint32_t pages_per_segment = 0;
    uint32_t usable_pages;
    if (kind == TOY_ARENA_GUARDED) {
        usable_pages = page_count;
    } else {
        kind = TOY_ARENA_TIERED;
        usable_pages = toy_arena_floor_pow2(page_count);
        uint32_t asked_segments = cfg.segment_count != 0 ? cfg.segment_count : 1;
        if (!toy_arena_is_pow2(asked_segments)) {
            return NULL;
        }
        /*
         * Keep every segment able to hold one 8-page block. A smaller arena
         * uses fewer segments, so large blocks stay allocatable.
         */
        uint32_t segment_limit = usable_pages / TOY_ARENA_MAX_BLOCK_PAGES;
        if (segment_limit == 0) {
            segment_limit = 1;
        }
        segment_count = asked_segments < segment_limit ? asked_segments : segment_limit;
        pages_per_segment = usable_pages / segment_count;
        if (!toy_arena_is_pow2(pages_per_segment)) {
            return NULL;
        }
    }

    size_t map_bytes = (size_t) usable_pages * page_size;
    void *base = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    toy_arena *arena = calloc(1, sizeof(*arena));
    if (arena == NULL) {
        munmap(base, map_bytes);
        return NULL;
    }
    arena->kind = kind;
    arena->coalesce = kind == TOY_ARENA_TIERED ? cfg.coalesce : false;
    arena->coalesce_unbounded = cfg.coalesce_unbounded;
    arena->base = base;
    arena->map_bytes = map_bytes;
    arena->page_size = page_size;
    arena->page_count = usable_pages;
    arena->segment_count = segment_count;
    arena->pages_per_segment = pages_per_segment;

    arena->page_prot = calloc(usable_pages, sizeof(uint8_t));
    if (arena->page_prot == NULL) {
        munmap(base, map_bytes);
        free(arena);
        return NULL;
    }
    if (pthread_mutex_init(&arena->stat_lock, NULL) != 0) {
        free(arena->page_prot);
        munmap(base, map_bytes);
        free(arena);
        return NULL;
    }

    if (kind == TOY_ARENA_GUARDED) {
        arena->guard_used = calloc(usable_pages, sizeof(uint8_t));
        if (arena->guard_used == NULL ||
            pthread_mutex_init(&arena->guard_lock, NULL) != 0) {
            free(arena->guard_used);
            pthread_mutex_destroy(&arena->stat_lock);
            free(arena->page_prot);
            munmap(base, map_bytes);
            free(arena);
            return NULL;
        }
        return arena;
    }

    arena->page_state = malloc((size_t) usable_pages * sizeof(int8_t));
    arena->segments = calloc(segment_count, sizeof(toy_arena_segment));
    if (arena->page_state == NULL || arena->segments == NULL) {
        free(arena->segments);
        free(arena->page_state);
        pthread_mutex_destroy(&arena->stat_lock);
        free(arena->page_prot);
        munmap(base, map_bytes);
        free(arena);
        return NULL;
    }

    bool init_ok = true;
    uint32_t inited_segments = 0;
    for (uint32_t s = 0; s < segment_count && init_ok; s++) {
        toy_arena_segment *segment = &arena->segments[s];
        segment->first_page = s * pages_per_segment;
        segment->page_count = pages_per_segment;
        segment->max_tier = toy_arena_tier_of_pow2(pages_per_segment);
        for (size_t t = 0; t < TOY_ARENA_TIER_LIMIT; t++) {
            segment->free_head[t] = -1;
        }
        segment->free_next = malloc((size_t) pages_per_segment * sizeof(int32_t));
        segment->free_prev = malloc((size_t) pages_per_segment * sizeof(int32_t));
        if (segment->free_next == NULL || segment->free_prev == NULL ||
            pthread_mutex_init(&segment->lock, NULL) != 0) {
            init_ok = false;
            break;
        }
        for (uint32_t i = 0; i < pages_per_segment; i++) {
            segment->free_next[i] = -1;
            segment->free_prev[i] = -1;
        }
        inited_segments++;
        toy_arena_segment_push_run(arena, segment, 0, segment->max_tier);
    }

    if (!init_ok) {
        for (uint32_t s = 0; s < segment_count; s++) {
            free(arena->segments[s].free_next);
            free(arena->segments[s].free_prev);
        }
        for (uint32_t s = 0; s < inited_segments; s++) {
            pthread_mutex_destroy(&arena->segments[s].lock);
        }
        free(arena->segments);
        free(arena->page_state);
        pthread_mutex_destroy(&arena->stat_lock);
        free(arena->page_prot);
        munmap(base, map_bytes);
        free(arena);
        return NULL;
    }
    return arena;
}

void toy_arena_destroy(toy_arena *arena)
{
    if (arena == NULL) {
        return;
    }
    if (arena->kind == TOY_ARENA_GUARDED) {
        pthread_mutex_destroy(&arena->guard_lock);
        free(arena->guard_used);
    } else {
        for (uint32_t s = 0; s < arena->segment_count; s++) {
            toy_arena_segment *segment = &arena->segments[s];
            pthread_mutex_destroy(&segment->lock);
            free(segment->free_next);
            free(segment->free_prev);
        }
        free(arena->segments);
        free(arena->page_state);
    }
    pthread_mutex_destroy(&arena->stat_lock);
    free(arena->page_prot);
    munmap(arena->base, arena->map_bytes);
    free(arena);
}

size_t toy_arena_page_size(const toy_arena *arena)
{
    return arena != NULL ? arena->page_size : 0;
}

size_t toy_arena_usable_bytes(const toy_arena *arena)
{
    return arena != NULL ? (size_t) arena->page_count * arena->page_size : 0;
}

toy_arena_kind toy_arena_kind_of(const toy_arena *arena)
{
    return arena != NULL ? arena->kind : TOY_ARENA_TIERED;
}

/*
 * Switches contiguous pages to the target protection. The page table records
 * the current state, so an unchanged state costs no system call.
 */
static bool toy_arena_protect(toy_arena *arena, uint32_t first_page, uint32_t pages,
                              uint8_t prot)
{
    bool need = false;
    for (uint32_t i = 0; i < pages; i++) {
        if (arena->page_prot[first_page + i] != prot) {
            need = true;
            break;
        }
    }
    if (!need) {
        return true;
    }
    int native_prot = PROT_READ | PROT_WRITE;
    if (prot == TOY_ARENA_PROT_RX) {
        native_prot = PROT_READ | PROT_EXEC;
    } else if (prot == TOY_ARENA_PROT_NONE) {
        native_prot = PROT_NONE;
    }
    if (mprotect(arena->base + (size_t) first_page * arena->page_size,
                 (size_t) pages * arena->page_size, native_prot) != 0) {
        return false;
    }
    for (uint32_t i = 0; i < pages; i++) {
        arena->page_prot[first_page + i] = prot;
    }
    toy_arena_bump_mprotect(arena);
    return true;
}

static bool toy_arena_tiered_alloc(toy_arena *arena, uint32_t pages, toy_arena_block *block)
{
    uint32_t tier = toy_arena_tier_of_pow2(pages);
    uint32_t home = toy_arena_pick_home_segment(arena);
    uint32_t chosen = UINT32_MAX;
    uint32_t local = 0;

    pthread_mutex_lock(&arena->segments[home].lock);
    bool ok = toy_arena_segment_alloc_run(arena, &arena->segments[home], tier, &local);
    pthread_mutex_unlock(&arena->segments[home].lock);
    if (ok) {
        chosen = home;
    }

    /* The home segment is empty. Try the other segments without blocking. */
    for (uint32_t i = 0; !ok && i < arena->segment_count; i++) {
        if (i == home) {
            continue;
        }
        toy_arena_segment *segment = &arena->segments[i];
        if (pthread_mutex_trylock(&segment->lock) != 0) {
            continue;
        }
        ok = toy_arena_segment_alloc_run(arena, segment, tier, &local);
        pthread_mutex_unlock(&segment->lock);
        if (ok) {
            chosen = i;
        }
    }

    /* Retry the home segment once, in case another thread released a run */
    if (!ok) {
        pthread_mutex_lock(&arena->segments[home].lock);
        ok = toy_arena_segment_alloc_run(arena, &arena->segments[home], tier, &local);
        pthread_mutex_unlock(&arena->segments[home].lock);
        if (ok) {
            chosen = home;
        }
    }

    if (!ok) {
        return false;
    }
    size_t first_page = (size_t) arena->segments[chosen].first_page + local;
    block->code = arena->base + first_page * arena->page_size;
    block->pages = pages;
    block->guard_pages = 0;
    /* A reused block can still be RX. Switch it back to RW first. */
    if (!toy_arena_protect(arena, (uint32_t) first_page, pages, TOY_ARENA_PROT_RW)) {
        toy_arena_free(arena, block);
        return false;
    }
    return true;
}

static bool toy_arena_guarded_alloc(toy_arena *arena, uint32_t pages, toy_arena_block *block)
{
    uint32_t span = pages + 1;
    uint32_t start = 0;

    pthread_mutex_lock(&arena->guard_lock);
    bool ok = toy_arena_guarded_find_run(arena, span, &start);
    if (ok) {
        toy_arena_guarded_mark(arena, start, span, 1);
        arena->guard_hint = start + span;
        if (arena->guard_hint >= arena->page_count) {
            arena->guard_hint = 0;
        }
        arena->guarded_alloc_count++;
    }
    pthread_mutex_unlock(&arena->guard_lock);
    if (!ok) {
        return false;
    }

    uint8_t *code = arena->base + (size_t) start * arena->page_size;
    uint8_t *guard = code + (size_t) pages * arena->page_size;
    if (!toy_arena_protect(arena, start, pages, TOY_ARENA_PROT_RW)) {
        goto rollback;
    }
    if (mprotect(guard, arena->page_size, PROT_NONE) != 0) {
        goto rollback;
    }
    arena->page_prot[start + pages] = (uint8_t) TOY_ARENA_PROT_NONE;
    toy_arena_bump_mprotect(arena);
    block->code = code;
    block->pages = pages;
    block->guard_pages = 1;
    return true;

rollback:
    pthread_mutex_lock(&arena->guard_lock);
    toy_arena_guarded_mark(arena, start, span, 0);
    pthread_mutex_unlock(&arena->guard_lock);
    return false;
}

bool toy_arena_alloc(toy_arena *arena, uint32_t pages, toy_arena_block *block)
{
    if (arena == NULL || block == NULL || !toy_arena_block_pages_valid(pages)) {
        return false;
    }
    bool ok;
    if (arena->kind == TOY_ARENA_GUARDED) {
        ok = toy_arena_guarded_alloc(arena, pages, block);
    } else {
        ok = toy_arena_tiered_alloc(arena, pages, block);
    }
    if (!ok) {
        pthread_mutex_lock(&arena->stat_lock);
        arena->alloc_failures++;
        pthread_mutex_unlock(&arena->stat_lock);
        block->code = NULL;
        block->pages = 0;
        block->guard_pages = 0;
    }
    return ok;
}

bool toy_arena_alloc_bytes(toy_arena *arena, size_t bytes, toy_arena_block *block)
{
    if (arena == NULL || block == NULL || bytes == 0) {
        return false;
    }
    /* Avoid an overflow in the rounding addition for a huge request */
    size_t pages = bytes / arena->page_size;
    if (bytes % arena->page_size != 0) {
        pages++;
    }
    if (pages == 0 || pages > TOY_ARENA_MAX_BLOCK_PAGES) {
        return false;
    }
    /* Byte counts round up to one of 1, 2, 4 and 8 pages */
    size_t tier_pages = 1;
    while (tier_pages < pages) {
        tier_pages <<= 1;
    }
    return toy_arena_alloc(arena, (uint32_t) tier_pages, block);
}

static bool toy_arena_locate_block(const toy_arena *arena, const toy_arena_block *block,
                                   uint32_t *first_page_out, uint32_t *span_out)
{
    if (block->code == NULL || block->pages == 0 ||
        block->pages > TOY_ARENA_MAX_BLOCK_PAGES) {
        return false;
    }
    uintptr_t base = (uintptr_t) arena->base;
    uintptr_t address = (uintptr_t) block->code;
    if (address < base || address >= base + arena->map_bytes) {
        return false;
    }
    size_t offset = (size_t) (address - base);
    if (offset % arena->page_size != 0) {
        return false;
    }
    uint32_t first_page = (uint32_t) (offset / arena->page_size);
    uint32_t span = block->pages + (arena->kind == TOY_ARENA_GUARDED ? 1u : 0u);
    if (first_page + span > arena->page_count) {
        return false;
    }
    *first_page_out = first_page;
    *span_out = span;
    return true;
}

void toy_arena_free(toy_arena *arena, toy_arena_block *block)
{
    if (arena == NULL || block == NULL || block->code == NULL) {
        return;
    }
    uint32_t first_page = 0;
    uint32_t span = 0;
    if (!toy_arena_locate_block(arena, block, &first_page, &span)) {
        fprintf(stderr, "toy_arena_free: block outside arena\n");
        abort();
    }
    if (!toy_arena_block_pages_valid(block->pages)) {
        fprintf(stderr, "toy_arena_free: invalid page count\n");
        abort();
    }

    if (arena->kind == TOY_ARENA_GUARDED) {
        pthread_mutex_lock(&arena->guard_lock);
        if (arena->guard_used[first_page] == 0) {
            pthread_mutex_unlock(&arena->guard_lock);
            fprintf(stderr, "toy_arena_free: double free\n");
            abort();
        }
        toy_arena_guarded_mark(arena, first_page, span, 0);
        arena->guarded_free_count++;
        pthread_mutex_unlock(&arena->guard_lock);
    } else {
        uint32_t tier = toy_arena_tier_of_pow2(block->pages);
        uint32_t segment_index = first_page / arena->pages_per_segment;
        toy_arena_segment *segment = &arena->segments[segment_index];
        uint32_t local = first_page - segment->first_page;
        if ((local & ((UINT32_C(1) << tier) - 1)) != 0 ||
            arena->page_state[first_page] != toy_arena_page_used) {
            fprintf(stderr, "toy_arena_free: invalid or double free\n");
            abort();
        }
        pthread_mutex_lock(&segment->lock);
        toy_arena_segment_free_run(arena, segment, local, tier);
        pthread_mutex_unlock(&segment->lock);
    }
    block->code = NULL;
    block->pages = 0;
    block->guard_pages = 0;
}

static void toy_arena_flush_icache(const uint8_t *code, size_t bytes)
{
#if defined(__GNUC__) || defined(__clang__)
    __builtin___clear_cache((char *) code, (char *) code + bytes);
#else
    (void) code;
    (void) bytes;
#endif
}

bool toy_arena_seal(toy_arena *arena, const toy_arena_block *block)
{
    uint32_t first_page = 0;
    uint32_t span = 0;
    if (arena == NULL || block == NULL || !toy_arena_block_pages_valid(block->pages) ||
        !toy_arena_locate_block(arena, block, &first_page, &span)) {
        return false;
    }
    toy_arena_flush_icache(block->code, (size_t) block->pages * arena->page_size);
    return toy_arena_protect(arena, first_page, block->pages, TOY_ARENA_PROT_RX);
}

bool toy_arena_unseal(toy_arena *arena, const toy_arena_block *block)
{
    uint32_t first_page = 0;
    uint32_t span = 0;
    if (arena == NULL || block == NULL || !toy_arena_block_pages_valid(block->pages) ||
        !toy_arena_locate_block(arena, block, &first_page, &span)) {
        return false;
    }
    return toy_arena_protect(arena, first_page, block->pages, TOY_ARENA_PROT_RW);
}

void toy_arena_get_stats(toy_arena *arena, toy_arena_stats *stats)
{
    if (arena == NULL || stats == NULL) {
        return;
    }
    memset(stats, 0, sizeof(*stats));
    stats->page_size = arena->page_size;
    stats->total_pages = arena->page_count;

    if (arena->kind == TOY_ARENA_GUARDED) {
        pthread_mutex_lock(&arena->guard_lock);
        stats->alloc_calls = arena->guarded_alloc_count;
        stats->free_calls = arena->guarded_free_count;
        uint64_t used = 0;
        for (uint32_t i = 0; i < arena->page_count; i++) {
            used += arena->guard_used[i] != 0 ? 1 : 0;
        }
        stats->used_pages = used;
        stats->free_pages = arena->page_count - used;

        uint64_t best = 0;
        uint64_t current = 0;
        for (uint32_t i = 0; i < arena->page_count; i++) {
            if (arena->guard_used[i] == 0) {
                current++;
                if (current > best) {
                    best = current;
                }
            } else {
                current = 0;
            }
        }
        stats->largest_free_run_pages = best;
        pthread_mutex_unlock(&arena->guard_lock);

        pthread_mutex_lock(&arena->stat_lock);
        stats->alloc_failures = arena->alloc_failures;
        stats->mprotect_calls = arena->mprotect_calls;
        pthread_mutex_unlock(&arena->stat_lock);
        return;
    }

    uint64_t used = 0;
    uint64_t most_free_pages = 0;
    for (uint32_t s = 0; s < arena->segment_count; s++) {
        toy_arena_segment *segment = &arena->segments[s];
        pthread_mutex_lock(&segment->lock);
        stats->alloc_calls += segment->counters.alloc_count;
        stats->free_calls += segment->counters.free_count;
        stats->split_count += segment->counters.split_count;
        stats->merge_count += segment->counters.merge_count;
        for (size_t i = 0; i < TOY_ARENA_MAX_BLOCK_PAGES; i++) {
            stats->tier_allocs[i] += segment->counters.tier_allocs[i];
        }
        for (uint32_t i = 0; i < segment->page_count; i++) {
            used += arena->page_state[segment->first_page + i] == toy_arena_page_used ? 1 : 0;
        }
        for (uint32_t tier = 0; tier <= segment->max_tier; tier++) {
            int32_t node = segment->free_head[tier];
            if (node < 0) {
                continue;
            }
            uint64_t count = 0;
            while (node >= 0) {
                count++;
                node = segment->free_next[node];
            }
            stats->free_run_counts[tier] += count;
            uint64_t tier_pages = UINT64_C(1) << tier;
            if (tier_pages > most_free_pages) {
                most_free_pages = tier_pages;
            }
        }
        pthread_mutex_unlock(&segment->lock);
    }
    stats->used_pages = used;
    stats->free_pages = arena->page_count - used;
    stats->largest_free_run_pages = most_free_pages;

    pthread_mutex_lock(&arena->stat_lock);
    stats->alloc_failures = arena->alloc_failures;
    stats->mprotect_calls = arena->mprotect_calls;
    pthread_mutex_unlock(&arena->stat_lock);
}

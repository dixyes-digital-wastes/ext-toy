/*
 * Standalone C tests for the kernel arena and the spec cache
 * This file does not link against PHP
 *
 * Build and run on POSIX:
 *   cc -std=gnu23 -Wall -Wextra -Werror -O2 -pthread -I. \
 *      tests/c/arena_test.c toy_arena.c toy_spec_cache.c -o arena_test
 *   ./arena_test
 *
 * Build and run on Windows from a vcvars64 shell:
 *   cl /nologo /std:clatest /W4 /WX /O2 toy_arena.c toy_spec_cache.c \
 *      tests\c\arena_test.c /Fe:arena_test.exe
 *   arena_test.exe
 */

#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
# include <process.h>
# include <windows.h>
#else
# include <pthread.h>
# include <signal.h>
# include <sys/wait.h>
# include <unistd.h>
#endif

#include "toy_arena.h"
#include "toy_spec_cache.h"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
# define TOY_TEST_HAVE_MACHINE_CODE 1
#else
# define TOY_TEST_HAVE_MACHINE_CODE 0
#endif

static int failures;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            failures++; \
        } \
    } while (0)

#if TOY_TEST_HAVE_MACHINE_CODE
/* mov eax, imm32; ret */
static void toy_test_write_return_stub(uint8_t *code, int value)
{
    code[0] = (uint8_t) 0xb8;
    code[1] = (uint8_t) (value & 0xff);
    code[2] = (uint8_t) ((value >> 8) & 0xff);
    code[3] = (uint8_t) ((value >> 16) & 0xff);
    code[4] = (uint8_t) ((value >> 24) & 0xff);
    code[5] = (uint8_t) 0xc3;
}

static int toy_test_call_code(const uint8_t *code)
{
    int (*function)(void);
    static_assert(sizeof(function) == sizeof(code), "a code pointer fits in a function pointer");
    memcpy(&function, &code, sizeof(function));
    return function();
}
#endif

typedef enum toy_test_protection {
    TOY_TEST_PROT_RW,
    TOY_TEST_PROT_RX,
    TOY_TEST_PROT_NONE,
} toy_test_protection;

/* Checks the page protection. Rejects a writable executable page. */
static bool toy_test_check_protection(const uint8_t *code, toy_test_protection expected)
{
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION info;
    if (VirtualQuery(code, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT) {
        return false;
    }
    DWORD expected_prot = PAGE_NOACCESS;
    if (expected == TOY_TEST_PROT_RW) {
        expected_prot = PAGE_READWRITE;
    } else if (expected == TOY_TEST_PROT_RX) {
        expected_prot = PAGE_EXECUTE_READ;
    }
    if (info.Protect != expected_prot) {
        return false;
    }
    return info.Protect != PAGE_EXECUTE_READWRITE && info.Protect != PAGE_EXECUTE_WRITECOPY;
#elif defined(__linux__)
    const char *expected_perms = "rw-p";
    if (expected == TOY_TEST_PROT_RX) {
        expected_perms = "r-xp";
    } else if (expected == TOY_TEST_PROT_NONE) {
        expected_perms = "---p";
    }
    FILE *maps = fopen("/proc/self/maps", "r");
    if (maps == NULL) {
        return false;
    }
    char line[512];
    unsigned long start = 0;
    unsigned long end = 0;
    char found[5] = { 0 };
    bool ok = false;
    while (fgets(line, sizeof(line), maps) != NULL) {
        if (sscanf(line, "%lx-%lx %4s", &start, &end, found) != 3) {
            continue;
        }
        if ((uintptr_t) code >= start && (uintptr_t) code < end) {
            bool no_write_execute = !(strchr(found, 'w') != NULL && strchr(found, 'x') != NULL);
            ok = strcmp(found, expected_perms) == 0 && no_write_execute;
            break;
        }
    }
    fclose(maps);
    return ok;
#else
    (void) code;
    (void) expected;
    return false;
#endif
}

#ifndef _WIN32
typedef int (*ToyTestChildAction)(void *);

/* Runs the action in a child. Returns the exit code or the negative signal. */
static int toy_test_run_child(ToyTestChildAction action, void *argument)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        _exit(action(argument));
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return -WTERMSIG(status);
    }
    return -1;
}
#endif

typedef struct toy_test_write_probe {
    uint8_t *code;
    size_t offset;
    uint8_t value;
} toy_test_write_probe;

#ifndef _WIN32
static int toy_test_child_write(void *argument)
{
    toy_test_write_probe *probe = argument;
    probe->code[probe->offset] = probe->value;
    return 0;
}

static int toy_test_child_execute(void *argument)
{
#if TOY_TEST_HAVE_MACHINE_CODE
    return toy_test_call_code(argument) & 0xff;
#else
    (void) argument;
    return 0;
#endif
}
#endif

/* Returns true when the write faults. The page must stay unchanged. */
static bool toy_test_write_faults(uint8_t *code, size_t offset, uint8_t value)
{
#ifdef _WIN32
    volatile bool faulted = false;
    __try {
        code[offset] = value;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        faulted = true;
    }
    return faulted;
#else
    toy_test_write_probe probe = { .code = code, .offset = offset, .value = value };
    int result = toy_test_run_child(toy_test_child_write, &probe);
    return result == -SIGSEGV || result == -SIGBUS;
#endif
}

/* Returns true when execution faults, as it does on a writable page. */
static bool toy_test_execute_faults(const uint8_t *code)
{
#ifdef _WIN32
    volatile bool faulted = false;
    volatile int result = 0;
    __try {
        int (*function)(void);
        memcpy(&function, &code, sizeof(function));
        result = function();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        faulted = true;
    }
    (void) result;
    return faulted;
#else
    int result = toy_test_run_child(toy_test_child_execute, (void *) code);
    return result == -SIGSEGV || result == -SIGBUS;
#endif
}

static uint64_t toy_test_next_random(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    return value;
}

static toy_arena *toy_test_create_arena(size_t bytes, toy_arena_kind kind, uint32_t segments)
{
    toy_arena_config config = {
        .total_bytes = bytes,
        .kind = kind,
        .segment_count = segments,
        .coalesce = true,
        .coalesce_unbounded = false,
    };
    return toy_arena_create(&config);
}

static void test_create(void)
{
    printf("test_create\n");

    toy_arena_config config = toy_arena_default_config();
    CHECK(config.total_bytes == ((size_t) 32 << 20));
    CHECK(config.kind == TOY_ARENA_TIERED);
    CHECK(config.segment_count == 8);
    CHECK(config.coalesce);
    CHECK(!config.coalesce_unbounded);

    toy_arena *arena = toy_arena_create(NULL);
    CHECK(arena != NULL);
    if (arena != NULL) {
        CHECK(toy_arena_page_size(arena) > 0);
        CHECK(toy_arena_usable_bytes(arena) == ((size_t) 32 << 20));
        CHECK(toy_arena_kind_of(arena) == TOY_ARENA_TIERED);
        toy_arena_destroy(arena);
    }

    arena = toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 4);
    CHECK(arena != NULL);
    if (arena != NULL) {
        CHECK(toy_arena_usable_bytes(arena) == ((size_t) 1 << 20));
        toy_arena_destroy(arena);
    }

    CHECK(toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 3) == NULL);

    /* A small arena uses fewer segments but keeps 8-page blocks available */
    arena = toy_test_create_arena((size_t) 64 << 10, TOY_ARENA_TIERED, 8);
    CHECK(arena != NULL);
    if (arena != NULL) {
        CHECK(toy_arena_usable_bytes(arena) == ((size_t) 64 << 10));
        toy_arena_block block = { 0 };
        CHECK(toy_arena_alloc(arena, 8, &block));
        toy_arena_free(arena, &block);
        toy_arena_destroy(arena);
    }

    /* A zero byte count selects the default size */
    arena = toy_test_create_arena(0, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena != NULL) {
        CHECK(toy_arena_usable_bytes(arena) == ((size_t) 32 << 20));
        toy_arena_destroy(arena);
    }
    CHECK(toy_test_create_arena(512, TOY_ARENA_TIERED, 1) == NULL);
    toy_arena_destroy(NULL);
}

static void test_tiers(void)
{
    printf("test_tiers\n");
    toy_arena *arena = toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }
    size_t page_size = toy_arena_page_size(arena);

    for (uint32_t pages = 1; pages <= 8; pages <<= 1) {
        toy_arena_block block = { 0 };
        CHECK(toy_arena_alloc(arena, pages, &block));
        CHECK(block.pages == pages);
        CHECK(block.guard_pages == 0);
        CHECK((uintptr_t) block.code % page_size == 0);
        CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RW));
        for (size_t page = 0; page < pages; page++) {
            block.code[page * page_size] = (uint8_t) (0x10 + page);
            block.code[page * page_size + page_size - 1] = 0x7f;
        }
        CHECK(toy_arena_seal(arena, &block));
        CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RX));
        CHECK(block.code[0] == 0x10);
        CHECK(toy_arena_unseal(arena, &block));
        CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RW));
#if TOY_TEST_HAVE_MACHINE_CODE
        toy_test_write_return_stub(block.code, (int) pages);
        CHECK(toy_arena_seal(arena, &block));
        CHECK(toy_test_call_code(block.code) == (int) pages);
#endif
        toy_arena_free(arena, &block);
        CHECK(block.code == NULL);
        CHECK(block.pages == 0);
    }

    toy_arena_block invalid = { 0 };
    CHECK(!toy_arena_alloc(arena, 0, &invalid));
    CHECK(!toy_arena_alloc(arena, 3, &invalid));
    CHECK(!toy_arena_alloc(arena, 16, &invalid));
    CHECK(invalid.code == NULL);

    toy_arena_block bytes_block = { 0 };
    CHECK(toy_arena_alloc_bytes(arena, 1, &bytes_block));
    CHECK(bytes_block.pages == 1);
    toy_arena_free(arena, &bytes_block);
    CHECK(toy_arena_alloc_bytes(arena, page_size + 1, &bytes_block));
    CHECK(bytes_block.pages == 2);
    toy_arena_free(arena, &bytes_block);
    CHECK(toy_arena_alloc_bytes(arena, page_size * 2 + 1, &bytes_block));
    CHECK(bytes_block.pages == 4);
    toy_arena_free(arena, &bytes_block);
    CHECK(toy_arena_alloc_bytes(arena, page_size * 8, &bytes_block));
    CHECK(bytes_block.pages == 8);
    toy_arena_free(arena, &bytes_block);
    CHECK(!toy_arena_alloc_bytes(arena, page_size * 8 + 1, &bytes_block));
    CHECK(!toy_arena_alloc_bytes(arena, 0, &bytes_block));

    toy_arena_stats stats;
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    CHECK(stats.free_pages == stats.total_pages);
    CHECK(stats.alloc_failures == 0);
    CHECK(stats.alloc_calls > 0);
    CHECK(stats.free_calls > 0);
    CHECK(stats.mprotect_calls > 0);
    toy_arena_destroy(arena);
}

static void test_wx(void)
{
    printf("test_wx\n");
#if !TOY_TEST_HAVE_MACHINE_CODE
    printf("test_wx: skipped, this target has no test machine code\n");
    return;
#else
    toy_arena *arena = toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }
    toy_arena_block block = { 0 };
    CHECK(toy_arena_alloc(arena, 2, &block));
    toy_test_write_return_stub(block.code, 42);
    CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RW));
    /* A writable page rejects execution */
    CHECK(toy_test_execute_faults(block.code));

    CHECK(toy_arena_seal(arena, &block));
    CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RX));
    CHECK(toy_test_call_code(block.code) == 42);
    CHECK(!toy_test_execute_faults(block.code));

    /* A sealed page rejects a write */
    CHECK(toy_test_write_faults(block.code, 0, 0x90));

    CHECK(toy_arena_unseal(arena, &block));
    CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RW));
    CHECK(toy_test_execute_faults(block.code));

    CHECK(toy_arena_seal(arena, &block));
    CHECK(toy_test_call_code(block.code) == 42);
    toy_arena_free(arena, &block);
    toy_arena_destroy(arena);
#endif
}

static void test_reuse(void)
{
    printf("test_reuse\n");
    toy_arena *arena = toy_test_create_arena((size_t) 64 << 10, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }

    toy_arena_block first = { 0 };
    CHECK(toy_arena_alloc(arena, 1, &first));
    uint8_t *address = first.code;
    first.code[0] = 0xaa;
    CHECK(toy_arena_seal(arena, &first));
    toy_arena_free(arena, &first);

    /* A freed block keeps its protection until the next allocation */
    toy_arena_block freed = { .code = address, .pages = 1, .guard_pages = 0 };
    CHECK(toy_test_check_protection(freed.code, TOY_TEST_PROT_RX));

    toy_arena_block second = { 0 };
    CHECK(toy_arena_alloc(arena, 1, &second));
    CHECK(second.code == address);
    CHECK(toy_test_check_protection(second.code, TOY_TEST_PROT_RW));
    second.code[0] = 0xbb;
    CHECK(second.code[0] == 0xbb);
    CHECK(toy_arena_seal(arena, &second));
    toy_arena_free(arena, &second);

    toy_arena_stats stats;
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    CHECK(stats.alloc_calls == 2);
    CHECK(stats.free_calls == 2);
    toy_arena_destroy(arena);
}

static void test_capacity(void)
{
    printf("test_capacity\n");
    toy_arena *arena = toy_test_create_arena((size_t) 64 << 10, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }
    size_t page_size = toy_arena_page_size(arena);
    uint64_t total_pages = (uint64_t) ((size_t) 64 << 10) / page_size;

    enum { block_count = 16 };
    CHECK(total_pages == block_count);

    toy_arena_block blocks[block_count];
    for (int i = 0; i < block_count; i++) {
        blocks[i] = (toy_arena_block) { 0 };
        CHECK(toy_arena_alloc(arena, 1, &blocks[i]));
        CHECK(blocks[i].code != NULL);
    }

    toy_arena_block exhausted = { 0 };
    CHECK(!toy_arena_alloc(arena, 1, &exhausted));
    CHECK(exhausted.code == NULL);

    toy_arena_stats stats;
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == total_pages);
    CHECK(stats.free_pages == 0);
    CHECK(stats.alloc_failures == 1);

    for (int i = 0; i < block_count; i++) {
        toy_arena_free(arena, &blocks[i]);
    }
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    CHECK(stats.free_pages == total_pages);

    /* Capacity returns after the release */
    for (int i = 0; i < block_count; i++) {
        CHECK(toy_arena_alloc(arena, 1, &blocks[i]));
    }
    for (int i = 0; i < block_count; i++) {
        toy_arena_free(arena, &blocks[i]);
    }
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    toy_arena_destroy(arena);
}

static void test_guarded(void)
{
    printf("test_guarded\n");
    toy_arena *arena = toy_test_create_arena((size_t) 128 << 10, TOY_ARENA_GUARDED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }
    size_t page_size = toy_arena_page_size(arena);

    toy_arena_block block = { 0 };
    CHECK(toy_arena_alloc(arena, 1, &block));
    CHECK(block.guard_pages == 1);
    CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RW));
    CHECK(toy_test_check_protection(block.code + page_size, TOY_TEST_PROT_NONE));
    block.code[0] = 0x55;
    CHECK(block.code[0] == 0x55);

    /* The guard page catches a write beyond the block */
    CHECK(toy_test_write_faults(block.code, page_size, 0x11));

    CHECK(toy_arena_seal(arena, &block));
    CHECK(toy_test_check_protection(block.code, TOY_TEST_PROT_RX));
    CHECK(toy_test_check_protection(block.code + page_size, TOY_TEST_PROT_NONE));

    toy_arena_stats stats;
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 2);

    toy_arena_free(arena, &block);
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    CHECK(stats.alloc_calls == 1);
    CHECK(stats.free_calls == 1);

    /* 128 KiB holds 16 one-page blocks with one guard page each */
    toy_arena_block blocks[16];
    int allocated = 0;
    for (int i = 0; i < 16; i++) {
        blocks[i] = (toy_arena_block) { 0 };
        CHECK(toy_arena_alloc(arena, 1, &blocks[i]));
        if (blocks[i].code != NULL) {
            allocated++;
        }
    }
    CHECK(allocated == 16);
    toy_arena_block exhausted = { 0 };
    CHECK(!toy_arena_alloc(arena, 1, &exhausted));
    for (int i = 0; i < 16; i++) {
        toy_arena_free(arena, &blocks[i]);
    }
    toy_arena_destroy(arena);
}

static int fill_calls;

static bool toy_test_fill(const uint8_t *key, void *context, uint8_t *code,
                          size_t capacity_bytes)
{
    (void) context;
    fill_calls++;
    if (capacity_bytes < 16) {
        return false;
    }
#if TOY_TEST_HAVE_MACHINE_CODE
    toy_test_write_return_stub(code, (int) key[0]);
#else
    code[0] = key[0];
#endif
    return true;
}

static bool toy_test_fill_fails(const uint8_t *key, void *context, uint8_t *code,
                                size_t capacity_bytes)
{
    (void) key;
    (void) context;
    (void) code;
    (void) capacity_bytes;
    fill_calls++;
    return false;
}

static void toy_test_make_key(uint8_t key[TOY_SPEC_KEY_SIZE], uint64_t id)
{
    memset(key, 0, TOY_SPEC_KEY_SIZE);
    memcpy(key, &id, sizeof(id));
}

static void test_cache(void)
{
    printf("test_cache\n");
    toy_arena *arena = toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 1);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }
    toy_spec_cache *cache = toy_spec_cache_create(arena, 64);
    CHECK(cache != NULL);
    if (cache == NULL) {
        toy_arena_destroy(arena);
        return;
    }

    uint8_t key[TOY_SPEC_KEY_SIZE];
    toy_spec_cache_stats cache_stats;
    toy_arena_stats arena_stats;

    toy_test_make_key(key, 0);
    toy_arena_block first = { 0 };
    fill_calls = 0;
    CHECK(toy_spec_cache_get_or_create(cache, key, 64, toy_test_fill, NULL, &first));
    CHECK(fill_calls == 1);
    CHECK(first.pages == 1);
    CHECK(first.guard_pages == 0);
    CHECK(toy_test_check_protection(first.code, TOY_TEST_PROT_RX));
#if TOY_TEST_HAVE_MACHINE_CODE
    CHECK(toy_test_call_code(first.code) == 0);
#endif

    /* A hit returns the same block and does not call the fill callback */
    toy_arena_block again = { 0 };
    CHECK(toy_spec_cache_get_or_create(cache, key, 64, toy_test_fill, NULL, &again));
    CHECK(fill_calls == 1);
    CHECK(again.code == first.code);

    toy_arena_block looked_up = { 0 };
    CHECK(toy_spec_cache_lookup(cache, key, &looked_up));
    CHECK(looked_up.code == first.code);

    /* The code pointer form returns the sealed block and never refills */
    CHECK(toy_spec_cache_get_code(cache, key, 64, toy_test_fill, NULL) == first.code);
    CHECK(fill_calls == 1);
    CHECK(toy_spec_cache_get_code(NULL, key, 64, toy_test_fill, NULL) == NULL);

    /* A key that differs in one byte is a different spec */
    uint8_t other_key[TOY_SPEC_KEY_SIZE];
    toy_test_make_key(other_key, 1);
    CHECK(!toy_spec_cache_lookup(cache, other_key, &looked_up));

    /* A failed fill returns no block and returns the pages to the arena */
    toy_arena_get_stats(arena, &arena_stats);
    uint64_t used_before_failure = arena_stats.used_pages;
    uint8_t failing_key[TOY_SPEC_KEY_SIZE];
    toy_test_make_key(failing_key, 200);
    fill_calls = 0;
    CHECK(!toy_spec_cache_get_or_create(cache, failing_key, 64, toy_test_fill_fails, NULL,
                                        &looked_up));
    CHECK(fill_calls == 1);
    toy_arena_get_stats(arena, &arena_stats);
    CHECK(arena_stats.used_pages == used_before_failure);
    toy_spec_cache_get_stats(cache, &cache_stats);
    CHECK(cache_stats.fill_failures == 1);

    /* An oversized kernel fails before the fill callback runs */
    toy_test_make_key(failing_key, 201);
    fill_calls = 0;
    CHECK(!toy_spec_cache_get_or_create(cache, failing_key, 1 << 16, toy_test_fill, NULL,
                                        &looked_up));
    CHECK(fill_calls == 0);
    CHECK(toy_spec_cache_get_code(cache, failing_key, 1 << 16, toy_test_fill, NULL) == NULL);
    CHECK(fill_calls == 0);

    /* Fill the table up to its 64 entries */
    for (uint64_t id = 1; id < 64; id++) {
        toy_test_make_key(key, id);
        toy_arena_block block = { 0 };
        CHECK(toy_spec_cache_get_or_create(cache, key, 64, toy_test_fill, NULL, &block));
    }
    toy_spec_cache_get_stats(cache, &cache_stats);
    CHECK(cache_stats.capacity == 64);
    CHECK(cache_stats.used == 64);
    CHECK(cache_stats.insert_failures == 0);

    /* The full table rejects the next insert and frees its block */
    toy_arena_get_stats(arena, &arena_stats);
    used_before_failure = arena_stats.used_pages;
    toy_test_make_key(key, 64);
    toy_arena_block rejected = { 0 };
    CHECK(!toy_spec_cache_get_or_create(cache, key, 64, toy_test_fill, NULL, &rejected));
    toy_spec_cache_get_stats(cache, &cache_stats);
    CHECK(cache_stats.insert_failures == 1);
    CHECK(cache_stats.used == 64);
    toy_arena_get_stats(arena, &arena_stats);
    CHECK(arena_stats.used_pages == used_before_failure);

    /* Every stored kernel stays executable and returns its own value */
    toy_test_make_key(key, 7);
    toy_arena_block seven = { 0 };
    CHECK(toy_spec_cache_lookup(cache, key, &seven));
#if TOY_TEST_HAVE_MACHINE_CODE
    CHECK(toy_test_call_code(seven.code) == 7);
#endif
    toy_test_make_key(key, 100);
    CHECK(!toy_spec_cache_lookup(cache, key, &seven));

    toy_spec_cache_destroy(cache);
    toy_arena_get_stats(arena, &arena_stats);
    CHECK(arena_stats.used_pages == 64);
    toy_arena_destroy(arena);

    CHECK(toy_spec_cache_create(NULL, 64) == NULL);
    toy_spec_cache_destroy(NULL);
}

typedef struct toy_test_thread_state {
    toy_arena *arena;
    uint64_t seed;
    int error;
} toy_test_thread_state;

static void *toy_test_thread_main(void *argument)
{
    toy_test_thread_state *state = argument;
    uint64_t random = state->seed;
    size_t page_size = toy_arena_page_size(state->arena);

    for (int i = 0; i < 100; i++) {
        random = toy_test_next_random(&random);
        uint32_t pages = UINT32_C(1) << (random % 4);
        toy_arena_block block = { 0 };
        if (!toy_arena_alloc(state->arena, pages, &block)) {
            state->error = 1;
            return NULL;
        }
        uint8_t pattern = (uint8_t) (random & 0xff);
        memset(block.code, pattern, (size_t) pages * page_size);
        if (!toy_arena_seal(state->arena, &block)) {
            state->error = 2;
            toy_arena_free(state->arena, &block);
            return NULL;
        }
        if (block.code[0] != pattern) {
            state->error = 3;
        }
        toy_arena_free(state->arena, &block);
    }
    return NULL;
}

#ifdef _WIN32
static unsigned __stdcall toy_test_thread_entry(void *argument)
{
    toy_test_thread_main(argument);
    return 0;
}
#endif

static void test_threads(void)
{
    printf("test_threads\n");
    toy_arena *arena = toy_test_create_arena((size_t) 1 << 20, TOY_ARENA_TIERED, 8);
    CHECK(arena != NULL);
    if (arena == NULL) {
        return;
    }

    enum { thread_count = 4 };
#ifdef _WIN32
    HANDLE threads[thread_count];
#else
    pthread_t threads[thread_count];
#endif
    bool started[thread_count] = { false };
    toy_test_thread_state states[thread_count];
    for (int i = 0; i < thread_count; i++) {
        states[i] = (toy_test_thread_state) {
            .arena = arena,
            .seed = (uint64_t) (i + 1) * UINT64_C(0x9e3779b97f4a7c15),
        };
#ifdef _WIN32
        threads[i] = (HANDLE) _beginthreadex(NULL, 0, toy_test_thread_entry, &states[i], 0, NULL);
        started[i] = threads[i] != NULL;
#else
        started[i] = pthread_create(&threads[i], NULL, toy_test_thread_main, &states[i]) == 0;
#endif
        if (!started[i]) {
            states[i].error = 1;
        }
    }
    for (int i = 0; i < thread_count; i++) {
        if (!started[i]) {
            continue;
        }
#ifdef _WIN32
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
#else
        pthread_join(threads[i], NULL);
#endif
    }
    for (int i = 0; i < thread_count; i++) {
        CHECK(states[i].error == 0);
    }

    toy_arena_stats stats;
    toy_arena_get_stats(arena, &stats);
    CHECK(stats.used_pages == 0);
    CHECK(stats.free_pages == stats.total_pages);
    CHECK(stats.alloc_failures == 0);
    CHECK(stats.alloc_calls == stats.free_calls);
    toy_arena_destroy(arena);
}

int main(void)
{
    test_create();
    test_tiers();
    test_wx();
    test_reuse();
    test_capacity();
    test_guarded();
    test_cache();
    test_threads();

    if (failures != 0) {
        fprintf(stderr, "%d arena check(s) failed\n", failures);
        return 1;
    }
    printf("all arena checks passed\n");
    return 0;
}

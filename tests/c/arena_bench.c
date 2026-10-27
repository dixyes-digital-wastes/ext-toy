/*
 * Standalone benchmark for the kernel arena
 * This file does not link against PHP and runs on POSIX only
 *
 * Every operation allocates a random block of 1, 2, 4 or 8 pages and frees it
 * at once. The seal variant also writes one byte and seals the block.
 *
 * Build and run:
 *   cc -std=gnu17 -Wall -Wextra -Werror -O2 -pthread -I. \
 *      tests/c/arena_bench.c arena.c -o arena_bench
 *   ./arena_bench [repeats]
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arena.h"

enum {
    TOY_BENCH_ARENA_BYTES = 32 << 20,
    TOY_BENCH_MAX_THREADS = 8,
    TOY_BENCH_ALLOC_OPS = 1000000,
    TOY_BENCH_SEAL_OPS = 10000,
    TOY_BENCH_REPEATS = 3,
    TOY_BENCH_WARMUP_OPS = 1000,
};

typedef struct bench_thread {
    arena *arena;
    uint64_t seed;
    uint64_t ops;
    bool seal;
    int error;
} bench_thread;

static pthread_barrier_t bench_barrier;

static uint64_t bench_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * UINT64_C(1000000000) + (uint64_t) ts.tv_nsec;
}

static uint64_t bench_next_random(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    return value;
}

static void bench_run_ops(bench_thread *thread)
{
    uint64_t random = thread->seed;

    for (uint64_t i = 0; i < thread->ops; i++) {
        random = bench_next_random(&random);
        uint32_t pages = UINT32_C(1) << (random & 3);
        arena_block block = { 0 };
        if (!arena_alloc(thread->arena, pages, &block)) {
            thread->error = 1;
            return;
        }
        if (thread->seal) {
            block.code[0] = (uint8_t) random;
            if (!arena_seal(thread->arena, &block)) {
                thread->error = 2;
                arena_free(thread->arena, &block);
                return;
            }
            if (block.code[0] != (uint8_t) random) {
                thread->error = 3;
            }
        }
        arena_free(thread->arena, &block);
    }
}

static void *bench_thread_main(void *argument)
{
    bench_thread *thread = argument;
    bench_thread warmup = *thread;

    warmup.ops = TOY_BENCH_WARMUP_OPS;
    bench_run_ops(&warmup);
    if (warmup.error != 0) {
        thread->error = warmup.error;
    }

    /* All workers and the main thread meet here, so the burst starts together */
    pthread_barrier_wait(&bench_barrier);
    bench_run_ops(thread);
    return NULL;
}

typedef struct bench_result {
    double ns_per_op;
    double mops;
    uint64_t mprotect_calls;
    bool ok;
} bench_result;

static bench_result bench_run(uint32_t segments, int thread_count, bool seal,
                                      uint64_t ops_per_thread)
{
    bench_result result = { 0 };
    arena_config config = arena_default_config();
    config.total_bytes = TOY_BENCH_ARENA_BYTES;
    config.segment_count = segments;

    arena *arena = arena_create(&config);
    if (arena == NULL) {
        fprintf(stderr, "FAIL cannot create an arena with %u segments\n", segments);
        return result;
    }

    pthread_t threads[TOY_BENCH_MAX_THREADS];
    bench_thread thread_args[TOY_BENCH_MAX_THREADS];
    pthread_barrier_init(&bench_barrier, NULL, (unsigned) thread_count + 1);

    for (int i = 0; i < thread_count; i++) {
        thread_args[i] = (bench_thread) {
            .arena = arena,
            .seed = (uint64_t) (i + 1) * UINT64_C(0x9e3779b97f4a7c15),
            .ops = ops_per_thread,
            .seal = seal,
            .error = 0,
        };
        if (pthread_create(&threads[i], NULL, bench_thread_main, &thread_args[i]) != 0) {
            /* The barrier already expects this thread, so the run cannot finish */
            fprintf(stderr, "FAIL cannot start thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    pthread_barrier_wait(&bench_barrier);
    uint64_t start_ns = bench_now_ns();
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    uint64_t elapsed_ns = bench_now_ns() - start_ns;
    pthread_barrier_destroy(&bench_barrier);

    uint64_t total_ops = ops_per_thread * (uint64_t) thread_count;
    bool ok = elapsed_ns > 0;
    for (int i = 0; i < thread_count; i++) {
        ok = ok && thread_args[i].error == 0;
    }

    arena_stats stats;
    arena_get_stats(arena, &stats);
    ok = ok && stats.used_pages == 0 && stats.alloc_failures == 0;

    result.ns_per_op = (double) elapsed_ns / (double) total_ops;
    result.mops = 1000.0 / result.ns_per_op;
    result.mprotect_calls = stats.mprotect_calls;
    result.ok = ok;
    arena_destroy(arena);
    return result;
}

static int bench_compare_double(const void *left, const void *right)
{
    double a = *(const double *) left;
    double b = *(const double *) right;
    return (a > b) - (a < b);
}

static double bench_median(double *values, int count)
{
    qsort(values, (size_t) count, sizeof(*values), bench_compare_double);
    return values[count / 2];
}

static void bench_report(const char *label, uint32_t segments, int thread_count,
                             bool seal, uint64_t ops_per_thread, int repeats)
{
    double samples[16];
    uint64_t mprotect_calls = 0;
    bool ok = true;
    if (repeats > (int) (sizeof(samples) / sizeof(samples[0]))) {
        repeats = (int) (sizeof(samples) / sizeof(samples[0]));
    }

    for (int i = 0; i < repeats; i++) {
        bench_result result = bench_run(segments, thread_count, seal, ops_per_thread);
        samples[i] = result.ns_per_op;
        mprotect_calls += result.mprotect_calls;
        ok = ok && result.ok;
    }
    double median = bench_median(samples, repeats);
    printf("%-10s segments=%u threads=%d  %9.1f ns/op  %6.2f Mops/s  "
           "mprotect=%llu  %s  samples:",
           label, segments, thread_count, median, 1000.0 / median,
           (unsigned long long) (mprotect_calls / (uint64_t) repeats), ok ? "ok" : "BAD");
    for (int i = 0; i < repeats; i++) {
        printf(" %.1f", samples[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    int repeats = TOY_BENCH_REPEATS;
    if (argc > 1) {
        repeats = atoi(argv[1]);
        if (repeats < 1 || repeats > 16) {
            fprintf(stderr, "repeats must be 1..16\n");
            return 2;
        }
    }

    arena *probe = arena_create(NULL);
    size_t page_size = arena_page_size(probe);
    arena_destroy(probe);

    printf("arena bytes=%d, page size=%zu, repeats=%d\n", TOY_BENCH_ARENA_BYTES, page_size,
           repeats);

    printf("\nalloc+free throughput\n");
    for (int thread_count = 1; thread_count <= TOY_BENCH_MAX_THREADS; thread_count *= 2) {
        bench_report("alloc+free", 1, thread_count, false, TOY_BENCH_ALLOC_OPS, repeats);
    }
    for (int thread_count = 1; thread_count <= TOY_BENCH_MAX_THREADS; thread_count *= 2) {
        bench_report("alloc+free", 8, thread_count, false, TOY_BENCH_ALLOC_OPS, repeats);
    }

    printf("\nalloc+write+seal+free throughput\n");
    for (int thread_count = 1; thread_count <= TOY_BENCH_MAX_THREADS; thread_count *= 2) {
        bench_report("seal", 1, thread_count, true, TOY_BENCH_SEAL_OPS, repeats);
    }
    for (int thread_count = 1; thread_count <= TOY_BENCH_MAX_THREADS; thread_count *= 2) {
        bench_report("seal", 8, thread_count, true, TOY_BENCH_SEAL_OPS, repeats);
    }
    return 0;
}

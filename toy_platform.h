#ifndef TOY_PLATFORM_H
#define TOY_PLATFORM_H

/*
 * Small platform layer for the arena and the spec cache
 *
 * Windows uses VirtualAlloc, VirtualProtect, FlushInstructionCache and
 * SRWLOCK. Other targets use mmap, mprotect, __builtin___clear_cache and
 * pthread mutexes.
 */

#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef enum toy_platform_prot {
    TOY_PLATFORM_PROT_RW = 0,
    TOY_PLATFORM_PROT_RX = 1,
    TOY_PLATFORM_PROT_NONE = 2,
} toy_platform_prot;

#ifdef _WIN32
typedef SRWLOCK toy_platform_lock;
#else
typedef pthread_mutex_t toy_platform_lock;
#endif

static inline size_t toy_platform_page_size(void)
{
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t) info.dwPageSize;
#else
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t) value : 0;
#endif
}

/* Maps a writable, non-executable anonymous region. Returns NULL on failure. */
static inline void *toy_platform_map(size_t bytes)
{
#ifdef _WIN32
    return VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    return base == MAP_FAILED ? NULL : base;
#endif
}

static inline void toy_platform_unmap(void *address, size_t bytes)
{
    if (address == NULL) {
        return;
    }
#ifdef _WIN32
    (void) bytes;
    VirtualFree(address, 0, MEM_RELEASE);
#else
    munmap(address, bytes);
#endif
}

static inline bool toy_platform_protect(void *address, size_t bytes, toy_platform_prot prot)
{
#ifdef _WIN32
    DWORD native_prot = PAGE_NOACCESS;
    if (prot == TOY_PLATFORM_PROT_RW) {
        native_prot = PAGE_READWRITE;
    } else if (prot == TOY_PLATFORM_PROT_RX) {
        native_prot = PAGE_EXECUTE_READ;
    }
    DWORD previous_prot = 0;
    return VirtualProtect(address, bytes, native_prot, &previous_prot) != 0;
#else
    int native_prot = PROT_NONE;
    if (prot == TOY_PLATFORM_PROT_RW) {
        native_prot = PROT_READ | PROT_WRITE;
    } else if (prot == TOY_PLATFORM_PROT_RX) {
        native_prot = PROT_READ | PROT_EXEC;
    }
    return mprotect(address, bytes, native_prot) == 0;
#endif
}

static inline void toy_platform_flush_icache(const void *code, size_t bytes)
{
#ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), code, bytes);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin___clear_cache((char *) code, (char *) code + bytes);
#else
    (void) code;
    (void) bytes;
#endif
}

static inline bool toy_platform_lock_init(toy_platform_lock *lock)
{
#ifdef _WIN32
    InitializeSRWLock(lock);
    return true;
#else
    return pthread_mutex_init(lock, NULL) == 0;
#endif
}

static inline void toy_platform_lock_destroy(toy_platform_lock *lock)
{
#ifdef _WIN32
    (void) lock;
#else
    pthread_mutex_destroy(lock);
#endif
}

static inline void toy_platform_lock_acquire(toy_platform_lock *lock)
{
#ifdef _WIN32
    AcquireSRWLockExclusive(lock);
#else
    pthread_mutex_lock(lock);
#endif
}

static inline bool toy_platform_lock_try_acquire(toy_platform_lock *lock)
{
#ifdef _WIN32
    return TryAcquireSRWLockExclusive(lock) != 0;
#else
    return pthread_mutex_trylock(lock) == 0;
#endif
}

static inline void toy_platform_lock_release(toy_platform_lock *lock)
{
#ifdef _WIN32
    ReleaseSRWLockExclusive(lock);
#else
    pthread_mutex_unlock(lock);
#endif
}

#endif /* TOY_PLATFORM_H */

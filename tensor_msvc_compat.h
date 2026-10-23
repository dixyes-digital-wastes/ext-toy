/*
 * MSVC compatibility header for the shipped PHP 8.4 headers
 *
 * PHP 8.4 typedefs zend_max_align_t from max_align_t once __STDC_VERSION__
 * reaches C11, but MSVC never provides max_align_t. The union mirrors the
 * fallback branch of Zend/zend_portability.h so the type keeps the same
 * layout. Force-include this header before any PHP header
 */

#ifndef TOY_MSVC_COMPAT_H
#define TOY_MSVC_COMPAT_H

#if defined(_MSC_VER)

typedef union {
    char c;
    short s;
    int i;
    long l;
    long long ll;
    float f;
    double d;
    long double ld;
    void *p;
    void (*fun)(void);
} max_align_t;

#endif

#endif /* TOY_MSVC_COMPAT_H */

#ifndef TOY_ATTRIBUTES_H
#define TOY_ATTRIBUTES_H

/*
 * Portable declaration attributes
 *
 * C17 has no attribute syntax, so the unused-result marker is a macro.
 * GCC and Clang warn when the result is ignored. MSVC has no equivalent
 * compiler warning outside its static analyzer
 */

#if defined(__GNUC__) || defined(__clang__)
# define TOY_NODISCARD __attribute__((warn_unused_result))
#else
# define TOY_NODISCARD
#endif

#endif /* TOY_ATTRIBUTES_H */

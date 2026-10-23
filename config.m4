dnl The extension is opt-in. The default is "no" for --enable-tensor.
PHP_ARG_ENABLE([tensor],
  [whether to enable the Toy tensor extension],
  [AS_HELP_STRING([--enable-tensor], [Enable the Toy tensor extension])],
  [no])

if test "$PHP_TENSOR" != "no"; then
  dnl The tensor sources use C23. Probe the actual compiler and CFLAGS first.
  dnl The probe appends the flag the same way the build does, so a user
  dnl -std=gnu17 in CFLAGS is overridden by the later -std=gnu23.
  AC_CACHE_CHECK([whether $CC compiles C23 with -std=gnu23], [php_cv_tensor_c23], [
    save_CFLAGS="$CFLAGS"
    CFLAGS="$CFLAGS -std=gnu23"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
      #include <stdbool.h>
      #include <stdalign.h>
      #if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L
      #error C23 is required
      #endif
      static_assert(1, "C23 static_assert");
      [[nodiscard]] static int toy_c23_probe_fn(void) { return 1; }
      struct toy_c23_probe { int value; };
    ]], [[
      bool flag = true;
      alignas(8) int aligned = 0;
      struct toy_c23_probe probe = { .value = aligned };
      return flag ? toy_c23_probe_fn() + probe.value : 0;
    ]])],
    [php_cv_tensor_c23=yes],
    [php_cv_tensor_c23=no])
    CFLAGS="$save_CFLAGS"
  ])

  if test "$php_cv_tensor_c23" != "yes"; then
    AC_MSG_ERROR([tensor requires C23 support; $CC does not compile C23 code with -std=gnu23])
  fi

  dnl The extra flags apply to this extension only. They come after CFLAGS
  dnl in the compile command, so they override a user -std= option.
  PHP_NEW_EXTENSION([tensor], m4_normalize([
    tensor.c
    tensor_ops.c
    toy_arena.c
    toy_kernel_arena.c
    toy_spec_cache.c
    toy_tensor.c
    toy_tensor_ops.c
  ]), [$ext_shared],, [-std=gnu23])
fi

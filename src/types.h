#ifndef GS_TYPES_H
#define GS_TYPES_H

#include <limits.h>

/* 
  Define the integer types used throughout the code,
  controlled by preprocessor macros.
  
  The integer type sint/uint (signed/unsigned) is used
  most frequently, e.g., for indexing into local arrays,
  and for processor ids. It can be one of
  
    macro                       sint/uint type
    
    (default)                   int
    GSLIB_USE_LONG              long
    GSLIB_USE_LONG_LONG         long long
    
  The slong/ulong type is used in relatively few places
  for global identifiers and indices. It can be one of

    macro                       slong/ulong type
    
    (default)                   int
    GSLIB_USE_GLOBAL_LONG       long
    GSLIB_USE_GLOBAL_LONG_LONG  long long

  Since the long long type is not ISO C90, it is never
  used unless explicitly asked for.

  The POSIX-standard limits.h header provides the
  LLONG_MAX and LLONG_MIN macros, which will be
  preferentially used.  

*/

#if defined(GSLIB_USE_LONG_LONG) || defined(GSLIB_USE_GLOBAL_LONG_LONG)
typedef long long long_long;
#  define GS_WHEN_LONG_LONG(x) x
#  if !defined(LLONG_MAX)
#    if defined(LONG_LONG_MAX)
#      define LLONG_MAX LONG_LONG_MAX
#    else
#      define LLONG_MAX 9223372036854775807
#    endif
#  endif
#  if !defined(LLONG_MIN)
#    if defined(LONG_LONG_MIN)
#      define LLONG_MIN LONG_LONG_MIN
#    else
#      define LLONG_MIN -9223372036854775807
#    endif
#  endif
#else
#  define GS_WHEN_LONG_LONG(x)
#endif

#if !defined(GSLIB_USE_LONG) && !defined(GSLIB_USE_LONG_LONG)
#  define GS_TYPE_LOCAL(i,l,ll) i
#elif defined(GSLIB_USE_LONG)
#  define GS_TYPE_LOCAL(i,l,ll) l
#elif defined(GSLIB_USE_LONG_LONG)
#  define GS_TYPE_LOCAL(i,l,ll) ll
#endif

#if !defined(GSLIB_USE_GLOBAL_LONG) && !defined(GSLIB_USE_GLOBAL_LONG_LONG)
#  define GS_TYPE_GLOBAL(i,l,ll) i
#elif defined(GSLIB_USE_GLOBAL_LONG)
#  define GS_TYPE_GLOBAL(i,l,ll) l
#else
#  define GS_TYPE_GLOBAL(i,l,ll) ll
#endif

/* local integer type: for quantities O(N/P) */
#define sint   signed GS_TYPE_LOCAL(int,long,long long)
#define uint unsigned GS_TYPE_LOCAL(int,long,long long)
#define iabs GS_TYPE_LOCAL(abs,labs,llabs)

/* global integer type: for quantities O(N) */
#define slong   signed GS_TYPE_GLOBAL(int,long,long long)
#define ulong unsigned GS_TYPE_GLOBAL(int,long,long long)
#define iabsl GS_TYPE_GLOBAL(abs,labs,llabs)

#endif


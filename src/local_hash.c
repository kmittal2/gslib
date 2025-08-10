#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "obbox.h"

struct uint_range { uint min, max; };

static struct dbl_range dbl_range_merge(struct dbl_range a, struct dbl_range b)
{
  struct dbl_range m;
  m.min = b.min<a.min?b.min:a.min,
  m.max = a.max>b.max?a.max:b.max;
  return m;
}


static sint ifloor(double x) { return floor(x); }
static sint iceil (double x) { return ceil (x); }
static uint local_hash_index_aux(double low, double fac, uint n, double x)
{
  const sint i = ifloor((x-low)*fac);
  return i<0 ? 0 : (n-1<(uint)i ? n-1 : (uint)i);
}


#define D 2
#define WHEN_3D(a)
#include "local_hash_imp.h"
#undef WHEN_3D
#undef D

#define D 3
#define WHEN_3D(a) a
#include "local_hash_imp.h"
#undef WHEN_3D
#undef D
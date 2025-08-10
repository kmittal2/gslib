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
#include "gs_defs.h"
#include "comm.h"
#include "crystal.h"
#include "sarray_transfer.h"
#include "sort.h"
#include "sarray_sort.h"
#include "local_hash.h"

struct ulong_range { ulong min, max; };
struct proc_index { uint proc, index; };

static struct dbl_range dbl_range_merge(struct dbl_range a, struct dbl_range b)
{
  struct dbl_range m;
  m.min = b.min<a.min?b.min:a.min,
  m.max = a.max>b.max?a.max:b.max;
  return m;
}

static slong lfloor(double x) { return floor(x); }
static slong lceil (double x) { return ceil (x); }
static sint ifloor(double x) { return floor(x); }
static sint iceil (double x) { return ceil (x); }

static uint hash_index_aux(double low, double fac, uint n, double x)
{
  const sint i = ifloor((x-low)*fac);
  return i<0 ? 0 : (n-1<(uint)i ? n-1 : (uint)i);
}


static void set_bit(unsigned char *const p, const uint i)
{
  const uint byte = i/CHAR_BIT;
  const unsigned bit = i%CHAR_BIT;
  p[byte] |= 1u<<bit;
}

static unsigned get_bit(const unsigned char *const p, const uint i)
{
  const uint byte = i/CHAR_BIT;
  const unsigned bit = i%CHAR_BIT;
  return p[byte]>>bit & 1u;
}

static unsigned byte_bits(const unsigned char x)
{
  unsigned bit, sum=0;
  for(bit=0;bit<CHAR_BIT;++bit) sum += x>>bit & 1u;
  return sum;
}

static uint count_bits(unsigned char *p, uint n)
{
  uint sum=0;
  for(;n;--n) sum+=byte_bits(*p++);
  return sum;
}


#define D 2
#define WHEN_3D(a)
#include "global_hash_imp.h"
#undef WHEN_3D
#undef D

#define D 3
#define WHEN_3D(a) a
#include "global_hash_imp.h"
#undef WHEN_3D
#undef D
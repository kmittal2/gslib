#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include <math.h>    /* for sqrt */
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "tensor.h"
#include "poly.h"
#include "lob_bnd.h"
#include <stdio.h>

#define obbox_calc_2  PREFIXED_NAME(obbox_calc_2)
#define obbox_calc_3  PREFIXED_NAME(obbox_calc_3)

#define obboxsurf_calc_2 PREFIXED_NAME(obboxsurf_calc_2)
#define obboxsurf_calc_3 PREFIXED_NAME(obboxsurf_calc_3)

struct obbox_2 { double c0[2], A[4];
                 struct dbl_range x[2]; };

struct obbox_3 { double c0[3], A[9];
                 struct dbl_range x[3]; };


static void copy_strided(double *out, const double *in,
                         unsigned g, unsigned s, unsigned n)
{
  if(g==1) for(;n;--n,in+=s) *out++ = *in;
  else {
    s *= g;
    for(;n;--n,in+=s) memcpy(out,in,g*sizeof(double)), out+=g;
  }
}

static void mat_inv_2(double inv[4], const double A[4])
{
  const double idet = 1/(A[0]*A[3]-A[1]*A[2]);
  inv[0] =   idet*A[3];
  inv[1] = -(idet*A[1]);
  inv[2] = -(idet*A[2]);
  inv[3] =   idet*A[0];
}

static void mat_inv_3(double inv[9], const double A[9])
{
  const double a = A[4]*A[8]-A[5]*A[7],
               b = A[5]*A[6]-A[3]*A[8],
               c = A[3]*A[7]-A[4]*A[6],
            idet = 1/(A[0]*a+A[1]*b+A[2]*c);
  inv[0] = idet*a;
  inv[1] = idet*(A[2]*A[7]-A[1]*A[8]);
  inv[2] = idet*(A[1]*A[5]-A[2]*A[4]);
  inv[3] = idet*b;
  inv[4] = idet*(A[0]*A[8]-A[2]*A[6]);
  inv[5] = idet*(A[2]*A[3]-A[0]*A[5]);
  inv[6] = idet*c;
  inv[7] = idet*(A[1]*A[6]-A[0]*A[7]);
  inv[8] = idet*(A[0]*A[4]-A[1]*A[3]);
}

static struct dbl_range dbl_range_merge(struct dbl_range a, struct dbl_range b)
{
  struct dbl_range m;
  m.min = b.min<a.min?b.min:a.min,
  m.max = a.max>b.max?a.max:b.max;
  return m;
}

static struct dbl_range dbl_range_expand(struct dbl_range b, double tol)
{
  double a = (b.min+b.max)/2, l = (b.max-b.min)*(1+tol)/2;
  struct dbl_range m;
  m.min = a-l, m.max = a+l;
  return m;
}

static void bbox_2_tfm(double *out, const double x0[2], const double Ji[4],
                       const double *x, const double *y, unsigned n)
{
  unsigned i;
  for(i=0;i<n;++i) {
    const double dx = x[i]-x0[0], dy = y[i]-x0[1];
    out[  i] = Ji[0]*dx + Ji[1]*dy;
    out[n+i] = Ji[2]*dx + Ji[3]*dy;
  }
}

static void bbox_3_tfm(double *out, const double x0[3], const double Ji[9],
                       const double *x, const double *y, const double *z,
                       unsigned n)
{
  unsigned i;
  for(i=0;i<n;++i) {
    const double dx = x[i]-x0[0], dy = y[i]-x0[1], dz = z[i]-x0[2];
    out[    i] = Ji[0]*dx + Ji[1]*dy + Ji[2]*dz;
    out[  n+i] = Ji[3]*dx + Ji[4]*dy + Ji[5]*dz;
    out[2*n+i] = Ji[6]*dx + Ji[7]*dy + Ji[8]*dz;
  }
}

#if 0

/* positive when possibly inside */
double obbox_axis_test_2(const struct obbox_2 *const b,
                         const double x, const double y)
{
  const double bx = (x-b->x[0].min)*(b->x[0].max-x);
  return bx<0 ? bx : (y-b->x[1].min)*(b->x[1].max-y);
}

/* positive when possibly inside */
double obbox_test_2(const struct obbox_2 *const b,
                    const double x, const double y)
{
  const double bxy = obbox_axis_test_2(b,x,y);
  if(bxy<0) return bxy; else {
    const double dx = x-b->c0[0], dy = y-b->c0[1];
    const double r = b->A[0]*dx + b->A[1]*dy,
                 s = b->A[2]*dx + b->A[3]*dy;
    const double br = (r+1)*(1-r);
    return br<0 ? br : (s+1)*(1-s);
  }
}

#endif

#define DO_MAX(a,b) do { unsigned temp = b; if(temp>a) a=temp; } while(0)

void obbox_calc_2(struct obbox_2 *out,
                  const double *const elx[2],
                  const unsigned n[2], uint nel,
                  const unsigned m[2], const double tol)
{
  const double *x = elx[0], *y = elx[1];
  const unsigned nr = n[0], ns = n[1];
  const unsigned mr = m[0], ms = m[1];

  const unsigned nrs = nr*ns;
  double *data;
  const unsigned lbsize0 = lob_bnd_size(nr,mr),
                 lbsize1 = lob_bnd_size(ns,ms);
  unsigned wsize = 4*ns+2*ms;
  DO_MAX(wsize,2*nr+2*mr);
  DO_MAX(wsize,gll_lag_size(nr));
  DO_MAX(wsize,gll_lag_size(ns));
  data = tmalloc(double, 2*(nr+ns)+lbsize0+lbsize1+wsize);

  {
    double *const I0r = data, *const I0s = data+2*nr;
    double *const lob_bnd_data_r = data+2*(nr+ns),
           *const lob_bnd_data_s = data+2*(nr+ns)+lbsize0;
    double *const work = data+2*(nr+ns)+lbsize0+lbsize1;

    #define SETUP_DIR(r) do { \
      lagrange_fun *const lag = gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)

    SETUP_DIR(r); SETUP_DIR(s);

    #undef SETUP_DIR

    for(;nel;--nel,x+=nrs,y+=nrs,++out) {
      double x0[2], J[4], Ji[4];
      struct dbl_range ab[2], tb[2];

      /* double work[2*nr] */
      x0[0] = tensor_ig2(J  , I0r,nr, I0s,ns, x, work);
      x0[1] = tensor_ig2(J+2, I0r,nr, I0s,ns, y, work);
      mat_inv_2(Ji, J);


      #define DO_BOUND(bnd,merge,r,x,work) do { \
        struct dbl_range b = \
        lob_bnd_1(lob_bnd_data_##r,n##r,m##r, x, work); \
        if(merge) bnd=dbl_range_merge(bnd,b); else bnd=b; \
      } while(0)

      #define DO_EDGE(merge,r,x,y,work) do { \
        DO_BOUND(ab[0],merge,r,x,work); \
        DO_BOUND(ab[1],merge,r,y,work); \
        bbox_2_tfm(work, x0,Ji, x,y,n##r); \
        DO_BOUND(tb[0],merge,r,(work)     ,(work)+2*n##r); \
        DO_BOUND(tb[1],merge,r,(work)+n##r,(work)+2*n##r); \
      } while(0)

      DO_EDGE(0,r,x,y,work);
      DO_EDGE(1,r,&x[nrs-nr],&y[nrs-nr],work);

      #define GET_EDGE(off) do { \
        copy_strided(work   , x+off,1,nr,ns); \
        copy_strided(work+ns, y+off,1,nr,ns); \
        DO_EDGE(1,s,work,work+ns,work+2*ns); \
      } while(0)

      GET_EDGE(0);
      GET_EDGE(nr-1);

      #undef GET_EDGE
      #undef DO_EDGE
      #undef DO_BOUND

      out->x[0] = dbl_range_expand(ab[0],tol),
      out->x[1] = dbl_range_expand(ab[1],tol);

      {
        const double av0=(tb[0].min+tb[0].max)/2, av1=(tb[1].min+tb[1].max)/2;
        out->c0[0] = x0[0] + J[0]*av0 + J[1]*av1;
        out->c0[1] = x0[1] + J[2]*av0 + J[3]*av1;
      }
      {
        const double di0 = 2/((1+tol)*(tb[0].max-tb[0].min)),
                     di1 = 2/((1+tol)*(tb[1].max-tb[1].min));
        out->A[0]=di0*Ji[0], out->A[1]=di0*Ji[1];
        out->A[2]=di1*Ji[2], out->A[3]=di1*Ji[3];
      }
    }
  }
  free(data);
}

void obbox_calc_3(struct obbox_3 *out,
                  const double *const elx[3],
                  const unsigned n[3], uint nel,
                  const unsigned m[3], const double tol)
{
  const double *x = elx[0], *y = elx[1], *z = elx[2];
  const unsigned nr = n[0], ns = n[1], nt = n[2];
  const unsigned mr = m[0], ms = m[1], mt = m[2];

  const unsigned nrs = nr*ns, nrst = nr*ns*nt;
  double *data;
  const unsigned lbsize0 = lob_bnd_size(nr,mr),
                 lbsize1 = lob_bnd_size(ns,ms),
                 lbsize2 = lob_bnd_size(nt,mt);
  unsigned wsize = 3*nr*ns+2*mr*(ns+ms+1);
  DO_MAX(wsize,6*nr*nt+2*mr*(nt+mt+1));
  DO_MAX(wsize,6*ns*nt+2*ms*(nt+mt+1));
  DO_MAX(wsize,2*nr*ns+3*nr);
  DO_MAX(wsize,gll_lag_size(nr));
  DO_MAX(wsize,gll_lag_size(ns));
  DO_MAX(wsize,gll_lag_size(nt));
  data = tmalloc(double, 2*(nr+ns+nt)+lbsize0+lbsize1+lbsize2+wsize);

  {
    double *const I0r = data, *const I0s = I0r+2*nr, *const I0t = I0s+2*ns;
    double *const lob_bnd_data_r = data+2*(nr+ns+nt),
           *const lob_bnd_data_s = data+2*(nr+ns+nt)+lbsize0,
           *const lob_bnd_data_t = data+2*(nr+ns+nt)+lbsize0+lbsize1;
    double *const work = data+2*(nr+ns+nt)+lbsize0+lbsize1+lbsize2;

    #define SETUP_DIR(r) do { \
      lagrange_fun *const lag = gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)

    SETUP_DIR(r); SETUP_DIR(s); SETUP_DIR(t);

    #undef SETUP_DIR

    for(;nel;--nel,x+=nrst,y+=nrst,z+=nrst,++out) {
      double x0[3], J[9], Ji[9];
      struct dbl_range ab[3], tb[3];

      /* double work[2*nrs+3*nr] */
      #define EVAL_AT_0(d,x) \
        x0[d] = tensor_ig3(J+3*d, I0r,nr, I0s,ns, I0t,nt, x, work)
      EVAL_AT_0(0,x); EVAL_AT_0(1,y); EVAL_AT_0(2,z);
      mat_inv_3(Ji, J);
      #undef EVAL_AT_0

      /* double work[2*m##r*(n##s+m##s+1)] */
      #define DO_BOUND(bnd,merge,r,s,x,work) do { \
        struct dbl_range b = \
        lob_bnd_2(lob_bnd_data_##r,n##r,m##r, \
                  lob_bnd_data_##s,n##s,m##s, x, work); \
        if(merge) bnd=dbl_range_merge(bnd,b); else bnd=b; \
      } while(0)

      /* double work[3*n##r*n##s+2*m##r*(n##s+m##s+1)] */
      #define DO_FACE(merge,r,s,x,y,z,work) do { \
        DO_BOUND(ab[0],merge,r,s,x,work); \
        DO_BOUND(ab[1],merge,r,s,y,work); \
        DO_BOUND(ab[2],merge,r,s,z,work); \
        bbox_3_tfm(work, x0,Ji, x,y,z,n##r*n##s); \
        DO_BOUND(tb[0],merge,r,s,(work)            ,(work)+3*n##r*n##s); \
        DO_BOUND(tb[1],merge,r,s,(work)+  n##r*n##s,(work)+3*n##r*n##s); \
        DO_BOUND(tb[2],merge,r,s,(work)+2*n##r*n##s,(work)+3*n##r*n##s); \
      } while(0)

      DO_FACE(0,r,s,x,y,z,work);
      DO_FACE(1,r,s,&x[nrst-nrs],&y[nrst-nrs],&z[nrst-nrs],work);

      /* double work[6*n##r*n##s+2*m##r*(n##s+m##s+1)] */
      #define GET_FACE(r,s,off,n1,n2,n3) do { \
        copy_strided(work            , x+off,n1,n2,n3); \
        copy_strided(work+  n##r*n##s, y+off,n1,n2,n3); \
        copy_strided(work+2*n##r*n##s, z+off,n1,n2,n3); \
        DO_FACE(1,r,s,work,work+n##r*n##s,work+2*n##r*n##s,work+3*n##r*n##s); \
      } while(0)

      GET_FACE(r,t,0     ,nr,ns,nt);
      GET_FACE(r,t,nrs-nr,nr,ns,nt);
      GET_FACE(s,t,0     , 1,nr,ns*nt);
      GET_FACE(s,t,nr-1  , 1,nr,ns*nt);

      #undef GET_FACE
      #undef DO_FACE
      #undef DO_BOUND

      out->x[0] = dbl_range_expand(ab[0],tol),
      out->x[1] = dbl_range_expand(ab[1],tol);
      out->x[2] = dbl_range_expand(ab[2],tol);

      {
        const double av0 = (tb[0].min+tb[0].max)/2,
                     av1 = (tb[1].min+tb[1].max)/2,
                     av2 = (tb[2].min+tb[2].max)/2;
        out->c0[0] = x0[0] + J[0]*av0 + J[1]*av1 + J[2]*av2;
        out->c0[1] = x0[1] + J[3]*av0 + J[4]*av1 + J[5]*av2;
        out->c0[2] = x0[2] + J[6]*av0 + J[7]*av1 + J[8]*av2;
      }
      {
        const double di0 = 2/((1+tol)*(tb[0].max-tb[0].min)),
                     di1 = 2/((1+tol)*(tb[1].max-tb[1].min)),
                     di2 = 2/((1+tol)*(tb[2].max-tb[2].min));
        out->A[0]=di0*Ji[0], out->A[1]=di0*Ji[1], out->A[2]=di0*Ji[2];
        out->A[3]=di1*Ji[3], out->A[4]=di1*Ji[4], out->A[5]=di1*Ji[5];
        out->A[6]=di2*Ji[6], out->A[7]=di2*Ji[7], out->A[8]=di2*Ji[8];
      }

    }
  }

  free(data);
}

/* Calculates the diagonal length of the bounding box and expands its bounds by
 * 0.5*len*tol at both its min and max values.
 * Returns the length of the diagonal (could be used for expanding obboxes).
 */
double dbl_range_diag_expand_2(struct dbl_range *m, struct dbl_range b[3], double tol)
{
  double l[2] = { b[0].max-b[0].min, b[1].max-b[1].min };
  double len = sqrt(l[0]*l[0] + l[1]*l[1])*0.5*tol;
  for (int i=0; i<2; i++) {
    m[i].min = b[i].min - len;
    m[i].max = b[i].max + len;
  }
  return len;
}
double dbl_range_diag_expand_3(struct dbl_range *m, struct dbl_range b[3], double tol)
{
  double l[3] = { b[0].max-b[0].min, b[1].max-b[1].min, b[2].max-b[2].min };
  double len = sqrt(l[0]*l[0] + l[1]*l[1] + l[2]*l[2])*0.5*tol;
  for (int i=0; i<3; i++) {
    m[i].min = b[i].min - len;
    m[i].max = b[i].max + len;
  }
  return len;
}

void obboxsurf_calc_2(        struct obbox_2 *out,
                       const double *const elx[2],
                              const unsigned n[1],
                                         uint nel,
                              const unsigned m[1],
                                 const double tol )
{
  const double *x   = elx[0],
               *y   = elx[1];
  const unsigned nr = n[0],
                 mr = m[0];

  double *data;
  const unsigned lbsize0 = lob_bnd_size(nr,mr);

  unsigned wsize = 2*nr+2*mr;
  DO_MAX(wsize,gll_lag_size(nr));

  // A big vector that stores all data related to bounds and all the work arrays
  data = tmalloc(double, 2*nr + lbsize0 + wsize);

  {
    double *const I0r = data,                          // 2*nr doubles
           *const lob_bnd_data_r = data + 2*nr,        // lbsize0 doubles
           *const work = data + 2*nr + lbsize0;        // wsize doubles

    #define SETUP_DIR(r) do { \
      lagrange_fun *const lag = gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)

    SETUP_DIR(r);
    #undef SETUP_DIR

    // Loop over all elements; note the decrementing nel
    uint nelorig = nel;
    for( ; nel; --nel,x+=nr,y+=nr,++out) {
      double x0[2], A[4];
      struct dbl_range ab[2], tb[2];

      x0[0] = tensor_ig1(A  ,I0r,nr,x);
      // A[0] = dx/dr, x0[0] = x(r=0), i.e., element center
      x0[1] = tensor_ig1(A+1,I0r,nr,y);
      // A[1] = dy/dr, x0[1] = y(r=0), i.e., element center
      A[2] = sqrt( A[0]*A[0] + A[1]*A[1] );
      A[0] = A[0]/A[2];
      A[1] = A[1]/A[2];
      A[2] = -A[1];
      A[3] =  A[0];
      /* At this stage, A has the rotation matrix that captures the rotation the
       * physical nodes require to align the tangent at element center with the
       * x-axis.
       */

      /* double work[2*m##r]
       * Find the bounds along a specific physical dimension.
       */
      #define DO_BOUND(bnd,r,x,work) do { \
        bnd = lob_bnd_1(lob_bnd_data_##r,n##r,m##r, x, work); \
      } while(0)

      /* double work[2*n##r + 2*m##r] */
      #define DO_EDGE(r,x,y,work) do { \
        DO_BOUND(ab[0],r,x,work); \
        DO_BOUND(ab[1],r,y,work); \
        bbox_2_tfm(work, x0,A, x,y,n##r); \
        DO_BOUND(tb[0],r,(work),(work)+2*n##r); \
        DO_BOUND(tb[1],r,(work)+n##r,(work)+2*n##r); \
      } while(0)
      DO_EDGE(r,x,y,work);
      #undef DO_EDGE
      #undef DO_BOUND

      double aabb_diag_len = dbl_range_diag_expand_2(out->x, ab, tol);

      const double av0 = (tb[0].min+tb[0].max)/2,
                   av1 = (tb[1].min+tb[1].max)/2;
      const double dx0 =  A[0]*av0 - A[1]*av1,
                   dx1 = -A[2]*av0 + A[3]*av1;
      out->c0[0] = x0[0] + dx0;
      out->c0[1] = x0[1] + dx1;

      // Expand by aabb_diag_len only if the element is possibly planar
      if (fabs(tb[1].max-tb[1].min) < 1e-10*aabb_diag_len)
      {
        tb[1].min -= aabb_diag_len;
        tb[1].max += aabb_diag_len;
      }
      else
      {
        tb[1] = dbl_range_expand(tb[1],tol);
      }
      tb[0] = dbl_range_expand(tb[0],tol);
      const double di0 = 2/(tb[0].max-tb[0].min),
                   di1 = 2/(tb[1].max-tb[1].min);
      out->A[0]=di0*A[0], out->A[1]=di0*A[1];
      out->A[2]=di1*A[2], out->A[3]=di1*A[3];
    }
  }
  free(data);
}

void obboxsurf_calc_3(        struct obbox_3 *out,
                       const double *const elx[3],
                              const unsigned n[2],
                                         uint nel,
                              const unsigned m[2],
                                 const double tol )
{
  const double *x = elx[0], *y = elx[1], *z = elx[2];
  const unsigned nr = n[0], ns = n[1];
  const unsigned mr = m[0], ms = m[1];

  const unsigned nrs = nr*ns;
  double *data;
  const unsigned lbsize0 = lob_bnd_size(nr,mr),
                 lbsize1 = lob_bnd_size(ns,ms);

  unsigned wsize = 3*nr*ns+2*mr*(ns+ms+1);
  DO_MAX(wsize,2*nr*ns+3*nr);
  DO_MAX(wsize,gll_lag_size(nr));
  DO_MAX(wsize,gll_lag_size(ns));

  data = tmalloc(double, 2*(nr+ns)+lbsize0+lbsize1+wsize);

  {
    double *const I0r            = data,
           *const I0s            = I0r  + 2*nr;
    double *const lob_bnd_data_r = data + 2*(nr+ns),
           *const lob_bnd_data_s = data + 2*(nr+ns) + lbsize0;
    double *const work           = data + 2*(nr+ns) + lbsize0 + lbsize1;

    /* All the calculation in SETUP_DIR is done for the reference space. So the
     * question arises: why r and s are treated separately?  This is because
     * James assumed that the r and s directions could have different number of
     * nodes. For us, that is not the case. So, we can just do this calculation
     * in one direction and use everywhere.
     * 1. lag would store a pointer to a function of type lagrange_fun,
     *    pointing to the correct function to evaluate the lagrange polynomials
     *    on nr GLL nodes.
     * 2. gll_lag_setup returns correct lag_coeff function and stores
     *    lag_coeffs (based on ref coords) in work for evaluating the lagrange
     *    polynomials.
     * 3. the coefficients in work are utilized to evaluate the lagrange
     *    polynomials and their 1st derivatives at x=0 (or r=0) in I0##r.
     * 4. lob_bnd_setup assigns to lob_bnd_data_##r, which is used to compute
     *    bounds
     */
    #define SETUP_DIR(r) do { \
      lagrange_fun *const lag = gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)
    SETUP_DIR(r);
    SETUP_DIR(s);
    #undef SETUP_DIR

    uint nelorig = nel;
    // Loop over all elements; note the decrementing nel
    for(; nel; --nel,x+=nrs,y+=nrs,z+=nrs,++out) {
      struct dbl_range ab[3];
      struct dbl_range tb[3];
      double x0[3], tv[9], A[9];

      /* double work[2*nr]
       * Find the center of the element (r=0 ref. coord.) in physical space
       * and store in x0.
       * tv[0], tv[1], tv[2]: kept empty at this point for convenience.
       * tv[3], tv[4]: dx/dr, dx/ds
       * tv[5], tv[6]: dy/dr, dy/ds
       * tv[7], tv[8]: dz/dr, dz/ds
       */
      x0[0] = tensor_ig2(tv+3, I0r,nr, I0s,ns, x, work);
      x0[1] = tensor_ig2(tv+5, I0r,nr, I0s,ns, y, work);
      x0[2] = tensor_ig2(tv+7, I0r,nr, I0s,ns, z, work);

      // tangent vector 1 moved to tv[0], tv[1], tv[2]
      tv[0] = tv[3], tv[1] = tv[5], tv[2] = tv[7];
      // tangent vector 2 moved to tv[3], tv[4], tv[5]
      tv[3] = tv[4], tv[4] = tv[6], tv[5] = tv[8];
      // normal vector to the plane formed by t1 and t2 (cross product)
      // is stored in tv[6], tv[7], tv[8]
      tv[6] = tv[1]*tv[5] - tv[2]*tv[4];
      tv[7] = tv[2]*tv[3] - tv[0]*tv[5];
      tv[8] = tv[0]*tv[4] - tv[1]*tv[3];
      // normalize the normal vector
      const double nmag  = sqrt( tv[6]*tv[6] + tv[7]*tv[7] + tv[8]*tv[8] );
      tv[6] = tv[6]/nmag;
      tv[7] = tv[7]/nmag;
      tv[8] = tv[8]/nmag;


      // // Rodrigues formula to compute the rotation matrix
      // double axis[3]; // cross product between normal vector and z-axis
      const double nmag2 = sqrt(tv[6]*tv[6] + tv[7]*tv[7]);
      tv[7] = tv[7]/nmag2;
      tv[6] = tv[6]/nmag2;
      #define kx tv[7]
      #define ky -tv[6]
      #define kz 0.0

      double ct = tv[8];
      double st = 1.0 - ct*ct;
      if (st > 0.0)
      {
        st = sqrt(st);
      }

      // row-major
      A[0] = 1.0 + st*0.0 + (1.0-ct)*(-ky*ky-kz*kz);
      A[1] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
      A[2] = 0.0 + st*(ky) + (1.0-ct)*(kx*kz);

      A[3] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
      A[4] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-kz*kz);
      A[5] = 0.0 + st*(-kx) + (1.0-ct)*(ky*kz);

      A[6] = 0.0 + st*(-ky) + (1.0-ct)*(kx*kz);
      A[7] = 0.0 + st*(kx) + (1.0-ct)*(ky*kz);
      A[8] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-ky*ky);


      /* At this stage, the normal vector has been aligned with the z-axis. We
       * still need to align the element in xy-plane. The process is as follows:
       * 1. We first translate all nodes to the element center.
       * 2. We then rotate the translated nodes by A.
       * 3. At this stage, we can obtain the z-bounds and expand them.
       * 4. We then calculate the Jacobian matrix at the rotated element center.
       *    This is done by applying A to the tangent vectors.
       * 5. We then calculate the inverse of this Jacobian matrix.
       * 6. Premultiplying the inverse Jacobian matrix to the rotated x,y
       *    coordinates gives their reference space coords.
       */

      /* double work[2*m##r*(n##s+m##s+1)] */
      #define DO_BOUND(bnd,r,s,x,work) do { \
        bnd = lob_bnd_2(lob_bnd_data_##r,n##r,m##r, \
                        lob_bnd_data_##s,n##s,m##s, \
                        x, work); \
      } while(0)

      DO_BOUND(ab[0],r,s,x,work);
      DO_BOUND(ab[1],r,s,y,work);
      DO_BOUND(ab[2],r,s,z,work);
      // expand bounding box based on (tol*diagonal_length) in each direction
      // to avoid 0 extent in 1 direction.
      double aabb_diag_len = dbl_range_diag_expand_3(out->x, ab, tol);

      double xtfm[3*nrs]; // xtfm[0]:x, xtfm[nrs]:y, xtfm[2*nrs]:z
      bbox_3_tfm(xtfm, x0,A, x,y,z,nrs);
      // The rotated z-coords are used to calculate z-bounds.
      DO_BOUND(tb[2],r,s,xtfm+2*nrs,work);
      // OBB - expand in z-direction by aabb_diag_len only if the rotate
      // element is possibly planar.
      if (fabs(tb[2].max-tb[2].min) < 1e-10*aabb_diag_len)
      {
        tb[2].min -= aabb_diag_len;
        tb[2].max += aabb_diag_len;
      }
      else
      {
        tb[2] = dbl_range_expand(tb[2],tol);
      }

      // Also apply A to the tangent vectors, which allows us to calculate the
      // Jacobian matrix at the rotated element center. NOTE that the z
      // components of the rotated tangent vectors will become zero, since the
      // normal vector is aligned with the z-axis.
      double J[4], Ji[4];
      J[0] = A[0]*tv[0] + A[1]*tv[1] + A[2]*tv[2]; // rotated dx/dr
      J[1] = A[0]*tv[3] + A[1]*tv[4] + A[2]*tv[5]; // rotated dx/ds
      J[2] = A[3]*tv[0] + A[4]*tv[1] + A[5]*tv[2]; // rotated dy/dr
      J[3] = A[3]*tv[3] + A[4]*tv[4] + A[5]*tv[5]; // rotated dy/ds
      mat_inv_2(Ji, J);

      // Now transform the already rotated x,y coordinates according to Ji to
      // their reference space.
      // Important to note that the nodes used here already have their element
      // center at (0,0). Hence, Ji can be directly applied to them.
      for(unsigned i=0;i<nrs;++i) {
        const double xt = xtfm[i], yt = xtfm[nrs+i];
        xtfm[    i] = Ji[0]*xt + Ji[1]*yt;
        xtfm[nrs+i] = Ji[2]*xt + Ji[3]*yt;
      }
      // Bound these reference space xy coordinates
      DO_BOUND(tb[0],r,s,xtfm    ,work);
      DO_BOUND(tb[1],r,s,xtfm+nrs,work);
      // Expand the bounds based on the tol
      tb[0] = dbl_range_expand(tb[0],tol);
      tb[1] = dbl_range_expand(tb[1],tol);
      #undef DO_BOUND

      /* We now have a BB whose bounds represent bounds of a OBB around the
       * original element.
       *
       * We calculate the center of the OBB in physical space by calculating the
       * center of this BB, which is the same as the displacement needed to move
       * from the element center in the transformed space to the BB center. This
       * displacement is then untransformed by applying (Ji.A)^-1 to it, and
       * added to known physical element center.
       *
       * This BB does not necessarily have known fixed size like [-1,1].
       * So, we premultiply a length scaling matrix, say L, to Ji.A to L.Ji.A.
       * This is the total transformation needed to move a physical location
       * that is inside the physical OBB to a location within [-1,1]^3.
       * Any transformed point not in [-1,1]^3 is outside the OBB.
       *
       * It must be noted: this transformation matrix is only applied to points
       * that have been translated by the physical OBB center.
       */
      {
        // The center of the BB in the transformed space
        const double av0 = (tb[0].min+tb[0].max)/2,
                     av1 = (tb[1].min+tb[1].max)/2,
                     av2 = (tb[2].min+tb[2].max)/2;
        // First untransform the x,y coordinates by J to obtain all components
        // in the rotated space (since z-component is in the rotated space).
        const double Jav0 = J[0]*av0 + J[1]*av1,
                     Jav1 = J[2]*av0 + J[3]*av1;
        // The physical displacement needed to move from the element center to
        // the OBB center is calculated by "un"rotating {Jav0,Jav1,av2} by
        // applying inverse of A.
        // The physical untransformed OBB center can then be obtained.
        out->c0[0] = x0[0] + A[0]*Jav0 + A[3]*Jav1 + A[6]*av2,
        out->c0[1] = x0[1] + A[1]*Jav0 + A[4]*Jav1 + A[7]*av2,
        out->c0[2] = x0[2] + A[2]*Jav0 + A[5]*Jav1 + A[8]*av2;
      }

      // Finally, obtain (L.Ji.A) and store it in out->A
      {
        // The scaling matrix L's diagonal terms, needed to scale the
        // transformation to [-1,1]^3.
        const double di0 = 2/(tb[0].max-tb[0].min),
                     di1 = 2/(tb[1].max-tb[1].min),
                     di2 = 2/(tb[2].max-tb[2].min);

        // We finally construct the final transformation matrix A=L.Ji.A.
        // This maps a position relative to OBB center to a position in
        // [-1,1]^3, if the position is inside the OBB.
        out->A[0]=di0*( Ji[0]*A[0] + Ji[1]*A[3] ),
        out->A[1]=di0*( Ji[0]*A[1] + Ji[1]*A[4] ),
        out->A[2]=di0*( Ji[0]*A[2] + Ji[1]*A[5] ),
        out->A[3]=di1*( Ji[2]*A[0] + Ji[3]*A[3] ),
        out->A[4]=di1*( Ji[2]*A[1] + Ji[3]*A[4] ),
        out->A[5]=di1*( Ji[2]*A[2] + Ji[3]*A[5] ),
        out->A[6]=di2*A[6],
        out->A[7]=di2*A[7],
        out->A[8]=di2*A[8];
      }
    }
  }
  free(data);
}
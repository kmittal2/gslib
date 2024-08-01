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

static void bbox_1_tfm(double *out, const double x0[2], const double Ji[4],
                       const double *x, const double *y, unsigned n)
{
  unsigned i;
  for(i=0;i<n;++i) {
    const double dx = x[i]-x0[0], dy = y[i]-x0[1];
    out[  i] = Ji[0]*dx + Ji[1]*dy;
  }
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

//printf("global bounding box (%g^%u):\n",(double)p->hash_n,D);

void printit(const double *p, const int size, char *myString)
{
    printf("Printing %s\n",myString);
    for (int i = 0; i < size;)
    {
        for (int j = 0; j < 16 && i < size; j++)
        {
            printf("%g ",p[i]);
            i++;
        }
        printf("\n");
    }
}

void printit_obbox_dbl_range(const struct dbl_range *p, char *myString)
{
    printf("dbl_range %s: (%g %g)\n", myString, p->min, p->max );
}

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

//  printit(x, nr*nr, "x coordinates");
//  printit(y, nr*nr, "y coordinates");

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

//    printit(lob_bnd_data_r, nr*mr, "lob_bnd_data_r");
    
    for(;nel;--nel,x+=nrs,y+=nrs,++out) {
      double x0[2], J[4], Ji[4];
      struct dbl_range ab[2], tb[2];
  
      /* double work[2*nr] */
      x0[0] = tensor_ig2(J  , I0r,nr, I0s,ns, x, work);
      x0[1] = tensor_ig2(J+2, I0r,nr, I0s,ns, y, work);
      mat_inv_2(Ji, J);
//      printit(J, 4, "Jacobian");
//      printit(Ji, 4, "Jacobian inverse");


      /* double work[2*m##r]
         Find the bounds along a specific physical dimension, and merge with
         existing bounds if specified.
      */
      #define DO_BOUND(bnd,merge,r,x,work) do { \
        struct dbl_range b = \
        lob_bnd_1(lob_bnd_data_##r,n##r,m##r, x, work); \
        if(merge) bnd=dbl_range_merge(bnd,b); else bnd=b; \
      } while(0)

      /* double work[2*n##r + 2*m##r]
         Find the bounds for a edge, for all its physical dimensions, and merge 
         with existing bounds if specified.
         We will explain every line of the macro:
         lines 1,2: Bound x and y physical dimensions separately and store in
                    ab[0] and ab[1] respectively.
         lines 3  : Transform the AABB in physical dimensions to the reference
                    frame, and store in work.
         lines 4,5: Bound the transformed AABB and store its ref space bounds in
                    tb[0] and tb[1] respectively.
      */
      #define DO_EDGE(merge,r,x,y,work) do { \
        DO_BOUND(ab[0],merge,r,x,work); \
        DO_BOUND(ab[1],merge,r,y,work); \
        bbox_2_tfm(work, x0,Ji, x,y,n##r); \
        DO_BOUND(tb[0],merge,r,(work)     ,(work)+2*n##r); \
        DO_BOUND(tb[1],merge,r,(work)+n##r,(work)+2*n##r); \
      } while(0)

      // Bound edge whose x and y coords start from memory pointed by "x" and "y".
      // This is the bottom lexicographic edge of the element.
      DO_EDGE(0,r,x,y,work);
      // printit_obbox_dbl_range(&ab[0],"ab0-E0");
      // printit_obbox_dbl_range(&ab[1],"ab1");

      // Beyond this point, any bound calculation would be merged with the existing
      // bounds. So, 1 is passed as the first argument to DO_EDGE.

      // This is the top lexicographic edge of the element.
      DO_EDGE(1,r,&x[nrs-nr],&y[nrs-nr],work);
      // printit_obbox_dbl_range(&ab[0],"ab0-E1");
      // printit_obbox_dbl_range(&ab[1],"ab1");

      /* double work[4*ns + 2*ms]
         Helper macro to call DO_EDGE for the left and right lexicographic edges
         whose coords are not contiguous in memory.
      */
      #define GET_EDGE(off) do { \
        copy_strided(work   , x+off,1,nr,ns); \
        copy_strided(work+ns, y+off,1,nr,ns); \
        DO_EDGE(1,s,work,work+ns,work+2*ns); \
      } while(0)

      // This is the left lexicographic edge of the element.
      GET_EDGE(0);
      // printit_obbox_dbl_range(&ab[0],"ab0-E2");
      // printit_obbox_dbl_range(&ab[1],"ab1");

      // This is the right lexicographic edge of the element.
      GET_EDGE(nr-1);
      // printit_obbox_dbl_range(&ab[0],"ab0-E3");
      // printit_obbox_dbl_range(&ab[1],"ab1");
  
      #undef GET_EDGE
      #undef DO_EDGE
      #undef DO_BOUND

      // set bbox bounds based on aabb bounds expanded based on tol
      out->x[0] = dbl_range_expand(ab[0],tol),
      out->x[1] = dbl_range_expand(ab[1],tol);
//      printit_obbox_dbl_range(&out->x[0],"ab0-expanded");
//      printit_obbox_dbl_range(&out->x[1],"ab1-expanded");


//      printit_obbox_dbl_range(&tb[0],"tb0-final");
//      printit_obbox_dbl_range(&tb[1],"tb1-final");
  
      {
        // av0 and av1 are the ref space mid-points of the reference frame bounding box
        const double av0=(tb[0].min+tb[0].max)/2, av1=(tb[1].min+tb[1].max)/2;
        // Calculate OBBOX center in physical space
        out->c0[0] = x0[0] + J[0]*av0 + J[1]*av1;
        out->c0[1] = x0[1] + J[2]*av0 + J[3]*av1;
      }
      {
        // Expand ref space bounding box based on tol
        const double di0 = 2/((1+tol)*(tb[0].max-tb[0].min)),
                     di1 = 2/((1+tol)*(tb[1].max-tb[1].min));
        // The same factor of expansion is applied to the Jacobian matrix
        // to get the OBBOX transformation matrix.
        out->A[0]=di0*Ji[0], out->A[1]=di0*Ji[1];
        out->A[2]=di1*Ji[2], out->A[3]=di1*Ji[3];
      }

//      printit(out->c0, 2, "center");
//      printit(out->A, 4, "def Jac");
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

void printit_coords( const double *x, const double *y, const double *z,
                      const int npts,   const int dim,  char *myString )
{
  printf("Printing coords %s\n",myString);
  for (int i = 0; i < npts; i++)
  {
    if (dim==2) printf("(%g,%g) ",x[i],y[i]);
    if (dim==3) printf("(%g,%g,%g) ",x[i],y[i],z[i]);
    if (i%6==0 && i!=0) printf("\n");
  }
  printf("\n");
}

/* Calculates the diagonal length of the bounding box and expands its bounds by
 * 0.5*len*tol at both its min and max values.
 * Returns the length of the diagonal (could be used for expanding obboxes).
 */
double dblsurf_range_expand_2(struct dbl_range *m, struct dbl_range b[2], double tol, double len)
{
  // amount of expansion in each physical dimension beyond current obbox bounds
  if (len<1e-12) { // FIXME: 1e-12 is arbitrary
    double l[2] = { b[0].max-b[0].min, b[1].max-b[1].min };
    len = sqrt( l[0]*l[0] + l[1]*l[1] )*0.5*tol;
  }
  for (int i=0; i<2; i++) {
    m[i].min = b[i].min - len;
    m[i].max = b[i].max + len;
  }
  return len;
}

/* Calculates the diagonal length of the bounding box and expands its bounds by
 * 0.5*len*tol at both its min and max values.
 * Returns the length of the diagonal (could be used for expanding obboxes).
 */
double dblsurf_range_expand_3(struct dbl_range *m, struct dbl_range b[3], double tol, double len)
{
  // amount of expansion in each physical dimension beyond current obbox bounds
  if (len<1e-12) { // FIXME: 1e-12 is arbitrary
    double l[3] = { b[0].max-b[0].min, b[1].max-b[1].min, b[2].max-b[2].min };
    len = sqrt( l[0]*l[0] + l[1]*l[1] + l[2]*l[2] )*0.5*tol;
  }
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

    /*
    All the calculation in SETUP_DIR is done for the reference space. So the
    question arises: why r and s are treated separately?  This is because James
    assumed that the r and s directions could have different number of nodes.
    For us, that is not the case. So, we can just do this calculation in one
    direction and use everywhere.
    1. lag would store a pointer to a function of type lagrange_fun, pointing to
       the correct function to evaluate the lagrange polynomials on nr GLL nodes
    2. gll_lag_setup returns correct lag_coeff function and stores lag_coeffs
       (based on ref coords) in work for evaluating the lagrange polynomials.
    3. the coefficients in work are utilized to evaluate the lagrange polynomials and 
       their 1st derivatives at x=0 (or r=0) in I0##r.
    4. lob_bnd_setup assigns to lob_bnd_data_##r, which is used to compute bounds
    */
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
      double x0[2], A[4];            //x0: element center; A: the transformation from phy. to ref. space
      struct dbl_range ab[2], tb[2]; //ab: aabb bounds; tb: transformed aabb bounds

      x0[0] = tensor_ig1(A  ,I0r,nr,x); // A[0] = dx/dr, x0[0] = x(r=0), i.e., element center
      x0[1] = tensor_ig1(A+1,I0r,nr,y); // A[1] = dy/dr, x0[1] = y(r=0), i.e., element center
      A[2] = sqrt( A[0]*A[0] + A[1]*A[1] );
      A[0] = A[0]/A[2];
      A[1] = A[1]/A[2];
      A[2] = -A[1];
      A[3] =  A[0];

      // printit(x0, 2, "center x0");
      // printit(A,  4, "Transformation matrix A");
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

      /* double work[2*n##r + 2*m##r]
       * Find the bounds for a edge, for all its physical dimensions.
       * We will explain every line of the macro:
       * lines 1,2: Bound x and y physical dimensions separately and store in
       *            ab[0] and ab[1] respectively.
       * lines 3  : Transform the gll nodes according to A and store in work.
       * lines 4,5: Bound the transformed gll nodes and store the bounds of 
       *            the transformation in tb[0] and tb[1].
       */
      #define DO_EDGE(r,x,y,work) do { \
        DO_BOUND(ab[0],r,x,work); \
        DO_BOUND(ab[1],r,y,work); \
        bbox_2_tfm(work, x0,A, x,y,n##r); \
        DO_BOUND(tb[0],r,(work),(work)+2*n##r); \
        DO_BOUND(tb[1],r,(work)+n##r,(work)+2*n##r); \
      } while(0)
      // the first edge for nodes in lexicographic order
      DO_EDGE(r,x,y,work);
      #undef DO_EDGE
      #undef DO_BOUND

      double tol_len;
      tol_len = dblsurf_range_expand_2(out->x, ab, tol, 0.0);

      // av0 and av1 are the rotated AABB center coordinates and are hence
      // the offset of the OBB center with respect to the element center
      // in the rotated space.
      const double av0 = (tb[0].min+tb[0].max)/2,
                   av1 = (tb[1].min+tb[1].max)/2;
      // We now "un"rotate av0 and av1 to get the OBB center in the original
      // physical space.
      const double dx0 =  A[0]*av0 - A[1]*av1,
                   dx1 = -A[2]*av0 + A[3]*av1;
      // Calculate the true untranslated OBBOX center in physical space
      out->c0[0] = x0[0] + dx0;
      out->c0[1] = x0[1] + dx1;

      tol_len = dblsurf_range_expand_2(tb, tb, tol, 0.0);
      // printit_obbox_dbl_range(&tb[0],"tb0-expanded");
      // printit_obbox_dbl_range(&tb[1],"tb1-expanded");
      const double di0 = 2/(tb[0].max-tb[0].min),
                   di1 = 2/(tb[1].max-tb[1].min);
      // The scaling matrix is premultiplied to the rotation matrix
      out->A[0]=di0*A[0], out->A[1]=di0*A[1];
      out->A[2]=di1*A[2], out->A[3]=di1*A[3];

      // printit(out->c0, 2, "center out->c0");
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
    
    #define SETUP_DIR(r) do { \
      lagrange_fun *const lag = gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)
    SETUP_DIR(r);
    SETUP_DIR(s);
    #undef SETUP_DIR
    
    uint nelorig = nel;
    for(; nel; --nel,x+=nrs,y+=nrs,z+=nrs,++out) {
      struct dbl_range ab[3];
      struct dbl_range tb[3];
      double x0[3], tv[9], A[9];
 
      /* double work[2*nr]
       * Find the center of the element (r=0 ref. coord.) in physical space
       */
      x0[0] = tensor_ig2(tv+3, I0r,nr, I0s,ns, x, work);
      x0[1] = tensor_ig2(tv+5, I0r,nr, I0s,ns, y, work);
      x0[2] = tensor_ig2(tv+7, I0r,nr, I0s,ns, z, work);

      // tangent vector 1
      tv[0] = tv[3], tv[1] = tv[5], tv[2] = tv[7];
      // tangent vector 2
      tv[3] = tv[4], tv[4] = tv[6], tv[5] = tv[8];
      // normal vector to the plane formed by t1 and t2 (cross product)
      tv[6] = tv[1]*tv[5] - tv[2]*tv[4];
      tv[7] = tv[2]*tv[3] - tv[0]*tv[5];
      tv[8] = tv[0]*tv[4] - tv[1]*tv[3];
      // normalize the normal vector
      const double nmag  = sqrt( tv[6]*tv[6] + tv[7]*tv[7] + tv[8]*tv[8] );
      tv[6] = tv[6]/nmag;
      tv[7] = tv[7]/nmag;
      tv[8] = tv[8]/nmag;
      // At this point, we have tv = [tangent_r tangent_s normal]^T, only 
      // the normal vector is normalized.

      // Calculate the anticlockwise rotation theta_x about x-axis needed to
      // bring the normal vector into the xz plane, say theta_x.
      // Also calculate the anticlockwise rotation about y-axis needed to 
      // align the normal vector to the z-axis, say theta_y.
      if (fabs(tv[7])>1e-12) { // if normal vector is not parallel to xz plane
        const double magyz   = sqrt( tv[7]*tv[7] + tv[8]*tv[8] ); // n's projection on the yz plane
        const double cthetax = tv[8]/magyz, // cos(theta_x)
                     sthetax = tv[7]/magyz, // sin(theta_x)
                     cthetay = magyz,       // cos(theta_y)
                     sthetay = -tv[6];      // sin(theta_y)
        A[0] =  cthetay, A[1] = sthetax*sthetay, A[2] = cthetax*sthetay;
        A[3] =  0,       A[4] = cthetax,         A[5] = -sthetax;
        A[6] = -sthetay, A[7] = sthetax*cthetay, A[8] = cthetax*cthetay;
      }
      // if normal vector is already parallel to xz plane, no rotation about
      // x-axis is needed.
      else {
        const double cthetay = tv[8],   // cos(theta_y)
                     sthetay = -tv[6];  // sin(theta_y)
        // The rotation matrix that rotates the normal vector to the z-axis
        A[0] =  cthetay, A[1] = 0, A[2] = sthetay;
        A[3] =  0,       A[4] = 1, A[5] = 0;
        A[6] = -sthetay, A[7] = 0, A[8] = cthetay;
      }

      /* At this stage, the normal vector has been aligned with the z-axis. We
       * still need to align the element in xy-plane. For that, we use the
       * inverse of the Jacobian at the rotated element center to transform the
       * projection of the rotated element on xy-plane to the reference space
       * [-1,1]x[-1,1]. This transformation is premultiplied to A to get the
       * a new transformation matrix.
       */

      /* double work[2*m##r*(n##s+m##s+1)] */
      #define DO_BOUND(bnd,r,s,x,work) do { \
        bnd = lob_bnd_2(lob_bnd_data_##r,n##r,m##r, \
                        lob_bnd_data_##s,n##s,m##s, \
                        x, work); \
      } while(0)

      // Obtain the axis aligned bounds
      DO_BOUND(ab[0],r,s,x,work);
      DO_BOUND(ab[1],r,s,y,work);
      DO_BOUND(ab[2],r,s,z,work);
      // Expand the AA-bounds based on the tol, and save the expansion length 
      // in tol_len.
      double tol_len = dblsurf_range_expand_3(out->x, ab, tol, 0.0);

      double xtfm[3*nrs]; // xtfm[0]:x, xtfm[nrs]:y, xtfm[2*nrs]:z
      // Obtain the GLL nodes, when rotated by A and translated to (0,0).
      bbox_3_tfm(xtfm, x0,A, x,y,z,nrs);
      // The rotated z-coords are used to calculate z-bounds.
      DO_BOUND(tb[2],r,s,xtfm+2*nrs,work);
      // expand in z-direction by AABB expansion length tol_len
      tb[2].min -= tol_len;
      tb[2].max += tol_len;

      // Also apply A to the tangent vectors, which allows us to calculate 
      // the Jacobian matrix at the rotated element center.
      // NOTE that the z components of the rotated tangent vectors will become
      // zero, since the normal vector is aligned with the z-axis.
      double J[4], Ji[4];
      J[0] = A[0]*tv[0] + A[1]*tv[1] + A[2]*tv[2]; // rotated dx/dr
      J[1] = A[0]*tv[3] + A[1]*tv[4] + A[2]*tv[5]; // rotated dx/ds
      J[2] = A[3]*tv[0] + A[4]*tv[1] + A[5]*tv[2]; // rotated dy/dr
      J[3] = A[3]*tv[3] + A[4]*tv[4] + A[5]*tv[5]; // rotated dy/ds
      mat_inv_2(Ji, J);

      // Now transform the already rotated x,y coordinates according to Ji
      // to their reference space.
      for(unsigned i=0;i<nrs;++i) {
        const double xt = xtfm[i], yt = xtfm[nrs+i];
        xtfm[    i] = Ji[0]*xt + Ji[1]*yt;
        xtfm[nrs+i] = Ji[2]*xt + Ji[3]*yt;
      }
      // Bound these reference space xy coordinates, and expand the 
      // corresponding bounds.
      DO_BOUND(tb[0],r,s,xtfm    ,work);
      DO_BOUND(tb[1],r,s,xtfm+nrs,work);
      tol_len = dblsurf_range_expand_2(tb,tb,tol,0.0);
      #undef DO_BOUND

      // printit_obbox_dbl_range(&tb[0],"tb0-expanded");
      // printit_obbox_dbl_range(&tb[1],"tb1-expanded");
      // printit_obbox_dbl_range(&tb[2],"tb2-expanded");

      /* Calculate the center of the OBBOX obtained from Ji.A.x element, and 
       * store it in {av0,av1,av2}.
       * Then calculate Ainv.J.{av0,av1,av2} to obtain the physical position
       * vector of OBBOX center relative to element center.
       * This position vector is then added to the element center to obtain the
       * OBBOX center in physical space.
       */
      {
        const double av0 = (tb[0].min+tb[0].max)/2,
                     av1 = (tb[1].min+tb[1].max)/2,
                     av2 = (tb[2].min+tb[2].max)/2;
        const double Jav0 = J[0]*av0 + J[1]*av1,
                     Jav1 = J[2]*av0 + J[3]*av1;
        out->c0[0] = x0[0] + A[0]*Jav0 + A[3]*Jav1 + A[6]*av2,
        out->c0[1] = x0[1] + A[1]*Jav0 + A[4]*Jav1 + A[7]*av2,
        out->c0[2] = x0[2] + A[2]*Jav0 + A[5]*Jav1 + A[8]*av2;
      }

      // Finally, we premultiply the scaling matrix to Ji.A to obtain the final
      // transformation matrix.
      {

        const double di0 = 2/(tb[0].max-tb[0].min),
                     di1 = 2/(tb[1].max-tb[1].min),
                     di2 = 2/(tb[2].max-tb[2].min);

        // We finally construct the final transformation matrix A=L.Ji.A,
        // where L is the scaling matrix.
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

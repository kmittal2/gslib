#include <float.h>
#define obbox               GS_TOKEN_PASTE(obbox_             ,D)
#define obbox_calc          GS_TOKEN_PASTE(GS_PREFIXED_NAME(obbox_calc_),D)
#define obbox_test          GS_TOKEN_PASTE(obbox_test_        ,D)
#define hash_data           GS_TOKEN_PASTE(findpts_local_hash_data_,D)
#define hash_index          GS_TOKEN_PASTE(hash_index_        ,D)
#define hash_setfac         GS_TOKEN_PASTE(hash_setfac_       ,D)
#define hash_range          GS_TOKEN_PASTE(hash_range_        ,D)
#define hash_count          GS_TOKEN_PASTE(hash_count_        ,D)
#define hash_opt_size       GS_TOKEN_PASTE(hash_opt_size_,D)
#define hash_bb             GS_TOKEN_PASTE(hash_bb_           ,D)
#define hash_build          GS_TOKEN_PASTE(hash_build_        ,D)
#define hash_free           GS_TOKEN_PASTE(hash_free_         ,D)
#define findpts_el_data     GS_TOKEN_PASTE(findpts_el_data_   ,D)
#define findpts_el_pt       GS_TOKEN_PASTE(findpts_el_pt_     ,D)
#define findpts_el_setup    GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_el_setup_),D)
#define findpts_el_free     GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_el_free_ ),D)
#define findpts_el          GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_el_      ),D)
#define findpts_el_eval     GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_el_eval_ ),D)
#define findpts_el_start    GS_TOKEN_PASTE(findpts_el_start_  ,D)
#define findpts_el_points   GS_TOKEN_PASTE(findpts_el_points_ ,D)
#define findpts_local_data  GS_TOKEN_PASTE(findpts_local_data_,D)
#define map_points_to_els   GS_TOKEN_PASTE(map_points_to_els_ ,D)

#define findptsms_local_setup GS_TOKEN_PASTE(GS_PREFIXED_NAME(findptsms_local_setup_),D)
#define findptsms_local_free  GS_TOKEN_PASTE(GS_PREFIXED_NAME(findptsms_local_free_ ),D)
#define findptsms_local       GS_TOKEN_PASTE(GS_PREFIXED_NAME(findptsms_local_      ),D)
#define findptsms_local_eval  GS_TOKEN_PASTE(GS_PREFIXED_NAME(findptsms_local_eval_ ),D)

#define findpts_local_setup GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_local_setup_),D)
#define findpts_local_free  GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_local_free_ ),D)
#define findpts_local       GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_local_      ),D)
#define findpts_local_eval  GS_TOKEN_PASTE(GS_PREFIXED_NAME(findpts_local_eval_ ),D)
/*--------------------------------------------------------------------------
   Point to Possible Elements Hashing

   Initializing the data:
     uint nel;        // number of elements
     uint max_size = nr*ns*nt*nel; // maximum size of hash table
     struct obbox *obb = ...; // bounding boxes for elements

     hash_data data;
     hash_build(&data, obb, nel, max_size);

   Using the data:
     double x[3];   // point to find

     uint index = hash_index_3(&data, x);
     uint i, b = data.offset[index], e = data.offset[index+1];

     // point may be in elements
     //   data.offset[b], data.offset[b+1], ... , data.offset[e-1]
     //
     // list has maximum size data.max (e.g., e-b <= data.max)

     for(i=b; i!=e; ++i) {
       uint el = data.offset[i];
       ...
     }

   When done:
     hash_free(&data);

  --------------------------------------------------------------------------*/

struct hash_data {
  uint hash_n;
  struct dbl_range bnd[D];
  double fac[D];
  uint *offset;
  uint max;
};

static uint hash_index(const struct hash_data *p, const double x[D])
{
  const uint n = p->hash_n;
  return ( WHEN_3D( hash_index_aux(p->bnd[2].min,p->fac[2],n,x[2])  *n )
                   +hash_index_aux(p->bnd[1].min,p->fac[1],n,x[1]) )*n
                   +hash_index_aux(p->bnd[0].min,p->fac[0],n,x[0]);
}

static void hash_setfac(struct hash_data *p, const uint n)
{
  unsigned d;
  p->hash_n = n;
  for(d=0;d<D;++d) p->fac[d] = n/(p->bnd[d].max-p->bnd[d].min);
}

static struct uint_range hash_range(const struct hash_data *p, unsigned d,
                                    const struct dbl_range r)
{
  struct uint_range ir;
  const sint i0 = ifloor( (r.min - p->bnd[d].min) * p->fac[d] );
  const uint i1 = iceil ( (r.max - p->bnd[d].min) * p->fac[d] );
  ir.min = i0<0 ? 0 : i0;
  ir.max = i1<p->hash_n ? i1 : p->hash_n;
  if(ir.max==ir.min) ++ir.max;
  return ir;
}

static uint hash_count(struct hash_data *p,
                       const struct obbox *const obb, const uint nel,
                       const uint n)
{
  uint i,count=0;
  hash_setfac(p,n);
  for(i=0;i<nel;++i) {
    struct uint_range ir; uint ci; unsigned d;
      ir=hash_range(p,0,obb[i].x[0]); ci  = ir.max-ir.min;
    for(d=1;d<D;++d)
      ir=hash_range(p,d,obb[i].x[d]), ci *= ir.max-ir.min;
    count+=ci;
  }
  return count;
}

uint hash_opt_size(struct hash_data *p,
                   const struct obbox *const obb, const uint nel,
                   const uint max_size)
{
  uint nl=1, nu=ceil(pow(max_size-nel,1.0/D));
  uint size_low=2+nel;
  while(nu-nl>1) {
    uint nm = nl+(nu-nl)/2, nmd = nm*nm, size;
    WHEN_3D(nmd *= nm);
    size = nmd+1+hash_count(p,obb,nel,nm);
    if(size<=max_size) nl=nm,size_low=size; else nu=nm;
  }
  hash_setfac(p,nl);
  return size_low;
}

static void hash_bb(struct hash_data *p,
                    const struct obbox *const obb, const uint nel)
{
  uint el; unsigned d;
  struct dbl_range bnd[D];
  if(nel) {
    for(d=0;d<D;++d) bnd[d]=obb[0].x[d];
    for(el=1;el<nel;++el)
      for(d=0;d<D;++d)
        bnd[d]=dbl_range_merge(bnd[d],obb[el].x[d]);
    for(d=0;d<D;++d) p->bnd[d]=bnd[d];
  } else {
    for(d=0;d<D;++d) p->bnd[d].max=p->bnd[d].min=0;
  }
}

static void hash_build(struct hash_data *p,
                       const struct obbox *const obb, const uint nel,
                       const uint max_size)
{
  uint i,el,size,hn,hnd,sum,max, *count;
  hash_bb(p,obb,nel);
  size = hash_opt_size(p,obb,nel,max_size);
  p->offset = tmalloc(uint,size);
  hn = p->hash_n;
  hnd = hn*hn; WHEN_3D(hnd*=hn);
  count = tcalloc(uint,hnd);
  for(el=0;el<nel;++el) {
    unsigned d; struct uint_range ir[D];
    for(d=0;d<D;++d) ir[d]=hash_range(p,d,obb[el].x[d]);
    #define FOR_LOOP() do { uint i,j; WHEN_3D(uint k;) \
      WHEN_3D(for(k=ir[2].min;k<ir[2].max;++k)) \
              for(j=ir[1].min;j<ir[1].max;++j) \
              for(i=ir[0].min;i<ir[0].max;++i) \
                ++count[(WHEN_3D(k*hn)+j)*hn+i]; \
    } while(0)
    FOR_LOOP();
    #undef FOR_LOOP
  }
  sum=hnd+1, max=count[0];
  p->offset[0]=sum;
  for(i=0;i<hnd;++i) {
    max = count[i]>max?count[i]:max;
    sum += count[i];
    p->offset[i+1] = sum;
  }
  p->max = max;
  for(el=0;el<nel;++el) {
    unsigned d; struct uint_range ir[D];
    for(d=0;d<D;++d) ir[d]=hash_range(p,d,obb[el].x[d]);
    #define FOR_LOOP() do { uint i,j; WHEN_3D(uint k;) \
      WHEN_3D(for(k=ir[2].min;k<ir[2].max;++k)) \
              for(j=ir[1].min;j<ir[1].max;++j) \
              for(i=ir[0].min;i<ir[0].max;++i) { \
                uint index = (WHEN_3D(k*hn)+j)*hn+i; \
                p->offset[p->offset[index+1]-count[index]]=el; \
                --count[index]; \
              } \
    } while(0)
    FOR_LOOP();
    #undef FOR_LOOP
  }
  free(count);
}

static void hash_free(struct hash_data *p) { free(p->offset); }

struct findpts_local_data {
  unsigned ntot;
  const double *elx[D];
  const unsigned *nsid;
  struct obbox *obb;
  struct hash_data hd;
  struct findpts_el_data fed;
  double tol;
  double *distrsti;
  const double *distfint;
  uint ims;
};

void findptsms_local_setup(struct findpts_local_data *const fd,
                         const double *const elx[D],
                         const unsigned *const nsid,
                         const double *const distfint,
                         const unsigned n[D], const uint nel,
                         const unsigned m[D], const double bbox_tol,
                         const uint max_hash_size,
                         const unsigned npt_max, const double newt_tol,const uint ims)
{
  unsigned d;
  unsigned ntot=n[0]; for(d=1;d<D;++d) ntot*=n[d];
  fd->ntot = ntot;
  for(d=0;d<D;++d) fd->elx[d]=elx[d];
  fd->nsid = nsid;
  fd->obb=tmalloc(struct obbox,nel);
  obbox_calc(fd->obb,elx,n,nel,m,bbox_tol);
  hash_build(&fd->hd,fd->obb,nel,max_hash_size);
  findpts_el_setup(&fd->fed,n,npt_max);
  fd->tol = newt_tol;
  fd->ims = ims;
  if (fd->ims==1) {
   fd->distrsti = tmalloc(double, npt_max);
   fd->distfint = distfint;
  }
}

void findptsms_local_free(struct findpts_local_data *const fd)
{
  findpts_el_free(&fd->fed);
  hash_free(&fd->hd);
  free(fd->obb);
  if (fd->ims==1) {
   free(fd->distrsti);
  }
}

#define   AT(T,var,i)   \
        (T*)(      (char*)var##_base   +(i)*var##_stride   )
#define  CAT(T,var,i) \
  (const T*)((const char*)var##_base   +(i)*var##_stride   )
#define CATD(T,var,i,d) \
  (const T*)((const char*)var##_base[d]+(i)*var##_stride[d])

static void map_points_to_els(
  struct array *const               map,
        uint   *const         code_base, const unsigned       code_stride,
  const double *const         x_base[D], const unsigned       x_stride[D],
  const uint   *const   session_id_base, const unsigned session_id_stride,
  const uint   *const  session_id_match, const uint                   npt, 
  const struct findpts_local_data *const fd, buffer *buf)
{
  uint index;
  const double *xp[D]; uint *code=code_base;
  unsigned d; for(d=0;d<D;++d) xp[d]=x_base[d];
  array_init(struct index_el,map,npt+(npt>>2)+1);
  uint sessm = *(session_id_match);

  const uint *sess_id; sess_id = session_id_base;
  for(index=0;index<npt;++index) {
    double x[D]; for(d=0;d<D;++d) x[d]=*xp[d];
    { const uint hi = hash_index(&fd->hd,x);
      const uint       *elp = fd->hd.offset + fd->hd.offset[hi  ],
                 *const ele = fd->hd.offset + fd->hd.offset[hi+1];
      *code = CODE_NOT_FOUND;
      for(; elp!=ele; ++elp) {
        const uint el = *elp;
        if (fd->ims==1 && sessm!=1 && *(fd->nsid) == *sess_id) continue;
        if (fd->ims==1 && sessm==1 && *(fd->nsid) != *sess_id) continue;
        if(obbox_test(&fd->obb[el],x)>=0) {
          struct index_el *const p =
            array_reserve(struct index_el,map,map->n+1);
          p[map->n].index = index;
          p[map->n].el = el;
          ++map->n;
        }
      }
    }
    for(d=0;d<D;++d)
    xp[d]    =(const double*)((const char*)xp[d]  +      x_stride[d]);
    code     =        (uint*)(      (char*)code   +      code_stride);
    sess_id  =(const uint*  )((const char*)sess_id+session_id_stride);
  }
  /* group by element */
  sarray_sort(struct index_el,map->ptr,map->n, el,0, buf);
  /* add sentinel */
  {
    struct index_el *const p =
      array_reserve(struct index_el,map,map->n+1);
    p[map->n].el = -(uint)1;
  }
}

void findptsms_local(
        uint   *const        code_base, const unsigned       code_stride,
        uint   *const          el_base, const unsigned         el_stride,
        double *const           r_base, const unsigned          r_stride,
        double *const       dist2_base, const unsigned      dist2_stride,
  const double *const        x_base[D], const unsigned       x_stride[D],
  const uint   *const  session_id_base, const unsigned session_id_stride,
        double *const       disti_base, const unsigned      disti_stride,
        uint   *const       elsid_base, const unsigned      elsid_stride,
  const uint   *const session_id_match, const uint                   npt, 
   struct findpts_local_data *const fd,  buffer *buf)
{
  struct findpts_el_data *const fed = &fd->fed;
  struct findpts_el_pt *const fpt = findpts_el_points(fed);
  struct array map; /* point -> element map */
  int rsid_stride = 1;
  map_points_to_els(&map, code_base,code_stride, x_base,x_stride, session_id_base, session_id_stride,session_id_match,npt, fd, buf);
  {
    const unsigned npt_max = fd->fed.npt_max;
    const struct index_el *p, *const pe = (struct index_el *)map.ptr+map.n;
    for(p=map.ptr;p!=pe;) {
      const uint el = p->el, el_off=el*fd->ntot;
      const double *elx[D];
      unsigned d;
      for(d=0;d<D;++d) elx[d]=fd->elx[d]+el_off;

      findpts_el_start(fed,elx);
      do {
        const struct index_el *q;
        unsigned i;
        for(i=0,q=p;i<npt_max && q->el==el;++q) {
          uint *code = AT(uint,code,q->index);
          if(*code==CODE_INTERNAL) continue;
          for(d=0;d<D;++d) fpt[i].x[d]=*CATD(double,x,q->index,d);
          ++i;
        }
        findpts_el(fed,i,fd->tol);
        if (fd->ims==1) {
           findpts_el_eval(fd->distrsti, sizeof(double),
                             &fpt[0].r[0], sizeof(struct findpts_el_pt), i,
                             fd->distfint+el*fd->ntot  ,fed);
         }

        for(i=0,q=p;i<npt_max && q->el==el;++q) {
          const uint index=q->index;
          uint *code = AT(uint,code,index);
          double *dist2 = AT(double,dist2,index);
          double *disti = (fd->ims) ? AT(double,disti,index) : disti_base;

          if(*code==CODE_INTERNAL) continue;
          if(*code==CODE_NOT_FOUND
          || fpt[i].flags==(1u<<(2*D)) /* converged, no constraints */
          || fpt[i].dist2<*dist2) {
            double *r = AT(double,r,index);
            uint *eli = AT(uint,el,index);
            uint *elsid = AT(uint,elsid,index);
            *eli   = el;    
            *code  = fpt[i].flags==(1u<<(2*D)) ? CODE_INTERNAL : CODE_BORDER;
            *dist2 = fpt[i].dist2;   
            *disti = (fd->ims==1) ? fd->distrsti[i] : 0.;
            *elsid = (fd->ims==1) ? fd->nsid[0]     : 0 ;
	    for(d=0;d<D;++d) r[d]=fpt[i].r[d];
          }
          ++i;
        }
        p=q;
      } while(p->el==el);
    }
  }
  array_free(&map);
}

/* assumes points are already grouped by elements */
void findptsms_local_eval(
        double *const out_base, const unsigned out_stride,
  const uint   *const  el_base, const unsigned  el_stride,
  const double *const   r_base, const unsigned   r_stride,
  const uint npt,
  const double *const in, struct findpts_local_data *const fd)
{
  struct findpts_el_data *const fed = &fd->fed;
  const unsigned npt_max = fed->npt_max;
  uint p;
  for(p=0;p<npt;) {
    const uint el = *CAT(uint,el,p);
    const double *const in_el = in+el*fd->ntot;
    do {
      unsigned i; uint q;
      for(i=0,q=p;i<npt_max && q<npt && *CAT(uint,el,q)==el;++q) ++i;
      findpts_el_eval( AT(double,out,p),out_stride,
                      CAT(double,  r,p),  r_stride, i,
                      in_el,fed);
      p=q;
    } while(p<npt && *CAT(uint,el,p)==el);
  }
}

void findpts_local_setup(struct findpts_local_data *const fd,
                         const double *const elx[D],
                         const unsigned n[D], const uint nel,
                         const unsigned m[D], const double bbox_tol,
                         const uint max_hash_size,
                         const unsigned npt_max, const double newt_tol)
{
  uint ims=0;
  unsigned int nsid = 0;
  double distfint = 0.;
  findptsms_local_setup(fd,
                        elx,
                       &nsid,
                       &distfint,
                        n,nel,
                        m,bbox_tol,
                        max_hash_size,
                        npt_max,newt_tol,ims);
}

void findpts_local_free(struct findpts_local_data *const fd)
{
  findptsms_local_free(fd);
}

void findpts_local(
        uint   *const  code_base   , const unsigned  code_stride   ,
        uint   *const    el_base   , const unsigned    el_stride   ,
        double *const     r_base   , const unsigned     r_stride   ,
        double *const dist2_base   , const unsigned dist2_stride   ,
  const double *const     x_base[D], const unsigned     x_stride[D],
  const uint npt, struct findpts_local_data *const fd,
  buffer *buf)
{
    unsigned int *sess_base = tmalloc(uint,1);
    double *disti_base = tmalloc(double,1);
    unsigned int *elsid_base = tmalloc(uint,1);
    unsigned int *sess_match = tmalloc(uint,1);
    *sess_base = 0;
    *sess_match = 0;
    *disti_base = 0;
    *elsid_base = 0;
    

    unsigned sess_stride=0;
    unsigned disti_stride=0;
    unsigned elsid_stride=0;
  findptsms_local(
         code_base, code_stride,
           el_base,   el_stride,
            r_base,    r_stride,
        dist2_base,dist2_stride,
            x_base,    x_stride,
         sess_base, sess_stride,
        disti_base,disti_stride,
        elsid_base,elsid_stride,
        sess_match,npt,fd,buf);
}

/* assumes points are already grouped by elements */
void findpts_local_eval(
        double *const out_base, const unsigned out_stride,
  const uint   *const  el_base, const unsigned  el_stride,
  const double *const   r_base, const unsigned   r_stride,
  const uint npt,
  const double *const in, struct findpts_local_data *const fd)
{
  findptsms_local_eval(
  out_base,out_stride,
   el_base,el_stride,
    r_base,r_stride,
       npt,
        in,fd);
}

#undef CATD
#undef CAT
#undef AT

#undef findptsms_local_eval
#undef findptsms_local
#undef findptsms_local_free
#undef findptsms_local_setup
#undef map_points_to_els
#undef findpts_local_data
#undef findpts_el_points
#undef findpts_el_start
#undef findpts_el_eval
#undef findpts_el
#undef findpts_el_free
#undef findpts_el_setup
#undef findpts_el_data
#undef hash_free
#undef hash_build
#undef hash_bb
#undef hash_opt_size
#undef hash_count
#undef hash_range
#undef hash_setfac
#undef hash_index
#undef hash_data
#undef obbox_test
#undef obbox_calc
#undef obbox

#undef findpts_local
#undef findpts_local_free
#undef findpts_local_setup
#undef findpts_local_eval

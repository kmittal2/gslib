#include <float.h>
#define obbox               GS_TOKEN_PASTE(obbox_             ,D)
#define obbox_calc          GS_TOKEN_PASTE(GS_PREFIXED_NAME(obbox_calc_),D)
#define obbox_test          GS_TOKEN_PASTE(obbox_test_        ,D)
#define hash_data           GS_TOKEN_PASTE(findpts_local_hash_data_,D)
#define local_hash_index          GS_TOKEN_PASTE(local_hash_index_        ,D)
#define hash_setfac         GS_TOKEN_PASTE(hash_setfac_       ,D)
#define hash_range          GS_TOKEN_PASTE(hash_range_        ,D)
#define hash_count          GS_TOKEN_PASTE(hash_count_        ,D)
#define hash_opt_size       GS_TOKEN_PASTE(hash_opt_size_,D)
#define hash_bb             GS_TOKEN_PASTE(hash_bb_           ,D)
#define local_hash_build          GS_TOKEN_PASTE(local_hash_build_        ,D)
#define local_hash_free           GS_TOKEN_PASTE(local_hash_free_         ,D)
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
  local_hash_build(&fd->hd,fd->obb,nel,max_hash_size);
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
  local_hash_free(&fd->hd);
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
    { const uint hi = local_hash_index(&fd->hd,x);
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
#undef local_hash_free
#undef local_hash_build
#undef hash_bb
#undef hash_opt_size
#undef hash_count
#undef hash_range
#undef hash_setfac
#undef local_hash_index
#undef hash_data
#undef obbox_test
#undef obbox_calc
#undef obbox

#undef findpts_local
#undef findpts_local_free
#undef findpts_local_setup
#undef findpts_local_eval

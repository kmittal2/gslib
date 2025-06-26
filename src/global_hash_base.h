#define obbox               GS_TOKEN_PASTE(obbox_             ,D)
#define local_hash_data GS_TOKEN_PASTE(findpts_local_hash_data_,D)
#define hash_data       GS_TOKEN_PASTE(findpts_hash_data_,D)
#define hash_index      GS_TOKEN_PASTE(hash_index_       ,D)
#define hash_setfac     GS_TOKEN_PASTE(hash_setfac_      ,D)
#define hash_range      GS_TOKEN_PASTE(hash_range_       ,D)
#define hash_bb         GS_TOKEN_PASTE(hash_bb_          ,D)
#define set_local_mask  GS_TOKEN_PASTE(set_local_mask_   ,D)
#define fill_hash       GS_TOKEN_PASTE(fill_hash_        ,D)
#define table_from_hash GS_TOKEN_PASTE(table_from_hash_  ,D)
#define hash_build      GS_TOKEN_PASTE(hash_build_       ,D)
#define hash_free       GS_TOKEN_PASTE(hash_free_        ,D)
#define hash_build_nx      GS_TOKEN_PASTE(hash_build_nx_       ,D)

struct hash_data {
  ulong hash_n;
  struct dbl_range bnd[D];
  double fac[D];
  uint *offset;
};

ulong hash_index(const struct hash_data *p, const double x[D])
{
  const ulong n = p->hash_n;
  return ( WHEN_3D( hash_index_aux(p->bnd[2].min,p->fac[2],n,x[2])  *n )
                   +hash_index_aux(p->bnd[1].min,p->fac[1],n,x[1]) )*n
                   +hash_index_aux(p->bnd[0].min,p->fac[0],n,x[0]);
}

void hash_setfac(struct hash_data *p, const ulong n)
{
  unsigned d;
  p->hash_n = n;
  for(d=0;d<D;++d) p->fac[d] = n/(p->bnd[d].max-p->bnd[d].min);
}

struct ulong_range hash_range(const struct hash_data *p, unsigned d,
                                     const struct dbl_range r)
{
  struct ulong_range ir;
  const slong i0 = lfloor( (r.min - p->bnd[d].min) * p->fac[d] );
  const ulong i1 = lceil ( (r.max - p->bnd[d].min) * p->fac[d] );
  ir.min = i0<0 ? 0 : i0;
  ir.max = i1<p->hash_n ? i1 : p->hash_n;
  if(ir.max==ir.min) ++ir.max;
  return ir;
}

void hash_bb(struct hash_data *p, const struct local_hash_data *lp,
                    const struct comm *comm, uint hash_size)
{
  double x[D], buf[D], ghs;
  unsigned d;
  for(d=0;d<D;++d) x[d]=lp->bnd[d].min;
  comm_allreduce(comm,gs_double,gs_min,x,D,buf);
  for(d=0;d<D;++d) p->bnd[d].min=x[d];

  for(d=0;d<D;++d) x[d]=lp->bnd[d].max;
  comm_allreduce(comm,gs_double,gs_max,x,D,buf);
  for(d=0;d<D;++d) p->bnd[d].max=x[d];

  ghs = hash_size; comm_allreduce(comm,gs_double,gs_add,&ghs,1,buf);
  hash_setfac(p,lceil(pow(ghs,1./D)));

  #ifdef DIAGNOSTICS
  if(comm->id==0) {
    printf("global bounding box (%g^%u):\n",(double)p->hash_n,D);
    for(d=0;d<D;++d) printf("  [%.17g, %.17g]\n",p->bnd[d].min,p->bnd[d].max);
  }
  #endif
}

void set_local_mask(unsigned char *const local_mask,
                           const ulong local_base[D], const uint local_n[D],
                           const struct hash_data *const p,
                           const struct obbox *const obb, const uint nel
                          )
{
  uint el;
  for(el=0;el<nel;++el) {
    struct ulong_range ir[D]; unsigned d;
    for(d=0;d<D;++d) ir[d]=hash_range(p,d,obb[el].x[d]);
    #define FOR_LOOP() do { ulong i,j; WHEN_3D(ulong k;) \
      WHEN_3D(for(k=ir[2].min;k<ir[2].max;++k)) \
              for(j=ir[1].min;j<ir[1].max;++j) \
              for(i=ir[0].min;i<ir[0].max;++i) \
                set_bit(local_mask, (WHEN_3D((k-local_base[2]) *local_n[1]) \
                                            +(j-local_base[1]))*local_n[0] \
                                            +(i-local_base[0]) \
                       ); \
    } while(0)
    FOR_LOOP();
    #undef FOR_LOOP
  }
}

void fill_hash(struct array *const hash,
                      const unsigned char *const local_mask,
                      const ulong local_base[D], const uint local_n[D],
                      const ulong hn, const uint np)
{
  struct proc_index *hp = hash->ptr;
  #define FOR_LOOP() do { uint bit=0,i,j; WHEN_3D(uint k;) \
    WHEN_3D(for(k=0;k<local_n[2];++k)) \
            for(j=0;j<local_n[1];++j) \
            for(i=0;i<local_n[0];++i) { ulong hi; \
              if(get_bit(local_mask,bit++)==0) continue; \
              hi = (WHEN_3D( (local_base[2]+k) *hn ) \
                            +(local_base[1]+j))*hn \
                            +(local_base[0]+i); \
              hp->proc = hi%np, hp->index = hi/np; \
              ++hp; \
            } \
  } while(0)
  FOR_LOOP();
  #undef FOR_LOOP
}

void table_from_hash(struct hash_data *const p,
                            struct array *const hash,
                            const uint np, buffer *buf)
{
  const ulong hn = p->hash_n;
  ulong hnd;
  uint ncell, *offset, i, next_cell;
  const struct proc_index *const hp = hash->ptr;
  const uint n = hash->n;
  hnd = hn*hn; WHEN_3D(hnd*=hn);
  ncell = (hnd-1)/np+1;
  p->offset = offset = tmalloc(uint,ncell+1+n);
  sarray_sort(struct proc_index,hash->ptr,n, index,0, buf);
  next_cell = 0;
  for(i=0;i<n;++i) {
    const uint cell = hp[i].index;
    const uint off = ncell+1+i;
    offset[off]=hp[i].proc;
    while(next_cell<=cell ) offset[next_cell++]=off;
  }
  { const uint off = ncell+1+i;
    while(next_cell<=ncell) offset[next_cell++]=off;
  }
}

void hash_build(struct hash_data *const p,
                       const struct local_hash_data *const lp,
                       const struct obbox *const obb, const uint nel,
                       const uint hash_size,
                       struct crystal *cr)
{
  ulong local_base[D]; uint local_n[D], local_ntot=1;
  unsigned char *local_mask;
  struct array hash; uint nc;
  unsigned d;
  hash_bb(p,lp,&cr->comm,hash_size);
  for(d=0;d<D;++d) {
    struct ulong_range rng=hash_range(p,d,lp->bnd[d]);
    local_base[d]=rng.min;
    local_n[d]=rng.max-rng.min;
    local_ntot*=local_n[d];
    #ifdef DIAGNOSTICS
    if(cr->comm.id==0) {
      printf("local_range %u: %lu to %lu\n",
             d,(unsigned long)rng.min,(unsigned long)rng.max);
    }
    #endif
  }
  local_mask = tcalloc(unsigned char, (local_ntot+CHAR_BIT-1)/CHAR_BIT);
  set_local_mask(local_mask,local_base,local_n,p,obb,nel);
  nc=count_bits(local_mask,(local_ntot+CHAR_BIT-1)/CHAR_BIT);
  #ifdef DIAGNOSTICS
  printf("findpts_hash(%u): local cells : %u / %u\n",cr->comm.id,nc,local_ntot);
  #endif
  array_init(struct proc_index,&hash,nc), hash.n=nc;
  fill_hash(&hash,local_mask,local_base,local_n,p->hash_n,cr->comm.np);
  free(local_mask);
  sarray_transfer(struct proc_index,&hash,proc,1,cr);
  table_from_hash(p,&hash,cr->comm.np,&cr->data);
  array_free(&hash);
}

void hash_build_nx(struct hash_data *const p,
                   const struct local_hash_data *const lp,
                   const struct obbox *const obb, const uint nel,
                   const uint nm,
                   struct crystal *cr)
{
  ulong local_base[D]; uint local_n[D], local_ntot=1;
  unsigned char *local_mask;
  struct array hash; uint nc;
  unsigned d;
  {
    double x[D], buf[D], ghs;
    unsigned d;
    for(d=0;d<D;++d) x[d]=lp->bnd[d].min;
    comm_allreduce(&cr->comm,gs_double,gs_min,x,D,buf);
    for(d=0;d<D;++d) p->bnd[d].min=x[d];

    for(d=0;d<D;++d) x[d]=lp->bnd[d].max;
    comm_allreduce(&cr->comm,gs_double,gs_max,x,D,buf);
    for(d=0;d<D;++d) p->bnd[d].max=x[d];

    ghs = nm; comm_allreduce(&cr->comm,gs_double,gs_max,&ghs,1,buf);
    hash_setfac(p,ghs);
  }
  for(d=0;d<D;++d) {
    struct ulong_range rng=hash_range(p,d,lp->bnd[d]);
    local_base[d]=rng.min;
    local_n[d]=rng.max-rng.min;
    local_ntot*=local_n[d];
    #ifdef DIAGNOSTICS
    if(cr->comm.id==0) {
      printf("local_range %u: %lu to %lu\n",
             d,(unsigned long)rng.min,(unsigned long)rng.max);
    }
    #endif
  }
  local_mask = tcalloc(unsigned char, (local_ntot+CHAR_BIT-1)/CHAR_BIT);
  set_local_mask(local_mask,local_base,local_n,p,obb,nel);
  nc=count_bits(local_mask,(local_ntot+CHAR_BIT-1)/CHAR_BIT);
  #ifdef DIAGNOSTICS
  printf("findpts_hash(%u): local cells : %u / %u\n",cr->comm.id,nc,local_ntot);
  #endif
  array_init(struct proc_index,&hash,nc), hash.n=nc;
  fill_hash(&hash,local_mask,local_base,local_n,p->hash_n,cr->comm.np);
  free(local_mask);
  sarray_transfer(struct proc_index,&hash,proc,1,cr);
  table_from_hash(p,&hash,cr->comm.np,&cr->data);
  array_free(&hash);
}

void hash_free(struct hash_data *p) { free(p->offset); }


#undef hash_free
#undef hash_build_nx
#undef hash_build
#undef table_from_hash
#undef fill_hash
#undef set_local_mask
#undef hash_bb
#undef hash_range
#undef hash_setfac
#undef hash_index
#undef hash_data
#undef local_hash_data
#undef obbox
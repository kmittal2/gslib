#define obbox                     GS_TOKEN_PASTE(obbox_             ,D)
#define local_hash_data           GS_TOKEN_PASTE(findpts_local_hash_data_,D)
#define local_hash_index          GS_TOKEN_PASTE(local_hash_index_        ,D)
#define local_hash_setfac         GS_TOKEN_PASTE(local_hash_setfac_       ,D)
#define local_hash_range          GS_TOKEN_PASTE(local_hash_range_        ,D)
#define local_hash_count          GS_TOKEN_PASTE(local_hash_count_        ,D)
#define local_hash_opt_size       GS_TOKEN_PASTE(local_hash_opt_size_,D)
#define local_hash_bb             GS_TOKEN_PASTE(local_hash_bb_           ,D)
#define local_hash_build          GS_TOKEN_PASTE(local_hash_build_        ,D)
#define local_hash_free           GS_TOKEN_PASTE(local_hash_free_         ,D)
#define local_hash_build_nx       GS_TOKEN_PASTE(local_hash_build_nx_       ,D)

struct local_hash_data {
  uint local_hash_n;
  struct dbl_range bnd[D];
  double fac[D];
  uint *offset;
  uint max;
};

uint local_hash_index(const struct local_hash_data *p, const double x[D])
{
  const uint n = p->local_hash_n;
  return ( WHEN_3D( local_hash_index_aux(p->bnd[2].min,p->fac[2],n,x[2])  *n )
                    +local_hash_index_aux(p->bnd[1].min,p->fac[1],n,x[1]) )*n
                    +local_hash_index_aux(p->bnd[0].min,p->fac[0],n,x[0]);
}

void local_hash_setfac(struct local_hash_data *p, const uint n)
{
  unsigned d;
  p->local_hash_n = n;
  for(d=0;d<D;++d) p->fac[d] = n/(p->bnd[d].max-p->bnd[d].min);
}

struct uint_range local_hash_range(const struct local_hash_data *p, unsigned d,
                                    const struct dbl_range r)
{
  struct uint_range ir;
  const sint i0 = ifloor( (r.min - p->bnd[d].min) * p->fac[d] );
  const uint i1 = iceil ( (r.max - p->bnd[d].min) * p->fac[d] );
  ir.min = i0<0 ? 0 : i0;
  ir.max = i1<p->local_hash_n ? i1 : p->local_hash_n;
  if(ir.max==ir.min) ++ir.max;
  return ir;
}

uint local_hash_count(struct local_hash_data *p,
                const struct obbox *const obb, const uint nel,
                const uint n)
{
  uint i,count=0;
  local_hash_setfac(p,n);
  for(i=0;i<nel;++i) {
    struct uint_range ir; uint ci; unsigned d;
      ir=local_hash_range(p,0,obb[i].x[0]); ci  = ir.max-ir.min;
    for(d=1;d<D;++d)
      ir=local_hash_range(p,d,obb[i].x[d]), ci *= ir.max-ir.min;
    count+=ci;
  }
  return count;
}

uint local_hash_opt_size(struct local_hash_data *p,
                    const struct obbox *const obb, const uint nel,
                    const uint max_size)
{
  uint nl=1, nu=ceil(pow(max_size-nel,1.0/D));
  uint size_low=2+nel;
  while(nu-nl>1) {
    uint nm = nl+(nu-nl)/2, nmd = nm*nm, size;
    WHEN_3D(nmd *= nm);
    size = nmd+1+local_hash_count(p,obb,nel,nm);
    if(size<=max_size) nl=nm,size_low=size; else nu=nm;
  }
  local_hash_setfac(p,nl);
  return size_low;
}

void local_hash_bb(struct local_hash_data *p,
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

void local_hash_build(struct local_hash_data *p,
                        const struct obbox *const obb, const uint nel,
                        const uint max_size)
{
  uint i,el,size,hn,hnd,sum,max, *count;
  local_hash_bb(p,obb,nel);
  size = local_hash_opt_size(p,obb,nel,max_size);
  p->offset = tmalloc(uint,size);
  hn = p->local_hash_n;
  hnd = hn*hn; WHEN_3D(hnd*=hn);
  count = tcalloc(uint,hnd);
  for(el=0;el<nel;++el) {
    unsigned d; struct uint_range ir[D];
    for(d=0;d<D;++d) ir[d]=local_hash_range(p,d,obb[el].x[d]);
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
    for(d=0;d<D;++d) ir[d]=local_hash_range(p,d,obb[el].x[d]);
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

void local_hash_build_nx(struct local_hash_data *p,
                         const struct obbox *const obb, const uint nel,
                         const uint nm)
{
  uint i,el,size,hn,hnd,sum,max, *count;
  local_hash_bb(p,obb,nel);

  uint nmd = nm*nm;
  WHEN_3D(nmd *= nm);
  size = nmd+1+local_hash_count(p,obb,nel,nm);
  local_hash_setfac(p,nm);
  p->offset = tmalloc(uint,size);
  hn = p->local_hash_n;
  hnd = hn*hn; WHEN_3D(hnd*=hn);
  count = tcalloc(uint,hnd);
  for(el=0;el<nel;++el) {
    unsigned d; struct uint_range ir[D];
    for(d=0;d<D;++d) ir[d]=local_hash_range(p,d,obb[el].x[d]);
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
    for(d=0;d<D;++d) ir[d]=local_hash_range(p,d,obb[el].x[d]);
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

void local_hash_free(struct local_hash_data *p) { free(p->offset); }

#undef obbox
#undef local_hash_free
#undef local_hash_build
#undef local_hash_build_nx
#undef local_hash_bb
#undef local_hash_opt_size
#undef local_hash_count
#undef local_hash_range
#undef local_hash_setfac
#undef local_hash_index
#undef local_hash_data
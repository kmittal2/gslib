#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"

#define gs_op gs_op_t   /* fix conflict with fortran */

#include "gs_defs.h"
#include "gs_local.h"
#include "comm.h"
#include "mem.h"
#include "sort.h"
#include "crystal.h"
#include "sarray_sort.h"
#include "sarray_transfer.h"

#define gs         GS_PREFIXED_NAME(gs       )
#define gs_vec     GS_PREFIXED_NAME(gs_vec   )
#define gs_many    GS_PREFIXED_NAME(gs_many  )
#define igs        GS_PREFIXED_NAME(igs      )
#define igs_vec    GS_PREFIXED_NAME(igs_vec  )
#define igs_many   GS_PREFIXED_NAME(igs_many )
#define gs_wait    GS_PREFIXED_NAME(gs_wait  )
#define gs_setup   GS_PREFIXED_NAME(gs_setup )
#define gs_free    GS_PREFIXED_NAME(gs_free  )
#define gs_unique  GS_PREFIXED_NAME(gs_unique)
#define gs_hf2c    GS_PREFIXED_NAME(gs_hf2c  )
#define pw_data_nmsg GS_PREFIXED_NAME(pw_data_nmsg )
#define pw_data_size GS_PREFIXED_NAME(pw_data_size )

GS_DEFINE_DOM_SIZES()

typedef enum { mode_plain, mode_vec, mode_many,
               mode_dry_run } gs_mode;

static buffer static_buffer = null_buffer;

static void gather_noop(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op)
{}

static void scatter_noop(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom)
{}

static void init_noop(
  void *out, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op)
{}

/*------------------------------------------------------------------------------
  Topology Discovery
------------------------------------------------------------------------------*/

struct gs_topology {
  ulong total_shared; /* number of globally unique shared ids */
  struct array nz; /* array of nonzero_id's, grouped by id,
                      sorted by primary index, then flag, then index */
  struct array sh; /* array of shared_id's, arbitrary ordering */
  struct array pr; /* array of primary_shared_id's */
};

static void gs_topology_free(struct gs_topology *top)
{
  array_free(&top->pr);
  array_free(&top->sh);
  array_free(&top->nz);
}

/************** Local topology **************/

/* nonzero_ids    (local part)

   Creates an array of s_nonzeros, one per nonzero in user id array. The
   output array is grouped by id. Within each group, non-flagged entries come
   first; otherwise the entries within the group are sorted by the index into
   the user id array. The first index in each group is the primary index, and
   is stored along with each entry. The groups themselves are ordered in
   increasing order of the primary index associated with the group (as opposed
   to the user id). */

struct nonzero_id {
  ulong id; uint i, flag, primary;
};

static void nonzero_ids(struct array *nz,
                        const slong *id, const uint n, buffer *buf)
{
  ulong last_id = ULONG_MAX;
  uint i, primary = UINT_MAX;
  struct nonzero_id *row, *end;
  array_init(struct nonzero_id,nz,n), end=row=nz->ptr;
  for(i=0;i<n;++i) {
    slong id_i = id[i], abs_id = iabsl(id_i);
    if(id_i==0) continue;
    end->i = i;
    end->id = abs_id;
    end->flag = id_i!=abs_id;
    ++end;
  }
  nz->n = end-row;
  array_resize(struct nonzero_id,nz,nz->n);
  sarray_sort_2(struct nonzero_id,nz->ptr,nz->n, id,1, flag,0, buf);
  for(row=nz->ptr,end=row+nz->n;row!=end;++row) {
    ulong this_id = row->id;
    if(this_id!=last_id) primary = row->i;
    row->primary = primary;
    last_id = this_id;
  }
  sarray_sort(struct nonzero_id,nz->ptr,nz->n, primary,0, buf);
}

/************** Global topology **************/

/* construct list of all unique id's on this proc */
struct unique_id { ulong id; uint work_proc, src_if; };
static void unique_ids(struct array *un, const struct array *nz, const uint np)
{
  struct unique_id *un_row;
  const struct nonzero_id *nz_row, *nz_end;
  array_init(struct unique_id,un,nz->n), un_row=un->ptr;
  for(nz_row=nz->ptr,nz_end=nz_row+nz->n;nz_row!=nz_end;++nz_row) {
    if(nz_row->i != nz_row->primary) continue;
    un_row->id = nz_row->id;
    un_row->work_proc = nz_row->id%np;
    un_row->src_if = nz_row->flag ? ~nz_row->i : nz_row->i;
    ++un_row;
  }
  un->n = un_row - (struct unique_id*)un->ptr;
}

/* shared_ids    (global part)

   Creates an array of shared_id's from an array of nonzero_id's. Each entry
   in the output identifies one id shared with one other processor p.
   Note: two procs share an id only when at least one of them has it unflagged.
   The primary index is i locally and ri remotely. Bit 1 of flags indicates
   the local flag, bit 2 indicates the remote flag. The output has no
   particular ordering.

   Also creates an array of primary_shared_id's, one for each shared id.
   This struct includes ord, a global rank of the id (arbitrary, but unique). */

#define FLAGS_LOCAL  1
#define FLAGS_REMOTE 2

/* i  : local primary index
   p  : remote proc
   ri : remote primary index
   bi : buffer index (set and used during pw setup) */
struct shared_id {
  ulong id; uint i, p, ri, bi; unsigned flags;
};

struct primary_shared_id {
  ulong id, ord; uint i; unsigned flag;
};



struct shared_id_work { ulong id,ord; uint p1, p2, i1f, i2f; };
static void shared_ids_aux(struct array *sh, struct array *pr, uint pr_n,
                           struct array *wa, buffer *buf)
{
  const struct shared_id_work *w, *we;
  struct shared_id *s;
  struct primary_shared_id *p;
  ulong last_id = ULONG_MAX;
  /* translate work array to output arrays */
  sarray_sort(struct shared_id_work,wa->ptr,wa->n, id,1, buf);
  array_init(struct shared_id,sh,wa->n), sh->n=wa->n, s=sh->ptr;
  array_init(struct primary_shared_id,pr,pr_n), p=pr->ptr;
  for(w=wa->ptr,we=w+wa->n;w!=we;++w) {
    uint i1f = w->i1f, i2f = w->i2f;
    uint i1 = ~i1f<i1f?~i1f:i1f, i2 = ~i2f<i2f?~i2f:i2f;
    s->id=w->id, s->i=i1, s->p=w->p2, s->ri=i2;
    s->flags = ((i2f^i2)&FLAGS_REMOTE) | ((i1f^i1)&FLAGS_LOCAL);
    ++s;
    if(w->id!=last_id) {
      last_id=w->id;
      p->id=last_id, p->ord=w->ord, p->i=i1, p->flag=(i1f^i1)&FLAGS_LOCAL;
      ++p;
    }
  }
  pr->n = p-(struct primary_shared_id*)pr->ptr;
  sarray_sort(struct primary_shared_id,pr->ptr,pr->n, i,0, buf);
}

static ulong shared_ids(struct array *sh, struct array *pr,
                        const struct array *nz, struct crystal *cr)
{
  struct array un; struct unique_id *un_row, *un_end, *other;
  ulong last_id = ULONG_MAX;
  ulong ordinal[2], n_shared=0, scan_buf[2];
  struct array wa; struct shared_id_work *w;
  uint n_unique;
  /* construct list of all unique id's on this proc */
  unique_ids(&un,nz,cr->comm.np);
  n_unique = un.n;
  /* transfer list to work procs */
  sarray_transfer(struct unique_id,&un, work_proc,1, cr);
  /* group by id, put flagged entries after unflagged (within each group) */
  sarray_sort_2(struct unique_id,un.ptr,un.n, id,1, src_if,0, &cr->data);
  /* count shared id's */
  for(un_row=un.ptr,un_end=un_row+un.n;un_row!=un_end;++un_row) {
    ulong id = un_row->id;
    if(~un_row->src_if<un_row->src_if) continue;
    if(id==last_id) continue;
    other=un_row+1;
    if(other!=un_end&&other->id==id) last_id=id, ++n_shared;
  }
  comm_scan(ordinal, &cr->comm,gs_slong,gs_add, &n_shared,1, scan_buf);
  /* there are ordinal[1] globally shared unique ids;
           and ordinal[0] of those are seen by work procs of lower rank;
     i.e., this work processor sees the range ordinal[0] + (0,n_shared-1) */
  /* construct list of shared ids */
  last_id = ULONG_MAX;
  array_init(struct shared_id_work,&wa,un.n), wa.n=0, w=wa.ptr;
  for(un_row=un.ptr,un_end=un_row+un.n;un_row!=un_end;++un_row) {
    ulong id = un_row->id;
    uint p1 = un_row->work_proc, i1f = un_row->src_if;
    if(~i1f<i1f) continue;
    for(other=un_row+1;other!=un_end&&other->id==id;++other) {
      uint p2 = other->work_proc, i2f = other->src_if;
      ulong ord;
      if(id!=last_id) last_id=id, ++ordinal[0];
      ord=ordinal[0]-1;
      if(wa.n+2>wa.max)
        array_reserve(struct shared_id_work,&wa,wa.n+2),
        w=(struct shared_id_work*)wa.ptr+wa.n;
      w->id=id, w->ord=ord, w->p1=p1, w->p2=p2, w->i1f=i1f, w->i2f=i2f, ++w;
      w->id=id, w->ord=ord, w->p1=p2, w->p2=p1, w->i1f=i2f, w->i2f=i1f, ++w;
      wa.n+=2;
    }
  }
  /* transfer shared list to source procs */
  sarray_transfer(struct shared_id_work,&wa, p1,0, cr);
  /* fill output arrays from work array */
  shared_ids_aux(sh,pr,n_unique,&wa,&cr->data);
  array_free(&un);
  array_free(&wa);
  return ordinal[1];
}

static void get_topology(struct gs_topology *top,
                         const slong *id, uint n, struct crystal *cr)
{
  nonzero_ids(&top->nz,id,n,&cr->data);
  top->total_shared = shared_ids(&top->sh,&top->pr, &top->nz,cr);
}

static void make_topology_unique(struct gs_topology *top, slong *id,
                                 uint pid, buffer *buf)
{
  struct array *const nz=&top->nz, *const sh=&top->sh, *const pr=&top->pr;
  struct nonzero_id *pnz;
  struct shared_id *pb, *pe, *e, *out;
  struct primary_shared_id *q;

  /* flag local non-primaries */
  sarray_sort(struct nonzero_id,nz->ptr,nz->n, i,0, buf);
  if(id) {
    struct nonzero_id *p,*e;
    for(p=nz->ptr,e=p+nz->n;p!=e;++p)
      if(p->i != p->primary) id[p->i]=-(slong)p->id,p->flag=1;
  } else {
    struct nonzero_id *p,*e;
    for(p=nz->ptr,e=p+nz->n;p!=e;++p)
      if(p->i != p->primary) p->flag=1;
  }
  sarray_sort(struct nonzero_id,nz->ptr,nz->n, primary,0, buf);

  /* assign owner among shared primaries */

  /* create sentinel with i = -1 */
  array_reserve(struct shared_id,sh,sh->n+1);
  ((struct shared_id*)sh->ptr)[sh->n].i = UINT_MAX;
  /* in the sorted list of procs sharing a given id,
     the owner is chosen to be the j^th unflagged proc,
     where j = id mod (length of list) */
  sarray_sort_2(struct shared_id,sh->ptr,sh->n, i,0, p,0, buf);
  out=sh->ptr; pnz=top->nz.ptr;
  for(pb=sh->ptr,e=pb+sh->n;pb!=e;pb=pe) {
    uint i = pb->i, lt=0,gt=0, owner; struct shared_id *p;
    while(pnz->i!=i) ++pnz;
    /* note: current proc not in list */
    for(pe=pb; pe->i==i && pe->p<pid; ++pe) if(!(pe->flags&FLAGS_REMOTE)) ++lt;
    for(     ; pe->i==i             ; ++pe) if(!(pe->flags&FLAGS_REMOTE)) ++gt;
    if(!(pb->flags&FLAGS_LOCAL)) {
      owner = pb->id%(lt+gt+1);
      if(owner==lt) goto make_sh_unique_mine;
      if(owner>lt) --owner;
    } else
      owner = pb->id%(lt+gt);
    /* we don't own pb->id */
    if(id) id[i] = -(slong)pb->id;
    pnz->flag=1;
    /* we only share this id with the owner now; remove the other entries */
    for(p=pb; p!=pe; ++p) if(!(p->flags&FLAGS_REMOTE) && !(owner--)) break;
    if(p!=pe) *out=*p, out->flags=FLAGS_LOCAL, ++out;
    continue;
  make_sh_unique_mine:
    /* we own pb->id */
    if(out==pb) { out=pe; for(p=pb; p!=pe; ++p) p->flags=FLAGS_REMOTE; }
    else        for(p=pb; p!=pe; ++p) *out=*p,out->flags=FLAGS_REMOTE,++out;
  }
  sh->n = out - ((struct shared_id*)sh->ptr);

  /* set primary_shared_id flags to match */
  ((struct shared_id*)sh->ptr)[sh->n].i = UINT_MAX;
  sarray_sort(struct shared_id,sh->ptr,sh->n, id,1, buf);
  sarray_sort(struct primary_shared_id,pr->ptr,pr->n, id,1, buf);
  q=pr->ptr;
  for(pb=sh->ptr,e=pb+sh->n;pb!=e;pb=pe) {
    uint i=pb->i;
    pe=pb; while(pe->i==i) ++pe;
    if(q->id!=pb->id) printf("FAIL!!!\n");
    q->flag=pb->flags&FLAGS_LOCAL;
    ++q;
  }
}

/*------------------------------------------------------------------------------
  Local setup
------------------------------------------------------------------------------*/

/* assumes nz is sorted by primary, then flag, then index */
static const uint *local_map(const struct array *nz, const int ignore_flagged,
                             uint *mem_size)
{
  uint *map, *p, count = 1;
  const struct nonzero_id *row, *other, *end;
#define DO_COUNT(cond) do \
    for(row=nz->ptr,end=row+nz->n;row!=end;) {                     \
      ulong row_id = row->id; int any=0;                           \
      for(other=row+1;other!=end&&other->id==row_id&&cond;++other) \
        any=2, ++count;                                            \
      count+=any, row=other;                                       \
    } while(0)
  if(ignore_flagged) DO_COUNT(other->flag==0); else DO_COUNT(1);
#undef DO_COUNT
  p = map = tmalloc(uint,count); *mem_size += count*sizeof(uint);
#define DO_SET(cond) do \
    for(row=nz->ptr,end=row+nz->n;row!=end;) {                     \
      ulong row_id = row->id; int any=0;                           \
      *p++ = row->i;                                               \
      for(other=row+1;other!=end&&other->id==row_id&&cond;++other) \
        any=1, *p++ = other->i;                                    \
      if(any) *p++ = UINT_MAX; else --p;                           \
      row=other;                                                   \
    } while(0)
  if(ignore_flagged) DO_SET(other->flag==0); else DO_SET(1);
#undef DO_SET
  *p = UINT_MAX;
  return map;
}

static const uint *flagged_primaries_map(const struct array *nz, uint *mem_size)
{
  uint *map, *p, count=1;
  const struct nonzero_id *row, *end;
  for(row=nz->ptr,end=row+nz->n;row!=end;++row)
    if(row->i==row->primary && row->flag==1) ++count;
  p = map = tmalloc(uint,count); *mem_size += count*sizeof(uint);
  for(row=nz->ptr,end=row+nz->n;row!=end;++row)
    if(row->i==row->primary && row->flag==1) *p++ = row->i;
  *p = UINT_MAX;
  return map;
}

/*------------------------------------------------------------------------------
  Remote execution and setup
------------------------------------------------------------------------------*/

typedef void exec_fun(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf);
typedef void fin_fun(void *data);

struct gs_remote {
  uint buffer_size, mem_size;
  void *data;
  exec_fun *exec;
  exec_fun *exec_irecv;
  exec_fun *exec_isend;
  exec_fun *exec_wait;
  fin_fun *fin;
};

typedef void setup_fun(struct gs_remote *r, struct gs_topology *top,
                       const struct comm *comm, buffer *buf);

/*------------------------------------------------------------------------------
  Pairwise Execution
------------------------------------------------------------------------------*/

struct pw_comm_data {
  uint n;      /* number of messages */
  uint *p;     /* message source/dest proc */
  uint *size;  /* size of message */
  uint total;  /* sum of message sizes */
};

struct pw_data {
  struct pw_comm_data comm[2];
  const uint *map[2];
  comm_req *req;
  uint buffer_size;
};

static char *pw_exec_recvs(char *buf, const unsigned unit_size,
                           const struct comm *comm,
                           const struct pw_comm_data *c, comm_req *req)
{
  const uint *p, *pe, *size=c->size;
  for(p=c->p,pe=p+c->n;p!=pe;++p) {
    size_t len = *(size++)*unit_size;
    comm_irecv(req++,comm,buf,len,*p,*p);
    buf += len;
  }
  return buf;
}

static char *pw_exec_sends(char *buf, const unsigned unit_size,
                           const struct comm *comm,
                           const struct pw_comm_data *c, comm_req *req)
{
  const uint *p, *pe, *size=c->size;
  for(p=c->p,pe=p+c->n;p!=pe;++p) {
    size_t len = *(size++)*unit_size;
    comm_isend(req++,comm,buf,len,*p,comm->id);
    buf += len;
  }
  return buf;
}

static void pw_exec(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct pw_data *pwd = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_gather_fun *const gather_from_buf[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec_to_many, &gather_noop };
  const unsigned recv = 0^transpose, send = 1^transpose;
  unsigned unit_size = vn*gs_dom_size[dom];
  char *sendbuf;
  /* post receives */
  sendbuf = pw_exec_recvs(buf,unit_size,comm,&pwd->comm[recv],pwd->req);
  /* fill send buffer */
  scatter_to_buf[mode](sendbuf,data,vn,pwd->map[send],dom);
  /* post sends */
  pw_exec_sends(sendbuf,unit_size,comm,&pwd->comm[send],
                &pwd->req[pwd->comm[recv].n]);
  comm_wait(pwd->req,pwd->comm[0].n+pwd->comm[1].n);
  /* gather using recv buffer */
  gather_from_buf[mode](data,buf,vn,pwd->map[recv],dom,op);
}

/*------------------------------------------------------------------------------
  Nonblocking Pairwise Execution
------------------------------------------------------------------------------*/
static void pw_exec_irecv(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct pw_data *pwd = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_gather_fun *const gather_from_buf[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec_to_many, &gather_noop };
  const unsigned recv = 0^transpose, send = 1^transpose;
  unsigned unit_size = vn*gs_dom_size[dom];
  /* post receives */
  char *sendbuf = pw_exec_recvs(buf,unit_size,comm,&pwd->comm[recv],pwd->req);
}

static void pw_exec_isend(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct pw_data *pwd = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_gather_fun *const gather_from_buf[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec_to_many, &gather_noop };
  const unsigned recv = 0^transpose, send = 1^transpose;
  unsigned unit_size = vn*gs_dom_size[dom];

  /* fill send buffer */
  char *sendbuf = buf+unit_size*pwd->comm[recv].total;
  scatter_to_buf[mode](sendbuf,data,vn,pwd->map[send],dom);
  /* post sends */
  pw_exec_sends(sendbuf,unit_size,comm,&pwd->comm[send],
                &pwd->req[pwd->comm[recv].n]);
}

static void pw_exec_wait(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct pw_data *pwd = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_gather_fun *const gather_from_buf[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec_to_many, &gather_noop };
  const unsigned recv = 0^transpose, send = 1^transpose;
  unsigned unit_size = vn*gs_dom_size[dom];

  comm_wait(pwd->req,pwd->comm[0].n+pwd->comm[1].n);
  /* gather using recv buffer */
  gather_from_buf[mode](data,buf,vn,pwd->map[recv],dom,op);
}

/*------------------------------------------------------------------------------
  Pairwise setup
------------------------------------------------------------------------------*/
static uint pw_comm_setup(struct pw_comm_data *data, struct array *sh,
                          const unsigned flags_mask, buffer *buf)
{
  uint n=0,count=0, lp=UINT_MAX, mem_size=0;
  struct shared_id *s, *se;
  /* sort by remote processor and id (a globally consistent ordering) */
  sarray_sort_2(struct shared_id,sh->ptr,sh->n, p,0, id,1, buf);
  /* assign index into buffer */
  for(s=sh->ptr,se=s+sh->n;s!=se;++s) {
    if(s->flags&flags_mask) { s->bi = UINT_MAX; continue; }
    s->bi = count++;
    if(s->p!=lp) lp=s->p, ++n;
  }
  data->n = n;
  data->p = tmalloc(uint,2*n); mem_size+=2*n*sizeof(uint);
  data->size = data->p + n;
  data->total = count;
  n = 0, lp=UINT_MAX;
  for(s=sh->ptr,se=s+sh->n;s!=se;++s) {
    if(s->flags&flags_mask) continue;
    if(s->p!=lp) {
      lp=s->p;
      if(n!=0) data->size[n-1] = count;
      count=0, data->p[n++]=lp;
    }
    ++count;
  }
  if(n!=0) data->size[n-1] = count;
  return mem_size;
}

static void pw_comm_free(struct pw_comm_data *data) { free(data->p); }

/* assumes that the bi field of sh is set */
static const uint *pw_map_setup(struct array *sh, buffer *buf, uint *mem_size)
{
  uint count=0, *map, *p;
  struct shared_id *s, *se;
  sarray_sort(struct shared_id,sh->ptr,sh->n, i,0, buf);
  /* calculate map size */
  count=1;
  for(s=sh->ptr,se=s+sh->n;s!=se;) {
    uint i=s->i;
    if(s->bi==UINT_MAX) { ++s; continue; }
    count+=3;
    for(++s;s!=se&&s->i==i;++s) if(s->bi!=UINT_MAX) ++count;
  }
  /* write map */
  p = map = tmalloc(uint,count); *mem_size += count*sizeof(uint);
  for(s=sh->ptr,se=s+sh->n;s!=se;) {
    uint i=s->i;
    if(s->bi==UINT_MAX) { ++s; continue; }
    *p++ = i, *p++ = s->bi;
    for(++s;s!=se&&s->i==i;++s) if(s->bi!=UINT_MAX) *p++ = s->bi;
    *p++ = UINT_MAX;
  }
  *p = UINT_MAX;
  return map;
}

static struct pw_data *pw_setup_aux(struct array *sh, buffer *buf,
                                    uint *mem_size)
{
  struct pw_data *pwd = tmalloc(struct pw_data,1);
  *mem_size = sizeof(struct pw_data);

  /* default behavior: receive only remotely unflagged data */
  *mem_size+=pw_comm_setup(&pwd->comm[0],sh, FLAGS_REMOTE, buf);
  pwd->map[0] = pw_map_setup(sh, buf, mem_size);

  /* default behavior: send only locally unflagged data */
  *mem_size+=pw_comm_setup(&pwd->comm[1],sh, FLAGS_LOCAL, buf);
  pwd->map[1] = pw_map_setup(sh, buf, mem_size);

  pwd->req = tmalloc(comm_req,pwd->comm[0].n+pwd->comm[1].n);
  *mem_size += (pwd->comm[0].n+pwd->comm[1].n)*sizeof(comm_req);
  pwd->buffer_size = pwd->comm[0].total + pwd->comm[1].total;
  return pwd;
}

static void pw_free(struct pw_data *data)
{
  pw_comm_free(&data->comm[0]);
  pw_comm_free(&data->comm[1]);
  free((uint*)data->map[0]);
  free((uint*)data->map[1]);
  free(data->req);
  free(data);
}

static void pw_setup(struct gs_remote *r, struct gs_topology *top,
                     const struct comm *comm, buffer *buf)
{
  struct pw_data *pwd = pw_setup_aux(&top->sh,buf, &r->mem_size);
  r->buffer_size = pwd->buffer_size;
  r->data = pwd;
  r->exec = (exec_fun*)&pw_exec;
  r->exec_irecv = (exec_fun*)&pw_exec_irecv;
  r->exec_isend = (exec_fun*)&pw_exec_isend;
  r->exec_wait = (exec_fun*)&pw_exec_wait;
  r->fin = (fin_fun*)&pw_free;
}

/*------------------------------------------------------------------------------
  Crystal-Router Execution
------------------------------------------------------------------------------*/
struct cr_stage {
  const uint *scatter_map, *gather_map;
  uint size_r, size_r1, size_r2;
  uint size_sk, size_s, size_total;
  uint p1, p2;
  unsigned nrecvn;
};

struct cr_data {
  struct cr_stage *stage[2];
  unsigned nstages;
  uint buffer_size, stage_buffer_size;
};

static void cr_exec(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct cr_data *crd = execdata;
  static gs_scatter_fun *const scatter_user_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_scatter_fun *const scatter_buf_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_vec, &gs_scatter };
  static gs_scatter_fun *const scatter_buf_to_user[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_vec_to_many, &scatter_noop };
  static gs_gather_fun *const gather_buf_to_user[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec_to_many, &gather_noop };
  static gs_gather_fun *const gather_buf_to_buf[] =
    { &gs_gather, &gs_gather_vec, &gs_gather_vec, &gs_gather };
  const unsigned unit_size = vn*gs_dom_size[dom], nstages=crd->nstages;
  unsigned k;
  char *sendbuf, *buf_old, *buf_new;
  const struct cr_stage *stage = crd->stage[transpose];
  buf_old = buf;
  buf_new = buf_old + unit_size*crd->stage_buffer_size;
  /* crystal router */
  for(k=0;k<nstages;++k) {
    comm_req req[3];
    if(stage[k].nrecvn)
      comm_irecv(&req[1],comm,buf_new,unit_size*stage[k].size_r1,
               stage[k].p1, comm->np+k);
    if(stage[k].nrecvn==2)
      comm_irecv(&req[2],comm,buf_new+unit_size*stage[k].size_r1,
               unit_size*stage[k].size_r2, stage[k].p2, comm->np+k);
    sendbuf = buf_new+unit_size*stage[k].size_r;
    if(k==0)
      scatter_user_to_buf[mode](sendbuf,data,vn,stage[0].scatter_map,dom);
    else
      scatter_buf_to_buf[mode](sendbuf,buf_old,vn,stage[k].scatter_map,dom),
      gather_buf_to_buf [mode](sendbuf,buf_old,vn,stage[k].gather_map ,dom,op);

    comm_isend(&req[0],comm,sendbuf,unit_size*stage[k].size_s,
               stage[k].p1, comm->np+k);
    comm_wait(&req[0],1+stage[k].nrecvn);
    { char *t = buf_old; buf_old=buf_new; buf_new=t; }
  }
  scatter_buf_to_user[mode](data,buf_old,vn,stage[k].scatter_map,dom);
  gather_buf_to_user [mode](data,buf_old,vn,stage[k].gather_map ,dom,op);
}

/*------------------------------------------------------------------------------
  Crystal-Router setup
------------------------------------------------------------------------------*/
static uint cr_schedule(struct cr_data *data, const struct comm *comm)
{
  uint mem_size = 0;
  const uint id = comm->id;
  uint bl=0, n=comm->np;
  unsigned k = 0;
  while(n>1) {
    uint nl = (n+1)/2, bh = bl+nl;
    if(id<bh) n=nl; else n-=nl,bl=bh;
    ++k;
  }
  data->nstages = k;
  data->stage[0] = tmalloc(struct cr_stage,2*(k+1));
  data->stage[1] = data->stage[0] + (k+1);
  mem_size += 2*(k+1)*sizeof(struct cr_stage);
  bl=0, n=comm->np, k=0;
  while(n>1) {
    uint nl = (n+1)/2, bh = bl+nl;
    uint targ; unsigned recvn;
    recvn = 1, targ = n-1-(id-bl)+bl;
    if(id==targ) targ=bh, recvn=0;
    if(n&1 && id==bh) recvn=2;
    data->stage[1][k].nrecvn=data->stage[0][k].nrecvn=recvn;
    data->stage[1][k].p1    =data->stage[0][k].p1    =targ;
    data->stage[1][k].p2    =data->stage[0][k].p2    =comm->id-1;
    if(id<bh) n=nl; else n-=nl,bl=bh;
    ++k;
  }
  return mem_size;
}

struct crl_id {
  ulong id; uint p, ri, si, bi, send;
};

/* assumes sh is grouped by i (e.g., sorted by i or by id) */
static void crl_work_init(struct array *cw, struct array *sh,
                          const unsigned send_mask, uint this_p)
{
  const unsigned recv_mask = send_mask^(FLAGS_REMOTE|FLAGS_LOCAL);
  uint last_i=UINT_MAX; int added_myself;
  uint cw_n = 0, cw_max = cw->max;
  struct crl_id *w = cw->ptr;
  struct shared_id *s, *se;

#define CW_ADD(aid,ap,ari,asi) do { \
    if(cw_n==cw_max)                                         \
      array_reserve(struct crl_id,cw,cw_n+1),cw_max=cw->max, \
      w=(struct crl_id*)cw->ptr+cw_n;                        \
    w->id=aid, w->p=ap, w->ri=ari, w->si=asi;                \
    ++w, ++cw_n;                                             \
  } while(0)

  for(s=sh->ptr,se=s+sh->n;s!=se;++s) {
    int send = (s->flags&send_mask)==0;
    int recv = (s->flags&recv_mask)==0;
    if(s->i!=last_i) last_i=s->i, added_myself=0;
    if(!added_myself && recv && (s->flags&FLAGS_LOCAL)==0) {
      added_myself=1;
      CW_ADD(s->id,this_p,s->i,s->i);
    }
    if(send) CW_ADD(s->id,s->p,s->ri,s->i);
  }
  cw->n=cw_n;
#undef CW_ADD
}

static uint crl_maps(struct cr_stage *stage, struct array *cw, buffer *buf)
{
  uint mem_size=0;
  struct crl_id *w, *we, *other;
  uint scount=1, gcount=1, *sp, *gp;
  sarray_sort_2(struct crl_id,cw->ptr,cw->n, bi,0, si,0, buf);
  for(w=cw->ptr,we=w+cw->n;w!=we;w=other) {
    uint bi=w->bi,any=0,si=w->si;
    scount+=3;
    for(other=w+1;other!=we&&other->bi==bi;++other)
      if(other->si!=si) si=other->si, any=2, ++gcount;
    gcount+=any;
  }
  stage->scatter_map = sp = tmalloc(uint,scount+gcount);
  stage->gather_map  = gp = sp + scount;
  mem_size += (scount+gcount)*sizeof(uint);
  for(w=cw->ptr,we=w+cw->n;w!=we;w=other) {
    uint bi=w->bi,any=0,si=w->si;
    *sp++ = w->si, *sp++ = bi;
    *gp++ = bi;
    for(other=w+1;other!=we&&other->bi==bi;++other)
      if(other->si!=si) si=other->si, any=1, *gp++ = si;
    if(any) *gp++ = UINT_MAX; else --gp;
    *sp++ = UINT_MAX;
  }
  *sp=UINT_MAX, *gp=UINT_MAX;
  return mem_size;
}

static uint crl_work_label(struct array *cw, struct cr_stage *stage,
                           uint cutoff, int send_hi, buffer *buf,
                           uint *mem_size)
{
  struct crl_id *w, *we, *start;
  uint nsend, nkeep = 0, nks = 0, bi=0;
  /* here w->send has a reverse meaning */
  if(send_hi) for(w=cw->ptr,we=w+cw->n;w!=we;++w) w->send = w->p< cutoff;
         else for(w=cw->ptr,we=w+cw->n;w!=we;++w) w->send = w->p>=cutoff;
  sarray_sort_2(struct crl_id,cw->ptr,cw->n, id,1, send,0, buf);
  for(start=cw->ptr,w=start,we=w+cw->n;w!=we;++w) {
    nkeep += w->send;
    if(w->id!=start->id) start=w;
    if(w->send!=start->send) w->send=0,w->bi=1, ++nks; else w->bi=0;
  }
  nsend = cw->n-nkeep;
  /* assign indices; sent ids have priority (hence w->send is reversed) */
  sarray_sort(struct crl_id,cw->ptr,cw->n, send,0, buf);
  for(start=cw->ptr,w=start,we=w+nsend+nks;w!=we;++w) {
    if(w->id!=start->id) start=w, ++bi;
    if(w->bi!=1) w->send=1;   /* switch back to the usual semantics */
    w->bi = bi;
  }
  stage->size_s = nsend+nks==0 ? 0 : bi+1;
  for(we=(struct crl_id*)cw->ptr+cw->n;w!=we;++w) {
    if(w->id!=start->id) start=w, ++bi;
    w->send = 0;              /* switch back to the usual semantics */
    w->bi = bi;
  }
  stage->size_sk = cw->n==0 ? 0 : bi+1;
  *mem_size += crl_maps(stage,cw,buf);
  return nsend;
}

static void crl_bi_to_si(struct crl_id *w, uint n, uint v) {
  for(;n;--n) w->si=w->bi+v, ++w;
}

static void crl_ri_to_bi(struct crl_id *w, uint n) {
  for(;n;--n) w->bi=w->ri, ++w;
}

static uint cr_learn(struct array *cw, struct cr_stage *stage,
                     const struct comm *comm, buffer *buf, uint *mem_size)
{
  comm_req req[3];
  const uint id = comm->id;
  uint bl=0, n=comm->np;
  uint size_max=0;
  uint tag = comm->np;
  while(n>1) {
    uint nl = (n+1)/2, bh = bl+nl;
    uint nkeep, nsend[2], nrecv[2][2] = {{0,0},{0,0}};
    struct crl_id *wrecv[2], *wsend;
    nsend[0] = crl_work_label(cw,stage,bh,id<bh,buf, mem_size);
    nsend[1] = stage->size_s;
    nkeep = cw->n - nsend[0];

    if(stage->nrecvn   ) comm_irecv(&req[1],comm,nrecv[0],2*sizeof(uint),
                                    stage->p1,tag);
    if(stage->nrecvn==2) comm_irecv(&req[2],comm,nrecv[1],2*sizeof(uint),
                                    stage->p2,tag);
    comm_isend(&req[0],comm,nsend,2*sizeof(uint),stage->p1,tag);
    comm_wait(req,1+stage->nrecvn),++tag;

    stage->size_r1 = nrecv[0][1], stage->size_r2 = nrecv[1][1];
    stage->size_r = stage->size_r1 + stage->size_r2;
    stage->size_total = stage->size_r + stage->size_sk;
    if(stage->size_total>size_max) size_max=stage->size_total;

    array_reserve(struct crl_id,cw,cw->n+nrecv[0][0]+nrecv[1][0]);
    wrecv[0] = cw->ptr, wrecv[0] += cw->n, wrecv[1] = wrecv[0]+nrecv[0][0];
    wsend = cw->ptr, wsend += nkeep;
    if(stage->nrecvn   )
      comm_irecv(&req[1],comm,wrecv[0],nrecv[0][0]*sizeof(struct crl_id),
                 stage->p1,tag);
    if(stage->nrecvn==2)
      comm_irecv(&req[2],comm,wrecv[1],nrecv[1][0]*sizeof(struct crl_id),
                 stage->p2,tag);
    sarray_sort_2(struct crl_id,cw->ptr,cw->n, send,0, bi,0, buf);
    comm_isend(&req[0],comm,wsend,nsend[0]*sizeof(struct crl_id),stage->p1,tag);
    comm_wait(req,1+stage->nrecvn),++tag;

    crl_bi_to_si(cw->ptr,nkeep,stage->size_r);
    if(stage->nrecvn)    crl_bi_to_si(wrecv[0],nrecv[0][0],0);
    if(stage->nrecvn==2) crl_bi_to_si(wrecv[1],nrecv[1][0],stage->size_r1);
    memmove(wsend,wrecv[0],(nrecv[0][0]+nrecv[1][0])*sizeof(struct crl_id));
    cw->n += nrecv[0][0] + nrecv[1][0];
    cw->n -= nsend[0];

    if(id<bh) n=nl; else n-=nl,bl=bh;
    ++stage;
  }
  crl_ri_to_bi(cw->ptr,cw->n);
  *mem_size += crl_maps(stage,cw,buf);
  return size_max;
}

static struct cr_data *cr_setup_aux(
  struct array *sh, const struct comm *comm, buffer *buf, uint *mem_size)
{
  uint size_max[2];
  struct array cw = null_array;
  struct cr_data *crd = tmalloc(struct cr_data,1);
  *mem_size = sizeof(struct cr_data);

  /* default behavior: receive only remotely unflagged data */
  /* default behavior: send only locally unflagged data */

  *mem_size += cr_schedule(crd,comm);

  sarray_sort(struct shared_id,sh->ptr,sh->n, i,0, buf);
  crl_work_init(&cw,sh, FLAGS_LOCAL , comm->id);
  size_max[0]=cr_learn(&cw,crd->stage[0],comm,buf, mem_size);
  crl_work_init(&cw,sh, FLAGS_REMOTE, comm->id);
  size_max[1]=cr_learn(&cw,crd->stage[1],comm,buf, mem_size);

  crd->stage_buffer_size = size_max[1]>size_max[0]?size_max[1]:size_max[0];

  array_free(&cw);

  crd->buffer_size = 2*crd->stage_buffer_size;
  return crd;
}

static void cr_free_stage_maps(struct cr_stage *stage, unsigned kmax)
{
  unsigned k;
  for(k=0; k<kmax; ++k) {
    free((uint*)stage->scatter_map);
    ++stage;
  }
  free((uint*)stage->scatter_map);
}

static void cr_free(struct cr_data *data)
{
  cr_free_stage_maps(data->stage[0],data->nstages);
  cr_free_stage_maps(data->stage[1],data->nstages);
  free(data->stage[0]);
  free(data);
}

static void cr_setup(struct gs_remote *r, struct gs_topology *top,
                     const struct comm *comm, buffer *buf)
{
  struct cr_data *crd = cr_setup_aux(&top->sh,comm,buf, &r->mem_size);
  r->buffer_size = crd->buffer_size;
  r->data = crd;
  r->exec = (exec_fun*)&cr_exec;
  r->fin = (fin_fun*)&cr_free;
}

/*------------------------------------------------------------------------------
  All-reduce Execution
------------------------------------------------------------------------------*/
struct allreduce_data {
  const uint *map_to_buf[2], *map_from_buf[2];
  uint buffer_size;
  comm_req *req;
};

static void allreduce_exec(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct allreduce_data *ard = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_scatter_fun *const scatter_from_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_vec_to_many, &scatter_noop };
  uint gvn = vn*(ard->buffer_size/2);
  unsigned unit_size = gs_dom_size[dom];
  char *ardbuf;
  ardbuf = buf+unit_size*gvn;
  /* user array -> buffer */
  gs_init_array(buf,gvn,dom,op);
  scatter_to_buf[mode](buf,data,vn,ard->map_to_buf[transpose],dom);
  /* all reduce */
  comm_allreduce(comm,dom,op, buf,gvn, ardbuf);
  /* buffer -> user array */
  scatter_from_buf[mode](data,buf,vn,ard->map_from_buf[transpose],dom);
}

/*------------------------------------------------------------------------------
  Nonblocking All-reduce Execution
------------------------------------------------------------------------------*/
static void allreduce_exec_i(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct allreduce_data *ard = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_scatter_fun *const scatter_from_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_vec_to_many, &scatter_noop };
  uint gvn = vn*(ard->buffer_size/2);
  unsigned unit_size = gs_dom_size[dom];
  char *ardbuf = buf+unit_size*gvn;

  /* user array -> buffer */
  gs_init_array(buf,gvn,dom,op);
  scatter_to_buf[mode](buf,data,vn,ard->map_to_buf[transpose],dom);
  /* all reduce */
  comm_iallreduce(ard->req,comm,dom,op,buf,gvn,ardbuf);
}

static void allreduce_exec_wait(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf)
{
  const struct allreduce_data *ard = execdata;
  static gs_scatter_fun *const scatter_to_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many_to_vec, &scatter_noop };
  static gs_scatter_fun *const scatter_from_buf[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_vec_to_many, &scatter_noop };
  uint gvn = vn*(ard->buffer_size/2);
  unsigned unit_size = gs_dom_size[dom];
  char *ardbuf = buf+unit_size*gvn;

  /* Why do I need this? Ugly */
  if (comm->np > 1)
    comm_wait(ard->req, 1);
#ifdef GSLIB_USE_MPI
  memcpy(buf,ardbuf,gvn*gs_dom_size[dom]);
#endif
  /* buffer -> user array */
  scatter_from_buf[mode](data,buf,vn,ard->map_from_buf[transpose],dom);
}

/*------------------------------------------------------------------------------
  All-reduce setup
------------------------------------------------------------------------------*/
static const uint *allreduce_map_setup(
  struct array *pr, const unsigned flags_mask, int to_buf, uint *mem_size)
{
  struct primary_shared_id *p, *pe;
  uint count=1, *map, *m;
  for(p=pr->ptr,pe=p+pr->n;p!=pe;++p)
    if((p->flag&flags_mask)==0) count+=3;
  m=map=tmalloc(uint,count); *mem_size += count*sizeof(uint);
  if(to_buf) {
    for(p=pr->ptr,pe=p+pr->n;p!=pe;++p)
      if((p->flag&flags_mask)==0)
        *m++ = p->i, *m++ = p->ord, *m++ = UINT_MAX;
  } else {
    for(p=pr->ptr,pe=p+pr->n;p!=pe;++p)
      if((p->flag&flags_mask)==0)
        *m++ = p->ord, *m++ = p->i, *m++ = UINT_MAX;
  }
  *m=UINT_MAX;
  return map;
}

static struct allreduce_data *allreduce_setup_aux(
  struct array *pr, ulong total_shared, uint *mem_size)
{
  struct allreduce_data *ard = tmalloc(struct allreduce_data,1);
  *mem_size = sizeof(struct allreduce_data);

  /* default behavior: reduce only unflagged data, copy to all */
  ard->map_to_buf  [0] = allreduce_map_setup(pr,1,1, mem_size);
  ard->map_from_buf[0] = allreduce_map_setup(pr,0,0, mem_size);

  /* transpose behavior: reduce all data, copy to unflagged */
  ard->map_to_buf  [1] = allreduce_map_setup(pr,0,1, mem_size);
  ard->map_from_buf[1] = allreduce_map_setup(pr,1,0, mem_size);

  ard->req = tmalloc(comm_req, 1);

  ard->buffer_size = total_shared*2;
  return ard;
}

static void allreduce_free(struct allreduce_data *ard)
{
  free((uint*)ard->map_to_buf[0]);
  free((uint*)ard->map_to_buf[1]);
  free((uint*)ard->map_from_buf[0]);
  free((uint*)ard->map_from_buf[1]);
  free(ard);
}

static void allreduce_setup(struct gs_remote *r, struct gs_topology *top,
                            const struct comm *comm, buffer *buf)
{
  struct allreduce_data *ard
    = allreduce_setup_aux(&top->pr,top->total_shared, &r->mem_size);
  r->buffer_size = ard->buffer_size;
  r->data = ard;
  r->exec = (exec_fun*)&allreduce_exec;
  r->exec_irecv = (exec_fun*)&allreduce_exec_i;
  r->exec_isend = NULL;
  r->exec_wait = (exec_fun*)&allreduce_exec_wait;
  r->fin = (fin_fun*)&allreduce_free;
}

/*------------------------------------------------------------------------------
  Automatic Setup --- dynamically picks the fastest method
------------------------------------------------------------------------------*/

static void dry_run_time(double times[3], const struct gs_remote *r,
                         const struct comm *comm, buffer *buf)
{
  int i; double t;
  buffer_reserve(buf,gs_dom_size[gs_double]*r->buffer_size);
  for(i= 2;i;--i)
    r->exec(0,mode_dry_run,1,gs_double,gs_add,0,r->data,comm,buf->ptr);
  comm_barrier(comm);
  t = comm_time();
  for(i=10;i;--i)
    r->exec(0,mode_dry_run,1,gs_double,gs_add,0,r->data,comm,buf->ptr);
  t = (comm_time() - t)/10;
  times[0] = t/comm->np, times[1] = t, times[2] = t;
  comm_allreduce(comm,gs_double,gs_add, &times[0],1, &t);
  comm_allreduce(comm,gs_double,gs_min, &times[1],1, &t);
  comm_allreduce(comm,gs_double,gs_max, &times[2],1, &t);
}

static void auto_setup(struct gs_remote *r, struct gs_topology *top,
                       const struct comm *comm, buffer *buf)
{
  pw_setup(r, top,comm,buf);

  if(comm->np>1) {
    const char *name = "pairwise";
    struct gs_remote r_alt;
    double time[2][3];

    #define DRY_RUN(i,gsr,str) do { \
      if(comm->id==0) printf("   " str ": "); \
      dry_run_time(time[i],gsr,comm,buf); \
      if(comm->id==0) \
        printf("%g %g %g\n",time[i][0],time[i][1],time[i][2]); \
    } while(0)

    #define DRY_RUN_CHECK(str,new_name) do { \
      DRY_RUN(1,&r_alt,str); \
      if(time[1][2]<time[0][2]) \
        time[0][2]=time[1][2], name=new_name, \
        r->fin(r->data), *r = r_alt; \
      else \
        r_alt.fin(r_alt.data); \
    } while(0)

    DRY_RUN(0, r, "pairwise times (avg, min, max)");

    cr_setup(&r_alt, top,comm,buf);
    DRY_RUN_CHECK(      "crystal router                ", "crystal router");

    if(top->total_shared<100000) {
      allreduce_setup(&r_alt, top,comm,buf);
      DRY_RUN_CHECK(    "all reduce                    ", "allreduce");
    }

    #undef DRY_RUN_CHECK
    #undef DRY_RUN

    if(comm->id==0) printf("   used all_to_all method: %s\n",name);
  }
}

/*------------------------------------------------------------------------------
  Main Execution
------------------------------------------------------------------------------*/
struct gs_data {
  struct comm comm;
  const uint *map_local[2]; /* 0=unflagged, 1=all */
  const uint *flagged_primaries;
  struct gs_remote r;
  uint handle_size;
};

/*------------------------------------------------------------------------------
  GS_AUX - blocking and non-blocking
------------------------------------------------------------------------------*/
static void gs_aux(
  void *u, gs_mode mode, unsigned vn, gs_dom dom, gs_op op, unsigned transpose,
  struct gs_data *gsh, buffer *buf)
{
  static gs_scatter_fun *const local_scatter[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many, &scatter_noop };
  static gs_gather_fun  *const local_gather [] =
    { &gs_gather,  &gs_gather_vec,  &gs_gather_many, &gather_noop  };
  static gs_init_fun *const init[] =
    { &gs_init, &gs_init_vec, &gs_init_many, &init_noop };
  if(!buf) buf = &static_buffer;
  buffer_reserve(buf,vn*gs_dom_size[dom]*gsh->r.buffer_size);
  local_gather [mode](u,u,vn,gsh->map_local[0^transpose],dom,op);
  if(transpose==0) init[mode](u,vn,gsh->flagged_primaries,dom,op);
  gsh->r.exec(u,mode,vn,dom,op,transpose,gsh->r.data,&gsh->comm,buf->ptr);
  local_scatter[mode](u,u,vn,gsh->map_local[1^transpose],dom);
}

static void gs_aux_irecv(
  void *u, gs_mode mode, unsigned vn, gs_dom dom, gs_op op, unsigned transpose,
  struct gs_data *gsh, buffer *buf)
{
  static gs_scatter_fun *const local_scatter[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many, &scatter_noop };
  static gs_gather_fun  *const local_gather [] =
    { &gs_gather,  &gs_gather_vec,  &gs_gather_many, &gather_noop  };
  static gs_init_fun *const init[] =
    { &gs_init, &gs_init_vec, &gs_init_many, &init_noop };
  if(!buf) buf = &static_buffer;
  buffer_reserve(buf,vn*gs_dom_size[dom]*gsh->r.buffer_size);
  local_gather [mode](u,u,vn,gsh->map_local[0^transpose],dom,op);
  if(transpose==0) init[mode](u,vn,gsh->flagged_primaries,dom,op);

  if (gsh->r.exec_irecv)
    gsh->r.exec_irecv(u,mode,vn,dom,op,transpose,gsh->r.data,&gsh->comm,buf->ptr);
}

static void gs_aux_isend(
  void *u, gs_mode mode, unsigned vn, gs_dom dom, gs_op op, unsigned transpose,
  struct gs_data *gsh, buffer *buf)
{
  static gs_scatter_fun *const local_scatter[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many, &scatter_noop };
  static gs_gather_fun  *const local_gather [] =
    { &gs_gather,  &gs_gather_vec,  &gs_gather_many, &gather_noop  };
  static gs_init_fun *const init[] =
    { &gs_init, &gs_init_vec, &gs_init_many, &init_noop };

  if(!buf) buf = &static_buffer;

  if (gsh->r.exec_isend)
    gsh->r.exec_isend(u,mode,vn,dom,op,transpose,gsh->r.data,&gsh->comm,buf->ptr);
}

static void gs_aux_wait(
  void *u, gs_mode mode, unsigned vn, gs_dom dom, gs_op op, unsigned transpose,
  struct gs_data *gsh, buffer *buf)
{
  static gs_scatter_fun *const local_scatter[] =
    { &gs_scatter, &gs_scatter_vec, &gs_scatter_many, &scatter_noop };
  static gs_gather_fun  *const local_gather [] =
    { &gs_gather,  &gs_gather_vec,  &gs_gather_many, &gather_noop  };
  static gs_init_fun *const init[] =
    { &gs_init, &gs_init_vec, &gs_init_many, &init_noop };

  if(!buf) buf = &static_buffer;

  if (gsh->r.exec_wait)
    gsh->r.exec_wait(u,mode,vn,dom,op,transpose,gsh->r.data,&gsh->comm,buf->ptr);

  local_scatter[mode](u,u,vn,gsh->map_local[1^transpose],dom);
}

/*------------------------------------------------------------------------------
  GS interface - blocking and non-blocking
------------------------------------------------------------------------------*/
struct nonblocking_private {
  void *u;
  gs_mode mode;
  gs_dom dom;
  gs_op op;
  unsigned transpose;
  struct gs_data *gsh;
  buffer *buf;
  unsigned vn;
};

typedef struct nonblocking_private* nblkng;

static nblkng *nblkng_dict;
static int nblkng_max = 0;
static int nblkng_n = 0;
static int nblkng_count = 0;

void gs(void *u, gs_dom dom, gs_op op, unsigned transpose,
        struct gs_data *gsh, buffer *buf)
{
  gs_aux(u,mode_plain,1,dom,op,transpose,gsh,buf);
}

void igs(void *u, gs_dom dom, gs_op op, unsigned transpose,
        struct gs_data *gsh, buffer *buf, int *handle)
{
  if(nblkng_n==nblkng_max) nblkng_max+=nblkng_max/2+1,
                     nblkng_dict=trealloc(nblkng,nblkng_dict,nblkng_max);

  nblkng_dict[nblkng_n] = tmalloc(struct nonblocking_private, 1);

  nblkng_dict[nblkng_n]->u = u;
  nblkng_dict[nblkng_n]->dom = dom;
  nblkng_dict[nblkng_n]->op = op;
  nblkng_dict[nblkng_n]->transpose = transpose;
  nblkng_dict[nblkng_n]->gsh = gsh;
  nblkng_dict[nblkng_n]->buf = buf;
  nblkng_dict[nblkng_n]->mode = mode_plain;
  nblkng_dict[nblkng_n]->vn = 1;

  *handle = nblkng_n++;
  nblkng_count++;

  gs_aux_irecv(u,mode_plain,1,dom,op,transpose,gsh,buf);
  gs_aux_isend(u,mode_plain,1,dom,op,transpose,gsh,buf);
}

void gs_wait(int handle)
{
  if(handle < nblkng_n) {
    gs_aux_wait(nblkng_dict[handle]->u,
	        nblkng_dict[handle]->mode,
	        nblkng_dict[handle]->vn,
	        nblkng_dict[handle]->dom,
	        nblkng_dict[handle]->op,
	        nblkng_dict[handle]->transpose,
	        nblkng_dict[handle]->gsh,
	        nblkng_dict[handle]->buf);
    free(nblkng_dict[handle]);
    nblkng_dict[handle] = 0;
    nblkng_count--;
  }

  if(nblkng_count == 0) {
    free(nblkng_dict);
    nblkng_dict = 0;
    nblkng_max = 0;
    nblkng_n = 0;
  }
}
/*------------------------------------------------------------------------------
  GS_VEC interface - blocking and non-blocking
------------------------------------------------------------------------------*/
void gs_vec(void *u, unsigned vn, gs_dom dom, gs_op op,
            unsigned transpose, struct gs_data *gsh, buffer *buf)
{
  gs_aux(u,mode_vec,vn,dom,op,transpose,gsh,buf);
}

void igs_vec(void *u, unsigned vn, gs_dom dom, gs_op op,
            unsigned transpose, struct gs_data *gsh, buffer *buf, int *handle)
{
  if(nblkng_n==nblkng_max) nblkng_max+=nblkng_max/2+1,
                     nblkng_dict=trealloc(nblkng,nblkng_dict,nblkng_max);

  nblkng_dict[nblkng_n] = tmalloc(struct nonblocking_private, 1);

  nblkng_dict[nblkng_n]->u = u;
  nblkng_dict[nblkng_n]->dom = dom;
  nblkng_dict[nblkng_n]->op = op;
  nblkng_dict[nblkng_n]->transpose = transpose;
  nblkng_dict[nblkng_n]->gsh = gsh;
  nblkng_dict[nblkng_n]->buf = buf;
  nblkng_dict[nblkng_n]->vn = vn;
  nblkng_dict[nblkng_n]->mode = mode_vec;

  *handle = nblkng_n++;
  nblkng_count++;

  gs_aux_irecv(u,mode_vec,vn,dom,op,transpose,gsh,buf);
  gs_aux_isend(u,mode_vec,vn,dom,op,transpose,gsh,buf);
}
/*------------------------------------------------------------------------------
  GS_MANY interface - blocking and non-blocking
------------------------------------------------------------------------------*/
void gs_many(void *const*u, unsigned vn, gs_dom dom, gs_op op,
             unsigned transpose, struct gs_data *gsh, buffer *buf)
{
  gs_aux((void*)u,mode_many,vn,dom,op,transpose,gsh,buf);
}

void igs_many(void *const*u, unsigned vn, gs_dom dom, gs_op op,
             unsigned transpose, struct gs_data *gsh, buffer *buf, int *handle)
{
  if(nblkng_n==nblkng_max) nblkng_max+=nblkng_max/2+1,
                     nblkng_dict=trealloc(nblkng,nblkng_dict,nblkng_max);

  nblkng_dict[nblkng_n] = tmalloc(struct nonblocking_private, 1);

  nblkng_dict[nblkng_n]->u = (void *)u;
  nblkng_dict[nblkng_n]->dom = dom;
  nblkng_dict[nblkng_n]->op = op;
  nblkng_dict[nblkng_n]->transpose = transpose;
  nblkng_dict[nblkng_n]->gsh = gsh;
  nblkng_dict[nblkng_n]->buf = buf;
  nblkng_dict[nblkng_n]->vn = vn;
  nblkng_dict[nblkng_n]->mode = mode_many;

  *handle = nblkng_n++;
  nblkng_count++;

  gs_aux_irecv((void *)u,mode_many,vn,dom,op,transpose,gsh,buf);
  gs_aux_isend((void *)u,mode_many,vn,dom,op,transpose,gsh,buf);
}

/*------------------------------------------------------------------------------
  Main Setup
------------------------------------------------------------------------------*/
typedef enum {gs_auto, gs_pairwise, gs_crystal_router, gs_all_reduce} gs_method;

static uint local_setup(struct gs_data *gsh, const struct array *nz)
{
  uint mem_size = 0;
  gsh->map_local[0] = local_map(nz,1, &mem_size);
  gsh->map_local[1] = local_map(nz,0, &mem_size);
  gsh->flagged_primaries = flagged_primaries_map(nz, &mem_size);
  return mem_size;
}

static void gs_setup_aux(struct gs_data *gsh, const slong *id, uint n,
                         int unique, gs_method method, int verbose)
{
  static setup_fun *const remote_setup[] =
    { &auto_setup, &pw_setup, &cr_setup, &allreduce_setup };

  struct gs_topology top;
  struct crystal cr;

  crystal_init(&cr,&gsh->comm);

  get_topology(&top, id,n, &cr);
  if(unique) make_topology_unique(&top,0,gsh->comm.id,&cr.data);

  gsh->handle_size = sizeof(struct gs_data);
  gsh->handle_size += local_setup(gsh,&top.nz);

  if(verbose && gsh->comm.id==0)
    printf("gs_setup: %ld unique labels shared\n",(long)top.total_shared);

  remote_setup[method](&gsh->r, &top,&gsh->comm,&cr.data);
  gsh->handle_size += gsh->r.mem_size;

  if(verbose) { /* report memory usage */

    double avg[2],td[2]; uint min[2],max[2],ti[2];
    avg[0] = min[0] = max[0] = gsh->handle_size;
    avg[1] = min[1] = max[1] = sizeof(double)*gsh->r.buffer_size;
    avg[0] /= gsh->comm.np; avg[1] /= gsh->comm.np;
    comm_allreduce(&gsh->comm,gs_double,gs_add, avg,2, td);
    comm_allreduce(&gsh->comm,gs_sint,gs_min, min,2, ti);
    comm_allreduce(&gsh->comm,gs_sint,gs_max, max,2, ti);
    if(gsh->comm.id==0) {
      printf("   " "handle bytes (avg, min, max)" ": " "%g %u %u\n",
        avg[0], (unsigned)min[0], (unsigned)max[0]);
      printf("   " "buffer bytes (avg, min, max)" ": " "%g %u %u\n",
        avg[1], (unsigned)min[1], (unsigned)max[1]);
    }
  }

  gs_topology_free(&top);
  crystal_free(&cr);
  fflush(stdout);
}

struct gs_data *gs_setup(const slong *id, uint n, const struct comm *comm,
                         int unique, gs_method method, int verbose)
{
  struct gs_data *gsh = tmalloc(struct gs_data,1);
  comm_dup(&gsh->comm,comm);
  gs_setup_aux(gsh,id,n,unique,method,verbose);
  return gsh;
}

void gs_free(struct gs_data *gsh)
{
  comm_free(&gsh->comm);
  free((uint*)gsh->map_local[0]), free((uint*)gsh->map_local[1]);
  free((uint*)gsh->flagged_primaries);
  gsh->r.fin(gsh->r.data);
  free(gsh);
}

void gs_unique(slong *id, uint n, const struct comm *comm)
{
  struct gs_topology top;
  struct crystal cr;
  crystal_init(&cr,comm);
  get_topology(&top, id,n, &cr);
  make_topology_unique(&top,id,comm->id,&cr.data);
  gs_topology_free(&top);
  crystal_free(&cr);
}

void pw_data_nmsg(struct gs_data *gsh, int *n)
{
  struct gs_remote *r = &gsh->r;
  const struct pw_data *pwd = r->data;
  *n = pwd->comm[0].n;
}

void pw_data_size(struct gs_data *gsh, int *n)
{
  int i;
  struct gs_remote *r = &gsh->r;
  const struct pw_data *pwd = r->data;
  int ns = pwd->comm[0].n;
  for(i=0; i< ns; ++i) n[i] = pwd->comm[0].size[i];
}

/*------------------------------------------------------------------------------
  FORTRAN interface
------------------------------------------------------------------------------*/

#undef gs_op

#undef gs_unique
#undef gs_free
#undef gs_setup
#undef gs_many
#undef gs_vec
#undef gs
#undef igs
#undef igs_vec
#undef igs_many
#undef gs_wait

#define cgs         GS_PREFIXED_NAME(gs      )
#define cgs_vec     GS_PREFIXED_NAME(gs_vec  )
#define cgs_many    GS_PREFIXED_NAME(gs_many )
#define cgs_setup   GS_PREFIXED_NAME(gs_setup)
#define cgs_free    GS_PREFIXED_NAME(gs_free )
#define cgs_unique  GS_PREFIXED_NAME(gs_unique)
#define cigs        GS_PREFIXED_NAME(igs     )
#define cigs_vec    GS_PREFIXED_NAME(igs_vec )
#define cigs_many   GS_PREFIXED_NAME(igs_many)
#define cgs_wait    GS_PREFIXED_NAME(gs_wait )

#define fgs_setup_pick GS_FORTRAN_NAME(gs_setup_pick,GS_SETUP_PICK)
#define fgs_setup      GS_FORTRAN_NAME(gs_setup     ,GS_SETUP     )
#define fgs            GS_FORTRAN_NAME(gs_op        ,GS_OP        )
#define fgs_vec        GS_FORTRAN_NAME(gs_op_vec    ,GS_OP_VEC    )
#define fgs_many       GS_FORTRAN_NAME(gs_op_many   ,GS_OP_MANY   )
#define figs           GS_FORTRAN_NAME(igs_op       ,IGS_OP       )
#define figs_vec       GS_FORTRAN_NAME(igs_op_vec   ,IGS_OP_VEC   )
#define figs_many      GS_FORTRAN_NAME(igs_op_many  ,IGS_OP_MANY  )
#define fgs_wait       GS_FORTRAN_NAME(gs_op_wait   ,GS_OP_WAIT   )
#define fgs_fields     GS_FORTRAN_NAME(gs_op_fields ,GS_OP_FIELDS )
#define fgs_free       GS_FORTRAN_NAME(gs_free      ,GS_FREE      )
#define fgs_unique     GS_FORTRAN_NAME(gs_unique    ,GS_UNIQUE    )

static struct gs_data **fgs_info = 0;
static int fgs_max = 0;
static int fgs_n = 0;

struct gs_data* gs_hf2c(const sint gsh)
{
  return fgs_info[gsh];
}

void fgs_setup_pick(sint *handle, const slong id[], const sint *n,
                    const MPI_Fint *comm, const sint *np, const sint *method)
{
  struct gs_data *gsh;
  if(fgs_n==fgs_max) fgs_max+=fgs_max/2+1,
                     fgs_info=trealloc(struct gs_data*,fgs_info,fgs_max);
  gsh=fgs_info[fgs_n]=tmalloc(struct gs_data,1);
  comm_init_check(&gsh->comm,*comm,*np);
  gs_setup_aux(gsh,id,*n,0,*method,1);
  *handle = fgs_n++;
}

void fgs_setup(sint *handle, const slong id[], const sint *n,
               const MPI_Fint *comm, const sint *np)
{
  const sint method = gs_auto;
  fgs_setup_pick(handle,id,n,comm,np,&method);
}

static void fgs_check_handle(sint handle, const char *func, unsigned line)
{
  if(handle<0 || handle>=fgs_n || !fgs_info[handle])
    fail(1,__FILE__,line,"%s: invalid handle", func);
}

static const gs_dom fgs_dom[4] = { gs_float, gs_double, gs_sint, gs_slong };

static void fgs_check_parms(sint handle, sint dom, sint op,
                            const char *func, unsigned line)
{
  if(dom<1 || dom>3)
    fail(1,__FILE__,line,"%s: datatype %d not in valid range 1-3",func,dom);
  if(op <1 || op >4)
    fail(1,__FILE__,line,"%s: op %d not in valid range 1-4",func,op);
  fgs_check_handle(handle,func,line);
}

void fgs(const sint *handle, void *u, const sint *dom, const sint *op,
         const sint *transpose)
{
  fgs_check_parms(*handle,*dom,*op,"gs_op",__LINE__);
  cgs(u,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,fgs_info[*handle],0);
}

void figs(const sint *handle, void *u, const sint *dom, const sint *op,
         const sint *transpose, int *wait)
{
  fgs_check_parms(*handle,*dom,*op,"gs_op",__LINE__);
  cigs(u,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,fgs_info[*handle],0,wait);
}

void fgs_vec(const sint *handle, void *u, const sint *n,
             const sint *dom, const sint *op, const sint *transpose)
{
  fgs_check_parms(*handle,*dom,*op,"gs_op_vec",__LINE__);
  cgs_vec(u,*n,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,
          fgs_info[*handle],0);
}

void figs_vec(const sint *handle, void *u, const sint *n,
             const sint *dom, const sint *op, const sint *transpose, int *wait)
{
  fgs_check_parms(*handle,*dom,*op,"gs_op_vec",__LINE__);
  cigs_vec(u,*n,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,
          fgs_info[*handle],0,wait);
}

void fgs_many(const sint *handle, void *u1, void *u2, void *u3,
              void *u4, void *u5, void *u6, const sint *n,
              const sint *dom, const sint *op, const sint *transpose)
{
  void *uu[6];
  uu[0]=u1,uu[1]=u2,uu[2]=u3,uu[3]=u4,uu[4]=u5,uu[5]=u6;
  fgs_check_parms(*handle,*dom,*op,"gs_op_many",__LINE__);
  cgs_many((void *const*)uu,*n,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,
           fgs_info[*handle],0);
}

void figs_many(const sint *handle, void *u1, void *u2, void *u3,
              void *u4, void *u5, void *u6, const sint *n,
              const sint *dom, const sint *op, const sint *transpose,
	      int *wait)
{
  void *uu[6];
  uu[0]=u1,uu[1]=u2,uu[2]=u3,uu[3]=u4,uu[4]=u5,uu[5]=u6;
  fgs_check_parms(*handle,*dom,*op,"gs_op_many",__LINE__);
  cigs_many((void *const*)uu,*n,fgs_dom[*dom],(gs_op_t)(*op-1),*transpose!=0,
           fgs_info[*handle],0,wait);
}

void fgs_wait(int *handle)
{
  cgs_wait(*handle);
}

static struct array fgs_fields_array = null_array;

void fgs_fields(const sint *handle,
                void *u, const sint *stride, const sint *n,
                const sint *dom, const sint *op, const sint *transpose)
{
  size_t offset;
  void **p;
  uint i;

  fgs_check_parms(*handle,*dom,*op,"gs_op_fields",__LINE__);
  if(*n<0) return;

  array_reserve(void*,&fgs_fields_array,*n);
  p = fgs_fields_array.ptr;
  offset = *stride * gs_dom_size[*dom-1];
  for(i=*n;i;--i) *p++ = u, u = (char*)u + offset;

  cgs_many((void *const*)fgs_fields_array.ptr,*n,
           fgs_dom[*dom],(gs_op_t)(*op-1),
           *transpose!=0, fgs_info[*handle],0);
}

void fgs_free(const sint *handle)
{
  fgs_check_handle(*handle,"gs_free",__LINE__);
  cgs_free(fgs_info[*handle]);
  fgs_info[*handle] = 0;
}

void fgs_unique(slong id[], const sint *n, const MPI_Fint *c, const sint *np)
{
  struct comm *comm;
  uint un = *n;
  comm = tmalloc(struct comm, 1);
  comm_init_check(comm, *c, *np);
  cgs_unique(id, un, comm);
  free(comm);
}

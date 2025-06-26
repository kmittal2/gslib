#ifndef GS_FINDPTS_LOCAL_HASH_H
#define GS_FINDPTS_LOCAL_HASH_H

#if !defined(GS_NAME_H)
#warning "local_hash.h" requires "name.h"
#endif

/*--------------------------------------------------------------------------
   Point to Possible Elements Hashing

   Initializing the data:
     uint nel;        // number of elements
     uint max_size = nr*ns*nt*nel; // maximum size of hash table
     struct obbox *obb = ...; // bounding boxes for elements

     hash_data data;
     local_hash_build(&data, obb, nel, max_size);

   Using the data:
     double x[3];   // point to find

     uint index = local_hash_index_3(&data, x);
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
     local_hash_free(&data);

  --------------------------------------------------------------------------*/

struct findpts_local_hash_data_2 {
  uint hash_n;
  struct dbl_range bnd[2];
  double fac[2];
  uint *offset;
  uint max;
};

uint local_hash_index_2(const struct findpts_local_hash_data_2 *p, const double x[2]);

void local_hash_build_2(struct findpts_local_hash_data_2 *p,
                  const struct obbox_2 *const obb, const uint nel,
                  const uint max_size);


void local_hash_build_nx_2(struct findpts_local_hash_data_2 *p,
                           const struct obbox_2 *const obb, const uint nel,
                           const uint nm);

void local_hash_free_2(struct findpts_local_hash_data_2 *p);

struct findpts_local_hash_data_3 {
  uint hash_n;
  struct dbl_range bnd[3];
  double fac[3];
  uint *offset;
  uint max;
};

uint local_hash_index_3(const struct findpts_local_hash_data_3 *p, const double x[3]);

void local_hash_build_3(struct findpts_local_hash_data_3 *p,
                  const struct obbox_3 *const obb, const uint nel,
                  const uint max_size);

void local_hash_build_nx_3(struct findpts_local_hash_data_3 *p,
                           const struct obbox_3 *const obb, const uint nel,
                           const uint nm);

void local_hash_free_3(struct findpts_local_hash_data_3 *p);

#endif
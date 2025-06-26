#ifndef GS_FINDPTS_HASH_H
#define GS_FINDPTS_HASH_H

#if !defined(GS_NAME_H)
#warning "global_hash.h" requires "name.h"
#endif

struct findpts_hash_data_2 {
  uint hash_n;
  struct dbl_range bnd[2];
  double fac[2];
  uint *offset;
};

uint hash_index_2(const struct findpts_hash_data_2 *p, const double x[2]);

void hash_build_2(struct findpts_hash_data_2 *const p,
                  const struct findpts_local_hash_data_2 *const lp,
                  const struct obbox_2 *const obb, const uint nel,
                  const uint hash_size,
                  struct crystal *cr);


void hash_build_nx_2(struct findpts_hash_data_2 *const p,
                     const struct findpts_local_hash_data_2 *const lp,
                     const struct obbox_2 *const obb, const uint nel,
                     const uint nm,
                     struct crystal *cr);

void hash_free_2(struct findpts_hash_data_2 *p);

struct findpts_hash_data_3 {
  uint hash_n;
  struct dbl_range bnd[3];
  double fac[3];
  uint *offset;
};

uint hash_index_3(const struct findpts_hash_data_3 *p, const double x[3]);

void hash_build_3(struct findpts_hash_data_3 *const p,
                  const struct findpts_local_hash_data_3 *const lp,
                  const struct obbox_3 *const obb, const uint nel,
                  const uint hash_size,
                  struct crystal *cr);


void hash_build_nx_3(struct findpts_hash_data_3 *const p,
                     const struct findpts_local_hash_data_3 *const lp,
                     const struct obbox_3 *const obb, const uint nel,
                     const uint nm,
                     struct crystal *cr);

void hash_free_3(struct findpts_hash_data_3 *p);

#endif
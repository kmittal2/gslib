#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "gslib.h"

/* Helper to check values relative to epsilon */
#define ABS(x) ((x)<0 ? -(x) : (x))
#define TEQ(a,b) (ABS((a)-(b)) < 1e-9)

int main(int narg, char *arg[])
{
  int i;
  comm_ext world; int np;
  struct comm comm;
  struct crystal cr;
  int id = 0;
#ifdef GSLIB_USE_MPI
  MPI_Init(&narg,&arg);
  world = MPI_COMM_WORLD;
  MPI_Comm_size(world,&np);
  MPI_Comm_rank(world,&id);
#else
  world=0, np=1;
#endif

  comm_init(&comm, world);
  crystal_init(&cr, &comm);

  /* Field 1: int, Field 2: uint [2], Field 3: double,
     Field 4: double[3] (coords)
  */
  int n_fields = 4;
  size_t sizes[] = {sizeof(int), sizeof(uint)*2, sizeof(double), 3*sizeof(double)};

  /* Each proc sends 1 message to every proc (including self) */
  uint n_in = (uint)np;

  /* Allocate inputs */
  uint *dest = tmalloc(uint, n_in);

  void **data_send = tmalloc(void*, n_fields);
  int *in_0 = tmalloc(int, n_in);
  uint *in_1 = tmalloc(uint, n_in * 2); /* 2 uints per entry */
  double *in_2 = tmalloc(double, n_in);
  double *in_3 = tmalloc(double, n_in * 3); /* 3D coords per entry */

  data_send[0] = in_0;
  data_send[1] = in_1;
  data_send[2] = in_2;
  data_send[3] = in_3;

  /* Fill Data */
  for(i=0; i<np; ++i) {
    int target = i;
    dest[i] = target;

    in_0[i] = id * 1000 + target;
    in_1[2*i] = (uint)(id * 10000 + target);
    in_1[2*i+1] = (uint)(id * 20000 + target);
    in_2[i] = (double)id + (double)target * 0.01;
    in_3[3*i + 0] = (double)id + 0.1;
    in_3[3*i + 1] = (double)id + 0.2;
    in_3[3*i + 2] = (double)id + 0.3;
  }

  printf("Proc %d sending %d items (All-to-All)\n", id, n_in);

  /* Packs data and performs transfer */
  uint n_out = sarray_transfer_soa_to_buffer(&cr, n_in, dest, n_fields, sizes, data_send);

  printf("Proc %d received %d items\n", id, n_out);
  comm_barrier(&comm);

  /* Allocate memory for output */
  uint *rank_recv = NULL;
  void **data_recv = tmalloc(void*, n_fields);

  if(n_out > 0) {
    rank_recv = tmalloc(uint, n_out);
    data_recv[0] = tmalloc(int, n_out);
    data_recv[1] = tmalloc(uint, n_out * 2);
    data_recv[2] = tmalloc(double, n_out);
    data_recv[3] = tmalloc(double, n_out * 3);
  } else {
    data_recv[0] = NULL;
    data_recv[1] = NULL;
    data_recv[2] = NULL;
    data_recv[3] = NULL;
  }

  /* Unpack buffer into user allocated memory */
  sarray_transfer_unpack_buffer_to_soa(&cr, n_out, n_fields, sizes, rank_recv, data_recv);

  /* Verification */
  if(n_out != (uint)np) {
    if(id==0) printf("Proc %d ERROR: Expected %d items, got %d\n", id, np, n_out);
  } else {
    for(i=0; i<n_out; ++i) {
      int src = rank_recv[i]; /* Should be source rank */

      int val0 = ((int*)data_recv[0])[i];
      uint val1_a = ((uint*)data_recv[1])[2*i];
      uint val1_b = ((uint*)data_recv[1])[2*i+1];
      double val2 = ((double*)data_recv[2])[i];
      double *val3 = &((double*)data_recv[3])[3*i];

      int exp0 = src * 1000 + id;
      uint exp1_a = src * 10000 + id;
      uint exp1_b = src * 20000 + id;
      double exp2 = (double)src + (double)id * 0.01;

      double exp3_x = (double)src + 0.1;
      double exp3_y = (double)src + 0.2;
      double exp3_z = (double)src + 0.3;

      if(val0 != exp0) printf("Proc %d ERR: Field 0 (int) expected %d got %d\n", id, exp0, val0);
      if(val1_a != exp1_a) printf("Proc %d ERR: Field 1 (uint) expected %u got %u\n", id, exp1_a, val1_a);
      if(val1_b != exp1_b) printf("Proc %d ERR: Field 1 (uint) expected %u got %u\n", id, exp1_b, val1_b);
      if(!TEQ(val2, exp2)) printf("Proc %d ERR: Field 2 (double) expected %f got %f\n", id, exp2, val2);

      if(!TEQ(val3[0], exp3_x)) printf("Proc %d ERR: Field 3 (x) expected %f got %f\n", id, exp3_x, val3[0]);
      if(!TEQ(val3[1], exp3_y)) printf("Proc %d ERR: Field 3 (y) expected %f got %f\n", id, exp3_y, val3[1]);
      if(!TEQ(val3[2], exp3_z)) printf("Proc %d ERR: Field 3 (z) expected %f got %f\n", id, exp3_z, val3[2]);
    }
  }

  printf("Proc %d finished verification.\n", id);

  /* Cleanup inputs */
  if(dest) free(dest);
  if(data_send) {
      free(in_0); free(in_1); free(in_2); free(in_3);
      free(data_send);
  }

  /* Cleanup outputs */
  if(rank_recv) free(rank_recv);
  if(data_recv[0]) free(data_recv[0]);
  if(data_recv[1]) free(data_recv[1]);
  if(data_recv[2]) free(data_recv[2]);
  if(data_recv[3]) free(data_recv[3]);
  if(data_recv) free(data_recv);

  crystal_free(&cr);
  comm_free(&comm);
  MPI_Finalize();
  return 0;
}

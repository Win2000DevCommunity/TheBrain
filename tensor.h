/* ============================================================
 * FILE: tensor.h
 * ============================================================ */
#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TB_MAX_RANK  8
#define TB_MAX_DIMS  8

typedef enum { DT_FLOAT32=0, DT_INT8=1, DT_INT32=2, DT_UINT8=3 } TBDtype;
typedef enum { DEV_CPU=0 } TBDevice;

#define TB_FLAG_VIEW   0x01
#define TB_FLAG_GRAD   0x02
#define TB_FLAG_PINNED 0x04
#define TB_FLAG_QUANT  0x08

typedef struct Tensor {
    void           *data;
    float          *grad;
    float          *quant_scales;
    int             shape[TB_MAX_RANK];
    int             strides[TB_MAX_RANK];
    int             rank;
    int             n_elem;
    TBDtype         dtype;
    TBDevice        device;
    unsigned char   flags;
    char            name[64];
} Tensor;

#define TB_POOL_MAX 4096

typedef struct TensorPool {
    Tensor   slots[TB_POOL_MAX];
    int      used;
    float   *data_arena;
    size_t   arena_used;
    size_t   arena_cap;
} TensorPool;

#ifdef __cplusplus
extern "C" {
#endif

int    tb_pool_init        (TensorPool *p, size_t arena_bytes);
void   tb_pool_reset       (TensorPool *p);
void   tb_pool_destroy     (TensorPool *p);
Tensor *tb_tensor_alloc    (TensorPool *p, int rank, const int *shape,
                             TBDtype dtype, const char *name);
Tensor *tb_tensor_view     (TensorPool *p, Tensor *src,
                             int rank, const int *shape, const char *name);
Tensor *tb_tensor_zeros    (TensorPool *p, int rank, const int *shape,
                             TBDtype dtype, const char *name);
Tensor *tb_tensor_like     (TensorPool *p, const Tensor *t, const char *name);
int    tb_tensor_n_elem    (int rank, const int *shape);
void   tb_tensor_compute_strides(int rank, const int *shape, int *strides);
int    tb_tensor_same_shape(const Tensor *a, const Tensor *b);
int    tb_tensor_alloc_grad(Tensor *t);
void   tb_tensor_zero_grad (Tensor *t);
void   tb_tensor_free_grad (Tensor *t);
float *tb_f32              (Tensor *t);
signed char *tb_i8         (Tensor *t);
void   tb_tensor_print_shape(const Tensor *t);

#ifdef __cplusplus
}
#endif
#endif /* TENSOR_H */

#include "tensor.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

int tb_tensor_n_elem(int rank, const int *shape)
{
    int i, n = 1;
    for (i = 0; i < rank; i++) n *= shape[i];
    return n;
}

void tb_tensor_compute_strides(int rank, const int *shape, int *strides)
{
    int i;
    strides[rank - 1] = 1;
    for (i = rank - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
}

int tb_tensor_same_shape(const Tensor *a, const Tensor *b)
{
    int i;
    if (a->rank != b->rank) return 0;
    for (i = 0; i < a->rank; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

float *tb_f32(Tensor *t)      { return (float *)t->data; }
signed char *tb_i8(Tensor *t) { return (signed char *)t->data; }

int tb_pool_init(TensorPool *p, size_t arena_bytes)
{
    memset(p, 0, sizeof(TensorPool));
    p->data_arena = (float *)malloc(arena_bytes);
    if (!p->data_arena) return 0;
    p->arena_cap  = arena_bytes / sizeof(float);
    p->arena_used = 0;
    p->used       = 0;
    return 1;
}

void tb_pool_reset(TensorPool *p)
{
    int i;
    for (i = 0; i < p->used; i++) {
        Tensor *t = &p->slots[i];
        if (t->grad && !(t->flags & TB_FLAG_VIEW)) {
            free(t->grad); t->grad = NULL;
        }
        if (t->quant_scales && !(t->flags & TB_FLAG_VIEW)) {
            free(t->quant_scales); t->quant_scales = NULL;
        }
    }
    p->used       = 0;
    p->arena_used = 0;
}

void tb_pool_destroy(TensorPool *p)
{
    tb_pool_reset(p);
    free(p->data_arena);
    p->data_arena = NULL;
    p->arena_cap  = 0;
}

static float *pool_arena_alloc(TensorPool *p, size_t n_float)
{
    float *ptr;
    size_t aligned = (n_float + 1) & ~(size_t)1;
    if (p->arena_used + aligned > p->arena_cap) return NULL;
    ptr = p->data_arena + p->arena_used;
    p->arena_used += aligned;
    return ptr;
}

Tensor *tb_tensor_alloc(TensorPool *p, int rank,
                         const int *shape, TBDtype dtype,
                         const char *name)
{
    Tensor *t;
    int     n;
    size_t  bytes;

    if (p->used >= TB_POOL_MAX) return NULL;
    if (rank < 1 || rank > TB_MAX_RANK) return NULL;

    t = &p->slots[p->used++];
    memset(t, 0, sizeof(Tensor));

    memcpy(t->shape, shape, rank * sizeof(int));
    t->rank   = rank;
    t->dtype  = dtype;
    t->device = DEV_CPU;
    t->n_elem = tb_tensor_n_elem(rank, shape);
    tb_tensor_compute_strides(rank, shape, t->strides);

    if (name) { strncpy(t->name, name, 63); t->name[63] = '\0'; }

    switch (dtype) {
        case DT_FLOAT32: bytes=(size_t)t->n_elem*sizeof(float); break;
        case DT_INT8:    bytes=(size_t)t->n_elem*sizeof(signed char); break;
        case DT_INT32:   bytes=(size_t)t->n_elem*sizeof(int); break;
        case DT_UINT8:   bytes=(size_t)t->n_elem*sizeof(unsigned char); break;
        default:         bytes=(size_t)t->n_elem*sizeof(float); break;
    }

    if (dtype == DT_FLOAT32) {
        float *arena_ptr = pool_arena_alloc(p, (size_t)t->n_elem);
        if (arena_ptr) {
            t->data  = arena_ptr;
            t->flags |= TB_FLAG_VIEW;
        }
    }

    if (!t->data) {
        t->data = malloc(bytes);
        if (!t->data) { p->used--; return NULL; }
        t->flags &= ~TB_FLAG_VIEW;
    }

    memset(t->data, 0, bytes);
    (void)n;
    return t;
}

Tensor *tb_tensor_zeros(TensorPool *p, int rank, const int *shape,
                         TBDtype dtype, const char *name)
{
    return tb_tensor_alloc(p, rank, shape, dtype, name);
}

Tensor *tb_tensor_view(TensorPool *p, Tensor *src,
                        int rank, const int *shape, const char *name)
{
    Tensor *t;
    int     new_n;

    if (!src || !src->data) return NULL;
    new_n = tb_tensor_n_elem(rank, shape);
    if (new_n != src->n_elem) return NULL;

    if (p->used >= TB_POOL_MAX) return NULL;
    t = &p->slots[p->used++];
    memset(t, 0, sizeof(Tensor));

    t->data   = src->data;
    t->dtype  = src->dtype;
    t->device = src->device;
    t->rank   = rank;
    t->n_elem = new_n;
    t->flags  = TB_FLAG_VIEW;
    memcpy(t->shape, shape, rank * sizeof(int));
    tb_tensor_compute_strides(rank, shape, t->strides);
    if (name) { strncpy(t->name, name, 63); t->name[63] = '\0'; }
    return t;
}

Tensor *tb_tensor_like(TensorPool *p, const Tensor *src, const char *name)
{
    return tb_tensor_alloc(p, src->rank, src->shape, src->dtype, name);
}

int tb_tensor_alloc_grad(Tensor *t)
{
    if (t->grad) return 1;
    if (t->dtype != DT_FLOAT32) return 0;
    t->grad = (float *)calloc((size_t)t->n_elem, sizeof(float));
    return t->grad != NULL;
}

void tb_tensor_zero_grad(Tensor *t)
{
    if (t->grad)
        memset(t->grad, 0, (size_t)t->n_elem * sizeof(float));
}

void tb_tensor_free_grad(Tensor *t)
{
    free(t->grad);
    t->grad = NULL;
    t->flags &= ~TB_FLAG_GRAD;
}

void tb_tensor_print_shape(const Tensor *t)
{
    int i;
    printf("[%s] rank=%d shape=(", t->name, t->rank);
    for (i = 0; i < t->rank; i++) {
        printf("%d", t->shape[i]);
        if (i < t->rank - 1) printf(",");
    }
    printf(") dtype=%d n_elem=%d\n", (int)t->dtype, t->n_elem);
}

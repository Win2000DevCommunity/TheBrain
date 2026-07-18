#include "tensor.h"
/* ============================================================
 * FILE: graph.h
 * ============================================================ */
#ifndef GRAPH_H
#define GRAPH_H

typedef enum {
    OP_NONE=0, OP_MATMUL=1, OP_MATMUL_T=2, OP_ADD=3,
    OP_SCALE=4, OP_RMSNORM=5, OP_LAYERNORM=6, OP_SOFTMAX=7,
    OP_GELU=8, OP_SWIGLU=9, OP_ROPE=10, OP_ATTN=11,
    OP_EMBED=12, OP_LM_HEAD=13, OP_DROPOUT=14, OP_CAUSAL_MASK=15,
    OP_QUANTIZE=16, OP_DEQUANTIZE=17, OP_COPY=18, OP_TRANSPOSE=19,
    OP_RESHAPE=20, OP_LOSS_CE=21, OP_MAX=32
} OpType;

#define GRAPH_MAX_INPUTS  8
#define GRAPH_MAX_OUTPUTS 4
#define GRAPH_MAX_NODES   4096
#define GRAPH_MAX_ATTRS   16

typedef union { float f; int i; double d; } AttrVal;

typedef struct {
    char    key[32];
    AttrVal val;
    int     is_float;
} NodeAttr;

typedef struct GraphNode {
    OpType   op;
    int      inputs[GRAPH_MAX_INPUTS];
    int      outputs[GRAPH_MAX_OUTPUTS];
    int      n_inputs;
    int      n_outputs;
    NodeAttr attrs[GRAPH_MAX_ATTRS];
    int      n_attrs;
    char     name[64];
    int      visited;
} GraphNode;

typedef struct Graph Graph;
typedef int (*OpForwardFn) (Graph *g, GraphNode *node);
typedef int (*OpBackwardFn)(Graph *g, GraphNode *node);

typedef struct {
    OpType       id;
    const char  *name;
    OpForwardFn  forward;
    OpBackwardFn backward;
} OpEntry;

struct Graph {
    GraphNode    nodes[GRAPH_MAX_NODES];
    int          n_nodes;
    Tensor      *tensors[TB_POOL_MAX];
    int          n_tensors;
    TensorPool  *pool;
    int          training;
    int          topo_order[GRAPH_MAX_NODES];
    int          topo_len;
};

#ifdef __cplusplus
extern "C" {
#endif

void   graph_register_op     (OpType id, const char *name,
                               OpForwardFn fwd, OpBackwardFn bwd);
const OpEntry *graph_get_op  (OpType id);
void   graph_init            (Graph *g, TensorPool *pool, int training);
void   graph_reset           (Graph *g);
int    graph_add_tensor      (Graph *g, Tensor *t);
Tensor *graph_tensor         (Graph *g, int idx);
int    graph_add_node        (Graph *g, OpType op,
                               const int *in_ids,  int n_in,
                               const int *out_ids, int n_out,
                               const char *name);
int    graph_node_set_attr_f (Graph *g, int node_idx,
                               const char *key, float val);
int    graph_node_set_attr_i (Graph *g, int node_idx,
                               const char *key, int val);
float  graph_node_get_attr_f (GraphNode *node, const char *key, float def);
int    graph_node_get_attr_i (GraphNode *node, const char *key, int def);
int    graph_topo_sort       (Graph *g);
int    graph_forward         (Graph *g);
int    graph_backward        (Graph *g);
int    g_matmul   (Graph *g, int A, int B, const char *name);
int    g_matmul_t (Graph *g, int A, int B, const char *name);
int    g_add      (Graph *g, int A, int B, const char *name);
int    g_scale    (Graph *g, int A, float s, const char *name);
int    g_rmsnorm  (Graph *g, int X, int W, float eps, const char *name);
int    g_layernorm(Graph *g, int X, int W, int B_t, float eps, const char *name);
int    g_softmax  (Graph *g, int X, int axis, const char *name);
int    g_gelu     (Graph *g, int X, const char *name);
int    g_swiglu   (Graph *g, int gate, int up, const char *name);
int    g_rope     (Graph *g, int X, int pos, int d_head, const char *name);
int    g_attention(Graph *g, int Q, int K, int V,
                   int causal, float scale, const char *name);
int    g_embed    (Graph *g, int tokens, int W, const char *name);
int    g_lm_head  (Graph *g, int X, int W, const char *name);

#ifdef __cplusplus
}
#endif
#endif /* GRAPH_H */

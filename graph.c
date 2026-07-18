#include "graph.h"
#include "tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static OpEntry g_op_registry[OP_MAX];

void graph_register_op(OpType id, const char *name,
                        OpForwardFn fwd, OpBackwardFn bwd)
{
    if (id < 0 || id >= OP_MAX) return;
    g_op_registry[id].id       = id;
    g_op_registry[id].name     = name;
    g_op_registry[id].forward  = fwd;
    g_op_registry[id].backward = bwd;
}

const OpEntry *graph_get_op(OpType id)
{
    if (id < 0 || id >= OP_MAX) return NULL;
    return &g_op_registry[id];
}

void graph_init(Graph *g, TensorPool *pool, int training)
{
    memset(g, 0, sizeof(Graph));
    g->pool     = pool;
    g->training = training;
}

void graph_reset(Graph *g)
{
    g->n_nodes   = 0;
    g->n_tensors = 0;
    g->topo_len  = 0;
    if (g->pool) tb_pool_reset(g->pool);
}

int graph_add_tensor(Graph *g, Tensor *t)
{
    if (g->n_tensors >= TB_POOL_MAX) return -1;
    g->tensors[g->n_tensors] = t;
    return g->n_tensors++;
}

Tensor *graph_tensor(Graph *g, int idx)
{
    if (idx < 0 || idx >= g->n_tensors) return NULL;
    return g->tensors[idx];
}

int graph_add_node(Graph *g, OpType op,
                    const int *in_ids, int n_in,
                    const int *out_ids, int n_out,
                    const char *name)
{
    GraphNode *node;
    int i;
    if (g->n_nodes >= GRAPH_MAX_NODES) return -1;
    node = &g->nodes[g->n_nodes];
    memset(node, 0, sizeof(GraphNode));
    node->op        = op;
    node->n_inputs  = n_in  < GRAPH_MAX_INPUTS  ? n_in  : GRAPH_MAX_INPUTS;
    node->n_outputs = n_out < GRAPH_MAX_OUTPUTS ? n_out : GRAPH_MAX_OUTPUTS;
    for (i=0;i<node->n_inputs;  i++) node->inputs[i]  = in_ids[i];
    for (i=0;i<node->n_outputs; i++) node->outputs[i] = out_ids[i];
    if (name) { strncpy(node->name, name, 63); node->name[63]='\0'; }
    return g->n_nodes++;
}

int graph_node_set_attr_f(Graph *g, int nidx, const char *key, float val)
{
    GraphNode *node; NodeAttr *a;
    if (nidx<0||nidx>=g->n_nodes) return 0;
    node=&g->nodes[nidx];
    if (node->n_attrs>=GRAPH_MAX_ATTRS) return 0;
    a=&node->attrs[node->n_attrs++];
    strncpy(a->key,key,31); a->key[31]='\0';
    a->val.f=val; a->is_float=1;
    return 1;
}

int graph_node_set_attr_i(Graph *g, int nidx, const char *key, int val)
{
    GraphNode *node; NodeAttr *a;
    if (nidx<0||nidx>=g->n_nodes) return 0;
    node=&g->nodes[nidx];
    if (node->n_attrs>=GRAPH_MAX_ATTRS) return 0;
    a=&node->attrs[node->n_attrs++];
    strncpy(a->key,key,31); a->key[31]='\0';
    a->val.i=val; a->is_float=0;
    return 1;
}

float graph_node_get_attr_f(GraphNode *node, const char *key, float def)
{
    int i;
    for (i=0;i<node->n_attrs;i++)
        if (strcmp(node->attrs[i].key,key)==0 && node->attrs[i].is_float)
            return node->attrs[i].val.f;
    return def;
}

int graph_node_get_attr_i(GraphNode *node, const char *key, int def)
{
    int i;
    for (i=0;i<node->n_attrs;i++)
        if (strcmp(node->attrs[i].key,key)==0 && !node->attrs[i].is_float)
            return node->attrs[i].val.i;
    return def;
}

int graph_topo_sort(Graph *g)
{
    int in_degree[GRAPH_MAX_NODES];
    int queue[GRAPH_MAX_NODES];
    int q_head=0,q_tail=0,i,j;

    memset(in_degree,0,g->n_nodes*sizeof(int));
    g->topo_len=0;

    for (j=0;j<g->n_nodes;j++) {
        for (i=0;i<j;i++) {
            int found=0,ki,kj;
            for (ki=0;ki<g->nodes[i].n_outputs&&!found;ki++)
                for (kj=0;kj<g->nodes[j].n_inputs&&!found;kj++)
                    if (g->nodes[i].outputs[ki]==g->nodes[j].inputs[kj])
                        found=1;
            if (found) in_degree[j]++;
        }
    }

    for (i=0;i<g->n_nodes;i++)
        if (in_degree[i]==0) queue[q_tail++]=i;

    while (q_head<q_tail) {
        int cur=queue[q_head++];
        g->topo_order[g->topo_len++]=cur;
        for (j=cur+1;j<g->n_nodes;j++) {
            int found=0,ki,kj;
            for (ki=0;ki<g->nodes[cur].n_outputs&&!found;ki++)
                for (kj=0;kj<g->nodes[j].n_inputs&&!found;kj++)
                    if (g->nodes[cur].outputs[ki]==g->nodes[j].inputs[kj])
                        found=1;
            if (found) {
                in_degree[j]--;
                if (in_degree[j]==0) queue[q_tail++]=j;
            }
        }
    }
    return (g->topo_len==g->n_nodes)?1:0;
}

int graph_forward(Graph *g)
{
    int i;
    if (g->topo_len==0)
        if (!graph_topo_sort(g)) return 0;
    for (i=0;i<g->topo_len;i++) {
        int ni=g->topo_order[i];
        GraphNode *n=&g->nodes[ni];
        const OpEntry *e=graph_get_op(n->op);
        if (!e||!e->forward) return 0;
        if (!e->forward(g,n)) return 0;
    }
    return 1;
}

int graph_backward(Graph *g)
{
    int i;
    if (!g->training) return 0;
    if (g->topo_len==0) return 0;
    for (i=g->topo_len-1;i>=0;i--) {
        int ni=g->topo_order[i];
        GraphNode *n=&g->nodes[ni];
        const OpEntry *e=graph_get_op(n->op);
        if (!e||!e->backward) continue;
        e->backward(g,n);
    }
    return 1;
}

static int alloc_output_f32(Graph *g, const char *name,
                              int rank, const int *shape)
{
    Tensor *t=tb_tensor_zeros(g->pool,rank,shape,DT_FLOAT32,name);
    if (!t) return -1;
    return graph_add_tensor(g,t);
}

int g_matmul(Graph *g, int A, int B, const char *name)
{
    Tensor *ta=graph_tensor(g,A),*tb_t=graph_tensor(g,B);
    int out_shape[2],node_idx,in_ids[2],out_ids[1],out;
    if (!ta||!tb_t) return -1;
    out_shape[0]=ta->shape[ta->rank-2];
    out_shape[1]=tb_t->shape[tb_t->rank-1];
    out=alloc_output_f32(g,name,2,out_shape);
    if (out<0) return -1;
    in_ids[0]=A; in_ids[1]=B; out_ids[0]=out;
    node_idx=graph_add_node(g,OP_MATMUL,in_ids,2,out_ids,1,name);
    (void)node_idx;
    return out;
}

int g_matmul_t(Graph *g, int A, int B, const char *name)
{
    Tensor *ta=graph_tensor(g,A),*tb_t=graph_tensor(g,B);
    int out_shape[2],in_ids[2],out_ids[1],out;
    if (!ta||!tb_t) return -1;
    out_shape[0]=ta->shape[ta->rank-2];
    out_shape[1]=tb_t->shape[tb_t->rank-2];
    out=alloc_output_f32(g,name,2,out_shape);
    if (out<0) return -1;
    in_ids[0]=A; in_ids[1]=B; out_ids[0]=out;
    graph_add_node(g,OP_MATMUL_T,in_ids,2,out_ids,1,name);
    return out;
}

int g_add(Graph *g, int A, int B, const char *name)
{
    Tensor *ta=graph_tensor(g,A);
    int out,in_ids[2],out_ids[1];
    if (!ta) return -1;
    out=alloc_output_f32(g,name,ta->rank,ta->shape);
    if (out<0) return -1;
    in_ids[0]=A; in_ids[1]=B; out_ids[0]=out;
    graph_add_node(g,OP_ADD,in_ids,2,out_ids,1,name);
    return out;
}

int g_scale(Graph *g, int A, float s, const char *name)
{
    Tensor *ta=graph_tensor(g,A);
    int out,in_ids[1],out_ids[1],nidx;
    if (!ta) return -1;
    out=alloc_output_f32(g,name,ta->rank,ta->shape);
    if (out<0) return -1;
    in_ids[0]=A; out_ids[0]=out;
    nidx=graph_add_node(g,OP_SCALE,in_ids,1,out_ids,1,name);
    graph_node_set_attr_f(g,nidx,"scale",s);
    return out;
}

int g_rmsnorm(Graph *g, int X, int W, float eps, const char *name)
{
    Tensor *tx=graph_tensor(g,X);
    int out,in_ids[2],out_ids[1],nidx;
    if (!tx) return -1;
    out=alloc_output_f32(g,name,tx->rank,tx->shape);
    if (out<0) return -1;
    in_ids[0]=X; in_ids[1]=W; out_ids[0]=out;
    nidx=graph_add_node(g,OP_RMSNORM,in_ids,2,out_ids,1,name);
    graph_node_set_attr_f(g,nidx,"eps",eps);
    return out;
}

int g_layernorm(Graph *g, int X, int W, int B_t, float eps, const char *name)
{
    Tensor *tx=graph_tensor(g,X);
    int out,in_ids[3],out_ids[1],nidx;
    if (!tx) return -1;
    out=alloc_output_f32(g,name,tx->rank,tx->shape);
    if (out<0) return -1;
    in_ids[0]=X; in_ids[1]=W; in_ids[2]=B_t; out_ids[0]=out;
    nidx=graph_add_node(g,OP_LAYERNORM,in_ids,3,out_ids,1,name);
    graph_node_set_attr_f(g,nidx,"eps",eps);
    return out;
}

int g_softmax(Graph *g, int X, int axis, const char *name)
{
    Tensor *tx=graph_tensor(g,X);
    int out,in_ids[1],out_ids[1],nidx;
    if (!tx) return -1;
    out=alloc_output_f32(g,name,tx->rank,tx->shape);
    if (out<0) return -1;
    in_ids[0]=X; out_ids[0]=out;
    nidx=graph_add_node(g,OP_SOFTMAX,in_ids,1,out_ids,1,name);
    graph_node_set_attr_i(g,nidx,"axis",axis);
    return out;
}

int g_gelu(Graph *g, int X, const char *name)
{
    Tensor *tx=graph_tensor(g,X);
    int out,in_ids[1],out_ids[1];
    if (!tx) return -1;
    out=alloc_output_f32(g,name,tx->rank,tx->shape);
    if (out<0) return -1;
    in_ids[0]=X; out_ids[0]=out;
    graph_add_node(g,OP_GELU,in_ids,1,out_ids,1,name);
    return out;
}

int g_swiglu(Graph *g, int gate, int up, const char *name)
{
    Tensor *tg=graph_tensor(g,gate);
    int out_shape[2],out,in_ids[2],out_ids[1];
    if (!tg) return -1;
    out=alloc_output_f32(g,name,tg->rank,tg->shape);
    if (out<0) return -1;
    in_ids[0]=gate; in_ids[1]=up; out_ids[0]=out;
    graph_add_node(g,OP_SWIGLU,in_ids,2,out_ids,1,name);
    (void)out_shape;
    return out;
}

int g_rope(Graph *g, int X, int pos, int d_head, const char *name)
{
    Tensor *tx=graph_tensor(g,X);
    int out,in_ids[2],out_ids[1],nidx;
    if (!tx) return -1;
    out=alloc_output_f32(g,name,tx->rank,tx->shape);
    if (out<0) return -1;
    in_ids[0]=X; in_ids[1]=pos; out_ids[0]=out;
    nidx=graph_add_node(g,OP_ROPE,in_ids,2,out_ids,1,name);
    graph_node_set_attr_i(g,nidx,"d_head",d_head);
    return out;
}

int g_attention(Graph *g, int Q, int K, int V,
                 int causal, float scale, const char *name)
{
    Tensor *tq=graph_tensor(g,Q);
    int out,in_ids[3],out_ids[1],nidx;
    if (!tq) return -1;
    out=alloc_output_f32(g,name,tq->rank,tq->shape);
    if (out<0) return -1;
    in_ids[0]=Q; in_ids[1]=K; in_ids[2]=V; out_ids[0]=out;
    nidx=graph_add_node(g,OP_ATTN,in_ids,3,out_ids,1,name);
    graph_node_set_attr_i(g,nidx,"causal",causal);
    graph_node_set_attr_f(g,nidx,"scale",scale);
    return out;
}

int g_embed(Graph *g, int tokens, int W, const char *name)
{
    Tensor *tt=graph_tensor(g,tokens),*tw=graph_tensor(g,W);
    int out_shape[2],out,in_ids[2],out_ids[1];
    if (!tt||!tw) return -1;
    out_shape[0]=tt->shape[0];
    out_shape[1]=tw->shape[1];
    out=alloc_output_f32(g,name,2,out_shape);
    if (out<0) return -1;
    in_ids[0]=tokens; in_ids[1]=W; out_ids[0]=out;
    graph_add_node(g,OP_EMBED,in_ids,2,out_ids,1,name);
    return out;
}

int g_lm_head(Graph *g, int X, int W, const char *name)
{
    Tensor *tw=graph_tensor(g,W);
    int out_shape[1],out,in_ids[2],out_ids[1];
    if (!tw) return -1;
    out_shape[0]=tw->shape[0];
    out=alloc_output_f32(g,name,1,out_shape);
    if (out<0) return -1;
    in_ids[0]=X; in_ids[1]=W; out_ids[0]=out;
    graph_add_node(g,OP_LM_HEAD,in_ids,2,out_ids,1,name);
    return out;
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 1
 *
 * Files covered:
 *   tensor.h / tensor.c  – full tensor abstraction layer
 *   graph.h  / graph.c   – computation graph + operator registry
 *
 * PART 2 will cover:
 *   sysinfo.h / sysinfo.c  – NEW: GlobalMemoryStatus probe,
 *                             RAM-tier detection, dynamic ModelConfig
 *                             builder, MaxEmbeds calculator,
 *                             CPU throttle helpers (tb_yield),
 *                             brain.log error sink
 *   ops.h / ops.c          – all operator implementations
 *                             (MatMul tiled + x86 FPU ASM,
 *                              RMSNorm, LayerNorm, Softmax,
 *                              GELU, SwiGLU, RoPE, Attention,
 *                              Embed, LMHead, Quant/Dequant,
 *                              qmatmul, graph wiring,
 *                              ops_register_all)
 * ─────────────────────────────────────────────────────────────
 */

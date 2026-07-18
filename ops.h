#include "tensor.h"
#include "graph.h"
/* ============================================================
 * FILE: ops.h
 * (identical structure to v12; new: tb_yield in attention loop)
 * ============================================================ */
#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void ops_register_all(void);

/* SIMD-tuned inner product (SSE / x87 ASM / scalar). Used by MT workers. */
float op_dot_f32(const float *a, const float *b, int k);

/* Runtime CPU capability detection (SSE fast paths, scalar fallback). */
void  tb_cpu_init         (void);
int   tb_cpu_has_sse      (void);
int   tb_cpu_has_sse2     (void);
void  tb_cpu_features_str (char *buf, int n);

void op_matmul_f32   (const float *A, const float *B, float *C,
                       int M, int K, int N);
void op_matmul_t_f32 (const float *A, const float *B, float *C,
                       int M, int K, int N);
void op_matmul_dA    (const float *dC, const float *B, float *dA,
                       int M, int K, int N);
void op_matmul_dB    (const float *dC, const float *A, float *dB,
                       int M, int K, int N);
void op_matmul_t_dA  (const float *dC, const float *B, float *dA,
                       int M, int K, int N);
void op_matmul_t_dB  (const float *dC, const float *A, float *dB,
                       int M, int K, int N);

void op_rmsnorm_f32  (const float *x, const float *w, float *y,
                       int n, float eps);
void op_rmsnorm_bwd  (const float *dy, const float *x,
                       const float *w, float *dx, float *dw,
                       int n, float eps);

void op_layernorm_f32(const float *x, const float *w, const float *b,
                       float *y, int n, float eps,
                       float *out_mean, float *out_inv_std);
void op_layernorm_bwd(const float *dy, const float *x,
                       const float *w,
                       float mean, float inv_std, int n,
                       float *dx, float *dw, float *db);

void op_softmax_f32  (float *x, int n);
void op_softmax_bwd  (const float *p, const float *dy, float *dx, int n);

void op_gelu_f32     (const float *x, float *y, int n);
void op_gelu_bwd     (const float *x, const float *dy, float *dx, int n);

void op_swiglu_f32   (const float *gate, const float *up, float *out, int n);
void op_swiglu_bwd   (const float *gate, const float *up,
                       const float *dout,
                       float *d_gate, float *d_up, int n);

void op_rope_f32     (float *x, int seq_len, int n_heads,
                       int d_head, int forward);

void op_attention_f32(const float *Q, const float *K, const float *V,
                       float *out, float *attn_w_out,
                       int seq, int n_heads, int d_head,
                       int causal, float scale);
void op_attention_bwd(const float *Q, const float *K, const float *V,
                       const float *attn_w, const float *dout,
                       float *dQ, float *dK, float *dV,
                       int seq, int n_heads, int d_head,
                       int causal, float scale);

void op_embed_f32    (const int *tokens, const float *W, float *out,
                       int seq, int d_model, int vocab);
void op_embed_bwd    (const int *tokens, const float *dout, float *dW,
                       int seq, int d_model);

void op_linear_f32   (const float *x, const float *W, float *out,
                       int d_model, int vocab);
void op_linear_bwd   (const float *x, const float *W, const float *dout,
                       float *dx, float *dW, int d_model, int vocab);

void op_quantize_i8  (const float *src, signed char *dst,
                       float *scales, int rows, int cols);
void op_dequantize_i8(const signed char *src, const float *scales,
                       float *dst, int rows, int cols);
void op_qmatmul      (const signed char *W, const float *scales,
                       const float *x, float *y, int rows, int cols);

#ifdef __cplusplus
}
#endif
#endif /* OPS_H */

#ifndef OPS_MT_H
#define OPS_MT_H

/* Win32 multi-core matmul / linear / attention (C89, Windows 2000+).
 * Falls back to serial automatically on single-core or tiny workloads. */

#ifdef __cplusplus
extern "C" {
#endif

void tb_mt_init     (void);
void tb_mt_shutdown (void);
int  tb_mt_nworkers (void);

/* Returns 1 if the parallel path ran, 0 => caller should use serial code. */
int tb_mt_try_matmul_t_f32(const float *A, const float *B, float *C,
                              int M, int K, int N);
int tb_mt_try_linear_f32  (const float *x, const float *W, float *out,
                              int d_model, int vocab);
int tb_mt_try_attention_f32(const float *Q, const float *K, const float *V,
                              float *out, float *attn_w_out,
                              int seq, int n_heads, int d_head,
                              int causal, float scale);

#ifdef __cplusplus
}
#endif
#endif /* OPS_MT_H */

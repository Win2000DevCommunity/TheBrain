#include "sysinfo.h"
#include "tensor.h"
/* ============================================================
 * FILE: model.h
 * ============================================================ */
#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>

/* sysinfo.h and tensor.h assumed already included via brain.h */

/* ── Memory budget ── */
#ifndef MEM_BUDGET_MB
#  define MEM_BUDGET_MB 256
#endif

/* ── Static defaults (overridden by DynModelCfg at runtime) ── */
#if MEM_BUDGET_MB >= 512
#  define CFG_LAYERS  8
#  define CFG_HEADS   8
#  define CFG_D_MODEL 512
#  define CFG_D_FF    2048
#  define CFG_CTX     1024
#else
#  define CFG_LAYERS  6
#  define CFG_HEADS   4
#  define CFG_D_MODEL 256
#  define CFG_D_FF    1024
#  define CFG_CTX     512
#endif

#define CFG_D_HEAD    (CFG_D_MODEL / CFG_HEADS)
#define CFG_VOCAB     32768
#define CFG_N_CLASSES 16
#define CFG_N_LANG    4

/* ── Special tokens ── */
#define TOKEN_PAD          0
#define TOKEN_BOS          1
#define TOKEN_EOS          2
#define TOKEN_UNK          3
#define TOKEN_LANG_C       4
#define TOKEN_LANG_PY      5
#define TOKEN_LANG_PAS     6
#define TOKEN_LANG_ASM     7
#define TOKEN_LANG_EN      8   /* NEW v13: English natural language   */
#define TOKEN_LANG_AR      9   /* NEW v13: Arabic natural language    */
#define TOKEN_LANG_FR     10   /* NEW v13: French natural language    */
#define TOKEN_SCRIPT_LATIN   11
#define TOKEN_SCRIPT_ARABIC  12
#define TOKEN_SCRIPT_CJK     13
#define TOKEN_SCRIPT_DEVA    14
#define TOKEN_SCRIPT_CYRIL   15
#define TOKEN_SPECIAL_END    16

/* ── Checkpoint format ── */
#define MODEL_MAGIC_V13  0x54423133UL   /* "TB13" */
#define MODEL_MAGIC_V12  0x54423132UL   /* "TB12" – can load  */
#define MODEL_VERSION    13

/* ── Error codes ── */
#define MODEL_OK              0
#define MODEL_ERR_OOM         1
#define MODEL_ERR_IO          2
#define MODEL_ERR_CORRUPT     3
#define MODEL_ERR_HMAC        4
#define MODEL_ERR_VER         5
#define MODEL_ERR_CFG_MISMATCH 6   /* NEW v13 */

/* ─────────────────────────────────────────────────────────────
 * ModelConfig
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    int   n_layers;
    int   n_heads;
    int   d_model;
    int   d_ff;
    int   ctx_len;
    int   vocab_size;
    int   n_classes;
    int   n_lang;
    float rms_eps;
    int   use_swiglu;
    int   use_rmsnorm;
    int   tie_embeddings;
} ModelConfig;

/* ─────────────────────────────────────────────────────────────
 * Per-layer weights
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    float *Wq_f32,*Wk_f32,*Wv_f32,*Wo_f32;
    float *Wq_m,*Wq_v, *Wk_m,*Wk_v, *Wv_m,*Wv_v, *Wo_m,*Wo_v;
    float *Wff1_f32,*Wff2_f32;
    float *Wgate_f32,*Wup_f32,*Wdown_f32;
    float *Wff1_m,*Wff1_v, *Wff2_m,*Wff2_v;
    float *Wgate_m,*Wgate_v, *Wup_m,*Wup_v, *Wdown_m,*Wdown_v;
    float *bff1,*bff2, *bff1_m,*bff1_v, *bff2_m,*bff2_v;
    float *norm1_w,*norm1_b, *norm2_w,*norm2_b;
    float *norm1_w_m,*norm1_w_v, *norm1_b_m,*norm1_b_v;
    float *norm2_w_m,*norm2_w_v, *norm2_b_m,*norm2_b_v;
    signed char *Wq_i8,*Wk_i8,*Wv_i8,*Wo_i8;
    float       *Wq_sc,*Wk_sc,*Wv_sc,*Wo_sc;
    signed char *Wff1_i8,*Wff2_i8;
    float       *Wff1_sc,*Wff2_sc;
    signed char *Wgate_i8,*Wup_i8,*Wdown_i8;
    float       *Wgate_sc,*Wup_sc,*Wdown_sc;
} LayerWeights;

/* ─────────────────────────────────────────────────────────────
 * Full model
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    ModelConfig   cfg;
    float  *embed_f32, *embed_m,  *embed_v;
    float  *norm_f_w,  *norm_f_b;
    float  *norm_f_w_m,*norm_f_w_v;
    float  *norm_f_b_m,*norm_f_b_v;
    float  *lm_head_f32, *lm_head_m, *lm_head_v;
    float  *cls_head,    *cls_bias;
    float  *cls_head_m,  *cls_head_v;
    float  *cls_bias_m,  *cls_bias_v;
    float  *lang_head,   *lang_bias;
    float  *lang_head_m, *lang_head_v;
    float  *lang_bias_m, *lang_bias_v;
    LayerWeights *layers;
    long   adam_t;
    long   total_tokens;
    float  best_val_ppl;
    long   best_val_step;
    int    trained;
    float  *scratch;
    float  *attn_scratch;
    size_t  scratch_bytes;
    /* Persistent per-token forward scratch (avoids malloc/free in the
     * inference/training hot loop -> big win on old CPUs / Win2000 heap). */
    float  *fwd_proj;   /* [d_model]  attention output projection         */
    float  *fwd_gate;   /* [d_ff]     FFN gate / hidden                   */
    float  *fwd_up;     /* [d_ff]     FFN up projection (SwiGLU)          */
    float  *fwd_ff;     /* [d_model]  FFN output                          */
} Model;

/* ─────────────────────────────────────────────────────────────
 * Forward pass output + activation cache
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    float *hidden;
    float *lm_logits;
    float *cls_logits;
    float *lang_logits;
} ModelOutput;

typedef struct {
    int    seq_len;
    float **x_in, **xnorm1, **xnorm2;
    float **Q, **K, **V;
    float **attn_w, **attn_out;
    float **ff_pre, **ff_act;
    float  *x_final;
    float  *x_normed_all;
    float  *x_final_all;
} ForwardCache;

/* ── API ── */
#ifdef __cplusplus
extern "C" {
#endif

ModelConfig   model_default_config  (void);
ModelConfig   model_cfg_from_dyn    (const DynModelCfg *d);  /* NEW v13 */
void          model_print_config    (const ModelConfig *cfg);
Model        *model_create          (const ModelConfig *cfg);
Model        *model_create_dynamic  (const DynModelCfg *d);   /* NEW v13 */
void          model_free            (Model *m);
void          model_init_xavier     (Model *m);
void          model_requantize      (Model *m);
int           model_forward         (Model *m,
                                     const int *tokens, int seq_len,
                                     ModelOutput *out,
                                     ForwardCache *cache);
ForwardCache *cache_create          (const ModelConfig *cfg, int seq_len);
void          cache_free            (ForwardCache *c, const ModelConfig *cfg);
int           model_save            (const Model *m, const char *path);
int           model_load            (Model *m,        const char *path);
int           model_cfg_compatible  (const ModelConfig *a,
                                     const ModelConfig *b); /* NEW v13 */

#ifdef __cplusplus
}
#endif
#endif /* MODEL_H */

#include "sysinfo.h"
#include "model.h"
#include "tokenizer.h"
/* ============================================================
 * FILE: train.h
 * ============================================================ */
#ifndef TRAIN_H
#define TRAIN_H

#include <stdio.h>

/* ── AdamW constants ── */
#define ADAMW_BETA1   0.9f
#define ADAMW_BETA2   0.999f
#define ADAMW_EPS     1e-8f
#define ADAMW_T_MAX   200000L

/* ── Training defaults ── */
#define TRAIN_LR_MAX        0.001f
#define TRAIN_LR_MIN        0.0001f
#define TRAIN_WARMUP_STEPS  500L
#define TRAIN_WEIGHT_DECAY  0.01f
#define TRAIN_GRAD_CLIP     1.0f
#define TRAIN_BATCH_SIZE    4
#define TRAIN_VALID_SPLIT   0.10f
#define TRAIN_PATIENCE      5
#define TRAIN_LANG_WEIGHT   0.1f
#define TRAIN_MAX_FILES     8192    /* doubled from v12 for mixed corpus */
#define TRAIN_LOG_FILE      "training_log.csv"

/* ── File type flags for mixed corpus ── */
#define FTYPE_UNKNOWN 0
#define FTYPE_CODE    1   /* .c .h .py .asm .pas */
#define FTYPE_CONV    2   /* .conv dialogue file  */
#define FTYPE_TEXT    3   /* .txt .md plain text  */

/* ─────────────────────────────────────────────────────────────
 * TrainConfig
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    float lr_max;
    float lr_min;
    long  warmup_steps;
    long  total_steps;
    float weight_decay;
    float grad_clip;
    int   batch_size;
    int   epochs;
    float valid_split;
    int   patience;
    float lang_loss_weight;
    char  log_file[256];
    int   save_best;
    char  checkpoint_path[256];
    /* NEW v13 */
    int   use_conv_files;    /* 1 = include .conv in training     */
    int   use_text_files;    /* 1 = include .txt/.md in training  */
    int   use_code_files;    /* 1 = include .c/.py code in training */
    float conv_loss_weight;  /* weight for conversation CE loss   */
} TrainConfig;

/* ─────────────────────────────────────────────────────────────
 * TrainState
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    long  global_step;
    int   epoch;
    float last_loss;
    float last_val_ppl;
    float best_val_ppl;
    long  best_step;
    int   patience_count;
    FILE *log_fp;
    /* NEW v13 */
    long  conv_steps;   /* steps taken on .conv files            */
    long  code_steps;   /* steps taken on code files             */
    long  text_steps;   /* steps taken on text files             */
} TrainState;

/* ─────────────────────────────────────────────────────────────
 * GradBuffer
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    float *d_embed;
    float *d_lm_head;
    float *d_cls_head;
    float *d_cls_bias;
    float *d_lang_head;
    float *d_lang_bias;
    float *d_norm_f_w;
    float *d_norm_f_b;
    float **d_Wq,  **d_Wk,  **d_Wv,  **d_Wo;
    float **d_Wff1,**d_Wff2;
    float **d_Wgate,**d_Wup,**d_Wdown;
    float **d_bff1,**d_bff2;
    float **d_norm1_w,**d_norm1_b;
    float **d_norm2_w,**d_norm2_b;
    int n_layers,d_model,d_ff,vocab,n_classes,n_lang;
    int use_swiglu,use_rmsnorm;
} GradBuffer;

/* ─────────────────────────────────────────────────────────────
 * Mixed corpus file entry  (NEW v13)
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    char path[512];
    int  ftype;          /* FTYPE_* constant */
    int  lang_token;     /* detected language prefix token */
    int  lang_class;     /* detected language class int */
} CorpusFile;

#ifdef __cplusplus
extern "C" {
#endif

/* Config helpers */
TrainConfig train_default_config (void);
int         train_config_validate(const TrainConfig *cfg); /* NEW v13 */

/* State */
void train_state_init    (TrainState *s, const TrainConfig *cfg);
void train_state_destroy (TrainState *s);

/* Gradient buffer */
GradBuffer *grad_alloc (const ModelConfig *cfg);
void        grad_free  (GradBuffer *g);
void        grad_zero  (GradBuffer *g);

/* AdamW */
void  adamw_update (float *w, float *m, float *v,
                    float grad, float lr, float wd,
                    long t, int is_weight);

/* LR schedule */
float lr_schedule (long step,
                   float lr_max, float lr_min,
                   long warmup, long total);

/* Gradient clipping */
float grad_global_norm (GradBuffer *g);
void  grad_clip_norm   (GradBuffer *g, float max_norm);

/* Loss */
float cross_entropy_loss (const float *logits, int target,
                           int vocab_size, float *d_logits_out);

/* Training steps */
float train_step      (Model *m, GradBuffer *g, ForwardCache *cache,
                       const int *tokens, int seq_len,
                       int lang_class, long step, float lr,
                       const TrainConfig *cfg);

float train_conv_step (Model *m, GradBuffer *g, ForwardCache *cache,
                       const Conversation *conv, int lang_token,
                       const BPETokenizer *tok,
                       long step, float lr,
                       const TrainConfig *cfg);  /* NEW v13 */

void  apply_gradients (Model *m, GradBuffer *g,
                       long step, float lr, const TrainConfig *cfg);

/* Validation */
float validate (Model *m, BPETokenizer *tok,
                CorpusFile *files, int n_files,   /* uses CorpusFile now */
                const ModelConfig *mcfg);

/* File collection  (NEW v13 signature – returns CorpusFile array) */
int collect_files_mixed (const char *dir,
                          CorpusFile *files,
                          int max_files,
                          int use_conv,
                          int use_text,
                          int use_code);

/* Training loops */
void train_loop       (Model *m, BPETokenizer *tok,
                       const char *corpus_dir,
                       const TrainConfig *tcfg,
                       TrainState *state,
                       volatile int *cancel_flag);  /* v12 compat */

void train_loop_mixed (Model *m, BPETokenizer *tok,
                       const char *corpus_dir,
                       const TrainConfig *tcfg,
                       TrainState *state,
                       volatile int *cancel_flag);  /* NEW v13 */

#ifdef __cplusplus
}
#endif
#endif /* TRAIN_H */

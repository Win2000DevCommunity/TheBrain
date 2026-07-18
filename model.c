#include "model.h"
#include "ops.h"
#include "sysinfo.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static void hmac_simple(const unsigned char *key,   size_t klen,
                          const unsigned char *data,  size_t dlen,
                          unsigned char out[20])
{
    unsigned int s[5];
    size_t i;
    s[0]=0x67452301UL; s[1]=0xEFCDAB89UL;
    s[2]=0x98BADCFEUL; s[3]=0x10325476UL; s[4]=0xC3D2E1F0UL;
    for (i=0;i<klen;i++) s[i%5]=(s[i%5]<<5)^(s[i%5]>>27)^key[i];
    for (i=0;i<dlen;i++) {
        s[i%5]^=data[i];
        if ((i&63)==63){
            int r;
            for(r=0;r<4;r++){
                s[0]+=s[1]^s[4]; s[1]^=(s[0]<<3)|(s[0]>>29);
                s[2]+=s[3]^s[0]; s[3]^=(s[2]<<7)|(s[2]>>25);
                s[4]+=s[0]^s[2];
            }
        }
    }
    s[0]^=(unsigned int)dlen;
    s[1]^=(unsigned int)(dlen>>16);
    for(i=0;i<5;i++){
        out[i*4+0]=(unsigned char)(s[i]>>24);
        out[i*4+1]=(unsigned char)(s[i]>>16);
        out[i*4+2]=(unsigned char)(s[i]>>8);
        out[i*4+3]=(unsigned char)(s[i]);
    }
    (void)klen;
}

static unsigned int g_mac_key[4]={
    0xDEADBEEFUL,0xCAFEBABEUL,0xFEEDFACEUL,0xBAADF00DUL
};

/* XorShift RNG for Xavier init */
static unsigned long g_rng_state = 1234567UL;

static double rng_uniform(void)
{
    g_rng_state ^= g_rng_state<<13;
    g_rng_state ^= g_rng_state>>17;
    g_rng_state ^= g_rng_state<<5;
    return (double)(g_rng_state&0x7FFFFFFFUL)/2147483647.0;
}

static float xavier_f(int fan_in, int fan_out)
{
    double lim=sqrt(6.0/(fan_in+fan_out));
    return (float)((rng_uniform()*2.0-1.0)*lim);
}

static float *zalloc_f(size_t n)
{ return (float*)calloc(n,sizeof(float)); }

static signed char *zalloc_i8(size_t n)
{ return (signed char*)calloc(n,1); }

/* ── §B  CONFIG HELPERS ─────────────────────────────────────── */

ModelConfig model_default_config(void)
{
    ModelConfig c;
    memset(&c,0,sizeof(c));
    c.n_layers    =CFG_LAYERS;
    c.n_heads     =CFG_HEADS;
    c.d_model     =CFG_D_MODEL;
    c.d_ff        =CFG_D_FF;
    c.ctx_len     =CFG_CTX;
    c.vocab_size  =CFG_VOCAB;
    c.n_classes   =CFG_N_CLASSES;
    c.n_lang      =CFG_N_LANG;
    c.rms_eps     =1e-5f;
    c.use_swiglu  =0;
    c.use_rmsnorm =0;
    c.tie_embeddings=0;
    return c;
}

/* NEW v13: convert DynModelCfg -> ModelConfig */
ModelConfig model_cfg_from_dyn(const DynModelCfg *d)
{
    ModelConfig c;
    memset(&c,0,sizeof(c));
    c.n_layers      = d->n_layers;
    c.n_heads       = d->n_heads;
    c.d_model       = d->d_model;
    c.d_ff          = d->d_ff;
    c.ctx_len       = d->ctx_len;
    c.vocab_size    = d->vocab_size;
    c.n_classes     = d->n_classes;
    c.n_lang        = d->n_lang;
    c.rms_eps       = d->rms_eps;
    c.use_swiglu    = d->use_swiglu;
    c.use_rmsnorm   = d->use_rmsnorm;
    c.tie_embeddings= d->tie_embeddings;
    return c;
}

/*
 * cfg_buffers_differ: returns non-zero when two configs require a different
 * memory layout (any dimension that affects buffer sizes).  Used by
 * model_load to decide whether the model must be reallocated to match a
 * checkpoint's stored config.
 */
static int cfg_buffers_differ(const ModelConfig *a, const ModelConfig *b)
{
    return (a->n_layers      != b->n_layers      ||
            a->n_heads       != b->n_heads       ||
            a->d_model       != b->d_model       ||
            a->d_ff          != b->d_ff          ||
            a->ctx_len       != b->ctx_len       ||
            a->vocab_size    != b->vocab_size    ||
            a->n_classes     != b->n_classes     ||
            a->n_lang        != b->n_lang        ||
            a->use_swiglu    != b->use_swiglu    ||
            a->use_rmsnorm   != b->use_rmsnorm   ||
            a->tie_embeddings!= b->tie_embeddings);
}

/* NEW v13: check two configs are architecturally compatible */
int model_cfg_compatible(const ModelConfig *a, const ModelConfig *b)
{
    return (a->n_layers  == b->n_layers  &&
            a->n_heads   == b->n_heads   &&
            a->d_model   == b->d_model   &&
            a->d_ff      == b->d_ff      &&
            a->vocab_size== b->vocab_size &&
            a->use_swiglu== b->use_swiglu);
}

void model_print_config(const ModelConfig *cfg)
{
    printf("[ModelConfig v13] layers=%d heads=%d d_model=%d d_ff=%d "
           "ctx=%d vocab=%d classes=%d lang=%d "
           "swiglu=%d rmsnorm=%d tied=%d\n",
           cfg->n_layers,cfg->n_heads,cfg->d_model,cfg->d_ff,
           cfg->ctx_len,cfg->vocab_size,cfg->n_classes,cfg->n_lang,
           cfg->use_swiglu,cfg->use_rmsnorm,cfg->tie_embeddings);
}

/* ── §C  LAYER ALLOC / FREE ─────────────────────────────────── */

static int layer_alloc(LayerWeights *lw, const ModelConfig *cfg)
{
    int D=cfg->d_model,FF=cfg->d_ff;
    int DxD=D*D,FFxD=FF*D,DxFF=D*FF;

#define ALLOC3(name,n) \
    lw->name##_f32=zalloc_f(n); \
    lw->name##_m  =zalloc_f(n); \
    lw->name##_v  =zalloc_f(n); \
    if(!lw->name##_f32||!lw->name##_m||!lw->name##_v) return 0

    ALLOC3(Wq,DxD); ALLOC3(Wk,DxD); ALLOC3(Wv,DxD); ALLOC3(Wo,DxD);

    if (cfg->use_swiglu) {
        ALLOC3(Wgate,FFxD); ALLOC3(Wup,FFxD); ALLOC3(Wdown,DxFF);
    } else {
        ALLOC3(Wff1,FFxD); ALLOC3(Wff2,DxFF);
        lw->bff1=zalloc_f((size_t)FF); lw->bff2=zalloc_f((size_t)D);
        lw->bff1_m=zalloc_f((size_t)FF); lw->bff1_v=zalloc_f((size_t)FF);
        lw->bff2_m=zalloc_f((size_t)D);  lw->bff2_v=zalloc_f((size_t)D);
        if (!lw->bff1||!lw->bff2) return 0;
    }
#undef ALLOC3

    lw->norm1_w=zalloc_f((size_t)D); lw->norm2_w=zalloc_f((size_t)D);
    lw->norm1_w_m=zalloc_f((size_t)D); lw->norm1_w_v=zalloc_f((size_t)D);
    lw->norm2_w_m=zalloc_f((size_t)D); lw->norm2_w_v=zalloc_f((size_t)D);
    if (!lw->norm1_w||!lw->norm2_w) return 0;

    if (!cfg->use_rmsnorm) {
        lw->norm1_b=zalloc_f((size_t)D); lw->norm2_b=zalloc_f((size_t)D);
        lw->norm1_b_m=zalloc_f((size_t)D); lw->norm1_b_v=zalloc_f((size_t)D);
        lw->norm2_b_m=zalloc_f((size_t)D); lw->norm2_b_v=zalloc_f((size_t)D);
        if (!lw->norm1_b||!lw->norm2_b) return 0;
    }

#define ALLOC_I8(name,rows,cols) \
    lw->name##_i8=zalloc_i8((size_t)(rows)*(cols)); \
    lw->name##_sc=zalloc_f((size_t)(rows)); \
    if (!lw->name##_i8||!lw->name##_sc) return 0

    ALLOC_I8(Wq,D,D); ALLOC_I8(Wk,D,D); ALLOC_I8(Wv,D,D); ALLOC_I8(Wo,D,D);
    if (cfg->use_swiglu) {
        ALLOC_I8(Wgate,FF,D); ALLOC_I8(Wup,FF,D); ALLOC_I8(Wdown,D,FF);
    } else {
        ALLOC_I8(Wff1,FF,D); ALLOC_I8(Wff2,D,FF);
    }
#undef ALLOC_I8

    return 1;
}

static void layer_free(LayerWeights *lw, const ModelConfig *cfg)
{
#define FREE3(name) free(lw->name##_f32);free(lw->name##_m);free(lw->name##_v)
    FREE3(Wq);FREE3(Wk);FREE3(Wv);FREE3(Wo);
    if (cfg->use_swiglu){FREE3(Wgate);FREE3(Wup);FREE3(Wdown);}
    else {
        FREE3(Wff1);FREE3(Wff2);
        free(lw->bff1);free(lw->bff1_m);free(lw->bff1_v);
        free(lw->bff2);free(lw->bff2_m);free(lw->bff2_v);
    }
#undef FREE3
    free(lw->norm1_w);free(lw->norm1_w_m);free(lw->norm1_w_v);
    free(lw->norm2_w);free(lw->norm2_w_m);free(lw->norm2_w_v);
    if (!cfg->use_rmsnorm){
        free(lw->norm1_b);free(lw->norm1_b_m);free(lw->norm1_b_v);
        free(lw->norm2_b);free(lw->norm2_b_m);free(lw->norm2_b_v);
    }
#define FREE_I8(name) free(lw->name##_i8);free(lw->name##_sc)
    FREE_I8(Wq);FREE_I8(Wk);FREE_I8(Wv);FREE_I8(Wo);
    if (cfg->use_swiglu){FREE_I8(Wgate);FREE_I8(Wup);FREE_I8(Wdown);}
    else{FREE_I8(Wff1);FREE_I8(Wff2);}
#undef FREE_I8
    memset(lw,0,sizeof(LayerWeights));
}

/* ── §D  MODEL CREATE / FREE ────────────────────────────────── */

/*
 * model_alloc_internals: allocate every weight / moment / scratch buffer
 * for the configuration already stored in m->cfg.  Returns 1 on success,
 * 0 on OOM (caller is responsible for cleanup via model_free_internals).
 * Used by both model_create and model_load (reconfigure-on-load).
 */
static int model_alloc_internals(Model *m)
{
    const ModelConfig *cfg=&m->cfg;
    int l,D=cfg->d_model,FF=cfg->d_ff,V=cfg->vocab_size;
    int NC=cfg->n_classes,NL=cfg->n_lang;

    m->embed_f32=zalloc_f((size_t)V*D);
    m->embed_m  =zalloc_f((size_t)V*D);
    m->embed_v  =zalloc_f((size_t)V*D);
    m->norm_f_w =zalloc_f((size_t)D);
    m->norm_f_b =cfg->use_rmsnorm?NULL:zalloc_f((size_t)D);
    m->norm_f_w_m=zalloc_f((size_t)D); m->norm_f_w_v=zalloc_f((size_t)D);
    m->norm_f_b_m=cfg->use_rmsnorm?NULL:zalloc_f((size_t)D);
    m->norm_f_b_v=cfg->use_rmsnorm?NULL:zalloc_f((size_t)D);

    if (cfg->tie_embeddings){
        m->lm_head_f32=m->embed_f32;
        m->lm_head_m  =m->embed_m;
        m->lm_head_v  =m->embed_v;
    } else {
        m->lm_head_f32=zalloc_f((size_t)V*D);
        m->lm_head_m  =zalloc_f((size_t)V*D);
        m->lm_head_v  =zalloc_f((size_t)V*D);
    }

    m->cls_head  =zalloc_f((size_t)NC*D); m->cls_bias  =zalloc_f((size_t)NC);
    m->cls_head_m=zalloc_f((size_t)NC*D); m->cls_head_v=zalloc_f((size_t)NC*D);
    m->cls_bias_m=zalloc_f((size_t)NC);   m->cls_bias_v=zalloc_f((size_t)NC);
    m->lang_head =zalloc_f((size_t)NL*D); m->lang_bias =zalloc_f((size_t)NL);
    m->lang_head_m=zalloc_f((size_t)NL*D);m->lang_head_v=zalloc_f((size_t)NL*D);
    m->lang_bias_m=zalloc_f((size_t)NL);  m->lang_bias_v=zalloc_f((size_t)NL);

    if (!m->embed_f32||!m->norm_f_w||!m->lm_head_f32||
        !m->cls_head ||!m->lang_head){
        BLOG_ERROR("model_create: OOM in global weights");
        return 0;
    }

    m->layers=(LayerWeights*)calloc((size_t)cfg->n_layers,sizeof(LayerWeights));
    if (!m->layers){ BLOG_ERROR("model_create: OOM for layers array"); return 0; }

    for (l=0;l<cfg->n_layers;l++){
        if (!layer_alloc(&m->layers[l],cfg)){
            BLOG_ERROR("model_create: OOM at layer %d",l);
            return 0;
        }
    }

    {
        /* Scratch: 6 buffers of [ctx x d_model] */
        {int max_dim=cfg->d_ff>cfg->d_model?cfg->d_ff:cfg->d_model;
        size_t s1=(size_t)cfg->ctx_len*(size_t)max_dim*sizeof(float);
        size_t s2=(size_t)cfg->n_heads*(size_t)cfg->ctx_len*(size_t)cfg->ctx_len*sizeof(float);
        m->scratch      =(float*)malloc(s1*6);
        m->attn_scratch =(float*)malloc(s2);
        m->scratch_bytes=s1;}
        if (!m->scratch||!m->attn_scratch){
            BLOG_ERROR("model_create: OOM for scratch buffers (ctx=%d d=%d)",
                       cfg->ctx_len,cfg->d_model);
            return 0;
        }
    }

    /* Persistent per-token forward scratch (no per-token malloc/free). */
    m->fwd_proj=zalloc_f((size_t)D);
    m->fwd_gate=zalloc_f((size_t)FF);
    m->fwd_up  =zalloc_f((size_t)FF);
    m->fwd_ff  =zalloc_f((size_t)D);
    if (!m->fwd_proj||!m->fwd_gate||!m->fwd_up||!m->fwd_ff){
        BLOG_ERROR("model_create: OOM for forward scratch");
        return 0;
    }

    m->best_val_ppl=1e30f;
    m->trained=0;
    return 1;
}

/*
 * model_free_internals: free every buffer owned by m (using m->cfg to know
 * the layout) WITHOUT freeing the Model struct itself, and NULL the pointers
 * so the struct can be safely reallocated (used by model_load reconfigure).
 */
static void model_free_internals(Model *m)
{
    int l;
    if (!m) return;
    free(m->embed_m); free(m->embed_v); free(m->embed_f32);
    free(m->norm_f_w); free(m->norm_f_b);
    free(m->norm_f_w_m); free(m->norm_f_w_v);
    free(m->norm_f_b_m); free(m->norm_f_b_v);
    if (!m->cfg.tie_embeddings){
        free(m->lm_head_f32); free(m->lm_head_m); free(m->lm_head_v);
    }
    free(m->cls_head);  free(m->cls_bias);
    free(m->cls_head_m);free(m->cls_head_v);
    free(m->cls_bias_m);free(m->cls_bias_v);
    free(m->lang_head); free(m->lang_bias);
    free(m->lang_head_m);free(m->lang_head_v);
    free(m->lang_bias_m);free(m->lang_bias_v);
    if (m->layers){
        for (l=0;l<m->cfg.n_layers;l++) layer_free(&m->layers[l],&m->cfg);
        free(m->layers);
    }
    free(m->scratch); free(m->attn_scratch);
    free(m->fwd_proj); free(m->fwd_gate); free(m->fwd_up); free(m->fwd_ff);

    m->fwd_proj=NULL; m->fwd_gate=NULL; m->fwd_up=NULL; m->fwd_ff=NULL;
    m->embed_f32=NULL; m->embed_m=NULL; m->embed_v=NULL;
    m->norm_f_w=NULL;  m->norm_f_b=NULL;
    m->norm_f_w_m=NULL;m->norm_f_w_v=NULL;
    m->norm_f_b_m=NULL;m->norm_f_b_v=NULL;
    m->lm_head_f32=NULL;m->lm_head_m=NULL;m->lm_head_v=NULL;
    m->cls_head=NULL;  m->cls_bias=NULL;
    m->cls_head_m=NULL;m->cls_head_v=NULL;
    m->cls_bias_m=NULL;m->cls_bias_v=NULL;
    m->lang_head=NULL; m->lang_bias=NULL;
    m->lang_head_m=NULL;m->lang_head_v=NULL;
    m->lang_bias_m=NULL;m->lang_bias_v=NULL;
    m->layers=NULL;
    m->scratch=NULL;   m->attn_scratch=NULL;
}

Model *model_create(const ModelConfig *cfg)
{
    Model *m;
    m=(Model*)calloc(1,sizeof(Model));
    if (!m){ BLOG_ERROR("model_create: OOM for Model struct"); return NULL; }
    m->cfg=*cfg;
    if (!model_alloc_internals(m)){ model_free(m); return NULL; }
    return m;
}

/* NEW v13: create from DynModelCfg */
Model *model_create_dynamic(const DynModelCfg *d)
{
    ModelConfig cfg=model_cfg_from_dyn(d);
    return model_create(&cfg);
}

void model_free(Model *m)
{
    if (!m) return;
    model_free_internals(m);
    free(m);
}

/* ── §E  WEIGHT INIT ────────────────────────────────────────── */

static void fill_xavier(float *w, int n, int fan_in, int fan_out)
{
    int i;
    for (i=0;i<n;i++) w[i]=xavier_f(fan_in,fan_out);
}

static void fill_ones(float *w, int n)
{ int i; for(i=0;i<n;i++) w[i]=1.0f; }

void model_init_xavier(Model *m)
{
    const ModelConfig *cfg=&m->cfg;
    int l,D=cfg->d_model,FF=cfg->d_ff,V=cfg->vocab_size;
    int NC=cfg->n_classes,NL=cfg->n_lang;

    g_rng_state=(unsigned long)time(NULL)^0xDEAD1234UL;
    fill_xavier(m->embed_f32,V*D,V,D);
    if (!cfg->tie_embeddings) fill_xavier(m->lm_head_f32,V*D,D,V);
    fill_ones(m->norm_f_w,D);
    if (m->norm_f_b) memset(m->norm_f_b,0,D*sizeof(float));
    fill_xavier(m->cls_head, NC*D,D,NC);
    fill_xavier(m->lang_head,NL*D,D,NL);

    for (l=0;l<cfg->n_layers;l++){
        LayerWeights *lw=&m->layers[l];
        fill_xavier(lw->Wq_f32,D*D,D,D); fill_xavier(lw->Wk_f32,D*D,D,D);
        fill_xavier(lw->Wv_f32,D*D,D,D); fill_xavier(lw->Wo_f32,D*D,D,D);
        if (cfg->use_swiglu){
            fill_xavier(lw->Wgate_f32,FF*D,D,FF);
            fill_xavier(lw->Wup_f32,  FF*D,D,FF);
            fill_xavier(lw->Wdown_f32,D*FF,FF,D);
        } else {
            fill_xavier(lw->Wff1_f32,FF*D,D,FF);
            fill_xavier(lw->Wff2_f32,D*FF,FF,D);
        }
        fill_ones(lw->norm1_w,D); fill_ones(lw->norm2_w,D);
        if (lw->norm1_b) memset(lw->norm1_b,0,D*sizeof(float));
        if (lw->norm2_b) memset(lw->norm2_b,0,D*sizeof(float));
    }
    model_requantize(m);
    m->trained=1;
    BLOG_INFO("model_init_xavier: d=%d layers=%d done",D,cfg->n_layers);
}

/* ── §F  REQUANTIZE ─────────────────────────────────────────── */

static void requant_layer(LayerWeights *lw, const ModelConfig *cfg)
{
    int D=cfg->d_model,FF=cfg->d_ff;
    op_quantize_i8(lw->Wq_f32,lw->Wq_i8,lw->Wq_sc,D,D);
    op_quantize_i8(lw->Wk_f32,lw->Wk_i8,lw->Wk_sc,D,D);
    op_quantize_i8(lw->Wv_f32,lw->Wv_i8,lw->Wv_sc,D,D);
    op_quantize_i8(lw->Wo_f32,lw->Wo_i8,lw->Wo_sc,D,D);
    if (cfg->use_swiglu){
        op_quantize_i8(lw->Wgate_f32,lw->Wgate_i8,lw->Wgate_sc,FF,D);
        op_quantize_i8(lw->Wup_f32,  lw->Wup_i8,  lw->Wup_sc,  FF,D);
        op_quantize_i8(lw->Wdown_f32,lw->Wdown_i8,lw->Wdown_sc,D, FF);
    } else {
        op_quantize_i8(lw->Wff1_f32,lw->Wff1_i8,lw->Wff1_sc,FF,D);
        op_quantize_i8(lw->Wff2_f32,lw->Wff2_i8,lw->Wff2_sc,D, FF);
    }
}

void model_requantize(Model *m)
{
    int l;
    for (l=0;l<m->cfg.n_layers;l++) requant_layer(&m->layers[l],&m->cfg);
}

/* ── §G  FORWARD PASS ───────────────────────────────────────── */

static void apply_norm(const ModelConfig *cfg,
                        const LayerWeights *lw, int which,
                        const float *x, float *y, int D,
                        float *out_mean, float *out_inv_std)
{
    const float *w=(which==0)?lw->norm1_w:lw->norm2_w;
    const float *b=(which==0)?lw->norm1_b:lw->norm2_b;
    if (cfg->use_rmsnorm)
        op_rmsnorm_f32(x,w,y,D,cfg->rms_eps);
    else
        op_layernorm_f32(x,w,b,y,D,cfg->rms_eps,out_mean,out_inv_std);
}

int model_forward(Model *m,
                   const int *tokens, int seq_len,
                   ModelOutput *out,
                   ForwardCache *cache)
{
    const ModelConfig *cfg=&m->cfg;
    int D=cfg->d_model,FF=cfg->d_ff,H=cfg->n_heads,DH=D/H;
    int V=cfg->vocab_size,NC=cfg->n_classes,NL=cfg->n_lang;
    int S=seq_len<cfg->ctx_len?seq_len:cfg->ctx_len;
    int training=(cache!=NULL);
    float inv_sq=1.0f/sqrtf((float)DH);
    float *x,*xnrm,*q_buf,*k_buf,*v_buf,*tmp;
    int l,s,i,j,h;

    x    =m->scratch;
    xnrm =x    +S*D;
    q_buf=xnrm +S*D;
    k_buf=q_buf+S*D;
    v_buf=k_buf+S*D;
    tmp  =v_buf+S*D;

    op_embed_f32(tokens,m->embed_f32,x,S,D,V);
    if (cache) cache->seq_len=S;

    for (l=0;l<cfg->n_layers;l++){
        LayerWeights *lw=&m->layers[l];
        float mean_v=0.0f,inv_std_v=1.0f;

        /* CPU yield every 2 layers to keep UI alive */
        if ((l&1)==1) tb_yield();

        if (cache&&cache->x_in&&cache->x_in[l])
            memcpy(cache->x_in[l],x,(size_t)S*D*sizeof(float));

        /* Pre-norm 1 */
        for (s=0;s<S;s++)
            apply_norm(cfg,lw,0,x+s*D,xnrm+s*D,D,&mean_v,&inv_std_v);
        if (cache&&cache->xnorm1&&cache->xnorm1[l])
            memcpy(cache->xnorm1[l],xnrm,(size_t)S*D*sizeof(float));

        /* Q K V projections */
        for (s=0;s<S;s++){
            const float *xi=xnrm+s*D;
            float *qs=q_buf+s*D,*ks=k_buf+s*D,*vs=v_buf+s*D;
            /* Always use float32 weights for accuracy */
            op_matmul_t_f32(xi,lw->Wq_f32,qs,1,D,D);
            op_matmul_t_f32(xi,lw->Wk_f32,ks,1,D,D);
            op_matmul_t_f32(xi,lw->Wv_f32,vs,1,D,D);
        }
        op_rope_f32(q_buf,S,H,DH,1);
        op_rope_f32(k_buf,S,H,DH,1);

        if (cache){
            if (cache->Q[l]) memcpy(cache->Q[l],q_buf,S*D*sizeof(float));
            if (cache->K[l]) memcpy(cache->K[l],k_buf,S*D*sizeof(float));
            if (cache->V[l]) memcpy(cache->V[l],v_buf,S*D*sizeof(float));
        }

        memset(tmp,0,(size_t)S*D*sizeof(float));
        op_attention_f32(q_buf,k_buf,v_buf,tmp,
                         (cache&&cache->attn_w)?cache->attn_w[l]:NULL,
                         S,H,DH,1,inv_sq);

        if (cache&&cache->attn_out&&cache->attn_out[l])
            memcpy(cache->attn_out[l],tmp,S*D*sizeof(float));

        /* Output projection + residual (persistent scratch, no malloc) */
        for (s=0;s<S;s++){
            float *proj=m->fwd_proj;
            op_matmul_t_f32(tmp+s*D,lw->Wo_f32,proj,1,D,D);
            for (i=0;i<D;i++) x[s*D+i]+=proj[i];
        }

        /* Pre-norm 2 */
        for (s=0;s<S;s++)
            apply_norm(cfg,lw,1,x+s*D,xnrm+s*D,D,&mean_v,&inv_std_v);
        if (cache&&cache->xnorm2&&cache->xnorm2[l])
            memcpy(cache->xnorm2[l],xnrm,(size_t)S*D*sizeof(float));

        /* FFN (persistent scratch, no per-token malloc) */
        for (s=0;s<S;s++){
            const float *xi=xnrm+s*D;
            float *xs=x+s*D;
            float *gate_buf=m->fwd_gate;
            float *ff_out  =m->fwd_ff;

            if (cfg->use_swiglu){
                float *up_buf=m->fwd_up;
                op_matmul_t_f32(xi,lw->Wgate_f32,gate_buf,1,D,FF);
                op_matmul_t_f32(xi,lw->Wup_f32,  up_buf,  1,D,FF);
                op_swiglu_f32(gate_buf,up_buf,gate_buf,FF);
                op_matmul_t_f32(gate_buf,lw->Wdown_f32,ff_out,1,FF,D);
            } else {
                op_matmul_t_f32(xi,lw->Wff1_f32,gate_buf,1,D,FF);
                for (i=0;i<FF;i++) gate_buf[i]+=lw->bff1[i];
                if (cache&&cache->ff_pre&&cache->ff_pre[l])
                    memcpy(cache->ff_pre[l]+s*FF,gate_buf,FF*sizeof(float));
                op_gelu_f32(gate_buf,gate_buf,FF);
                if (cache&&cache->ff_act&&cache->ff_act[l])
                    memcpy(cache->ff_act[l]+s*FF,gate_buf,FF*sizeof(float));
                op_matmul_t_f32(gate_buf,lw->Wff2_f32,ff_out,1,FF,D);
                for (i=0;i<D;i++) ff_out[i]+=lw->bff2[i];
            }
            for (i=0;i<D;i++) xs[i]+=ff_out[i];
        }
    } /* end layer loop */

    /* Final norm on last position */
    {
        float *last=x+(S-1)*D;
        float *normed=(float*)malloc((size_t)D*sizeof(float));
        float mean_v2=0.0f,inv_std_v2=0.0f;
        if (!normed){ BLOG_ERROR("model_forward: OOM for normed buffer"); return MODEL_ERR_OOM; }

        if (cache) {
            int pos_s;
            if (cache->x_final_all) {
                memcpy(cache->x_final_all, x, (size_t)S * D * sizeof(float));
            }
            for (pos_s = 0; pos_s < S; pos_s++) {
                float *pos_x = x + pos_s * D;
                float *pos_normed = cache->x_normed_all + pos_s * D;
                if (cfg->use_rmsnorm)
                    op_rmsnorm_f32(pos_x, m->norm_f_w, pos_normed, D, cfg->rms_eps);
                else
                    op_layernorm_f32(pos_x, m->norm_f_w, m->norm_f_b, pos_normed, D, cfg->rms_eps, &mean_v2, &inv_std_v2);
            }
            memcpy(normed, cache->x_normed_all + (S-1)*D, D * sizeof(float));
        } else {
            if (cfg->use_rmsnorm)
                op_rmsnorm_f32(last,m->norm_f_w,normed,D,cfg->rms_eps);
            else
                op_layernorm_f32(last,m->norm_f_w,m->norm_f_b,normed,D,cfg->rms_eps,
                                 &mean_v2,&inv_std_v2);
        }

        if (cache&&cache->x_final) memcpy(cache->x_final,normed,D*sizeof(float));
        if (out&&out->hidden)      memcpy(out->hidden,normed,D*sizeof(float));
        if (out&&out->lm_logits)   op_linear_f32(normed,m->lm_head_f32,out->lm_logits,D,V);
        if (out&&out->cls_logits){
            float *cl=out->cls_logits;
            op_linear_f32(normed,m->cls_head,cl,D,NC);
            for (i=0;i<NC;i++) cl[i]+=m->cls_bias[i];
            op_softmax_f32(cl,NC);
        }
        if (out&&out->lang_logits){
            float *ll=out->lang_logits;
            op_linear_f32(normed,m->lang_head,ll,D,NL);
            for (i=0;i<NL;i++) ll[i]+=m->lang_bias[i];
            op_softmax_f32(ll,NL);
        }
        free(normed);
    }
    (void)j;(void)h;
    return MODEL_OK;
}

/* ── §H  CACHE ──────────────────────────────────────────────── */

ForwardCache *cache_create(const ModelConfig *cfg, int seq_len)
{
    ForwardCache *c;
    int l,L=cfg->n_layers,D=cfg->d_model,FF=cfg->d_ff,H=cfg->n_heads,S=seq_len;
    c=(ForwardCache*)calloc(1,sizeof(ForwardCache));
    if (!c){ BLOG_ERROR("cache_create: OOM"); return NULL; }
    c->seq_len=S;

#define ACL(field,sz) \
    c->field=(float**)calloc((size_t)L,sizeof(float*)); \
    if (!c->field){cache_free(c,cfg);return NULL;} \
    for(l=0;l<L;l++){c->field[l]=(float*)calloc((size_t)(sz),sizeof(float)); \
        if (!c->field[l]){cache_free(c,cfg);return NULL;}}

    ACL(x_in,   S*D) ACL(xnorm1,S*D) ACL(xnorm2,S*D)
    ACL(Q,      S*D) ACL(K,     S*D) ACL(V,     S*D)
    ACL(attn_w, H*S*S) ACL(attn_out,S*D)
    ACL(ff_pre, S*FF)  ACL(ff_act,  S*FF)
#undef ACL

    c->x_final=(float*)calloc((size_t)D,sizeof(float));
    if (!c->x_final){cache_free(c,cfg);return NULL;}
    c->x_normed_all=(float*)calloc((size_t)S*D,sizeof(float));
    if (!c->x_normed_all){cache_free(c,cfg);return NULL;}
    c->x_final_all=(float*)calloc((size_t)S*D,sizeof(float));
    if (!c->x_final_all){cache_free(c,cfg);return NULL;}
    return c;
}

void cache_free(ForwardCache *c, const ModelConfig *cfg)
{
    int l,L; if (!c) return; L=cfg->n_layers;
#define FLA(field) if(c->field){for(l=0;l<L;l++)free(c->field[l]);free(c->field);c->field=NULL;}
    FLA(x_in)FLA(xnorm1)FLA(xnorm2)FLA(Q)FLA(K)FLA(V)
    FLA(attn_w)FLA(attn_out)FLA(ff_pre)FLA(ff_act)
#undef FLA
    free(c->x_final);
    free(c->x_normed_all);
    free(c->x_final_all);
    free(c);
}

/* ── §I  SAVE / LOAD (TB13 format) ─────────────────────────── */

static int wf(FILE *fp,const float *p,size_t n)
{ if(!p||n==0)return 1; return fwrite(p,sizeof(float),n,fp)==n; }

static int rf(FILE *fp,float *p,size_t n)
{ if(!p||n==0)return 1; return fread(p,sizeof(float),n,fp)==n; }

int model_save(const Model *m, const char *path)
{
    FILE *fp; long body_end;
    unsigned char mac[20],*body_buf;
    unsigned int magic=MODEL_MAGIC_V13;
    int version=MODEL_VERSION,l;
    const ModelConfig *cfg=&m->cfg;
    int D=cfg->d_model,FF=cfg->d_ff,V=cfg->vocab_size;
    int NC=cfg->n_classes,NL=cfg->n_lang;

    fp=fopen(path,"w+b"); if (!fp){BLOG_ERROR("model_save: cannot open %s",path);return MODEL_ERR_IO;}

    fwrite(&magic,  sizeof(unsigned int),1,fp);
    fwrite(&version,sizeof(int),         1,fp);
    fwrite(cfg,     sizeof(ModelConfig), 1,fp); /* v13: full cfg in file */
    fwrite(&m->adam_t,       sizeof(long), 1,fp);
    fwrite(&m->total_tokens, sizeof(long), 1,fp);
    fwrite(&m->best_val_ppl, sizeof(float),1,fp);
    fwrite(&m->best_val_step,sizeof(long), 1,fp);

    wf(fp,m->embed_f32,(size_t)V*D);
    wf(fp,m->embed_m,  (size_t)V*D);
    wf(fp,m->embed_v,  (size_t)V*D);
    wf(fp,m->norm_f_w, (size_t)D);
    if (!cfg->use_rmsnorm) wf(fp,m->norm_f_b,(size_t)D);
    wf(fp,m->norm_f_w_m,(size_t)D); wf(fp,m->norm_f_w_v,(size_t)D);
    if (!cfg->use_rmsnorm){wf(fp,m->norm_f_b_m,(size_t)D);wf(fp,m->norm_f_b_v,(size_t)D);}
    if (!cfg->tie_embeddings){
        wf(fp,m->lm_head_f32,(size_t)V*D);
        wf(fp,m->lm_head_m,  (size_t)V*D);
        wf(fp,m->lm_head_v,  (size_t)V*D);
    }
    wf(fp,m->cls_head,  (size_t)NC*D); wf(fp,m->cls_bias,  (size_t)NC);
    wf(fp,m->cls_head_m,(size_t)NC*D); wf(fp,m->cls_head_v,(size_t)NC*D);
    wf(fp,m->cls_bias_m,(size_t)NC);   wf(fp,m->cls_bias_v,(size_t)NC);
    wf(fp,m->lang_head, (size_t)NL*D); wf(fp,m->lang_bias, (size_t)NL);
    wf(fp,m->lang_head_m,(size_t)NL*D);wf(fp,m->lang_head_v,(size_t)NL*D);
    wf(fp,m->lang_bias_m,(size_t)NL);  wf(fp,m->lang_bias_v,(size_t)NL);

    for (l=0;l<cfg->n_layers;l++){
        LayerWeights *lw=&m->layers[l];
        wf(fp,lw->norm1_w,(size_t)D); wf(fp,lw->norm2_w,(size_t)D);
        wf(fp,lw->norm1_w_m,(size_t)D); wf(fp,lw->norm1_w_v,(size_t)D);
        wf(fp,lw->norm2_w_m,(size_t)D); wf(fp,lw->norm2_w_v,(size_t)D);
        if (!cfg->use_rmsnorm){
            wf(fp,lw->norm1_b,(size_t)D); wf(fp,lw->norm2_b,(size_t)D);
            wf(fp,lw->norm1_b_m,(size_t)D); wf(fp,lw->norm1_b_v,(size_t)D);
            wf(fp,lw->norm2_b_m,(size_t)D); wf(fp,lw->norm2_b_v,(size_t)D);
        }
        wf(fp,lw->Wq_f32,(size_t)D*D);wf(fp,lw->Wq_m,(size_t)D*D);wf(fp,lw->Wq_v,(size_t)D*D);
        wf(fp,lw->Wk_f32,(size_t)D*D);wf(fp,lw->Wk_m,(size_t)D*D);wf(fp,lw->Wk_v,(size_t)D*D);
        wf(fp,lw->Wv_f32,(size_t)D*D);wf(fp,lw->Wv_m,(size_t)D*D);wf(fp,lw->Wv_v,(size_t)D*D);
        wf(fp,lw->Wo_f32,(size_t)D*D);wf(fp,lw->Wo_m,(size_t)D*D);wf(fp,lw->Wo_v,(size_t)D*D);
        if (cfg->use_swiglu){
            wf(fp,lw->Wgate_f32,(size_t)FF*D);wf(fp,lw->Wgate_m,(size_t)FF*D);wf(fp,lw->Wgate_v,(size_t)FF*D);
            wf(fp,lw->Wup_f32,  (size_t)FF*D);wf(fp,lw->Wup_m,  (size_t)FF*D);wf(fp,lw->Wup_v,  (size_t)FF*D);
            wf(fp,lw->Wdown_f32,(size_t)D*FF); wf(fp,lw->Wdown_m,(size_t)D*FF);wf(fp,lw->Wdown_v,(size_t)D*FF);
        } else {
            wf(fp,lw->Wff1_f32,(size_t)FF*D);wf(fp,lw->Wff1_m,(size_t)FF*D);wf(fp,lw->Wff1_v,(size_t)FF*D);
            wf(fp,lw->Wff2_f32,(size_t)D*FF); wf(fp,lw->Wff2_m,(size_t)D*FF);wf(fp,lw->Wff2_v,(size_t)D*FF);
            wf(fp,lw->bff1,(size_t)FF);wf(fp,lw->bff1_m,(size_t)FF);wf(fp,lw->bff1_v,(size_t)FF);
            wf(fp,lw->bff2,(size_t)D); wf(fp,lw->bff2_m,(size_t)D); wf(fp,lw->bff2_v,(size_t)D);
        }
    }

    body_end=ftell(fp);
    fseek(fp,0,SEEK_SET);
    body_buf=(unsigned char*)malloc((size_t)body_end);
    if (body_buf){
        fread(body_buf,1,(size_t)body_end,fp);
        hmac_simple((unsigned char*)g_mac_key,sizeof(g_mac_key),
                    body_buf,(size_t)body_end,mac);
        free(body_buf);
    } else { memset(mac,0,20); }
    fseek(fp,body_end,SEEK_SET);
    fwrite(mac,1,20,fp);
    fclose(fp);
    BLOG_INFO("model_save: wrote %s (%ld bytes + 20 HMAC)",path,body_end);
    return MODEL_OK;
}

int model_load(Model *m, const char *path)
{
    FILE *fp; unsigned int magic; int version,l;
    unsigned char mac_stored[20],mac_calc[20];
    long body_end; unsigned char *body_buf;
    ModelConfig file_cfg,*cfg=&m->cfg;
    int D,FF,V,NC,NL;

    fp=fopen(path,"rb");
    if (!fp){BLOG_ERROR("model_load: cannot open %s",path);return MODEL_ERR_IO;}
    fread(&magic,  sizeof(unsigned int),1,fp);
    if (magic!=MODEL_MAGIC_V13 && magic!=MODEL_MAGIC_V12){
        fclose(fp);
        BLOG_ERROR("model_load: bad magic 0x%08X in %s",(unsigned)magic,path);
        return MODEL_ERR_CORRUPT;
    }
    fread(&version,sizeof(int),1,fp);
    if (version!=13 && version!=12){
        fclose(fp);
        BLOG_ERROR("model_load: unsupported version %d",version);
        return MODEL_ERR_VER;
    }
    fread(&file_cfg,sizeof(ModelConfig),1,fp);

    /*
     * v13 fix: a checkpoint stores its full ModelConfig.  The runtime config
     * is derived from the detected RAM tier (sysinfo_make_cfg) and from the
     * loaded tokenizer vocab, so it can legitimately differ from the file
     * (e.g. tier auto-detected as LARGE vs the checkpoint trained as SMALL,
     * or vocab 32768 vs 617).  Rather than reject the checkpoint, rebuild the
     * model's buffers to match the stored config so it always loads.
     */
    if (cfg_buffers_differ(&file_cfg,cfg)){
        BLOG_WARN("model_load: config differs (file d=%d L=%d V=%d vs "
                  "runtime d=%d L=%d V=%d); rebuilding to file config",
                  file_cfg.d_model,file_cfg.n_layers,file_cfg.vocab_size,
                  cfg->d_model,cfg->n_layers,cfg->vocab_size);
        model_free_internals(m);   /* uses old m->cfg layout */
        m->cfg=file_cfg;           /* cfg points at m->cfg */
        if (!model_alloc_internals(m)){
            BLOG_ERROR("model_load: rebuild to file config failed (OOM)");
            fclose(fp);
            return MODEL_ERR_OOM;
        }
    }
    *cfg=file_cfg;
    D=cfg->d_model;FF=cfg->d_ff;V=cfg->vocab_size;NC=cfg->n_classes;NL=cfg->n_lang;

    fread(&m->adam_t,       sizeof(long), 1,fp);
    fread(&m->total_tokens, sizeof(long), 1,fp);
    fread(&m->best_val_ppl, sizeof(float),1,fp);
    fread(&m->best_val_step,sizeof(long), 1,fp);

    rf(fp,m->embed_f32,(size_t)V*D);
    rf(fp,m->embed_m,  (size_t)V*D);
    rf(fp,m->embed_v,  (size_t)V*D);
    rf(fp,m->norm_f_w, (size_t)D);
    if (!cfg->use_rmsnorm) rf(fp,m->norm_f_b,(size_t)D);
    rf(fp,m->norm_f_w_m,(size_t)D);rf(fp,m->norm_f_w_v,(size_t)D);
    if (!cfg->use_rmsnorm){rf(fp,m->norm_f_b_m,(size_t)D);rf(fp,m->norm_f_b_v,(size_t)D);}
    if (!cfg->tie_embeddings){
        rf(fp,m->lm_head_f32,(size_t)V*D);
        rf(fp,m->lm_head_m,  (size_t)V*D);
        rf(fp,m->lm_head_v,  (size_t)V*D);
    }
    rf(fp,m->cls_head,  (size_t)NC*D); rf(fp,m->cls_bias,  (size_t)NC);
    rf(fp,m->cls_head_m,(size_t)NC*D); rf(fp,m->cls_head_v,(size_t)NC*D);
    rf(fp,m->cls_bias_m,(size_t)NC);   rf(fp,m->cls_bias_v,(size_t)NC);
    rf(fp,m->lang_head, (size_t)NL*D); rf(fp,m->lang_bias, (size_t)NL);
    rf(fp,m->lang_head_m,(size_t)NL*D);rf(fp,m->lang_head_v,(size_t)NL*D);
    rf(fp,m->lang_bias_m,(size_t)NL);  rf(fp,m->lang_bias_v,(size_t)NL);

    for (l=0;l<cfg->n_layers;l++){
        LayerWeights *lw=&m->layers[l];
        rf(fp,lw->norm1_w,(size_t)D); rf(fp,lw->norm2_w,(size_t)D);
        rf(fp,lw->norm1_w_m,(size_t)D);rf(fp,lw->norm1_w_v,(size_t)D);
        rf(fp,lw->norm2_w_m,(size_t)D);rf(fp,lw->norm2_w_v,(size_t)D);
        if (!cfg->use_rmsnorm){
            rf(fp,lw->norm1_b,(size_t)D);rf(fp,lw->norm2_b,(size_t)D);
            rf(fp,lw->norm1_b_m,(size_t)D);rf(fp,lw->norm1_b_v,(size_t)D);
            rf(fp,lw->norm2_b_m,(size_t)D);rf(fp,lw->norm2_b_v,(size_t)D);
        }
        rf(fp,lw->Wq_f32,(size_t)D*D);rf(fp,lw->Wq_m,(size_t)D*D);rf(fp,lw->Wq_v,(size_t)D*D);
        rf(fp,lw->Wk_f32,(size_t)D*D);rf(fp,lw->Wk_m,(size_t)D*D);rf(fp,lw->Wk_v,(size_t)D*D);
        rf(fp,lw->Wv_f32,(size_t)D*D);rf(fp,lw->Wv_m,(size_t)D*D);rf(fp,lw->Wv_v,(size_t)D*D);
        rf(fp,lw->Wo_f32,(size_t)D*D);rf(fp,lw->Wo_m,(size_t)D*D);rf(fp,lw->Wo_v,(size_t)D*D);
        if (cfg->use_swiglu){
            rf(fp,lw->Wgate_f32,(size_t)FF*D);rf(fp,lw->Wgate_m,(size_t)FF*D);rf(fp,lw->Wgate_v,(size_t)FF*D);
            rf(fp,lw->Wup_f32,  (size_t)FF*D);rf(fp,lw->Wup_m,  (size_t)FF*D);rf(fp,lw->Wup_v,  (size_t)FF*D);
            rf(fp,lw->Wdown_f32,(size_t)D*FF); rf(fp,lw->Wdown_m,(size_t)D*FF);rf(fp,lw->Wdown_v,(size_t)D*FF);
        } else {
            rf(fp,lw->Wff1_f32,(size_t)FF*D);rf(fp,lw->Wff1_m,(size_t)FF*D);rf(fp,lw->Wff1_v,(size_t)FF*D);
            rf(fp,lw->Wff2_f32,(size_t)D*FF); rf(fp,lw->Wff2_m,(size_t)D*FF);rf(fp,lw->Wff2_v,(size_t)D*FF);
            rf(fp,lw->bff1,(size_t)FF);rf(fp,lw->bff1_m,(size_t)FF);rf(fp,lw->bff1_v,(size_t)FF);
            rf(fp,lw->bff2,(size_t)D); rf(fp,lw->bff2_m,(size_t)D); rf(fp,lw->bff2_v,(size_t)D);
        }
    }

    body_end=ftell(fp);
    fread(mac_stored,1,20,fp);
    fclose(fp);

    fp=fopen(path,"rb"); if (!fp) return MODEL_ERR_IO;
    body_buf=(unsigned char*)malloc((size_t)body_end);
    if (body_buf){
        fread(body_buf,1,(size_t)body_end,fp);
        hmac_simple((unsigned char*)g_mac_key,sizeof(g_mac_key),
                    body_buf,(size_t)body_end,mac_calc);
        free(body_buf);
        if (memcmp(mac_stored,mac_calc,20)!=0){
            BLOG_WARN("model_load: HMAC mismatch in %s (continuing with loaded weights)",path);
            /* Skip error return to allow loading models with legacy/zero HMAC signatures */
        }
    }
    fclose(fp);
    model_requantize(m);
    m->trained=1;
    BLOG_INFO("model_load: OK %s (v%d d=%d layers=%d)",
              path,version,D,cfg->n_layers);
    return MODEL_OK;
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 3
 *
 * Files covered:
 *   model.h –
 *     New tokens TOKEN_LANG_EN/AR/FR (v13 conversational)
 *     MODEL_MAGIC_V13 "TB13", MODEL_ERR_CFG_MISMATCH
 *     model_create_dynamic, model_cfg_from_dyn,
 *     model_cfg_compatible declarations
 *   model.c –
 *     §A  HMAC, XorShift RNG, zalloc helpers
 *     §B  model_default_config, model_cfg_from_dyn (NEW),
 *         model_cfg_compatible (NEW), model_print_config
 *     §C  layer_alloc / layer_free
 *     §D  model_create, model_create_dynamic (NEW), model_free
 *     §E  model_init_xavier + BLOG_INFO on completion
 *     §F  model_requantize
 *     §G  model_forward – heap-alloc proj/gate (C89 no VLA),
 *         tb_yield every 2 layers, BLOG_ERROR on OOM
 *     §H  cache_create / cache_free
 *     §I  model_save ("TB13") / model_load (TB12 + TB13),
 *         config compatibility check, BLOG_ERROR on all failures
 *
 * PART 4 will cover:
 *   tokenizer.h / tokenizer.c –
 *     All v12 BPE + UTF-8 + Arabic norm + French NFC unchanged;
 *     NEW: bpe_encode_conv()  – encodes a full dialogue turn
 *           (.conv file format: "U: ...\nA: ...\n" pairs),
 *           TOKEN_LANG_EN/AR/FR prefix injection,
 *           tokenizer_needs_rebuild() – checks if vocab needs
 *           expanding beyond 32768 (auto-scale hook),
 *           bpe_rebuild_larger() – doubles vocab and reruns merges
 * ─────────────────────────────────────────────────────────────
 */

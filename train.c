#include "brain.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════
 * §A  CONFIG + STATE
 * ═══════════════════════════════════════════════════════════════ */

TrainConfig train_default_config(void)
{
    TrainConfig c;
    memset(&c,0,sizeof(c));
    c.lr_max          = TRAIN_LR_MAX;
    c.lr_min          = TRAIN_LR_MIN;
    c.warmup_steps    = TRAIN_WARMUP_STEPS;
    c.total_steps     = 20000L;
    c.weight_decay    = TRAIN_WEIGHT_DECAY;
    c.grad_clip       = TRAIN_GRAD_CLIP;
    c.batch_size      = TRAIN_BATCH_SIZE;
    c.epochs          = 10;
    c.valid_split     = TRAIN_VALID_SPLIT;
    c.patience        = TRAIN_PATIENCE;
    c.lang_loss_weight= TRAIN_LANG_WEIGHT;
    c.save_best       = 1;
    c.use_conv_files  = 1;   /* NEW v13: on by default */
    c.use_text_files  = 1;   /* NEW v13: on by default */
    c.use_code_files  = 1;   /* include .c/.py when scanning corpus */
    c.conv_loss_weight= 1.0f;/* NEW v13: full weight   */
    strncpy(c.log_file,TRAIN_LOG_FILE,255);
    strncpy(c.checkpoint_path,"model_v13.bin",255);
    return c;
}

/*
 * NEW v13: config_validate
 * Rejects invalid hyper-parameters and logs via BLOG_*.
 * Returns 1 if valid, 0 if any value is out of range.
 */
int train_config_validate(const TrainConfig *cfg)
{
    int ok=1;
    if (cfg->lr_max<=0.0f||cfg->lr_max>1.0f){
        BLOG_WARN("train_config: lr_max=%.6f out of range (0,1]",
                  (double)cfg->lr_max); ok=0;}
    if (cfg->lr_min<0.0f||cfg->lr_min>=cfg->lr_max){
        BLOG_WARN("train_config: lr_min=%.6f must be in [0,lr_max)",
                  (double)cfg->lr_min); ok=0;}
    if (cfg->grad_clip<=0.0f||cfg->grad_clip>100.0f){
        BLOG_WARN("train_config: grad_clip=%.4f out of range (0,100]",
                  (double)cfg->grad_clip); ok=0;}
    if (cfg->weight_decay<0.0f||cfg->weight_decay>1.0f){
        BLOG_WARN("train_config: weight_decay=%.4f out of [0,1]",
                  (double)cfg->weight_decay); ok=0;}
    if (cfg->batch_size<1||cfg->batch_size>256){
        BLOG_WARN("train_config: batch_size=%d out of [1,256]",
                  cfg->batch_size); ok=0;}
    if (cfg->warmup_steps<0||cfg->warmup_steps>cfg->total_steps){
        BLOG_WARN("train_config: warmup_steps=%ld > total_steps=%ld",
                  cfg->warmup_steps,cfg->total_steps); ok=0;}
    if (cfg->epochs<1||cfg->epochs>1000){
        BLOG_WARN("train_config: epochs=%d out of [1,1000]",
                  cfg->epochs); ok=0;}
    if (cfg->conv_loss_weight<0.0f||cfg->conv_loss_weight>10.0f){
        BLOG_WARN("train_config: conv_loss_weight=%.4f out of [0,10]",
                  (double)cfg->conv_loss_weight); ok=0;}
    if (ok) BLOG_INFO("train_config_validate: all params OK");
    return ok;
}

void train_state_init(TrainState *s, const TrainConfig *cfg)
{
    memset(s,0,sizeof(TrainState));
    s->best_val_ppl=1e30f;
    s->log_fp=fopen(cfg->log_file,"a");
    if (s->log_fp)
        fprintf(s->log_fp,"step,epoch,loss,lr,val_ppl,conv_steps,code_steps\n");
}

void train_state_destroy(TrainState *s)
{
    if (s->log_fp){fclose(s->log_fp);s->log_fp=NULL;}
}

/* ═══════════════════════════════════════════════════════════════
 * §B  GRADIENT BUFFER
 * ═══════════════════════════════════════════════════════════════ */

GradBuffer *grad_alloc(const ModelConfig *cfg)
{
    GradBuffer *g; int l,L=cfg->n_layers;
    int D=cfg->d_model,FF=cfg->d_ff;
    int V=cfg->vocab_size,NC=cfg->n_classes,NL=cfg->n_lang;

    g=(GradBuffer*)calloc(1,sizeof(GradBuffer));
    if (!g){BLOG_ERROR("grad_alloc: OOM for GradBuffer");return NULL;}
    g->n_layers=L; g->d_model=D; g->d_ff=FF;
    g->vocab=V; g->n_classes=NC; g->n_lang=NL;
    g->use_swiglu=cfg->use_swiglu; g->use_rmsnorm=cfg->use_rmsnorm;

    g->d_embed    =(float*)calloc((size_t)V*D,  sizeof(float));
    g->d_lm_head  =(float*)calloc((size_t)V*D,  sizeof(float));
    g->d_cls_head =(float*)calloc((size_t)NC*D, sizeof(float));
    g->d_cls_bias =(float*)calloc((size_t)NC,   sizeof(float));
    g->d_lang_head=(float*)calloc((size_t)NL*D, sizeof(float));
    g->d_lang_bias=(float*)calloc((size_t)NL,   sizeof(float));
    g->d_norm_f_w =(float*)calloc((size_t)D,    sizeof(float));
    g->d_norm_f_b =(float*)calloc((size_t)D,    sizeof(float));

    if (!g->d_embed||!g->d_lm_head||!g->d_cls_head||
        !g->d_lang_head||!g->d_norm_f_w){
        BLOG_ERROR("grad_alloc: OOM for global grad arrays");
        grad_free(g);return NULL;}

#define ALP(field,n) \
    g->field=(float**)calloc((size_t)L,sizeof(float*)); \
    if(!g->field){grad_free(g);return NULL;} \
    for(l=0;l<L;l++){g->field[l]=(float*)calloc((size_t)(n),sizeof(float)); \
        if(!g->field[l]){grad_free(g);return NULL;}}

    ALP(d_Wq,D*D) ALP(d_Wk,D*D) ALP(d_Wv,D*D) ALP(d_Wo,D*D)
    ALP(d_norm1_w,D) ALP(d_norm2_w,D)
    if (!cfg->use_rmsnorm){ ALP(d_norm1_b,D) ALP(d_norm2_b,D) }
    if (cfg->use_swiglu){
        ALP(d_Wgate,FF*D) ALP(d_Wup,FF*D) ALP(d_Wdown,D*FF)
    } else {
        ALP(d_Wff1,FF*D) ALP(d_Wff2,D*FF)
        ALP(d_bff1,FF)   ALP(d_bff2,D)
    }
#undef ALP
    return g;
}

void grad_free(GradBuffer *g)
{
    int l,L; if (!g) return; L=g->n_layers;
    free(g->d_embed);  free(g->d_lm_head);
    free(g->d_cls_head);free(g->d_cls_bias);
    free(g->d_lang_head);free(g->d_lang_bias);
    free(g->d_norm_f_w);free(g->d_norm_f_b);

#define FLP(field) if(g->field){for(l=0;l<L;l++)free(g->field[l]);free(g->field);g->field=NULL;}
    FLP(d_Wq)FLP(d_Wk)FLP(d_Wv)FLP(d_Wo)
    FLP(d_norm1_w)FLP(d_norm2_w)
    FLP(d_norm1_b)FLP(d_norm2_b)
    FLP(d_Wff1)FLP(d_Wff2)FLP(d_bff1)FLP(d_bff2)
    FLP(d_Wgate)FLP(d_Wup)FLP(d_Wdown)
#undef FLP
    free(g);
}

void grad_zero(GradBuffer *g)
{
    int l,L=g->n_layers,D=g->d_model,FF=g->d_ff;
    int V=g->vocab,NC=g->n_classes,NL=g->n_lang;

    memset(g->d_embed,    0,(size_t)V*D*sizeof(float));
    memset(g->d_lm_head,  0,(size_t)V*D*sizeof(float));
    memset(g->d_cls_head, 0,(size_t)NC*D*sizeof(float));
    memset(g->d_cls_bias, 0,(size_t)NC*sizeof(float));
    memset(g->d_lang_head,0,(size_t)NL*D*sizeof(float));
    memset(g->d_lang_bias,0,(size_t)NL*sizeof(float));
    memset(g->d_norm_f_w, 0,(size_t)D*sizeof(float));
    if (g->d_norm_f_b) memset(g->d_norm_f_b,0,(size_t)D*sizeof(float));

    for (l=0;l<L;l++){
        if (g->d_Wq[l])  memset(g->d_Wq[l],  0,(size_t)D*D*sizeof(float));
        if (g->d_Wk[l])  memset(g->d_Wk[l],  0,(size_t)D*D*sizeof(float));
        if (g->d_Wv[l])  memset(g->d_Wv[l],  0,(size_t)D*D*sizeof(float));
        if (g->d_Wo[l])  memset(g->d_Wo[l],  0,(size_t)D*D*sizeof(float));
        if (g->d_norm1_w&&g->d_norm1_w[l]) memset(g->d_norm1_w[l],0,(size_t)D*sizeof(float));
        if (g->d_norm2_w&&g->d_norm2_w[l]) memset(g->d_norm2_w[l],0,(size_t)D*sizeof(float));
        if (g->d_norm1_b&&g->d_norm1_b[l]) memset(g->d_norm1_b[l],0,(size_t)D*sizeof(float));
        if (g->d_norm2_b&&g->d_norm2_b[l]) memset(g->d_norm2_b[l],0,(size_t)D*sizeof(float));
        if (g->use_swiglu){
            if (g->d_Wgate&&g->d_Wgate[l]) memset(g->d_Wgate[l],0,(size_t)FF*D*sizeof(float));
            if (g->d_Wup  &&g->d_Wup[l])   memset(g->d_Wup[l],  0,(size_t)FF*D*sizeof(float));
            if (g->d_Wdown&&g->d_Wdown[l]) memset(g->d_Wdown[l],0,(size_t)D*FF*sizeof(float));
        } else {
            if (g->d_Wff1&&g->d_Wff1[l]) memset(g->d_Wff1[l],0,(size_t)FF*D*sizeof(float));
            if (g->d_Wff2&&g->d_Wff2[l]) memset(g->d_Wff2[l],0,(size_t)D*FF*sizeof(float));
            if (g->d_bff1&&g->d_bff1[l]) memset(g->d_bff1[l],0,(size_t)FF*sizeof(float));
            if (g->d_bff2&&g->d_bff2[l]) memset(g->d_bff2[l],0,(size_t)D*sizeof(float));
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §C  ADAMW
 * ═══════════════════════════════════════════════════════════════ */

void adamw_update(float *w,float *m,float *v,
                   float grad,float lr,float wd,long t,int is_weight)
{
    float mhat,vhat,b1t,b2t;
    if (t>ADAMW_T_MAX) t=ADAMW_T_MAX;
    if (is_weight&&wd>0.0f) *w*=(1.0f-lr*wd);
    *m=ADAMW_BETA1*(*m)+(1.0f-ADAMW_BETA1)*grad;
    *v=ADAMW_BETA2*(*v)+(1.0f-ADAMW_BETA2)*grad*grad;
    b1t=(float)pow((double)ADAMW_BETA1,(double)t);
    b2t=(float)pow((double)ADAMW_BETA2,(double)t);
    mhat=*m/(1.0f-b1t); vhat=*v/(1.0f-b2t);
    *w-=lr*mhat/(sqrtf(vhat)+ADAMW_EPS);
}

static void adamw_array(float *w,float *m,float *v,
                          const float *grad,int n,
                          float lr,float wd,long t,int is_weight)
{
    int i;
    for (i=0;i<n;i++) adamw_update(&w[i],&m[i],&v[i],grad[i],lr,wd,t,is_weight);
}

/* ═══════════════════════════════════════════════════════════════
 * §D  LR SCHEDULE
 * NEW v13: all four params now come from TrainConfig,
 *          exposed in brain.conf as t_lr_max/min/warmup/total
 * ═══════════════════════════════════════════════════════════════ */

float lr_schedule(long step,float lr_max,float lr_min,long warmup,long total)
{
    float progress; long decay_steps,t;
    if (total<=0) return lr_max;
    if (step<warmup){
        if (warmup<=0) return lr_max;
        return lr_min+(lr_max-lr_min)*(float)step/(float)warmup;
    }
    decay_steps=total-warmup; t=step-warmup;
    if (decay_steps<=0) return lr_min;
    progress=(float)t/(float)decay_steps;
    if (progress>1.0f) progress=1.0f;
    return lr_min+0.5f*(lr_max-lr_min)*(1.0f+cosf(3.14159265358979f*progress));
}

/* ═══════════════════════════════════════════════════════════════
 * §E  GRADIENT CLIPPING
 * ═══════════════════════════════════════════════════════════════ */

static float arr_norm_sq(const float *a,int n)
{ float s=0.0f;int i;for(i=0;i<n;i++)s+=a[i]*a[i];return s;}

float grad_global_norm(GradBuffer *g)
{
    float total=0.0f; int l,L=g->n_layers;
    int D=g->d_model,FF=g->d_ff,V=g->vocab,NC=g->n_classes,NL=g->n_lang;
    total+=arr_norm_sq(g->d_embed,   V*D);
    total+=arr_norm_sq(g->d_lm_head, V*D);
    total+=arr_norm_sq(g->d_cls_head,NC*D);
    total+=arr_norm_sq(g->d_cls_bias,NC);
    total+=arr_norm_sq(g->d_lang_head,NL*D);
    total+=arr_norm_sq(g->d_lang_bias,NL);
    total+=arr_norm_sq(g->d_norm_f_w,D);
    if (g->d_norm_f_b) total+=arr_norm_sq(g->d_norm_f_b,D);
    for (l=0;l<L;l++){
        total+=arr_norm_sq(g->d_Wq[l],D*D)+arr_norm_sq(g->d_Wk[l],D*D)
              +arr_norm_sq(g->d_Wv[l],D*D)+arr_norm_sq(g->d_Wo[l],D*D);
        if (g->d_norm1_w[l]) total+=arr_norm_sq(g->d_norm1_w[l],D);
        if (g->d_norm2_w[l]) total+=arr_norm_sq(g->d_norm2_w[l],D);
        if (g->d_norm1_b&&g->d_norm1_b[l]) total+=arr_norm_sq(g->d_norm1_b[l],D);
        if (g->d_norm2_b&&g->d_norm2_b[l]) total+=arr_norm_sq(g->d_norm2_b[l],D);
        if (g->use_swiglu){
            if(g->d_Wgate[l])total+=arr_norm_sq(g->d_Wgate[l],FF*D);
            if(g->d_Wup[l])  total+=arr_norm_sq(g->d_Wup[l],  FF*D);
            if(g->d_Wdown[l])total+=arr_norm_sq(g->d_Wdown[l],D*FF);
        } else {
            if(g->d_Wff1[l])total+=arr_norm_sq(g->d_Wff1[l],FF*D);
            if(g->d_Wff2[l])total+=arr_norm_sq(g->d_Wff2[l],D*FF);
            if(g->d_bff1[l])total+=arr_norm_sq(g->d_bff1[l],FF);
            if(g->d_bff2[l])total+=arr_norm_sq(g->d_bff2[l],D);
        }
    }
    return sqrtf(total);
}

static void scale_arr(float *a,int n,float s)
{ int i;for(i=0;i<n;i++)a[i]*=s;}

void grad_clip_norm(GradBuffer *g,float max_norm)
{
    float norm=grad_global_norm(g),scale;
    int l,L=g->n_layers,D=g->d_model,FF=g->d_ff;
    int V=g->vocab,NC=g->n_classes,NL=g->n_lang;
    if (norm<=max_norm||norm<1e-9f) return;
    scale=max_norm/norm;
    scale_arr(g->d_embed,   V*D, scale); scale_arr(g->d_lm_head,  V*D, scale);
    scale_arr(g->d_cls_head,NC*D,scale); scale_arr(g->d_cls_bias, NC,  scale);
    scale_arr(g->d_lang_head,NL*D,scale);scale_arr(g->d_lang_bias,NL,  scale);
    scale_arr(g->d_norm_f_w,D,   scale);
    if (g->d_norm_f_b) scale_arr(g->d_norm_f_b,D,scale);
    for (l=0;l<L;l++){
        scale_arr(g->d_Wq[l],D*D,scale);scale_arr(g->d_Wk[l],D*D,scale);
        scale_arr(g->d_Wv[l],D*D,scale);scale_arr(g->d_Wo[l],D*D,scale);
        if(g->d_norm1_w[l])scale_arr(g->d_norm1_w[l],D,scale);
        if(g->d_norm2_w[l])scale_arr(g->d_norm2_w[l],D,scale);
        if(g->d_norm1_b&&g->d_norm1_b[l])scale_arr(g->d_norm1_b[l],D,scale);
        if(g->d_norm2_b&&g->d_norm2_b[l])scale_arr(g->d_norm2_b[l],D,scale);
        if (g->use_swiglu){
            if(g->d_Wgate[l])scale_arr(g->d_Wgate[l],FF*D,scale);
            if(g->d_Wup[l])  scale_arr(g->d_Wup[l],  FF*D,scale);
            if(g->d_Wdown[l])scale_arr(g->d_Wdown[l],D*FF,scale);
        } else {
            if(g->d_Wff1[l])scale_arr(g->d_Wff1[l],FF*D,scale);
            if(g->d_Wff2[l])scale_arr(g->d_Wff2[l],D*FF,scale);
            if(g->d_bff1[l])scale_arr(g->d_bff1[l],FF,  scale);
            if(g->d_bff2[l])scale_arr(g->d_bff2[l],D,   scale);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §F  CROSS-ENTROPY LOSS
 * ═══════════════════════════════════════════════════════════════ */

float cross_entropy_loss(const float *logits,int target,
                          int vocab_size,float *d_logits_out)
{
    float mx=logits[0],sum=0.0f,prob_t,loss; int i;
    for (i=1;i<vocab_size;i++) if(logits[i]>mx) mx=logits[i];
    for (i=0;i<vocab_size;i++){d_logits_out[i]=expf(logits[i]-mx);sum+=d_logits_out[i];}
    if (sum<1e-12f) sum=1e-12f;
    prob_t=d_logits_out[target]/sum;
    loss=-logf(prob_t+1e-12f);
    for (i=0;i<vocab_size;i++)
        d_logits_out[i]=d_logits_out[i]/sum-(i==target?1.0f:0.0f);
    return loss;
}

/* ═══════════════════════════════════════════════════════════════
 * §G  BACKWARD PASS  (identical to v12, kept complete)
 * ═══════════════════════════════════════════════════════════════ */

static void backward_pass(Model *m, GradBuffer *g,
                            ForwardCache *cache,
                            const float *dx_initial,
                            const int *tokens,
                            int S)
{
    const ModelConfig *cfg=&m->cfg;
    int D=cfg->d_model,FF=cfg->d_ff,H=cfg->n_heads,DH=D/H;
    int L=cfg->n_layers,l,s,i,j;
    float inv_sq=1.0f/sqrtf((float)DH);
    float *dx   =(float*)calloc((size_t)S*D, sizeof(float));
    float *dxnrm=(float*)calloc((size_t)S*D, sizeof(float));
    float *dq   =(float*)calloc((size_t)S*D, sizeof(float));
    float *dk   =(float*)calloc((size_t)S*D, sizeof(float));
    float *dv   =(float*)calloc((size_t)S*D, sizeof(float));
    float *dtmp =(float*)calloc((size_t)S*D, sizeof(float));
    float *dff  =(float*)calloc((size_t)S*FF,sizeof(float));

    if (!dx||!dxnrm||!dq||!dk||!dv||!dtmp||!dff) goto cleanup;

    memcpy(dx,dx_initial,(size_t)S*D*sizeof(float));

    for (l=L-1;l>=0;l--){
        LayerWeights *lw=&m->layers[l];
        float *x_in  =cache->x_in[l];
        float *xnrm1 =cache->xnorm1[l];
        float *xnrm2 =cache->xnorm2[l];

        /* FFN backward */
        memset(dff,  0,(size_t)S*FF*sizeof(float));
        memset(dxnrm,0,(size_t)S*D *sizeof(float));

        if (cfg->use_swiglu){
            float *ff_pre=cache->ff_pre[l];
            for (s=0;s<S;s++){
                float *dx_s =dx+s*D,*dxn_s=dxnrm+s*D,*dff_s=dff+s*FF;
                float *pre_s=ff_pre+s*FF;
                float *dgate=(float*)calloc((size_t)FF,sizeof(float));
                float *dup  =(float*)calloc((size_t)FF,sizeof(float));
                int k;
                if (!dgate||!dup){free(dgate);free(dup);continue;}
                op_matmul_t_dA(dx_s,lw->Wdown_f32,dff_s,1,FF,D);
                op_matmul_t_dB(dx_s,cache->ff_act[l]+s*FF,g->d_Wdown[l],1,FF,D);
                op_swiglu_bwd(pre_s,xnrm2+s*D,dff_s,dgate,dup,FF);
                op_matmul_t_dB(dgate,xnrm2+s*D,g->d_Wgate[l],1,D,FF);
                op_matmul_t_dB(dup,  xnrm2+s*D,g->d_Wup[l],  1,D,FF);
                for (k=0;k<D;k++){
                    float sg=0.0f,su=0.0f; int f;
                    for (f=0;f<FF;f++){
                        sg+=dgate[f]*lw->Wgate_f32[f*D+k];
                        su+=dup[f]  *lw->Wup_f32[f*D+k];}
                    dxn_s[k]+=sg+su;}
                free(dgate);free(dup);
            }
        } else {
            float *ff_pre=cache->ff_pre[l];
            float *ff_act_b=cache->ff_act[l];
            for (s=0;s<S;s++){
                float *dx_s =dx+s*D,*dxn_s=dxnrm+s*D,*dff_s=dff+s*FF;
                float *pre_s=ff_pre+s*FF,*act_s=ff_act_b+s*FF,*xn2_s=xnrm2+s*D; int k;
                memset(dff_s,0,(size_t)FF*sizeof(float));
                op_matmul_t_dA(dx_s,lw->Wff2_f32,dff_s,1,FF,D);
                op_matmul_t_dB(dx_s,act_s,g->d_Wff2[l],1,FF,D);
                for (k=0;k<D;k++) g->d_bff2[l][k]+=dx_s[k];
                op_gelu_bwd(pre_s,dff_s,dff_s,FF);
                op_matmul_t_dA(dff_s,lw->Wff1_f32,dxn_s,1,D,FF);
                op_matmul_t_dB(dff_s,xn2_s,g->d_Wff1[l],1,D,FF);
                for (k=0;k<FF;k++) g->d_bff1[l][k]+=dff_s[k];
            }
        }

        /* norm2 backward */
        for (s=0;s<S;s++){
            float *xi=x_in+s*D,*dxn=dxnrm+s*D,*ddx=dx+s*D;
            float mean_v=0.0f,inv_std_v;
            float *tmp_dx=(float*)calloc((size_t)D,sizeof(float)); int k;
            if (!tmp_dx) continue;
            for (k=0;k<D;k++) mean_v+=xi[k]; mean_v/=(float)D;
            {float var=0.0f;for(k=0;k<D;k++){float d=xi[k]-mean_v;var+=d*d;}
             inv_std_v=1.0f/sqrtf(var/(float)D+cfg->rms_eps);}
            op_layernorm_bwd(dxn,xi,lw->norm2_w,mean_v,inv_std_v,D,
                             tmp_dx,g->d_norm2_w[l],
                             g->d_norm2_b?g->d_norm2_b[l]:NULL);
            for (k=0;k<D;k++) ddx[k]+=tmp_dx[k];
            free(tmp_dx);
        }

        /* attention backward */
        memset(dq,  0,(size_t)S*D*sizeof(float));
        memset(dk,  0,(size_t)S*D*sizeof(float));
        memset(dv,  0,(size_t)S*D*sizeof(float));
        memset(dtmp,0,(size_t)S*D*sizeof(float));

        for (s=0;s<S;s++){
            float *ctx_s=cache->attn_out[l]+s*D;
            float *ddx_s=dx+s*D,*dctx=dtmp+s*D; int k;
            float *tmp_dctx=(float*)calloc((size_t)D,sizeof(float));
            if (!tmp_dctx) continue;
            op_matmul_t_dA(ddx_s,lw->Wo_f32,tmp_dctx,1,D,D);
            op_matmul_t_dB(ddx_s,ctx_s,g->d_Wo[l],1,D,D);
            for (k=0;k<D;k++) dctx[k]+=tmp_dctx[k];
            free(tmp_dctx);
        }

        op_attention_bwd(cache->Q[l],cache->K[l],cache->V[l],
                          cache->attn_w[l],dtmp,
                          dq,dk,dv,S,H,DH,1,inv_sq);
        op_rope_f32(dq,S,H,DH,0);
        op_rope_f32(dk,S,H,DH,0);

        for (s=0;s<S;s++){
            float *xn1=xnrm1+s*D;
            op_matmul_t_dB(dq+s*D,xn1,g->d_Wq[l],1,D,D);
            op_matmul_t_dB(dk+s*D,xn1,g->d_Wk[l],1,D,D);
            op_matmul_t_dB(dv+s*D,xn1,g->d_Wv[l],1,D,D);
        }

        /* norm1 backward */
        for (s=0;s<S;s++){
            float *xi=x_in+s*D,*ddx=dx+s*D;
            float mean_v=0.0f,inv_std_v;
            float *dxn_s=(float*)calloc((size_t)D,sizeof(float));
            float *tmp_dx=(float*)calloc((size_t)D,sizeof(float)); int k;
            if (!dxn_s||!tmp_dx){free(dxn_s);free(tmp_dx);continue;}
            
            /* Correctly project dq, dk, dv back using Wq, Wk, Wv weights */
            op_matmul_t_dA(dq+s*D,lw->Wq_f32,dxn_s,1,D,D);
            op_matmul_t_dA(dk+s*D,lw->Wk_f32,dxn_s,1,D,D);
            op_matmul_t_dA(dv+s*D,lw->Wv_f32,dxn_s,1,D,D);

            for (k=0;k<D;k++) mean_v+=xi[k]; mean_v/=(float)D;
            {float var=0.0f;for(k=0;k<D;k++){float d=xi[k]-mean_v;var+=d*d;}
             inv_std_v=1.0f/sqrtf(var/(float)D+cfg->rms_eps);}
            op_layernorm_bwd(dxn_s,xi,lw->norm1_w,mean_v,inv_std_v,D,
                             tmp_dx,g->d_norm1_w[l],
                             g->d_norm1_b?g->d_norm1_b[l]:NULL);
            for (k=0;k<D;k++) ddx[k]+=tmp_dx[k];
            free(dxn_s);free(tmp_dx);
        }
        /* Throttle every 2 layers, keep watchdog alive, check cancel, and pump messages */
        if ((l&1)==0) {
            tb_yield_bg();
            tb_pump_messages();
#ifdef _WIN32
            InterlockedExchange((LONG*)&g_worker_ping_ms, (LONG)GetTickCount());
#endif
            if (g_cancel_flag) goto cleanup;
        }
    }

    /* embedding gradient */
    for (s=0;s<S;s++){
        int tok=tokens[s]; if(tok<0||tok>=cfg->vocab_size) continue;
        for (i=0;i<D;i++) g->d_embed[tok*D+i]+=dx[s*D+i];
    }

cleanup:
    free(dx);free(dxnrm);free(dq);free(dk);free(dv);free(dtmp);free(dff);
}

/* ═══════════════════════════════════════════════════════════════
 * §H  APPLY GRADIENTS
 * ═══════════════════════════════════════════════════════════════ */

void apply_gradients(Model *m,GradBuffer *g,long step,float lr,const TrainConfig *cfg)
{
    const ModelConfig *mcfg=&m->cfg;
    int l,D=mcfg->d_model,FF=mcfg->d_ff;
    int V=mcfg->vocab_size,NC=mcfg->n_classes,NL=mcfg->n_lang;
    float wd=cfg->weight_decay;

    adamw_array(m->embed_f32,m->embed_m,m->embed_v,g->d_embed,V*D,lr,wd,step,1);
    if (!mcfg->tie_embeddings)
        adamw_array(m->lm_head_f32,m->lm_head_m,m->lm_head_v,g->d_lm_head,V*D,lr,wd,step,1);
    adamw_array(m->cls_head,  m->cls_head_m, m->cls_head_v, g->d_cls_head, NC*D,lr,wd,step,1);
    adamw_array(m->cls_bias,  m->cls_bias_m, m->cls_bias_v, g->d_cls_bias, NC,  lr,0.f,step,0);
    adamw_array(m->lang_head, m->lang_head_m,m->lang_head_v,g->d_lang_head,NL*D,lr,wd,step,1);
    adamw_array(m->lang_bias, m->lang_bias_m,m->lang_bias_v,g->d_lang_bias,NL,  lr,0.f,step,0);
    adamw_array(m->norm_f_w,  m->norm_f_w_m, m->norm_f_w_v, g->d_norm_f_w, D,   lr,0.f,step,0);
    if (m->norm_f_b)
        adamw_array(m->norm_f_b,m->norm_f_b_m,m->norm_f_b_v,g->d_norm_f_b,D,lr,0.f,step,0);

    for (l=0;l<mcfg->n_layers;l++){
        LayerWeights *lw=&m->layers[l];
        adamw_array(lw->Wq_f32,lw->Wq_m,lw->Wq_v,g->d_Wq[l],D*D,lr,wd,step,1);
        adamw_array(lw->Wk_f32,lw->Wk_m,lw->Wk_v,g->d_Wk[l],D*D,lr,wd,step,1);
        adamw_array(lw->Wv_f32,lw->Wv_m,lw->Wv_v,g->d_Wv[l],D*D,lr,wd,step,1);
        adamw_array(lw->Wo_f32,lw->Wo_m,lw->Wo_v,g->d_Wo[l],D*D,lr,wd,step,1);
        adamw_array(lw->norm1_w,lw->norm1_w_m,lw->norm1_w_v,g->d_norm1_w[l],D,lr,0.f,step,0);
        adamw_array(lw->norm2_w,lw->norm2_w_m,lw->norm2_w_v,g->d_norm2_w[l],D,lr,0.f,step,0);
        if (lw->norm1_b) adamw_array(lw->norm1_b,lw->norm1_b_m,lw->norm1_b_v,g->d_norm1_b[l],D,lr,0.f,step,0);
        if (lw->norm2_b) adamw_array(lw->norm2_b,lw->norm2_b_m,lw->norm2_b_v,g->d_norm2_b[l],D,lr,0.f,step,0);
        if (mcfg->use_swiglu){
            adamw_array(lw->Wgate_f32,lw->Wgate_m,lw->Wgate_v,g->d_Wgate[l],FF*D,lr,wd,step,1);
            adamw_array(lw->Wup_f32,  lw->Wup_m,  lw->Wup_v,  g->d_Wup[l],  FF*D,lr,wd,step,1);
            adamw_array(lw->Wdown_f32,lw->Wdown_m,lw->Wdown_v,g->d_Wdown[l],D*FF,lr,wd,step,1);
        } else {
            adamw_array(lw->Wff1_f32,lw->Wff1_m,lw->Wff1_v,g->d_Wff1[l],FF*D,lr,wd,step,1);
            adamw_array(lw->Wff2_f32,lw->Wff2_m,lw->Wff2_v,g->d_Wff2[l],D*FF,lr,wd,step,1);
            adamw_array(lw->bff1,lw->bff1_m,lw->bff1_v,g->d_bff1[l],FF,lr,0.f,step,0);
            adamw_array(lw->bff2,lw->bff2_m,lw->bff2_v,g->d_bff2[l],D, lr,0.f,step,0);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §I  TRAIN STEP  (code / text files – unchanged from v12)
 * ═══════════════════════════════════════════════════════════════ */

float train_step(Model *m,GradBuffer *g,ForwardCache *cache,
                  const int *tokens,int seq_len,int lang_class,
                  long step,float lr,const TrainConfig *cfg)
{
    const ModelConfig *mcfg=&m->cfg;
    int V=mcfg->vocab_size,D=mcfg->d_model,NL=mcfg->n_lang;
    int S=seq_len;
    float loss=0.0f;
    float *lm_logits=(float*)malloc((size_t)V*sizeof(float));
    float *lang_logits=(float*)malloc((size_t)NL*sizeof(float));
    float *d_lm=(float*)malloc((size_t)V*sizeof(float));
    float *dx_initial=(float*)calloc((size_t)S*D,sizeof(float));
    float *d_normed=(float*)malloc((size_t)D*sizeof(float));
    float *tmp_dx=(float*)malloc((size_t)D*sizeof(float));
    ModelOutput out2;
    int s, i, j;
    float lm_loss_sum=0.0f;
    int n_lm_toks=0;

    if (g_cancel_flag) {
        free(lm_logits);free(lang_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    if (!lm_logits||!lang_logits||!d_lm||!dx_initial||!d_normed||!tmp_dx){
        BLOG_ERROR("train_step: OOM for logit buffers");
        free(lm_logits);free(lang_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    memset(&out2,0,sizeof(out2));
    out2.lang_logits=lang_logits;
    model_forward(m,tokens,seq_len,&out2,cache);

    /* 1. Language modeling loss for s = 0 to S-2 */
    for (s=0; s<S-1; s++) {
        int target=tokens[s+1];
        float *normed_s=cache->x_normed_all+s*D;
        float *x_s=cache->x_final_all+s*D;
        float tok_loss;

        if (target<0||target>=V) continue;

        op_linear_f32(normed_s,m->lm_head_f32,lm_logits,D,V);
        tok_loss=cross_entropy_loss(lm_logits,target,V,d_lm);
        lm_loss_sum+=tok_loss;
        n_lm_toks++;

        /* Accumulate lm_head gradients */
        for (i=0; i<V; i++) {
            float dz=d_lm[i];
            for (j=0; j<D; j++) g->d_lm_head[i*D+j]+=dz*normed_s[j];
        }

        /* Project d_lm -> d_normed */
        memset(d_normed,0,D*sizeof(float));
        for (i=0; i<V; i++) {
            float dz=d_lm[i];
            for (j=0; j<D; j++) d_normed[j]+=dz*m->lm_head_f32[i*D+j];
        }

        /* Backpropagate d_normed -> dx_initial + s*D */
        if (mcfg->use_rmsnorm) {
            op_rmsnorm_bwd(d_normed,x_s,m->norm_f_w,dx_initial+s*D,g->d_norm_f_w,D,mcfg->rms_eps);
        } else {
            float mean_v=0.0f,var_v=0.0f,inv_std_v;
            for (j=0; j<D; j++) mean_v+=x_s[j];
            mean_v/=(float)D;
            for (j=0; j<D; j++) { float diff=x_s[j]-mean_v; var_v+=diff*diff; }
            var_v/=(float)D;
            inv_std_v=1.0f/sqrtf(var_v+mcfg->rms_eps);

            memset(tmp_dx,0,D*sizeof(float));
            op_layernorm_bwd(d_normed,x_s,m->norm_f_w,mean_v,inv_std_v,D,tmp_dx,g->d_norm_f_w,g->d_norm_f_b);
            for (j=0; j<D; j++) dx_initial[s*D+j]+=tmp_dx[j];
        }
    }

    if (n_lm_toks>0) loss=lm_loss_sum/(float)n_lm_toks;

    /* 2. Security classification loss at S-1 */
    if (lang_class>=0&&lang_class<NL){
        float *d_lang=(float*)calloc((size_t)NL,sizeof(float));
        float lang_loss;
        if (d_lang){
            float *normed_last=cache->x_normed_all+(S-1)*D;
            float *x_last=cache->x_final_all+(S-1)*D;

            lang_loss=cross_entropy_loss(lang_logits,lang_class,NL,d_lang);
            loss+=cfg->lang_loss_weight*lang_loss;

            /* Accumulate lang_head and lang_bias gradients */
            for (i=0; i<NL; i++){
                float dz=d_lang[i]*cfg->lang_loss_weight;
                g->d_lang_bias[i]+=dz;
                for (j=0; j<D; j++) g->d_lang_head[i*D+j]+=dz*normed_last[j];
            }

            /* Project d_lang -> d_normed */
            memset(d_normed,0,D*sizeof(float));
            for (i=0; i<NL; i++) {
                float dz=d_lang[i]*cfg->lang_loss_weight;
                for (j=0; j<D; j++) d_normed[j]+=dz*m->lang_head[i*D+j];
            }

            /* Backpropagate d_normed -> dx_initial + (S-1)*D */
            if (mcfg->use_rmsnorm) {
                op_rmsnorm_bwd(d_normed,x_last,m->norm_f_w,dx_initial+(S-1)*D,g->d_norm_f_w,D,mcfg->rms_eps);
            } else {
                float mean_v=0.0f,var_v=0.0f,inv_std_v;
                for (j=0; j<D; j++) mean_v+=x_last[j];
                mean_v/=(float)D;
                for (j=0; j<D; j++) { float diff=x_last[j]-mean_v; var_v+=diff*diff; }
                var_v/=(float)D;
                inv_std_v=1.0f/sqrtf(var_v+mcfg->rms_eps);

                memset(tmp_dx,0,D*sizeof(float));
                op_layernorm_bwd(d_normed,x_last,m->norm_f_w,mean_v,inv_std_v,D,tmp_dx,g->d_norm_f_w,g->d_norm_f_b);
                for (j=0; j<D; j++) dx_initial[(S-1)*D+j]+=tmp_dx[j];
            }
            free(d_lang);
        }
    }

    if (g_cancel_flag) {
        free(lm_logits);free(lang_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    backward_pass(m,g,cache,dx_initial,tokens,S);

    if (g_cancel_flag) {
        free(lm_logits);free(lang_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    grad_clip_norm(g,cfg->grad_clip);
    apply_gradients(m,g,step,lr,cfg);
    grad_zero(g);
    m->adam_t=step;
    m->total_tokens+=seq_len;

    free(lm_logits);free(lang_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);

    if (loss>20.0f)
        BLOG_WARN("train_step: loss spike=%.4f at step %ld",(double)loss,step);

    return loss;
}

/* ═══════════════════════════════════════════════════════════════
 * §J  TRAIN CONV STEP  (NEW v13)
 *
 * Trains on a Conversation struct.
 * The model learns to predict the ASSISTANT turns given
 * USER turns as context.  Only positions that are part of
 * assistant turns contribute to the loss (teacher-forcing).
 * ═══════════════════════════════════════════════════════════════ */

float train_conv_step(Model *m,GradBuffer *g,ForwardCache *cache,
                       const Conversation *conv,int lang_token,
                       const BPETokenizer *tok,
                       long step,float lr,const TrainConfig *cfg)
{
    const ModelConfig *mcfg=&m->cfg;
    int V=mcfg->vocab_size,D=mcfg->d_model;
    int ctx=cache->seq_len;
    int *tok_buf=(int*)malloc((size_t)ctx*sizeof(int));
    int *asst_mask=(int*)calloc((size_t)ctx,sizeof(int));
    float *lm_logits=(float*)malloc((size_t)V*sizeof(float));
    float *d_lm=(float*)malloc((size_t)V*sizeof(float));
    float *dx_initial=(float*)calloc((size_t)ctx*D,sizeof(float));
    float *d_normed=(float*)malloc((size_t)D*sizeof(float));
    float *tmp_dx=(float*)malloc((size_t)D*sizeof(float));
    ModelOutput out2;
    int n_toks=0,i,t,j;
    float total_loss=0.0f;
    int n_loss_toks=0;

    if (g_cancel_flag) {
        free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    if (!tok_buf||!asst_mask||!lm_logits||!d_lm||!dx_initial||!d_normed||!tmp_dx){
        BLOG_ERROR("train_conv_step: OOM for token/logit buffers");
        free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    /* Encode conversation */
    for (t=0;t<conv->n_turns&&n_toks<ctx-8;t++){
        int turn_start=n_toks;
        int n=bpe_encode_conv_turn(tok,&conv->turns[t],lang_token,
                                    tok_buf+n_toks,ctx-n_toks-2);
        n_toks+=n;
        if (conv->turns[t].role==1){
            int k;
            for (k=turn_start;k<n_toks&&k<ctx;k++) asst_mask[k]=1;
        }
    }

    if (n_toks<2){
        free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    memset(&out2,0,sizeof(out2));
    model_forward(m,tok_buf,n_toks,&out2,cache);

    /* Sum loss over all assistant next-token predictions */
    for (i=0; i<n_toks-1; i++) {
        if (!asst_mask[i+1]) continue;
        {
            int target=tok_buf[i+1];
            float *normed_i=cache->x_normed_all+i*D;
            float *x_i=cache->x_final_all+i*D;
            float tok_loss;

            if (target<0||target>=V) continue;

            op_linear_f32(normed_i,m->lm_head_f32,lm_logits,D,V);
            tok_loss=cross_entropy_loss(lm_logits,target,V,d_lm);
            total_loss+=tok_loss;
            n_loss_toks++;

            /* Accumulate lm_head gradients, scaled by conv_loss_weight */
            for (t=0; t<V; t++) {
                float dz=d_lm[t]*cfg->conv_loss_weight;
                for (j=0; j<D; j++) g->d_lm_head[t*D+j]+=dz*normed_i[j];
            }

            /* Project d_lm -> d_normed */
            memset(d_normed,0,D*sizeof(float));
            for (t=0; t<V; t++) {
                float dz=d_lm[t]*cfg->conv_loss_weight;
                for (j=0; j<D; j++) d_normed[j]+=dz*m->lm_head_f32[t*D+j];
            }

            /* Backpropagate d_normed -> dx_initial + i*D */
            if (mcfg->use_rmsnorm) {
                op_rmsnorm_bwd(d_normed,x_i,m->norm_f_w,dx_initial+i*D,g->d_norm_f_w,D,mcfg->rms_eps);
            } else {
                float mean_v=0.0f,var_v=0.0f,inv_std_v;
                for (j=0; j<D; j++) mean_v+=x_i[j];
                mean_v/=(float)D;
                for (j=0; j<D; j++) { float diff=x_i[j]-mean_v; var_v+=diff*diff; }
                var_v/=(float)D;
                inv_std_v=1.0f/sqrtf(var_v+mcfg->rms_eps);

                memset(tmp_dx,0,D*sizeof(float));
                op_layernorm_bwd(d_normed,x_i,m->norm_f_w,mean_v,inv_std_v,D,tmp_dx,g->d_norm_f_w,g->d_norm_f_b);
                for (j=0; j<D; j++) dx_initial[i*D+j]+=tmp_dx[j];
            }
        }
    }

    if (n_loss_toks>0) total_loss/=(float)n_loss_toks;

    if (g_cancel_flag) {
        free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    backward_pass(m,g,cache,dx_initial,tok_buf,n_toks);

    if (g_cancel_flag) {
        free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);
        return 0.0f;
    }

    grad_clip_norm(g,cfg->grad_clip);
    apply_gradients(m,g,step,lr,cfg);
    grad_zero(g);
    m->adam_t=step;
    m->total_tokens+=n_toks;

    free(tok_buf);free(asst_mask);free(lm_logits);free(d_lm);free(dx_initial);free(d_normed);free(tmp_dx);

    if (total_loss>20.0f)
        BLOG_WARN("train_conv_step: loss spike=%.4f at step %ld",(double)total_loss,step);

    return total_loss;
}

/* ═══════════════════════════════════════════════════════════════
 * §K  FILE COLLECTION  (NEW v13 mixed corpus)
 * ═══════════════════════════════════════════════════════════════ */

static int get_ftype(const char *filename)
{
    const char *dot=strrchr(filename,'.');
    if (!dot) return FTYPE_TEXT;
#ifdef _WIN32
#  define ICMP _stricmp
#else
#  define ICMP strcasecmp
#endif
    if (!ICMP(dot,".conv")) return FTYPE_CONV;
    if (!ICMP(dot,".c")||!ICMP(dot,".h")||!ICMP(dot,".cpp")||
        !ICMP(dot,".py")||!ICMP(dot,".asm")||!ICMP(dot,".pas"))
        return FTYPE_CODE;
    return FTYPE_TEXT;
#undef ICMP
}

int collect_files_mixed(const char *dir,CorpusFile *files,
                          int max_files,int use_conv,int use_text,int use_code)
{
#ifdef _WIN32
    WIN32_FIND_DATAA fd; char pattern[520]; HANDLE h; int n=0;
    _snprintf(pattern,sizeof(pattern)-1,"%s\\*.*",dir);
    pattern[sizeof(pattern)-1]='\0';
    h=FindFirstFileA(pattern,&fd);
    if (h==INVALID_HANDLE_VALUE) return 0;
    do {
        if (fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){
            if (strcmp(fd.cFileName,".")&&strcmp(fd.cFileName,"..")){
                /* BUG 2 FIX: heap-allocate sub[] to prevent stack overflow */
                CorpusFile *sub=(CorpusFile*)malloc(512*sizeof(CorpusFile));
                int ns,k;
                char subdir[520];
                if(sub){
                    _snprintf(subdir,sizeof(subdir)-1,"%s\\%s",dir,fd.cFileName);
                    ns=collect_files_mixed(subdir,sub,512,use_conv,use_text,use_code);
                    for (k=0;k<ns&&n<max_files;k++) files[n++]=sub[k];
                    free(sub);
                }
            }
            continue;
        }
        if (n>=max_files) break;
        {
            int ft;
            _snprintf(files[n].path,511,"%s\\%s",dir,fd.cFileName);
            files[n].path[511]='\0';
            ft=get_ftype(fd.cFileName);
            if (ft==FTYPE_CONV&&!use_conv) continue;
            if (ft==FTYPE_TEXT&&!use_text) continue;
            if (ft==FTYPE_CODE&&!use_code) continue;
            files[n].ftype=ft;
            if (ft == FTYPE_CONV)
                files[n].lang_token = detect_lang_token_from_file(files[n].path);
            else
                files[n].lang_token = detect_lang_token(files[n].path);
            files[n].lang_class=detect_lang_class(files[n].path);
            n++;
        }
    } while (FindNextFileA(h,&fd));
    FindClose(h);
    return n;
#else
    (void)dir;(void)files;(void)max_files;(void)use_conv;(void)use_text;(void)use_code;
    return 0;
#endif
}

static void shuffle_corpus(CorpusFile *arr, int n)
{
    static unsigned long seed=9876543UL; int i,j;
    for (i=n-1;i>0;i--){
        CorpusFile tmp;
        seed=seed*1103515245UL+12345UL;
        j=(int)((seed>>16)%(unsigned long)(i+1));
        tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §L  VALIDATION  (updated to use CorpusFile array)
 * ═══════════════════════════════════════════════════════════════ */

float validate(Model *m,BPETokenizer *tok,
                CorpusFile *files,int n_files,
                const ModelConfig *mcfg)
{
    double nll=0.0; long ntok=0; int fi;
    float *lm_logits=(float*)malloc((size_t)mcfg->vocab_size*sizeof(float));
    int   *tok_buf  =(int*)  malloc((size_t)mcfg->ctx_len*sizeof(int));
    ModelOutput out2;
    int D = mcfg->d_model;
    if (!lm_logits||!tok_buf){free(lm_logits);free(tok_buf);return 9999.0f;}
    memset(&out2,0,sizeof(out2));

    for (fi=0;fi<n_files;fi++){
        int n_ids=0,s;
        int *asst_mask=NULL;
        if (g_cancel_flag) break;
        if (files[fi].ftype==FTYPE_CONV){
            Conversation conv;
            if (bpe_parse_conv_file(files[fi].path,&conv)>0){
                asst_mask=(int*)calloc((size_t)mcfg->ctx_len,sizeof(int));
                if (asst_mask) {
                    int t;
                    for (t=0;t<conv.n_turns&&n_ids<mcfg->ctx_len-8;t++){
                        int turn_start=n_ids;
                        int n=bpe_encode_conv_turn(tok,&conv.turns[t],
                                                    files[fi].lang_token>=0?files[fi].lang_token:TOKEN_LANG_EN,
                                                    tok_buf+n_ids,mcfg->ctx_len-n_ids-2);
                        n_ids+=n;
                        if (conv.turns[t].role==1){
                            int k;
                            for (k=turn_start;k<n_ids&&k<mcfg->ctx_len;k++) asst_mask[k]=1;
                        }
                    }
                    if (n_ids<mcfg->ctx_len) tok_buf[n_ids++]=TOKEN_EOS;
                } else {
                    n_ids=bpe_encode_conv(tok,&conv,
                                           files[fi].lang_token>=0?files[fi].lang_token:TOKEN_LANG_EN,
                                           tok_buf,mcfg->ctx_len);
                }
            }
        } else {
            FILE *fp; long fsz; char *text;
            fp=fopen(files[fi].path,"rb"); if (!fp) continue;
            fseek(fp,0,SEEK_END); fsz=ftell(fp); rewind(fp);
            if (fsz<=0||fsz>64L*1024*1024){fclose(fp);continue;}
            text=(char*)malloc((size_t)fsz+1); if (!text){fclose(fp);continue;}
            fread(text,1,(size_t)fsz,fp); fclose(fp); text[fsz]='\0';
            n_ids=bpe_encode_multilingual(tok,text,files[fi].lang_token,
                                           tok_buf,mcfg->ctx_len);
            free(text);
        }
        if (n_ids<2) {
            if (asst_mask) free(asst_mask);
            continue;
        }
        {
            ForwardCache *v_cache = cache_create(mcfg, n_ids);
            if (v_cache) {
                model_forward(m,tok_buf,n_ids,&out2,v_cache);
                for (s=0;s<n_ids-1;s++){
                    if (asst_mask && !asst_mask[s+1]) continue;
                    {
                        int tgt=tok_buf[s+1];
                        float *normed_s=v_cache->x_normed_all+s*D;
                        float mx,sm=0.0f; int ii;

                        if (tgt<0||tgt>=mcfg->vocab_size) continue;

                        op_linear_f32(normed_s,m->lm_head_f32,lm_logits,D,mcfg->vocab_size);
                        mx=lm_logits[0];
                        for (ii=1;ii<mcfg->vocab_size;ii++) if(lm_logits[ii]>mx)mx=lm_logits[ii];
                        for (ii=0;ii<mcfg->vocab_size;ii++){lm_logits[ii]=expf(lm_logits[ii]-mx);sm+=lm_logits[ii];}
                        if (sm<1e-12f)sm=1e-12f;
                        nll-=log((double)(lm_logits[tgt]/sm)+1e-12);
                        ntok++;
                    }
                }
                cache_free(v_cache, mcfg);
            }
        }
        if (asst_mask) free(asst_mask);
        tb_yield_bg();
        tb_pump_messages();
#ifdef _WIN32
        InterlockedExchange((LONG*)&g_worker_ping_ms, (LONG)GetTickCount());
#endif
    }
    free(lm_logits);free(tok_buf);
    if (ntok==0) return 9999.0f;
    return (float)exp(nll/(double)ntok);
}

/* Forward declare the WM_APP_PROGRESS post helper used in brain.c */
#ifdef _WIN32
extern HWND g_hMain;
extern HWND g_hProgress;
#define REPORT_PROG(pct) \
    if(g_hMain&&IsWindow(g_hMain)) PostMessage(g_hMain,WM_APP+2,(WPARAM)(pct),0)
#else
#define REPORT_PROG(pct) ((void)(pct))
#endif

void train_loop_mixed(Model *m,BPETokenizer *tok,
                       const char *corpus_dir,
                       const TrainConfig *tcfg,
                       TrainState *state,
                       volatile int *cancel_flag)
{
    CorpusFile *all_files;
    int n_all=0,n_train,n_val,epoch,fi;
    long step=state->global_step;
    int patience=0;
    GradBuffer   *grad;
    ForwardCache *cache;
    int *tok_buf;
    const ModelConfig *mcfg=&m->cfg;
    char msg[512];
    int last_pct;
    int prev_pct;
    long total_steps;
    long warmup_steps;
    long actual_total_steps;

    /* Validate config before starting */
    if (!train_config_validate(tcfg)){
        BLOG_WARN("train_loop_mixed: config invalid, aborting");
        return;
    }

    /* Allocate tok_buf for train_ctx=256, not full ctx_len=1024 */
    tok_buf=(int*)malloc(256*sizeof(int));
    if (!tok_buf) return;

    all_files=(CorpusFile*)malloc((size_t)TRAIN_MAX_FILES*sizeof(CorpusFile));
    if (!all_files){free(tok_buf);return;}

    n_all=collect_files_mixed(corpus_dir,all_files,TRAIN_MAX_FILES,
                               tcfg->use_conv_files,tcfg->use_text_files,
                               tcfg->use_code_files);
    if (n_all==0){
        safe_fmt(msg,sizeof(msg),"train_loop_mixed: no files in %s\r\n",corpus_dir);
        app_warn(msg);
        free(tok_buf);free(all_files);return;}

    shuffle_corpus(all_files,n_all);
    n_val  =(int)((float)n_all*tcfg->valid_split);
    if (n_val<1) n_val=1;
    n_train=n_all-n_val;

    /* Scale total_steps and warmup_steps for small datasets */
    total_steps = tcfg->total_steps;
    warmup_steps = tcfg->warmup_steps;
    actual_total_steps = (long)n_train * tcfg->epochs;
    if (actual_total_steps < total_steps) {
        total_steps = actual_total_steps;
        warmup_steps = total_steps / 10;
        if (warmup_steps < 1) warmup_steps = 1;
    }

    /* ROOT CAUSE FIX: train_ctx=256 used for BOTH cache creation AND encoding.
     * Previous bug: cache created for 256 tokens but encoding used ctx_len=1024.
     * Writing 1024 tokens into 256-token cache arrays = heap overflow = ntdll crash. */
    {
        int train_ctx = (mcfg->ctx_len < 256) ? mcfg->ctx_len : 256;
        grad  = grad_alloc(mcfg);
        cache = cache_create(mcfg, train_ctx);
        if (!grad || !cache){
            app_danger("train_loop_mixed: OOM for grad/cache (need ~300MB free)\r\n");
            BLOG_ERROR("train_loop_mixed: OOM for grad/cache (need ~300MB free)");
            grad_free(grad); cache_free(cache, mcfg);
            free(tok_buf); free(all_files); return;
        }
        /* tctx is now stored in cache->seq_len - use that everywhere */
    }

    {
        char start_msg[512];
        safe_fmt(start_msg, sizeof(start_msg),
                 "Training started: files=%d (train=%d, val=%d) | epochs=%d | ctx=%d\r\n"
                 "  LR schedule: [%.5f -> %.5f] | warmup: %ld | total: %ld steps\r\n",
                 n_all, n_train, n_val, tcfg->epochs, cache->seq_len,
                 (double)tcfg->lr_max, (double)tcfg->lr_min,
                 warmup_steps, total_steps);
        app_info(start_msg);
    }
    BLOG_INFO("train_loop_mixed started: files=%d epochs=%d ctx=%d",
              n_all, tcfg->epochs, cache->seq_len);


    for (epoch=0;
         epoch<tcfg->epochs&&(!cancel_flag||!*cancel_flag)&&patience<tcfg->patience;
         epoch++)
    {
        double epoch_loss=0.0; int epoch_steps=0;
        state->epoch=epoch+1;
        shuffle_corpus(all_files,n_train);
        last_pct = -10;
        prev_pct = -1;

        for (fi=0;fi<n_train;fi++){
            FILE *fp; long fsz; char *text;
            int n_ids; float cur_lr,step_loss;
            CorpusFile *cf=&all_files[fi];

            cache->seq_len = (mcfg->ctx_len < 256) ? mcfg->ctx_len : 256;

            if (cancel_flag&&*cancel_flag) break;

            cur_lr=lr_schedule(step,tcfg->lr_max,tcfg->lr_min,
                               warmup_steps,total_steps);

            /* ── Throttled percentage progress logging ── */
            {
                int pct = (fi * 100) / n_train;
                if (pct >= last_pct + 10) {
                    char progress_msg[256];
                    double ml = epoch_steps > 0 ? epoch_loss / epoch_steps : 0.0;
                    safe_fmt(progress_msg, sizeof(progress_msg),
                             "  [Epoch %d/%d] Progress: %d%% | Step: %ld | Loss: %.4f | LR: %.6f\r\n",
                             epoch + 1, tcfg->epochs, pct, step, ml, (double)cur_lr);
                    app_info(progress_msg);
                    last_pct = pct;
                }
                if (pct != prev_pct) {
                    REPORT_PROG(pct);
                    prev_pct = pct;
                }
            }

            /* ── CPU throttle: yield every file (NEW v13) ── */
            tb_yield_bg();
            tb_pump_messages();
#ifdef _WIN32
            InterlockedExchange((LONG*)&g_worker_ping_ms, (LONG)GetTickCount());
#endif

            /* ── .conv file: use train_conv_step ── */
            if (cf->ftype==FTYPE_CONV){
                Conversation conv;
                if (bpe_parse_conv_file(cf->path,&conv)>0){
                    step++;
                    step_loss=train_conv_step(m,grad,cache,&conv,
                                              cf->lang_token>=0?cf->lang_token:TOKEN_LANG_EN,
                                              tok,step,cur_lr,tcfg);
                    epoch_loss+=(double)step_loss; epoch_steps++;
                    state->conv_steps++;
                }
                continue;
            }

            /* ── Code / text file: use train_step ── */
            fp=fopen(cf->path,"rb"); if (!fp) continue;
            fseek(fp,0,SEEK_END); fsz=ftell(fp); rewind(fp);
            if (fsz<=0||fsz>64L*1024*1024){fclose(fp);continue;}
            text=(char*)malloc((size_t)fsz+1); if (!text){fclose(fp);continue;}
            fread(text,1,(size_t)fsz,fp); fclose(fp); text[fsz]='\0';
            n_ids=bpe_encode_multilingual(tok,text,cf->lang_token,
                                           tok_buf,cache->seq_len);
            /* SAFETY:  */
            if (n_ids>cache->seq_len) n_ids=cache->seq_len;
            free(text);
            if (n_ids<2) continue;

            step++;
            step_loss=train_step(m,grad,cache,tok_buf,n_ids,
                                  cf->lang_class,step,cur_lr,tcfg);
            epoch_loss+=(double)step_loss; epoch_steps++;
            if (cf->ftype==FTYPE_CODE) state->code_steps++;
            else                        state->text_steps++;

            if (fi%25==0){
                double ml=epoch_steps>0?epoch_loss/epoch_steps:0.0;
                printf("  [ep%d/%d] fi=%d/%d step=%ld loss=%.4f lr=%.6f\n",
                       epoch+1,tcfg->epochs,fi+1,n_train,step,ml,(double)cur_lr);
            }
        }

        /* ── Validation ── */
        {
            float val_ppl=validate(m,tok,all_files+n_train,n_val,mcfg);
            state->last_loss   =epoch_steps>0?(float)(epoch_loss/epoch_steps):0.0f;
            state->last_val_ppl=val_ppl;

            _snprintf(msg,sizeof(msg)-1,
                      "[EPOCH %d/%d] loss=%.4f val_ppl=%.2f "
                      "tokens=%ld conv=%ld code=%ld text=%ld\n",
                      epoch+1,tcfg->epochs,
                      state->last_loss,val_ppl,m->total_tokens,
                      state->conv_steps,state->code_steps,state->text_steps);
            printf("%s",msg);
            BLOG_INFO("%s",msg);

            if (state->log_fp){
                fprintf(state->log_fp,"%ld,%d,%.6f,%.8f,%.4f,%ld,%ld\n",
                        step,epoch+1,(double)state->last_loss,
                        (double)lr_schedule(step,tcfg->lr_max,tcfg->lr_min,
                                            warmup_steps,total_steps),
                        (double)val_ppl,state->conv_steps,state->code_steps);
                fflush(state->log_fp);
            }

            if (val_ppl<state->best_val_ppl){
                state->best_val_ppl=val_ppl;
                state->best_step=step;
                patience=0;
                if (tcfg->save_best) model_save(m,tcfg->checkpoint_path);
                printf("  New best ppl=%.2f  saved to %s\n",
                       val_ppl,tcfg->checkpoint_path);
            } else {
                patience++;
                printf("  patience=%d/%d\n",patience,tcfg->patience);
                if (patience>=tcfg->patience) {
                    char stop_msg[256];
                    printf("  [Early stopping]\n");
                    safe_fmt(stop_msg, sizeof(stop_msg),
                             "  Early stop at epoch %d/%d "
                             "(best val_ppl=%.2f step=%ld). "
                             "Small val set can pick a weak checkpoint.\r\n",
                             epoch + 1, tcfg->epochs,
                             (double)state->best_val_ppl, state->best_step);
                    app_warn(stop_msg);
                }
            }
        }

        model_requantize(m);
        /* Rebuild vocab if approaching capacity */
        if (tok&&tokenizer_needs_rebuild(tok)&&corpus_dir){
            BLOG_WARN("train_loop_mixed: vocab near cap=%d, rebuilding",tok->vocab_size);
            bpe_rebuild_larger(tok,corpus_dir,2000);
        }
    }

    state->global_step  =step;
    state->patience_count=patience;
    REPORT_PROG(100);
    grad_free(grad);cache_free(cache,mcfg);
    free(tok_buf);free(all_files);
    printf("train_loop_mixed done. best_ppl=%.2f at step=%ld\n",
           state->best_val_ppl,state->best_step);
    BLOG_INFO("train_loop_mixed done. best_ppl=%.2f step=%ld",
              (double)state->best_val_ppl,state->best_step);
}

/* ── v12-compatible train_loop wrapper ─────────────────────── */
void train_loop(Model *m,BPETokenizer *tok,
                 const char *corpus_dir,
                 const TrainConfig *tcfg,
                 TrainState *state,
                 volatile int *cancel_flag)
{
    /* Route to mixed-corpus loop (handles all file types) */
    train_loop_mixed(m,tok,corpus_dir,tcfg,state,cancel_flag);
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 5
 *
 * Files covered:
 *   train.h –
 *     TrainConfig: use_conv_files, use_text_files, conv_loss_weight
 *     TrainState: conv_steps, code_steps, text_steps
 *     CorpusFile struct: path, ftype, lang_token, lang_class
 *     FTYPE_CODE / CONV / TEXT constants
 *     train_config_validate (NEW), train_conv_step (NEW),
 *     collect_files_mixed (NEW), train_loop_mixed (NEW)
 *
 *   train.c –
 *     §A  train_default_config + train_config_validate (NEW)
 *         Rejects lr/clip/batch/epochs/warmup out of range
 *     §B  grad_alloc / grad_free / grad_zero (unchanged)
 *     §C  adamw_update + adamw_array (unchanged)
 *     §D  lr_schedule – all 4 params from TrainConfig (v13)
 *     §E  grad_global_norm / grad_clip_norm (unchanged)
 *     §F  cross_entropy_loss (unchanged)
 *     §G  backward_pass – heap alloc (C89), tb_yield_bg/2 layers,
 *         BLOG_ERROR on OOM
 *     §H  apply_gradients (unchanged)
 *     §I  train_step – BLOG_WARN on loss spike >20
 *     §J  train_conv_step (NEW) – assistant-masked CE loss,
 *         bpe_encode_conv_turn, per-position d_lm accumulation
 *     §K  collect_files_mixed (NEW) – recursive Win32, FTYPE_*,
 *         use_conv/use_text flags; shuffle_corpus Fisher-Yates
 *     §L  validate – uses CorpusFile, skips FTYPE_CONV, tb_yield_bg
 *     §M  train_loop_mixed (NEW) – unified code+conv+text loop,
 *         per-file REPORT_PROG, tb_yield_bg, config validation,
 *         tokenizer auto-rebuild, CSV log with conv/code columns,
 *         train_loop() wrapper for v12 compatibility
 *
 * PART 6 will cover:
 *   converse.h / converse.c (NEW in v13) –
 *     ConvHistory struct (ring buffer of ConvTurn)
 *     cmd_converse() – the real conversational AI engine:
 *       prompt normalise -> lang detect -> history prefix ->
 *       bpe_encode_with_history -> model_forward (inference) ->
 *       top-k + temperature sampling -> decode -> append to history
 *     cmd_converse_reset() – clears history
 *     Facts store: fact_add / fact_search (keyword index) for
 *       grounding answers in known security facts
 *     Security fact seed corpus (50+ hard-coded facts)
 *     NLU extension: all non-command input -> INTENT_CHAT ->
 *       cmd_converse()
 *     Token streaming: per-token PostMessage to UI
 * ─────────────────────────────────────────────────────────────
 */

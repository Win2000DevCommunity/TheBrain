/* ============================================================
 * TheBrain v13.0 - brain_ml.c
 * ML subsystem: MLP, NaiveBayes, IsoForest, OCSVM,
 * feature extraction, ensemble, undo stack,
 * cmd_predict/train/scan/explain/reason/anomaly/
 * cmd_generate/cmd_summarize
 * Pure C89, Windows 2000+
 * ============================================================ */
#include "brain.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════
 * §A  STATIC HELPERS  (sq_dist_b, normalize_feat, rand helpers)
 * ═══════════════════════════════════════════════════════════════ */

static unsigned long g_xstate_ml = 9876543UL;
static double rand_uniform_ml(void){
    g_xstate_ml^=g_xstate_ml<<13;
    g_xstate_ml^=g_xstate_ml>>17;
    g_xstate_ml^=g_xstate_ml<<5;
    return (double)(g_xstate_ml&0x7FFFFFFFUL)/2147483647.0;
}
static double rand_normal_ml(void){
    double u1=rand_uniform_ml()+1e-12,u2=rand_uniform_ml();
    return sqrt(-2.0*log(u1))*cos(6.28318530717959*u2);
}

double sq_dist_b(const double *a,const double *b,int n){
    double s=0.0;int i;for(i=0;i<n;i++){double d=a[i]-b[i];s+=d*d;}return s;}

void compute_normstats(void){
    int i,j;
    memset(&g_norm,0,sizeof(g_norm));
    if(g_nsamples==0)return;
    for(j=0;j<MAX_FEATURES;j++){
        double sum=0.0,sq=0.0,m,v;
        for(i=0;i<g_nsamples;i++) sum+=g_samples[i].features[j];
        m=sum/(double)g_nsamples;
        for(i=0;i<g_nsamples;i++){double d=g_samples[i].features[j]-m;sq+=d*d;}
        v=sq/(double)(g_nsamples>1?g_nsamples-1:1);
        g_norm.mean[j]=m; g_norm.std[j]=sqrt(v)+1e-9;
    }
}

void normalize_feat(double *feat,double *out){
    int j;
    for(j=0;j<MAX_FEATURES;j++)
        out[j]=(feat[j]-g_norm.mean[j])/(g_norm.std[j]+1e-9);
}

/* ═══════════════════════════════════════════════════════════════
 * §B  FEATURE EXTRACTION
 * ═══════════════════════════════════════════════════════════════ */

static double buf_entropy_ml(const unsigned char *buf,size_t n){
    double freq[256],e=0.0;size_t i;
    if(!n)return 0.0;
    memset(freq,0,sizeof(freq));
    for(i=0;i<n;i++)freq[buf[i]]+=1.0;
    for(i=0;i<256;i++){double p=freq[i]/(double)n;if(p>1e-12)e-=p*log(p)/0.693147180559945;}
    return e;
}

static DWORD crc32_ml(const unsigned char *data,size_t len){
    DWORD c=0xFFFFFFFFUL;size_t i;int j;
    for(i=0;i<len;i++){c^=data[i];for(j=0;j<8;j++)c=(c&1)?(c>>1)^0xEDB88320UL:(c>>1);}
    return c^0xFFFFFFFFUL;
}

void extract_features(const char *file, double *feat){
    HANDLE hf;DWORD fsz,done,i,n_secs;
    unsigned char *buf;MY_DOS *dos;MY_NTHDRS *nt;MY_SECHDR *sec;
    DWORD path_crc;DWORD mtime_lo=0,mtime_hi=0;
    int ci;FeatCacheEntry *ce;
    WIN32_FILE_ATTRIBUTE_DATA fa;

    memset(feat,0,MAX_FEATURES*sizeof(double));
    path_crc=crc32_ml((const unsigned char*)file,strlen(file));

    /* Check feature cache */
    EnterCriticalSection(&g_cs_cache);
    for(ci=0;ci<g_n_feat_cache;ci++){
        if(g_feat_cache[ci].path_crc==path_crc){
            memcpy(feat,g_feat_cache[ci].features,MAX_FEATURES*sizeof(double));
            g_feat_cache[ci].lru_stamp=g_lru_clock++;
            LeaveCriticalSection(&g_cs_cache);
            return;
        }
    }
    LeaveCriticalSection(&g_cs_cache);

    /* File size */
    if(GetFileAttributesExA(file,GetFileExInfoStandard,&fa)){
        feat[0]=(double)(((__int64)fa.nFileSizeHigh<<32)|fa.nFileSizeLow);
        mtime_lo=fa.ftLastWriteTime.dwLowDateTime;
        mtime_hi=fa.ftLastWriteTime.dwHighDateTime;
    }

    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){feat[0]=0;return;}
    fsz=GetFileSize(hf,NULL);
    if(!fsz||fsz>PE_MAX_MAP_SIZE){CloseHandle(hf);return;}

    buf=(unsigned char*)malloc(fsz);
    if(!buf){CloseHandle(hf);return;}
    ReadFile(hf,buf,fsz,&done,NULL);CloseHandle(hf);

    feat[0]=(double)done;
    feat[15]=buf_entropy_ml(buf,(size_t)done);

    dos=(MY_DOS*)buf;
    if(done>sizeof(MY_DOS)&&dos->e_magic==MY_IMAGE_DOS_SIGNATURE&&
       (DWORD)dos->e_lfanew+sizeof(MY_NTHDRS)<=done){

        nt=(MY_NTHDRS*)(buf+dos->e_lfanew);
        if(nt->Sig==MY_IMAGE_NT_SIGNATURE){
            n_secs=nt->File.NumSecs;
            feat[1]=(double)n_secs;
            feat[2]=(double)nt->Opt.EP;
            feat[3]=(double)nt->Opt.ImageBase;
            feat[6]=(double)nt->Opt.SzCode;
            feat[7]=(double)nt->Opt.SzInitData;

            sec=(MY_SECHDR*)((unsigned char*)nt+sizeof(DWORD)+sizeof(MY_FILEHDR)+nt->File.SzOpt);
            {double ent_sum=0.0,ent_max=0.0;int n_high=0;
             for(i=0;i<n_secs;i++){
                 DWORD rp=sec[i].RawPtr,rs=sec[i].RawSz;
                 double se;
                 if(!rs||rp+rs>done)continue;
                 se=buf_entropy_ml(buf+rp,rs);
                 ent_sum+=se; if(se>ent_max)ent_max=se;
                 if(se>7.0)n_high++;
             }
             feat[16]=ent_max;
             feat[17]=(n_secs>0)?ent_sum/(double)n_secs:0.0;
             feat[18]=(double)n_high;
            }
        }
    }

    /* Opcode scan */
    {int call_n=0,jmp_n=0,push_n=0,pop_n=0,ret_n=0,int3_n=0;
     DWORD scan_max=done<0x10000?done:0x10000;
     for(i=0;i<scan_max;i++){
         switch(buf[i]){
         case 0xE8:call_n++;break; case 0xE9:case 0xEB:jmp_n++;break;
         case 0x50:case 0x51:case 0x52:case 0x53:
         case 0x54:case 0x55:case 0x56:case 0x57:push_n++;break;
         case 0x58:case 0x59:case 0x5A:case 0x5B:
         case 0x5C:case 0x5D:case 0x5E:case 0x5F:pop_n++;break;
         case 0xC3:case 0xCB:case 0xC2:ret_n++;break;
         case 0xCC:int3_n++;break;
         default:break;
         }
     }
     feat[8]=(double)call_n; feat[9]=(double)jmp_n;
     feat[10]=(double)push_n;feat[11]=(double)pop_n;
     feat[12]=(double)ret_n; feat[13]=(double)int3_n;
    }

    /* Suspicious API count */
    {int sus=0;int j;
     if(done>sizeof(MY_DOS)&&dos->e_magic==MY_IMAGE_DOS_SIGNATURE){
         /* quick string scan for sus apis */
         for(i=0;i<done-8;i++){
             for(j=0;g_sus_apis[j];j++){
                 size_t al=strlen(g_sus_apis[j]);
                 if(i+al<=done&&memcmp(buf+i,g_sus_apis[j],al)==0)sus++;
             }
         }
     }
     feat[5]=(double)sus;
    }

    free(buf);
    g_perf.files_scanned++;

    /* Store in cache */
    EnterCriticalSection(&g_cs_cache);
    if(g_n_feat_cache<FEAT_CACHE_MAX){
        ce=&g_feat_cache[g_n_feat_cache++];
    }else{
        /* evict LRU */
        DWORD oldest=0xFFFFFFFFUL;int oldest_i=0;
        for(ci=0;ci<FEAT_CACHE_MAX;ci++)
            if(g_feat_cache[ci].lru_stamp<oldest){oldest=g_feat_cache[ci].lru_stamp;oldest_i=ci;}
        ce=&g_feat_cache[oldest_i];
    }
    ce->path_crc=path_crc;
    ce->mtime_lo=mtime_lo; ce->mtime_hi=mtime_hi;
    ce->lru_stamp=g_lru_clock++;
    memcpy(ce->features,feat,MAX_FEATURES*sizeof(double));
    LeaveCriticalSection(&g_cs_cache);
}

/* ═══════════════════════════════════════════════════════════════
 * §C  MLP
 * ═══════════════════════════════════════════════════════════════ */

void mlp_init(void){
    int i,j;
    memset(&g_mlp,0,sizeof(g_mlp));
    for(i=0;i<MLP_HIDDEN;i++){
        for(j=0;j<MAX_FEATURES;j++)
            g_mlp.w1[i][j]=(rand_normal_ml()*0.1);
    }
    for(i=0;i<MLP_OUT;i++)
        for(j=0;j<MLP_HIDDEN;j++)
            g_mlp.w2[i][j]=(rand_normal_ml()*0.1);
}

static double relu(double x){return x>0.0?x:0.0;}
static double relu_d(double x){return x>0.0?1.0:0.0;}

static void mlp_forward(double *fn, double *h, double *out){
    int i,j;
    for(i=0;i<MLP_HIDDEN;i++){
        double s=g_mlp.b1[i];
        for(j=0;j<MAX_FEATURES;j++) s+=g_mlp.w1[i][j]*fn[j];
        h[i]=relu(s);
    }
    for(i=0;i<MLP_OUT;i++){
        double s=g_mlp.b2[i];
        for(j=0;j<MLP_HIDDEN;j++) s+=g_mlp.w2[i][j]*h[j];
        out[i]=s;
    }
    /* softmax */
    {double mx=out[0],sm=0.0;
     if(out[1]>mx)mx=out[1];
     out[0]=exp(out[0]-mx);out[1]=exp(out[1]-mx);
     sm=out[0]+out[1]; if(sm<1e-12)sm=1e-12;
     out[0]/=sm;out[1]/=sm;
    }
}

int mlp_predict(double *fn, double *conf){
    double h[MLP_HIDDEN],out[MLP_OUT];
    mlp_forward(fn,h,out);
    if(conf)*conf=out[1]>out[0]?out[1]:out[0];
    return out[1]>out[0]?1:0;
}

double mlp_train_internal(void){
    double total_loss=0.0; int i,j,k;
    double lr=g_cfg.lr,wd=g_cfg.dropout;
    if(g_nsamples==0)return 0.0;
    compute_normstats();
    for(i=0;i<g_nsamples;i++){
        double fn[MAX_FEATURES],h[MLP_HIDDEN],out[MLP_OUT];
        double dout[MLP_OUT],dh[MLP_HIDDEN];
        int lbl=g_samples[i].label>0?1:0;
        float w=(float)(g_samples[i].weight>0?g_samples[i].weight:1.0f);
        normalize_feat(g_samples[i].features,fn);
        mlp_forward(fn,h,out);
        total_loss-=log(out[lbl]+1e-12);
        dout[0]=(lbl==0?out[0]-1.0:out[0])*(double)w;
        dout[1]=(lbl==1?out[1]-1.0:out[1])*(double)w;
        /* bwd w2,b2 */
        for(k=0;k<MLP_OUT;k++){
            for(j=0;j<MLP_HIDDEN;j++){
                double g2=dout[k]*h[j];
                g_mlp.mw2[k][j]=0.9*g_mlp.mw2[k][j]+0.1*g2;
                g_mlp.vw2[k][j]=0.999*g_mlp.vw2[k][j]+0.001*g2*g2;
                g_mlp.w2[k][j]-=lr*g_mlp.mw2[k][j]/(sqrt(g_mlp.vw2[k][j])+1e-8)-lr*wd*g_mlp.w2[k][j];
            }
            g_mlp.mb2[k]=0.9*g_mlp.mb2[k]+0.1*dout[k];
            g_mlp.vb2[k]=0.999*g_mlp.vb2[k]+0.001*dout[k]*dout[k];
            g_mlp.b2[k]-=lr*g_mlp.mb2[k]/(sqrt(g_mlp.vb2[k])+1e-8);
        }
        /* dh */
        for(j=0;j<MLP_HIDDEN;j++){
            double s=0.0;
            for(k=0;k<MLP_OUT;k++) s+=g_mlp.w2[k][j]*dout[k];
            dh[j]=s*relu_d(h[j]);
        }
        /* bwd w1,b1 */
        for(j=0;j<MLP_HIDDEN;j++){
            for(k=0;k<MAX_FEATURES;k++){
                double g1=dh[j]*fn[k];
                g_mlp.mw1[j][k]=0.9*g_mlp.mw1[j][k]+0.1*g1;
                g_mlp.vw1[j][k]=0.999*g_mlp.vw1[j][k]+0.001*g1*g1;
                g_mlp.w1[j][k]-=lr*g_mlp.mw1[j][k]/(sqrt(g_mlp.vw1[j][k])+1e-8)-lr*wd*g_mlp.w1[j][k];
            }
            g_mlp.mb1[j]=0.9*g_mlp.mb1[j]+0.1*dh[j];
            g_mlp.vb1[j]=0.999*g_mlp.vb1[j]+0.001*dh[j]*dh[j];
            g_mlp.b1[j]-=lr*g_mlp.mb1[j]/(sqrt(g_mlp.vb1[j])+1e-8);
        }
    }
    g_mlp.adam_t++;
    return total_loss/(double)(g_nsamples>0?g_nsamples:1);
}

/* ═══════════════════════════════════════════════════════════════
 * §D  NAIVE BAYES
 * ═══════════════════════════════════════════════════════════════ */

void nb_train(void){
    int i,j,c,cnt[2]={0,0};
    memset(&g_nb,0,sizeof(g_nb));
    if(g_nsamples<2)return;
    compute_normstats();
    for(i=0;i<g_nsamples;i++){c=g_samples[i].label>0?1:0;cnt[c]++;}
    g_nb.prior[0]=(cnt[0]+1.0)/(g_nsamples+2.0);
    g_nb.prior[1]=(cnt[1]+1.0)/(g_nsamples+2.0);
    for(j=0;j<MAX_FEATURES;j++){
        double sum[2]={0,0},sq[2]={0,0};
        for(i=0;i<g_nsamples;i++){
            c=g_samples[i].label>0?1:0;
            sum[c]+=g_samples[i].features[j];
            sq[c]+=g_samples[i].features[j]*g_samples[i].features[j];
        }
        for(c=0;c<2;c++){
            double n=(double)(cnt[c]>0?cnt[c]:1);
            double m=sum[c]/n;
            double v=sq[c]/n-m*m;
            g_nb.mean[c][j]=m;
            g_nb.var[c][j]=v>1e-9?v:1e-9;
        }
    }
    g_nb.trained=1;
}

int nb_predict(double *feat, double *conf){
    double lp[2]; int j,c,best;
    if(!g_nb.trained){if(conf)*conf=0.5;return 0;}
    for(c=0;c<2;c++){
        lp[c]=log(g_nb.prior[c]+1e-12);
        for(j=0;j<MAX_FEATURES;j++){
            double m=g_nb.mean[c][j],v=g_nb.var[c][j];
            double d=feat[j]-m;
            lp[c]+=-0.5*log(2.0*3.14159265358979*v)-0.5*d*d/v;
        }
    }
    best=lp[1]>lp[0]?1:0;
    if(conf){
        double mx=lp[0]>lp[1]?lp[0]:lp[1];
        double p0=exp(lp[0]-mx),p1=exp(lp[1]-mx);
        *conf=(best?p1:p0)/(p0+p1+1e-12);
    }
    return best;
}

void lrp_mlp(const double *fn, double *out){
    /* Simplified LRP: absolute input×weight contribution */
    int j,k; double h[MLP_HIDDEN],fwd[MLP_OUT];
    double fn2[MAX_FEATURES];
    memcpy(fn2,fn,MAX_FEATURES*sizeof(double));
    {double *fn3=(double*)fn2;
    mlp_forward(fn3,h,fwd);
    }
    memset(out,0,MAX_FEATURES*sizeof(double));
    for(j=0;j<MLP_HIDDEN;j++){
        double rel=0.0;
        for(k=0;k<MLP_OUT;k++) rel+=g_mlp.w2[k][j]*h[j];
        for(k=0;k<MAX_FEATURES;k++){
            double contrib=fabs(g_mlp.w1[j][k]*fn[k]);
            out[k]+=contrib*(rel>0.0?rel:0.0);
        }
    }
}

void nb_log_odds(const double *feat, double *out){
    int j;
    for(j=0;j<MAX_FEATURES;j++){
        double m0=g_nb.mean[0][j],v0=g_nb.var[0][j];
        double m1=g_nb.mean[1][j],v1=g_nb.var[1][j];
        double d0=feat[j]-m0, d1=feat[j]-m1;
        double log0=-0.5*log(2.0*3.14159265358979*v0)-0.5*d0*d0/v0;
        double log1=-0.5*log(2.0*3.14159265358979*v1)-0.5*d1*d1/v1;
        out[j]=log1-log0;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §E  ISOLATION FOREST (simple)
 * ═══════════════════════════════════════════════════════════════ */

static double iso_path(IsoNode *nodes, int node, double *feat, int depth){
    int i=node; int lim=0;
    while(nodes[i].left>=0&&lim<ISO_NODES){
        if(feat[nodes[i].feature]<nodes[i].split) i=nodes[i].left;
        else i=nodes[i].right;
        lim++;
    }
    return (double)(depth+lim);
}

static void build_tree(IsoTree *tree, int *sample_idx, int n, int depth){
    IsoNode *nd; int fidx,i,left_cnt,right_cnt;
    int *left_buf,*right_buf;
    double fmin,fmax,split;
    if(tree->n_nodes>=ISO_NODES||n<=1||depth>20){
        if(tree->n_nodes<ISO_NODES){
            nd=&tree->nodes[tree->n_nodes++];
            nd->left=nd->right=-1; nd->size=n;
        }
        return;
    }
    nd=&tree->nodes[tree->n_nodes++];
    nd->left=nd->right=-1; nd->size=n;
    fidx=(int)(rand_uniform_ml()*(double)MAX_FEATURES)%MAX_FEATURES;
    fmin=g_samples[sample_idx[0]].features[fidx];
    fmax=fmin;
    for(i=1;i<n;i++){
        double v=g_samples[sample_idx[i]].features[fidx];
        if(v<fmin)fmin=v; if(v>fmax)fmax=v;
    }
    if(fmax<=fmin){nd->left=nd->right=-1;return;}
    split=fmin+rand_uniform_ml()*(fmax-fmin);
    nd->feature=fidx; nd->split=split;
    left_buf=(int*)malloc((size_t)n*sizeof(int));
    right_buf=(int*)malloc((size_t)n*sizeof(int));
    if(!left_buf||!right_buf){free(left_buf);free(right_buf);return;}
    left_cnt=right_cnt=0;
    for(i=0;i<n;i++){
        if(g_samples[sample_idx[i]].features[fidx]<split) left_buf[left_cnt++]=sample_idx[i];
        else right_buf[right_cnt++]=sample_idx[i];
    }
    nd->left=tree->n_nodes;
    build_tree(tree,left_buf,left_cnt,depth+1);
    nd->right=tree->n_nodes;
    build_tree(tree,right_buf,right_cnt,depth+1);
    free(left_buf);free(right_buf);
}

static void iso_train(void){
    int t,i;
    int sub[ISO_SUBSAMPLE];
    memset(&g_isoforest,0,sizeof(g_isoforest));
    g_isoforest.avg_path=0.0;
    for(t=0;t<ISO_TREES;t++){
        int n=g_nsamples<ISO_SUBSAMPLE?g_nsamples:ISO_SUBSAMPLE;
        for(i=0;i<n;i++) sub[i]=(int)(rand_uniform_ml()*(double)g_nsamples)%g_nsamples;
        g_isoforest.trees[t].n_nodes=0;
        g_isoforest.trees[t].root=0;
        build_tree(&g_isoforest.trees[t],sub,n,0);
    }
    g_isoforest.threshold=0.6;
    g_isoforest.trained=1;
}

static double iso_score(double *feat){
    int t; double total=0.0;
    for(t=0;t<ISO_TREES;t++)
        total+=iso_path(g_isoforest.trees[t].nodes,0,feat,0);
    return total/(double)ISO_TREES;
}

/* ═══════════════════════════════════════════════════════════════
 * §F  ENSEMBLE PREDICT + TRAIN_ALL
 * ═══════════════════════════════════════════════════════════════ */

int ensemble_predict(double *feat, double *conf){
    double fn[MAX_FEATURES]; double c_mlp=0.5,c_nb=0.5;
    int p_mlp,p_nb,vote;
    normalize_feat(feat,fn);
    p_mlp=mlp_predict(fn,&c_mlp);
    p_nb =nb_predict(feat,&c_nb);
    vote=(p_mlp+p_nb)>=1?1:0;
    if(conf)*conf=(c_mlp+c_nb)/2.0;
    g_perf.inferences_total++;
    return vote;
}

void train_all(void){
    int ep,e;
    char msg[256];
    if(g_nsamples<2){app_warn("train_all: need >=2 samples\r\n");return;}
    compute_normstats();
    mlp_init();
    for(ep=0;ep<g_cfg.epochs;ep++){
        double loss=mlp_train_internal();
        if(g_cancel_flag) break;
        if(ep%50==0){
            safe_fmt(msg,sizeof(msg),"  [ep %d/%d] loss=%.4f\r\n",ep+1,g_cfg.epochs,loss);
            app_colored(msg,COL_TRAIN);
        }
        tb_yield_bg();
        tb_pump_messages();
        InterlockedExchange((LONG*)&g_worker_ping_ms, (LONG)GetTickCount());
    }
    nb_train();
    iso_train();
    g_perf.train_cycles++;
    (void)e;
    safe_fmt(msg,sizeof(msg),"Train done: %d samples\r\n",g_nsamples);
    app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §G  cmd_anomalytrain / cmd_anomaly
 * ═══════════════════════════════════════════════════════════════ */

void cmd_anomalytrain(void){
    app_info("Anomaly train: building IsoForest...\r\n");
    iso_train();
    {double thresh=0.0;int i,n=0;
     for(i=0;i<g_nsamples;i++){double s=iso_score(g_samples[i].features);thresh+=s;n++;}
     g_isoforest.threshold=(n>0?thresh/(double)n:0.6)*g_cfg.iso_thresh_max;
     app_safe("IsoForest trained.\r\n");}
}

void cmd_anomaly(const char *file){
    double feat[MAX_FEATURES]; double score; char msg[512];
    extract_features(file,feat);
    score=g_isoforest.trained?iso_score(feat):0.5;
    safe_fmt(msg,sizeof(msg),"Anomaly: %s  score=%.4f  %s\r\n",
             file,score,score>g_isoforest.threshold?"[ANOMALY]":"[NORMAL]");
    if(score>g_isoforest.threshold)app_danger(msg); else app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §H  cmd_predict / cmd_train / cmd_scan
 * ═══════════════════════════════════════════════════════════════ */

void cmd_predict(const char *file){
    double feat[MAX_FEATURES]; double conf=0.0; int label; char msg[512];
    extract_features(file,feat);
    label=ensemble_predict(feat,&conf);
    safe_fmt(msg,sizeof(msg),"Predict: %s\r\n  Label: %s  Conf: %.1f%%\r\n",
             file,label?g_class_names[1]:"SAFE",(double)(conf*100.0));
    if(label)app_danger(msg); else app_safe(msg);
    safe_strcpy(g_last_file,file,sizeof(g_last_file));
}

void cmd_train(const char *file, int label){
    double feat[MAX_FEATURES]; char msg[512];
    /* Validate file exists */
    {FILE *_chk=fopen(file,"rb");
     if(!_chk){safe_fmt(msg,sizeof(msg),"train ERROR: cannot open:\r\n  %s\r\n",file);app_danger(msg);return;}
     fclose(_chk);}
    undo_push();
    extract_features(file,feat);
    EnterCriticalSection(&g_cs_samples);
    if(g_nsamples<MAX_SAMPLES){
        memcpy(g_samples[g_nsamples].features,feat,MAX_FEATURES*sizeof(double));
        g_samples[g_nsamples].label=label; g_samples[g_nsamples].weight=1.0f;
        safe_strcpy(g_samples[g_nsamples].filename,file,sizeof(g_samples[0].filename));
        g_nsamples++;
    }
    LeaveCriticalSection(&g_cs_samples);
    safe_fmt(msg,sizeof(msg),"Train: %s label=%s samples=%d\r\n",
             file,label?"DANGEROUS":"SAFE",g_nsamples);
    app_colored(msg,COL_TRAIN);
    train_all();
    safe_strcpy(g_last_file,file,sizeof(g_last_file));
}

void cmd_scan(const char *dir){
    WIN32_FIND_DATAA fd; char pattern[520],full[520],msg[256]; HANDLE h; int n=0;
    safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("scan: cannot open dir\r\n");return;}
    safe_fmt(msg,sizeof(msg),"=== SCAN: %s ===\r\n",dir); app_info(msg);
    do{if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
       safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
       cmd_predict(full); n++;
       tb_yield_bg(); if(g_cancel_flag)break;}
    while(FindNextFileA(h,&fd));
    FindClose(h);
    safe_fmt(msg,sizeof(msg),"Scan done: %d files\r\n",n); app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §I  cmd_explain / cmd_reason
 * ═══════════════════════════════════════════════════════════════ */

void cmd_explain(const char *file){
    double feat[MAX_FEATURES],fn[MAX_FEATURES];
    double lrp[MAX_FEATURES],nlo[MAX_FEATURES];
    double conf=0.0; int label,i; char msg[512];
    extract_features(file,feat);
    normalize_feat(feat,fn);
    label=mlp_predict(fn,&conf);
    lrp_mlp(fn,lrp);
    nb_log_odds(feat,nlo);
    safe_fmt(msg,sizeof(msg),"=== EXPLAIN: %s ===\r\n  Label=%s Conf=%.1f%%\r\n",
             file,label?"DANGEROUS":"SAFE",conf*100.0); app_info(msg);
    app("  Top contributing features:\r\n");
    for(i=0;i<MAX_FEATURES&&i<EXPLAIN_TOP_N;i++){
        safe_fmt(msg,sizeof(msg),"    %-16s val=%-8.2f lrp=%+.4f nlo=%+.4f\r\n",
                 g_feat_names[i],feat[i],lrp[i],nlo[i]);
        if(lrp[i]>0.1||nlo[i]>0.5)app_warn(msg); else app(msg);
    }
    safe_strcpy(g_last_file,file,sizeof(g_last_file));
}

void cmd_reason(const char *file, int top_n){
    double feat[MAX_FEATURES],fn[MAX_FEATURES];
    double lrp[MAX_FEATURES]; double conf=0.0; int i,j; char msg[512];
    /* sort indices */
    int order[MAX_FEATURES]; double sd[MAX_FEATURES]; double td; int ti;
    extract_features(file,feat);
    normalize_feat(feat,fn);
    mlp_predict(fn,&conf);
    lrp_mlp(fn,lrp);
    for(i=0;i<MAX_FEATURES;i++){order[i]=i;sd[i]=lrp[i];}
    for(i=0;i<MAX_FEATURES-1;i++)
        for(j=i+1;j<MAX_FEATURES;j++)
            if(sd[j]>sd[i]){td=sd[i];sd[i]=sd[j];sd[j]=td;ti=order[i];order[i]=order[j];order[j]=ti;}
    safe_fmt(msg,sizeof(msg),"=== REASON: %s ===\r\n",file); app_info(msg);
    if(top_n>MAX_FEATURES)top_n=MAX_FEATURES;
    for(i=0;i<top_n;i++){
        int oi=order[i];
        safe_fmt(msg,sizeof(msg),"  %d. %-16s = %.4f (importance %.4f)\r\n",
                 i+1,g_feat_names[oi],feat[oi],sd[i]);
        if(sd[i]>0.1)app_danger(msg); else app(msg);
    }
    safe_strcpy(g_last_file,file,sizeof(g_last_file));
    (void)td;(void)ti;
}

/* ═══════════════════════════════════════════════════════════════
 * §J  cmd_generate / cmd_summarize
 * ═══════════════════════════════════════════════════════════════ */

void cmd_generate(const GenRequest *req){
    int *toks; int nt,gen_len=0,gvocab; char *out; char msg[256];
    float *logits; ModelOutput fwd;
    if(!g_model||!g_model->trained){app_warn("generate: model not ready\r\n");return;}
    if(!g_tokenizer.trained){app_warn("generate: tokenizer not ready\r\n");return;}
    toks=(int*)malloc((size_t)g_model->cfg.ctx_len*sizeof(int));
    out=(char*)malloc(8192);
    logits=(float*)malloc((size_t)g_model->cfg.vocab_size*sizeof(float));
    if(!toks||!out||!logits){free(toks);free(out);free(logits);app_warn("generate: OOM\r\n");return;}
    /* Clamp sampling to decodable tokenizer vocab (avoid untrained tokens) */
    gvocab=g_model->cfg.vocab_size;
    if(g_tokenizer.vocab_size>TOKEN_SPECIAL_END&&g_tokenizer.vocab_size<gvocab)
        gvocab=g_tokenizer.vocab_size;
    nt=bpe_encode_with_lang(&g_tokenizer,req->prompt,TOKEN_LANG_EN,toks,g_model->cfg.ctx_len-4);
    memset(&fwd,0,sizeof(fwd)); fwd.lm_logits=logits;
    app_colored("=== GENERATE ===\r\n",COL_GENERATE);
    app_colored(req->prompt,COL_GENERATE);
    while(gen_len<req->max_new_tokens&&!g_cancel_flag){
        int next; float mx,sm,r,cs; int i,k;
        float *probs=(float*)malloc((size_t)g_model->cfg.vocab_size*sizeof(float));
        if(!probs)break;
        EnterCriticalSection(&g_cs_model);
        model_forward(g_model,toks,nt,&fwd,NULL);
        LeaveCriticalSection(&g_cs_model);
        mx=logits[0];
        for(i=1;i<gvocab;i++)if(logits[i]>mx)mx=logits[i];
        sm=0.0f;
        for(i=0;i<gvocab;i++){probs[i]=expf((logits[i]-mx)/req->temperature);sm+=probs[i];}
        if(sm<1e-12f)sm=1e-12f;
        for(i=0;i<gvocab;i++)probs[i]/=sm;
        /* top-k */
        {int top_k=req->top_k<gvocab?req->top_k:gvocab;
         for(i=0;i<top_k;i++){int best=i;float tp;for(k=i+1;k<gvocab;k++)if(probs[k]>probs[best])best=k;tp=probs[i];probs[i]=probs[best];probs[best]=tp;}}
        r=(float)(rand_uniform_ml()*(double)sm);cs=0.0f;next=0;
        for(k=0;k<req->top_k&&k<gvocab;k++){cs+=probs[k];if(r<=cs){next=k;break;}}
        free(probs);
        if(next==TOKEN_EOS||next==TOKEN_PAD)break;
        if(nt<g_model->cfg.ctx_len){toks[nt++]=next;}
        else{memmove(toks,toks+1,(size_t)(nt-1)*sizeof(int));toks[nt-1]=next;}
        gen_len++;
        if(gen_len%8==0||gen_len==1){
            bpe_decode(&g_tokenizer,toks+(nt-gen_len>0?nt-gen_len:0),gen_len,out,8192);
            app_colored(out,COL_GENERATE);
        }
        tb_yield();
    }
    app("\r\n");
    safe_fmt(msg,sizeof(msg),"[generated %d tokens]\r\n",gen_len); app_colored(msg,COL_GREY);
    g_perf.tokens_generated+=gen_len;
    free(toks);free(out);free(logits);
}

void cmd_summarize(const char *file){
    FILE *fp; long fsz; char *text; GenRequest req;
    fp=fopen(file,"rb"); if(!fp){app_warn("summarize: cannot open\r\n");return;}
    fseek(fp,0,SEEK_END);fsz=ftell(fp);rewind(fp);
    if(fsz<=0||fsz>65536){fclose(fp);app_warn("summarize: file too large (max 64KB)\r\n");return;}
    text=(char*)malloc((size_t)fsz+64);
    if(!text){fclose(fp);return;}
    fread(text,1,(size_t)fsz,fp);fclose(fp);text[fsz]='\0';
    memset(&req,0,sizeof(req));
    req.max_new_tokens=256; req.temperature=0.7f; req.top_k=40;
    safe_fmt(req.prompt,sizeof(req.prompt),"Summarize: %.*s",(int)(sizeof(req.prompt)-20),text);
    free(text);
    app_info("=== SUMMARIZE ===\r\n");
    cmd_generate(&req);
}

/* ═══════════════════════════════════════════════════════════════
 * §K  UNDO STACK
 * ═══════════════════════════════════════════════════════════════ */

void undo_push(void){
    UndoSnapshot *snap=&g_undo.ring[g_undo.head%UNDO_DEPTH];
    EnterCriticalSection(&g_cs_samples);
    memcpy(snap->samples,g_samples,(size_t)g_nsamples*sizeof(Sample));
    snap->nsamples=g_nsamples;
    LeaveCriticalSection(&g_cs_samples);
    g_undo.head=(g_undo.head+1)%UNDO_DEPTH;
    if(g_undo.count<UNDO_DEPTH)g_undo.count++;
}

int undo_pop(void){
    UndoSnapshot *snap;
    if(g_undo.count==0){app_warn("Nothing to undo.\r\n");return 0;}
    g_undo.head=(g_undo.head-1+UNDO_DEPTH)%UNDO_DEPTH;
    g_undo.count--;
    snap=&g_undo.ring[g_undo.head];
    EnterCriticalSection(&g_cs_samples);
    memcpy(g_samples,snap->samples,(size_t)snap->nsamples*sizeof(Sample));
    g_nsamples=snap->nsamples;
    LeaveCriticalSection(&g_cs_samples);
    app_safe("Undo: previous state restored.\r\n");
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * §L  EMBEDSCAN
 * ═══════════════════════════════════════════════════════════════ */

void cmd_embedscan(const char *dir){
    WIN32_FIND_DATAA fd; char pattern[520],full[520],msg[256]; HANDLE h; int n=0;
    float *logits; ModelOutput out2; int *toks;
    if(!g_model||!g_model->trained){app_warn("embedscan: model not ready\r\n");return;}
    logits=(float*)malloc((size_t)g_model->cfg.d_model*sizeof(float));
    toks=(int*)malloc((size_t)g_model->cfg.ctx_len*sizeof(int));
    if(!logits||!toks){free(logits);free(toks);return;}
    safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){free(logits);free(toks);return;}
    safe_fmt(msg,sizeof(msg),"=== EMBEDSCAN: %s ===\r\n",dir);app_info(msg);
    do{FILE *fp; long fsz; char *text; int nt;
       if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
       if(g_n_embeds>=g_dyn_max_embeds)break;
       safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
       fp=fopen(full,"rb"); if(!fp)continue;
       fseek(fp,0,SEEK_END);fsz=ftell(fp);rewind(fp);
       if(fsz<=0||fsz>65536){fclose(fp);continue;}
       text=(char*)malloc((size_t)fsz+1); if(!text){fclose(fp);continue;}
       fread(text,1,(size_t)fsz,fp);fclose(fp);text[fsz]='\0';
       nt=bpe_encode_multilingual(&g_tokenizer,text,detect_lang_token(full),toks,g_model->cfg.ctx_len);
       free(text);
       memset(&out2,0,sizeof(out2)); out2.hidden=logits;
       EnterCriticalSection(&g_cs_model);
       model_forward(g_model,toks,nt,&out2,NULL);
       LeaveCriticalSection(&g_cs_model);
       EnterCriticalSection(&g_cs_embeds);
       {FileEmbed *e=&g_embeds[g_n_embeds++];
        memcpy(e->vec,logits,(size_t)g_model->cfg.d_model*sizeof(float));
        safe_strcpy(e->filename,full,sizeof(e->filename));
        e->label=0; e->crc=0;}
       LeaveCriticalSection(&g_cs_embeds);
       n++; tb_yield_bg();}
    while(FindNextFileA(h,&fd));
    FindClose(h);
    free(logits);free(toks);
    safe_fmt(msg,sizeof(msg),"Embedscan done: %d files\r\n",n);app_safe(msg);
}

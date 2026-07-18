#include "ops.h"
#include "ops_mt.h"
#include "sysinfo.h"
#include "tensor.h"
#include "graph.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

/* SSE intrinsics are available on MSVC/GCC x86; only *used* at runtime
 * when CPUID confirms the CPU supports SSE, so genuine pre-SSE Win2000
 * boxes (Pentium II / early) safely fall back to the scalar path. */
#if (defined(_MSC_VER) && defined(_M_IX86)) || \
    (defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)))
#  include <xmmintrin.h>
#  define TB_HAVE_SSE_INTRIN 1
#endif

/* ── §A0  RUNTIME CPU CAPABILITY DETECTION ──────────────────────
 * One-time CPUID probe.  Drives the SIMD dispatch in dot_f32_asm and
 * is reported in the startup banner so the user sees the active path. */
static int g_cpu_inited = 0;
static int g_cpu_sse    = 0;
static int g_cpu_sse2   = 0;

static void cpu_detect(void)
{
    unsigned int feat_edx = 0;
#if defined(_MSC_VER) && defined(_M_IX86)
    unsigned int _edx = 0;
    __asm {
        mov eax, 1
        cpuid
        mov _edx, edx
    }
    feat_edx = _edx;
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    unsigned int a, b, c, d;
    __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1):);
    feat_edx = d;
#else
    feat_edx = 0;
#endif
    g_cpu_sse  = (feat_edx & (1u << 25)) ? 1 : 0;
    g_cpu_sse2 = (feat_edx & (1u << 26)) ? 1 : 0;
    g_cpu_inited = 1;
}

void tb_cpu_init(void)
{
    if (!g_cpu_inited) cpu_detect();
    tb_mt_init();
}
int  tb_cpu_has_sse(void)    { if (!g_cpu_inited) cpu_detect(); return g_cpu_sse; }
int  tb_cpu_has_sse2(void)   { if (!g_cpu_inited) cpu_detect(); return g_cpu_sse2; }

void tb_cpu_features_str(char *buf, int n)
{
    int nw;
    if (n <= 0) return;
    if (!g_cpu_inited) cpu_detect();
    nw = tb_mt_nworkers();
    if (nw > 1) {
        _snprintf(buf, (size_t)n - 1, "%s x%d",
                  g_cpu_sse2 ? "SSE2" : (g_cpu_sse ? "SSE" : "scalar"),
                  nw);
    } else {
        _snprintf(buf, (size_t)n - 1, "%s",
                  g_cpu_sse2 ? "SSE2" : (g_cpu_sse ? "SSE" : "scalar (x87)"));
    }
    buf[n - 1] = '\0';
}

#ifdef TB_HAVE_SSE_INTRIN
/* SSE (single-precision) 4-wide dot product. Unaligned loads tolerate
 * the model's non-16B-aligned weight rows. */
static float dot_sse(const float *a, const float *b, int K)
{
    __m128 acc;
    float  tmp[4], s;
    int    k, k4, i;
    acc = _mm_setzero_ps();
    k4  = K & ~3;
    for (k = 0; k < k4; k += 4)
        acc = _mm_add_ps(acc,
                  _mm_mul_ps(_mm_loadu_ps(a + k), _mm_loadu_ps(b + k)));
    _mm_storeu_ps(tmp, acc);
    s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (i = k4; i < K; i++) s += a[i] * b[i];
    return s;
}
#endif

#if defined(_MSC_VER) && defined(_M_IX86)
/* MSVC x87 inline ASM dot product (pre-SSE Pentium / Win2000 era). */
static float dot_x87_msvc(const float *a, const float *b, int K)
{
    float  result = 0.0f;
    int    k4 = K / 4;
    int    kr = K - k4 * 4;
    int    i;

    if (k4 > 0) {
        __asm {
            mov esi, a
            mov edi, b
            mov ecx, k4
            fldz
        tb_dot4:
            fld  dword ptr [esi]
            fmul dword ptr [edi]
            fld  dword ptr [esi+4]
            fmul dword ptr [edi+4]
            faddp st(1), st(0)
            fld  dword ptr [esi+8]
            fmul dword ptr [edi+8]
            faddp st(1), st(0)
            fld  dword ptr [esi+12]
            fmul dword ptr [edi+12]
            faddp st(1), st(0)
            add  esi, 16
            add  edi, 16
            loop tb_dot4
            fstp dword ptr [result]
        }
        a += k4 * 4;
        b += k4 * 4;
    }
    for (i = 0; i < kr; i++) result += a[i] * b[i];
    return result;
}
#endif

/* Hot inner product. Dispatches to SSE / x87 ASM / scalar. */
float op_dot_f32(const float *a, const float *b, int K)
{
    if (!g_cpu_inited) cpu_detect();
#ifdef TB_HAVE_SSE_INTRIN
    if (g_cpu_sse) return dot_sse(a, b, K);
#endif
#if defined(_MSC_VER) && defined(_M_IX86)
    if (!g_cpu_sse) return dot_x87_msvc(a, b, K);
#endif
    {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int   k4 = K & ~3, i;
        for (i = 0; i < k4; i += 4) {
            s0 += a[i]   * b[i];
            s1 += a[i+1] * b[i+1];
            s2 += a[i+2] * b[i+2];
            s3 += a[i+3] * b[i+3];
        }
        for (i = k4; i < K; i++) s0 += a[i] * b[i];
        return s0 + s1 + s2 + s3;
    }
}

void op_matmul_f32(const float *A, const float *B, float *C,
                    int M, int K, int N)
{
    int TILE_K=64,TILE_N=64;
    int i,j,k,kk,jj;
    memset(C,0,(size_t)M*N*sizeof(float));
    for (kk=0;kk<K;kk+=TILE_K) {
        int k_end=kk+TILE_K<K?kk+TILE_K:K;
        for (jj=0;jj<N;jj+=TILE_N) {
            int j_end=jj+TILE_N<N?jj+TILE_N:N;
            for (i=0;i<M;i++) {
                const float *Ai=A+i*K; float *Ci=C+i*N;
                for (k=kk;k<k_end;k++) {
                    float aik=Ai[k]; const float *Bk=B+k*N;
                    for (j=jj;j<j_end;j++) Ci[j]+=aik*Bk[j];
                }
            }
        }
    }
}

void op_matmul_t_f32(const float *A, const float *B, float *C,
                      int M, int K, int N)
{
    int i,j;
#ifdef _WIN32
    if (tb_mt_try_matmul_t_f32(A, B, C, M, K, N)) return;
#endif
    for (i=0;i<M;i++) {
        const float *Ai=A+i*K; float *Ci=C+i*N;
        for (j=0;j<N;j++) Ci[j]=op_dot_f32(Ai,B+j*K,K);
    }
}

void op_matmul_dA(const float *dC,const float *B,float *dA,int M,int K,int N)
{
    int i,j,k;
    for (i=0;i<M;i++)
        for (j=0;j<N;j++) {
            float dc=dC[i*N+j];
            for (k=0;k<K;k++) dA[i*K+k]+=dc*B[k*N+j];
        }
}

void op_matmul_dB(const float *dC,const float *A,float *dB,int M,int K,int N)
{
    int i,j,k;
    for (k=0;k<K;k++)
        for (i=0;i<M;i++) {
            float aik=A[i*K+k];
            for (j=0;j<N;j++) dB[k*N+j]+=aik*dC[i*N+j];
        }
}

void op_matmul_t_dA(const float *dC,const float *B,float *dA,int M,int K,int N)
{
    int i,j,k;
    for (i=0;i<M;i++)
        for (j=0;j<N;j++) {
            float dc=dC[i*N+j];
            for (k=0;k<K;k++) dA[i*K+k]+=dc*B[j*K+k];
        }
}

void op_matmul_t_dB(const float *dC,const float *A,float *dB,int M,int K,int N)
{
    int i,j,k;
    for (j=0;j<N;j++)
        for (i=0;i<M;i++) {
            float dc=dC[i*N+j];
            for (k=0;k<K;k++) dB[j*K+k]+=dc*A[i*K+k];
        }
}

/* ── §B  RMSNORM ────────────────────────────────────────────── */

void op_rmsnorm_f32(const float *x,const float *w,float *y,int n,float eps)
{
    float ms=0.0f,rms,inv_rms; int i;
    for (i=0;i<n;i++) ms+=x[i]*x[i];
    ms/=(float)n; rms=sqrtf(ms+eps); inv_rms=1.0f/rms;
    for (i=0;i<n;i++) y[i]=x[i]*inv_rms*w[i];
}

void op_rmsnorm_bwd(const float *dy,const float *x,
                     const float *w,float *dx,float *dw,
                     int n,float eps)
{
    float ms=0.0f,rms,inv_rms,inv_rms3,dot=0.0f; int i;
    for (i=0;i<n;i++) ms+=x[i]*x[i];
    ms/=(float)n; rms=sqrtf(ms+eps);
    inv_rms=1.0f/rms; inv_rms3=inv_rms*inv_rms*inv_rms;
    for (i=0;i<n;i++) dot+=dy[i]*w[i]*x[i];
    for (i=0;i<n;i++) {
        dw[i]+=dy[i]*x[i]*inv_rms;
        dx[i]+=dy[i]*w[i]*inv_rms - x[i]*inv_rms3/(float)n*dot;
    }
}

/* ── §C  LAYERNORM ──────────────────────────────────────────── */

void op_layernorm_f32(const float *x,const float *w,const float *b,
                       float *y,int n,float eps,
                       float *out_mean,float *out_inv_std)
{
    float mean=0.0f,var=0.0f,inv_std; int i;
    for (i=0;i<n;i++) mean+=x[i];
    mean/=(float)n;
    for (i=0;i<n;i++){float d=x[i]-mean;var+=d*d;}
    var/=(float)n; inv_std=1.0f/sqrtf(var+eps);
    if (out_mean)    *out_mean=mean;
    if (out_inv_std) *out_inv_std=inv_std;
    for (i=0;i<n;i++) y[i]=(x[i]-mean)*inv_std*w[i]+b[i];
}

void op_layernorm_bwd(const float *dy,const float *x,
                       const float *w,
                       float mean,float inv_std,int n,
                       float *dx,float *dw,float *db)
{
    float sum_dy=0.0f,sum_dy_xh=0.0f,inv_n=1.0f/(float)n; int i;
    for (i=0;i<n;i++) {
        float xhat=(x[i]-mean)*inv_std;
        dw[i]+=dy[i]*xhat; db[i]+=dy[i];
        sum_dy+=dy[i]*w[i]; sum_dy_xh+=dy[i]*w[i]*xhat;
    }
    for (i=0;i<n;i++) {
        float xhat=(x[i]-mean)*inv_std;
        dx[i]=inv_std*(dy[i]*w[i]-inv_n*sum_dy-inv_n*xhat*sum_dy_xh);
    }
}

/* ── §D  SOFTMAX ────────────────────────────────────────────── */

void op_softmax_f32(float *x,int n)
{
    float mx=x[0],sum=0.0f; int i;
    for (i=1;i<n;i++) if (x[i]>mx) mx=x[i];
    for (i=0;i<n;i++){x[i]=expf(x[i]-mx);sum+=x[i];}
    if (sum<1e-12f) sum=1e-12f;
    for (i=0;i<n;i++) x[i]/=sum;
}

void op_softmax_bwd(const float *p,const float *dy,float *dx,int n)
{
    float dot=0.0f; int i;
    for (i=0;i<n;i++) dot+=p[i]*dy[i];
    for (i=0;i<n;i++) dx[i]=p[i]*(dy[i]-dot);
}

/* ── §E  GELU ───────────────────────────────────────────────── */

#define GELU_K 0.7978845608f
#define GELU_A 0.044715f

void op_gelu_f32(const float *x,float *y,int n)
{
    int i;
    for (i=0;i<n;i++) {
        float xi=x[i],t=GELU_K*(xi+GELU_A*xi*xi*xi);
        y[i]=0.5f*xi*(1.0f+tanhf(t));
    }
}

void op_gelu_bwd(const float *x,const float *dy,float *dx,int n)
{
    int i;
    for (i=0;i<n;i++) {
        float xi=x[i],kx=GELU_K*(xi+GELU_A*xi*xi*xi);
        float tanh_kx=tanhf(kx),sech2=1.0f-tanh_kx*tanh_kx;
        float dgelu=0.5f*(1.0f+tanh_kx)
                  +0.5f*xi*sech2*GELU_K*(1.0f+3.0f*GELU_A*xi*xi);
        dx[i]=dy[i]*dgelu;
    }
}

/* ── §F  SWIGLU ─────────────────────────────────────────────── */

static float silu_f(float x){return x/(1.0f+expf(-x));}
static float silu_bwd_f(float x){float s=1.0f/(1.0f+expf(-x));return s*(1.0f+x*(1.0f-s));}

void op_swiglu_f32(const float *gate,const float *up,float *out,int n)
{
    int i;
    for (i=0;i<n;i++) out[i]=silu_f(gate[i])*up[i];
}

void op_swiglu_bwd(const float *gate,const float *up,
                    const float *dout,
                    float *d_gate,float *d_up,int n)
{
    int i;
    for (i=0;i<n;i++) {
        float sg=silu_f(gate[i]),dsg=silu_bwd_f(gate[i]);
        d_gate[i]+=dout[i]*dsg*up[i];
        d_up[i]  +=dout[i]*sg;
    }
}

/* ── §G  ROPE ───────────────────────────────────────────────── */

static void rope_single_head(float *x,int pos,int d_head,int forward)
{
    int i;
    for (i=0;i<d_head;i+=2) {
        float theta=(float)pos/powf(10000.0f,(float)i/(float)d_head);
        float c=cosf(theta),s=sinf(theta);
        float x0=x[i],x1=x[i+1];
        if (forward) { x[i]=x0*c-x1*s; x[i+1]=x1*c+x0*s; }
        else         { x[i]=x0*c+x1*s; x[i+1]=x1*c-x0*s; }
    }
}

void op_rope_f32(float *x,int seq_len,int n_heads,int d_head,int forward)
{
    int s,h;
    for (s=0;s<seq_len;s++) {
        float *pos_vec=x+s*(n_heads*d_head);
        for (h=0;h<n_heads;h++)
            rope_single_head(pos_vec+h*d_head,s,d_head,forward);
    }
}

/* ── §H  ATTENTION (with CPU throttle every 64 heads×rows) ─── */

void op_attention_f32(const float *Q,const float *K,const float *V,
                       float *out,float *attn_w_out,
                       int seq,int n_heads,int d_head,
                       int causal,float scale)
{
    float *attn_row;
    int    h,s,j,i;
    int    throttle_ctr=0;

#ifdef _WIN32
    if (tb_mt_try_attention_f32(Q, K, V, out, attn_w_out,
                                 seq, n_heads, d_head, causal, scale))
        return;
#endif

    attn_row=(float*)malloc((size_t)seq*seq*sizeof(float));
    if (!attn_row) return;
    memset(out,0,(size_t)seq*n_heads*d_head*sizeof(float));

    for (h=0;h<n_heads;h++) {
        int hoff=h*d_head;

        /* Throttle every 8 heads to keep UI responsive */
        if (++throttle_ctr % 8 == 0) tb_yield();

        for (s=0;s<seq;s++) {
            const float *qs=Q+s*n_heads*d_head+hoff;
            float       *ar=attn_row+s*seq;
            for (j=0;j<seq;j++) {
                if (causal&&j>s){ar[j]=-1e9f;continue;}
                ar[j]=op_dot_f32(qs,K+j*n_heads*d_head+hoff,d_head)*scale;
            }
            op_softmax_f32(ar,seq);
        }

        if (attn_w_out)
            memcpy(attn_w_out+h*seq*seq,attn_row,(size_t)seq*seq*sizeof(float));

        for (s=0;s<seq;s++) {
            float *ar=attn_row+s*seq;
            float *outs=out+s*n_heads*d_head+hoff;
            for (j=0;j<=(causal?s:seq-1);j++) {
                float a=ar[j];
                const float *vs=V+j*n_heads*d_head+hoff;
                for (i=0;i<d_head;i++) outs[i]+=a*vs[i];
            }
        }
    }
    free(attn_row);
}

void op_attention_bwd(const float *Q,const float *K,const float *V,
                       const float *attn_w,const float *dout,
                       float *dQ,float *dK,float *dV,
                       int seq,int n_heads,int d_head,
                       int causal,float scale)
{
    float *dattn=(float*)malloc((size_t)seq*seq*sizeof(float));
    int    h,s,j,i;
    if (!dattn) return;

    for (h=0;h<n_heads;h++) {
        int         hoff=h*d_head;
        const float *aw=attn_w+h*seq*seq;

        for (s=0;s<seq;s++) {
            const float *dout_s=dout+s*n_heads*d_head+hoff;
            const float *aw_row=aw+s*seq;
            float       *dat_row=dattn+s*seq;
            int          j_end=causal?s+1:seq;

            for (j=0;j<j_end;j++) {
                float *dVjs=dV+j*n_heads*d_head+hoff;
                for (i=0;i<d_head;i++) dVjs[i]+=aw_row[j]*dout_s[i];
            }
            for (j=0;j<j_end;j++) {
                const float *vs=V+j*n_heads*d_head+hoff;
                float da=0.0f;
                for (i=0;i<d_head;i++) da+=dout_s[i]*vs[i];
                dat_row[j]=da;
            }
            for (j=j_end;j<seq;j++) dat_row[j]=0.0f;
            op_softmax_bwd(aw_row,dat_row,dat_row,seq);

            for (j=0;j<j_end;j++) {
                float ds=dat_row[j]*scale;
                float *dQs=dQ+s*n_heads*d_head+hoff;
                float *dKjs=dK+j*n_heads*d_head+hoff;
                const float *ks=K+j*n_heads*d_head+hoff;
                const float *qs=Q+s*n_heads*d_head+hoff;
                for (i=0;i<d_head;i++){dQs[i]+=ds*ks[i]; dKjs[i]+=ds*qs[i];}
            }
        }
        /* Throttle in backward too */
        if ((h&7)==7) tb_yield();
    }
    free(dattn);
}

/* ── §I  EMBED ──────────────────────────────────────────────── */

void op_embed_f32(const int *tokens,const float *W,float *out,
                   int seq,int d_model,int vocab)
{
    int s;
    (void)vocab;
    for (s=0;s<seq;s++) {
        int tok=tokens[s];
        if (tok<0) tok=0;
        if (tok>=vocab) tok=0;
        memcpy(out+s*d_model,W+tok*d_model,(size_t)d_model*sizeof(float));
    }
}

void op_embed_bwd(const int *tokens,const float *dout,
                   float *dW,int seq,int d_model)
{
    int s,i;
    for (s=0;s<seq;s++) {
        int tok=tokens[s];
        if (tok<0) continue;
        for (i=0;i<d_model;i++) dW[tok*d_model+i]+=dout[s*d_model+i];
    }
}

/* ── §J  LINEAR (LM head) ───────────────────────────────────── */

void op_linear_f32(const float *x,const float *W,float *out,
                    int d_model,int vocab)
{
    int v;
#ifdef _WIN32
    if (tb_mt_try_linear_f32(x, W, out, d_model, vocab)) return;
#endif
    for (v=0;v<vocab;v++) out[v]=op_dot_f32(W+v*d_model,x,d_model);
}

void op_linear_bwd(const float *x,const float *W,const float *dout,
                    float *dx,float *dW,int d_model,int vocab)
{
    int v,d;
    for (v=0;v<vocab;v++) {
        float dov=dout[v];
        const float *Wv=W+v*d_model;
        float *dWv=dW+v*d_model;
        for (d=0;d<d_model;d++){dx[d]+=Wv[d]*dov; dWv[d]+=x[d]*dov;}
    }
}

/* ── §K  QUANTIZE / DEQUANTIZE / QMATMUL ────────────────────── */

void op_quantize_i8(const float *src,signed char *dst,
                     float *scales,int rows,int cols)
{
    int i,j;
    for (i=0;i<rows;i++) {
        float mx=0.0f;
        const float *row=src+i*cols; signed char *dr=dst+i*cols;
        for (j=0;j<cols;j++) if (fabsf(row[j])>mx) mx=fabsf(row[j]);
        scales[i]=mx>0.0f?mx/127.0f:1.0f;
        for (j=0;j<cols;j++) dr[j]=(signed char)(row[j]/scales[i]);
    }
}

void op_dequantize_i8(const signed char *src,const float *scales,
                       float *dst,int rows,int cols)
{
    int i,j;
    for (i=0;i<rows;i++) {
        float s=scales[i];
        const signed char *sr=src+i*cols; float *dr=dst+i*cols;
        for (j=0;j<cols;j++) dr[j]=(float)sr[j]*s;
    }
}

void op_qmatmul(const signed char *W,const float *scales,
                 const float *x,float *y,int rows,int cols)
{
#if 0
    int i,j;
    for (i=0;i<rows;i++) {
        const signed char *Wr=W+i*cols; float acc=0.0f;
        int c4=cols/4,cr=cols%4;
        __asm {
            mov esi, Wr
            mov edi, x
            mov ecx, c4
            fldz
        qloop:
            movsx eax, byte ptr [esi]
            push eax
            fild dword ptr [esp]
            fmul dword ptr [edi]
            faddp st(1), st(0)
            add esp, 4

            movsx eax, byte ptr [esi+1]
            push eax
            fild dword ptr [esp]
            fmul dword ptr [edi+4]
            faddp st(1), st(0)
            add esp, 4

            movsx eax, byte ptr [esi+2]
            push eax
            fild dword ptr [esp]
            fmul dword ptr [edi+8]
            faddp st(1), st(0)
            add esp, 4

            movsx eax, byte ptr [esi+3]
            push eax
            fild dword ptr [esp]
            fmul dword ptr [edi+12]
            faddp st(1), st(0)
            add esp, 4

            add esi, 4
            add edi, 16
            loop qloop
            fstp dword ptr [acc]
        }
        for (j=c4*4;j<cols;j++) acc+=(float)Wr[j]*x[j];
        y[i]=acc*scales[i];
    }
#else
    int i,j;
    for (i=0;i<rows;i++) {
        float acc=0.0f;
        const signed char *Wr=W+i*cols;
        for (j=0;j<cols;j++) acc+=(float)Wr[j]*x[j];
        y[i]=acc*scales[i];
    }
#endif
}

/* ── §L  OPERATOR GRAPH WIRING ─────────────────────────────── */

static int fwd_matmul(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    int M=A->shape[A->rank-2],K=A->shape[A->rank-1],N=B->shape[B->rank-1];
    op_matmul_f32(tb_f32(A),tb_f32(B),tb_f32(C),M,K,N); return 1;}

static int bwd_matmul(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    int M=A->shape[A->rank-2],K=A->shape[A->rank-1],N=B->shape[B->rank-1];
    if (!C->grad) return 1;
    if (A->grad) op_matmul_dA(C->grad,tb_f32(B),A->grad,M,K,N);
    if (B->grad) op_matmul_dB(C->grad,tb_f32(A),B->grad,M,K,N); return 1;}

static int fwd_matmul_t(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    int M=A->shape[A->rank-2],K=A->shape[A->rank-1],N=B->shape[B->rank-2];
    op_matmul_t_f32(tb_f32(A),tb_f32(B),tb_f32(C),M,K,N); return 1;}

static int bwd_matmul_t(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    int M=A->shape[A->rank-2],K=A->shape[A->rank-1],N=B->shape[B->rank-2];
    if (!C->grad) return 1;
    if (A->grad) op_matmul_t_dA(C->grad,tb_f32(B),A->grad,M,K,N);
    if (B->grad) op_matmul_t_dB(C->grad,tb_f32(A),B->grad,M,K,N); return 1;}

static int fwd_add(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    float *a=tb_f32(A),*b=tb_f32(B),*c=tb_f32(C); int i;
    for (i=0;i<C->n_elem;i++) c[i]=a[i]+b[i]; return 1;}

static int bwd_add(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*B=graph_tensor(g,n->inputs[1]),*C=graph_tensor(g,n->outputs[0]);
    int i;
    if (!C->grad) return 1;
    if (A->grad) for(i=0;i<A->n_elem;i++) A->grad[i]+=C->grad[i];
    if (B->grad) for(i=0;i<B->n_elem;i++) B->grad[i]+=C->grad[i]; return 1;}

static int fwd_scale(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*C=graph_tensor(g,n->outputs[0]);
    float s=graph_node_get_attr_f(n,"scale",1.0f),*a=tb_f32(A),*c=tb_f32(C); int i;
    for (i=0;i<C->n_elem;i++) c[i]=a[i]*s; return 1;}

static int bwd_scale(Graph *g,GraphNode *n){
    Tensor *A=graph_tensor(g,n->inputs[0]),*C=graph_tensor(g,n->outputs[0]);
    float s=graph_node_get_attr_f(n,"scale",1.0f); int i;
    if (!C->grad||!A->grad) return 1;
    for (i=0;i<A->n_elem;i++) A->grad[i]+=C->grad[i]*s; return 1;}

static int fwd_rmsnorm(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    float eps=graph_node_get_attr_f(n,"eps",1e-5f);
    int seq=X->rank>=2?X->shape[0]:1,d=X->shape[X->rank-1],s;
    for (s=0;s<seq;s++) op_rmsnorm_f32(tb_f32(X)+s*d,tb_f32(W),tb_f32(Y)+s*d,d,eps);
    return 1;}

static int bwd_rmsnorm(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    float eps=graph_node_get_attr_f(n,"eps",1e-5f);
    int seq=X->rank>=2?X->shape[0]:1,d=X->shape[X->rank-1],s;
    if (!Y->grad) return 1;
    for (s=0;s<seq;s++)
        op_rmsnorm_bwd(Y->grad+s*d,tb_f32(X)+s*d,tb_f32(W),
                       X->grad?X->grad+s*d:NULL,W->grad,d,eps);
    return 1;}

static int fwd_gelu(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    op_gelu_f32(tb_f32(X),tb_f32(Y),X->n_elem); return 1;}

static int bwd_gelu(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    if (!Y->grad||!X->grad) return 1;
    op_gelu_bwd(tb_f32(X),Y->grad,X->grad,X->n_elem); return 1;}

static int fwd_swiglu(Graph *g,GraphNode *n){
    Tensor *gate=graph_tensor(g,n->inputs[0]),*up=graph_tensor(g,n->inputs[1]),*out=graph_tensor(g,n->outputs[0]);
    op_swiglu_f32(tb_f32(gate),tb_f32(up),tb_f32(out),gate->n_elem); return 1;}

static int bwd_swiglu(Graph *g,GraphNode *n){
    Tensor *gate=graph_tensor(g,n->inputs[0]),*up=graph_tensor(g,n->inputs[1]),*out=graph_tensor(g,n->outputs[0]);
    if (!out->grad) return 1;
    op_swiglu_bwd(tb_f32(gate),tb_f32(up),out->grad,gate->grad,up->grad,gate->n_elem); return 1;}

static int fwd_softmax(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    int seq=X->rank>=2?X->shape[0]:1,d=X->shape[X->rank-1],s;
    memcpy(tb_f32(Y),tb_f32(X),(size_t)X->n_elem*sizeof(float));
    for (s=0;s<seq;s++) op_softmax_f32(tb_f32(Y)+s*d,d); return 1;}

static int bwd_softmax(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    int seq=X->rank>=2?X->shape[0]:1,d=X->shape[X->rank-1],s;
    if (!Y->grad||!X->grad) return 1;
    for (s=0;s<seq;s++) op_softmax_bwd(tb_f32(Y)+s*d,Y->grad+s*d,X->grad+s*d,d); return 1;}

static int fwd_rope(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    int d_head=graph_node_get_attr_i(n,"d_head",64),n_heads=graph_node_get_attr_i(n,"n_heads",4),seq=X->shape[0];
    memcpy(tb_f32(Y),tb_f32(X),(size_t)X->n_elem*sizeof(float));
    op_rope_f32(tb_f32(Y),seq,n_heads,d_head,1); return 1;}

static int bwd_rope(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    int d_head=graph_node_get_attr_i(n,"d_head",64),n_heads=graph_node_get_attr_i(n,"n_heads",4),seq=X->shape[0];
    if (!Y->grad||!X->grad) return 1;
    memcpy(X->grad,Y->grad,(size_t)Y->n_elem*sizeof(float));
    op_rope_f32(X->grad,seq,n_heads,d_head,0); return 1;}

static int fwd_embed(Graph *g,GraphNode *n){
    Tensor *tokens=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    op_embed_f32((int*)tokens->data,tb_f32(W),tb_f32(Y),tokens->shape[0],W->shape[1],W->shape[0]); return 1;}

static int bwd_embed(Graph *g,GraphNode *n){
    Tensor *tokens=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    if (!Y->grad||!W->grad) return 1;
    op_embed_bwd((int*)tokens->data,Y->grad,W->grad,tokens->shape[0],W->shape[1]); return 1;}

static int fwd_lm_head(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    op_linear_f32(tb_f32(X),tb_f32(W),tb_f32(Y),W->shape[1],W->shape[0]); return 1;}

static int bwd_lm_head(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*W=graph_tensor(g,n->inputs[1]),*Y=graph_tensor(g,n->outputs[0]);
    int d=W->shape[1],v=W->shape[0];
    if (!Y->grad) return 1;
    op_linear_bwd(tb_f32(X),tb_f32(W),Y->grad,X->grad,W->grad,d,v); return 1;}

static int fwd_quantize(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    int rows=X->shape[0],cols=X->shape[1];
    if (!Y->quant_scales) Y->quant_scales=(float*)malloc((size_t)rows*sizeof(float));
    if (!Y->quant_scales) return 0;
    op_quantize_i8(tb_f32(X),tb_i8(Y),Y->quant_scales,rows,cols);
    Y->flags|=TB_FLAG_QUANT; return 1;}

static int fwd_dequantize(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    if (!X->quant_scales) return 0;
    op_dequantize_i8(tb_i8(X),X->quant_scales,tb_f32(Y),X->shape[0],X->shape[1]); return 1;}

static int bwd_quantize(Graph *g,GraphNode *n){
    Tensor *X=graph_tensor(g,n->inputs[0]),*Y=graph_tensor(g,n->outputs[0]);
    if (X->grad&&Y->grad) memcpy(X->grad,Y->grad,(size_t)X->n_elem*sizeof(float)); return 1;}

void ops_register_all(void)
{
    graph_register_op(OP_MATMUL,   "MatMul",    fwd_matmul,    bwd_matmul);
    graph_register_op(OP_MATMUL_T, "MatMulT",   fwd_matmul_t,  bwd_matmul_t);
    graph_register_op(OP_ADD,      "Add",        fwd_add,       bwd_add);
    graph_register_op(OP_SCALE,    "Scale",      fwd_scale,     bwd_scale);
    graph_register_op(OP_RMSNORM,  "RMSNorm",    fwd_rmsnorm,   bwd_rmsnorm);
    graph_register_op(OP_GELU,     "GELU",       fwd_gelu,      bwd_gelu);
    graph_register_op(OP_SWIGLU,   "SwiGLU",     fwd_swiglu,    bwd_swiglu);
    graph_register_op(OP_SOFTMAX,  "Softmax",    fwd_softmax,   bwd_softmax);
    graph_register_op(OP_ROPE,     "RoPE",       fwd_rope,      bwd_rope);
    graph_register_op(OP_EMBED,    "Embed",      fwd_embed,     bwd_embed);
    graph_register_op(OP_LM_HEAD,  "LMHead",     fwd_lm_head,   bwd_lm_head);
    graph_register_op(OP_QUANTIZE, "Quantize",   fwd_quantize,  bwd_quantize);
    graph_register_op(OP_DEQUANTIZE,"Dequantize",fwd_dequantize,NULL);
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 2
 *
 * Files covered:
 *   sysinfo.h / sysinfo.c  – GlobalMemoryStatus probe,
 *     RAM_TIER_* constants, DynModelCfg builder,
 *     max_embeds auto-size, arena_mb suggestion,
 *     tb_yield / tb_yield_bg / tb_thread_set_bg,
 *     brain.log sink (blog_init/write/close),
 *     BLOG_INFO/WARN/ERROR macros
 *   ops.h / ops.c           – all operators + graph wiring;
 *     new: tb_yield every 8 heads in op_attention_f32/bwd
 *
 * PART 3 will cover:
 *   model.h / model.c  – TransformerConfig extended with
 *     DynModelCfg import helper, model_create_dynamic,
 *     full forward pass (train+inference), cache,
 *     HMAC-signed "TB13" binary save/load,
 *     model_requantize, config stored in checkpoint
 * ─────────────────────────────────────────────────────────────
 */

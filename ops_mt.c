/*
 * ops_mt.c - Win32 multi-core BLAS-style kernels for TheBrain v13
 *
 * Persistent worker thread pool (CreateThread + Events).  Compatible with
 * Windows 2000 / NT4+ (GetSystemInfo, WaitForMultipleObjects).
 * Pure C89.  No C runtime thread APIs required.
 */
#include "ops_mt.h"
#include "ops.h"
#include "sysinfo.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0500
#  endif
#  include <windows.h>
#endif

#define TB_MT_MAX_WORKERS  8
#define TB_MT_MIN_COLS     32   /* min N before matmul_t threads */
#define TB_MT_MIN_VOCAB    64   /* min vocab before linear threads */
#define TB_MT_MIN_ATTN_SEQ 12   /* min seq before attention threads */

#define TB_MT_JOB_NONE     0
#define TB_MT_JOB_MATMUL_T 1
#define TB_MT_JOB_LINEAR   2
#define TB_MT_JOB_ATTN     3

typedef struct TB_MtJobTag {
    int           kind;
    int           i0, i1;       /* row or head range [i0,i1) */
    int           j0, j1;       /* col range [j0,j1) when M==1 */
    const float  *A;
    const float  *B;
    const float  *W;
    const float  *x;
    const float  *Q;
    const float  *K;
    const float  *V;
    float        *C;
    float        *out;
    float        *attn_row;
    float        *attn_w_out;
    int           M, dim_k, N;
    int           d_model;
    int           vocab;
    int           seq;
    int           n_heads;
    int           d_head;
    int           causal;
    float         scale;
} TB_MtJob;

#ifdef _WIN32

static CRITICAL_SECTION g_mt_cs;
static HANDLE           g_mt_go  [TB_MT_MAX_WORKERS];
static HANDLE           g_mt_done[TB_MT_MAX_WORKERS];
static HANDLE           g_mt_thr [TB_MT_MAX_WORKERS];
static TB_MtJob         g_mt_job [TB_MT_MAX_WORKERS];
static volatile LONG    g_mt_shutdown = 0;
static int              g_mt_nworkers  = 0;
static int              g_mt_ready     = 0;

/* ── worker helpers (one job chunk) ── */

static void mt_matmul_t_chunk(const TB_MtJob *job)
{
    int i, j;
    if (job->M <= 1) {
        const float *Ai = job->A;
        float       *Ci = job->C;
        for (j = job->j0; j < job->j1; j++)
            Ci[j] = op_dot_f32(Ai, job->B + j * job->dim_k, job->dim_k);
    } else {
        for (i = job->i0; i < job->i1; i++) {
            const float *Ai = job->A + i * job->dim_k;
            float       *Ci = job->C + i * job->N;
            for (j = 0; j < job->N; j++)
                Ci[j] = op_dot_f32(Ai, job->B + j * job->dim_k, job->dim_k);
        }
    }
}

static void mt_linear_chunk(const TB_MtJob *job)
{
    int v;
    for (v = job->i0; v < job->i1; v++)
        job->out[v] = op_dot_f32(job->W + v * job->d_model,
                                  job->x, job->d_model);
}

static void mt_attn_head_chunk(const TB_MtJob *job)
{
    int h, s, j, i;
    for (h = job->i0; h < job->i1; h++) {
        int hoff = h * job->d_head;
        for (s = 0; s < job->seq; s++) {
            const float *qs = job->Q + s * job->n_heads * job->d_head + hoff;
            float       *ar = job->attn_row + (size_t)s * (size_t)job->seq;
            for (j = 0; j < job->seq; j++) {
                if (job->causal && j > s) { ar[j] = -1e9f; continue; }
                ar[j] = op_dot_f32(qs,
                                     job->K + j * job->n_heads * job->d_head + hoff,
                                     job->d_head) * job->scale;
            }
            op_softmax_f32(ar, job->seq);
        }
        if (job->attn_w_out)
            memcpy(job->attn_w_out + (size_t)h * (size_t)job->seq * (size_t)job->seq,
                   job->attn_row,
                   (size_t)job->seq * (size_t)job->seq * sizeof(float));
        for (s = 0; s < job->seq; s++) {
            float       *ar   = job->attn_row + (size_t)s * (size_t)job->seq;
            float       *outs = job->out + s * job->n_heads * job->d_head + hoff;
            int          j_end = job->causal ? s : job->seq - 1;
            for (j = 0; j <= j_end; j++) {
                float a = ar[j];
                const float *vs = job->V + j * job->n_heads * job->d_head + hoff;
                for (i = 0; i < job->d_head; i++)
                    outs[i] += a * vs[i];
            }
        }
    }
}

static DWORD WINAPI tb_mt_worker_proc(LPVOID param)
{
    int wid = (int)(INT_PTR)param;
    for (;;) {
        if (WaitForSingleObject(g_mt_go[wid], INFINITE) != WAIT_OBJECT_0)
            return 0;
        if (g_mt_shutdown)
            return 0;
        switch (g_mt_job[wid].kind) {
        case TB_MT_JOB_MATMUL_T: mt_matmul_t_chunk(&g_mt_job[wid]); break;
        case TB_MT_JOB_LINEAR:   mt_linear_chunk  (&g_mt_job[wid]); break;
        case TB_MT_JOB_ATTN:     mt_attn_head_chunk(&g_mt_job[wid]); break;
        default: break;
        }
        SetEvent(g_mt_done[wid]);
    }
    /* not reached */
}

static void tb_mt_dispatch(int n_used)
{
    HANDLE waits[TB_MT_MAX_WORKERS];
    int i;
    for (i = 0; i < n_used; i++) {
        ResetEvent(g_mt_done[i]);
        SetEvent(g_mt_go[i]);
        waits[i] = g_mt_done[i];
    }
    WaitForMultipleObjects((DWORD)n_used, waits, TRUE, INFINITE);
}

void tb_mt_init(void)
{
    SYSTEM_INFO si;
    int i, n;

    if (g_mt_ready) return;
    InitializeCriticalSection(&g_mt_cs);
    EnterCriticalSection(&g_mt_cs);
    if (g_mt_ready) {
        LeaveCriticalSection(&g_mt_cs);
        return;
    }

    memset(&si, 0, sizeof(si));
    GetSystemInfo(&si);
    n = (int)si.dwNumberOfProcessors;
    if (n < 1) n = 1;
    if (n > TB_MT_MAX_WORKERS) n = TB_MT_MAX_WORKERS;
    g_mt_nworkers = n;

    if (g_mt_nworkers > 1) {
        for (i = 0; i < g_mt_nworkers; i++) {
            g_mt_go[i]   = CreateEvent(NULL, FALSE, FALSE, NULL);
            g_mt_done[i] = CreateEvent(NULL, TRUE,  FALSE, NULL);
            g_mt_thr[i]  = CreateThread(NULL, 0, tb_mt_worker_proc,
                                        (LPVOID)(INT_PTR)i, 0, NULL);
            if (!g_mt_go[i] || !g_mt_done[i] || !g_mt_thr[i]) {
                g_mt_nworkers = 1;
                break;
            }
            tb_thread_set_bg(g_mt_thr[i]);
        }
    } else {
        g_mt_nworkers = 1;
    }

    g_mt_ready = 1;
    LeaveCriticalSection(&g_mt_cs);
    BLOG_INFO("tb_mt_init: workers=%d", g_mt_nworkers);
}

void tb_mt_shutdown(void)
{
    int i;
    if (!g_mt_ready || g_mt_nworkers <= 1) return;
    InterlockedExchange(&g_mt_shutdown, 1);
    for (i = 0; i < g_mt_nworkers; i++) {
        if (g_mt_go[i]) SetEvent(g_mt_go[i]);
    }
    for (i = 0; i < g_mt_nworkers; i++) {
        if (g_mt_thr[i]) {
            WaitForSingleObject(g_mt_thr[i], 5000);
            CloseHandle(g_mt_thr[i]);
            g_mt_thr[i] = NULL;
        }
        if (g_mt_go[i])   { CloseHandle(g_mt_go[i]);   g_mt_go[i]   = NULL; }
        if (g_mt_done[i]) { CloseHandle(g_mt_done[i]); g_mt_done[i] = NULL; }
    }
    DeleteCriticalSection(&g_mt_cs);
    g_mt_ready = 0;
    g_mt_nworkers = 0;
}

int tb_mt_nworkers(void)
{
    if (!g_mt_ready) tb_mt_init();
    return g_mt_nworkers;
}

int tb_mt_try_matmul_t_f32(const float *A, const float *B, float *C,
                              int M, int K, int N)
{
    int nw, w, chunk, used;

    if (!g_mt_ready) tb_mt_init();
    nw = g_mt_nworkers;
    if (nw <= 1) return 0;
    if (M <= 1 && N < TB_MT_MIN_COLS) return 0;
    if (M > 1 && M < nw) return 0;

    used = nw;
    if (M <= 1) {
        chunk = (N + nw - 1) / nw;
        for (w = 0; w < nw; w++) {
            g_mt_job[w].kind = TB_MT_JOB_MATMUL_T;
            g_mt_job[w].M = M; g_mt_job[w].dim_k = K; g_mt_job[w].N = N;
            g_mt_job[w].A = A; g_mt_job[w].B = B; g_mt_job[w].C = C;
            g_mt_job[w].j0 = w * chunk;
            g_mt_job[w].j1 = g_mt_job[w].j0 + chunk;
            if (g_mt_job[w].j1 > N) g_mt_job[w].j1 = N;
            if (g_mt_job[w].j0 >= N) { used = w; break; }
        }
    } else {
        chunk = (M + nw - 1) / nw;
        for (w = 0; w < nw; w++) {
            g_mt_job[w].kind = TB_MT_JOB_MATMUL_T;
            g_mt_job[w].M = M; g_mt_job[w].dim_k = K; g_mt_job[w].N = N;
            g_mt_job[w].A = A; g_mt_job[w].B = B; g_mt_job[w].C = C;
            g_mt_job[w].i0 = w * chunk;
            g_mt_job[w].i1 = g_mt_job[w].i0 + chunk;
            if (g_mt_job[w].i1 > M) g_mt_job[w].i1 = M;
            if (g_mt_job[w].i0 >= M) { used = w; break; }
        }
    }
    if (used <= 1) return 0;
    tb_mt_dispatch(used);
    return 1;
}

int tb_mt_try_linear_f32(const float *x, const float *W, float *out,
                          int d_model, int vocab)
{
    int nw, w, chunk, used;

    if (!g_mt_ready) tb_mt_init();
    nw = g_mt_nworkers;
    if (nw <= 1 || vocab < TB_MT_MIN_VOCAB) return 0;

    chunk = (vocab + nw - 1) / nw;
    used = nw;
    for (w = 0; w < nw; w++) {
        g_mt_job[w].kind     = TB_MT_JOB_LINEAR;
        g_mt_job[w].x        = x;
        g_mt_job[w].W        = W;
        g_mt_job[w].out      = out;
        g_mt_job[w].d_model  = d_model;
        g_mt_job[w].vocab    = vocab;
        g_mt_job[w].i0       = w * chunk;
        g_mt_job[w].i1       = g_mt_job[w].i0 + chunk;
        if (g_mt_job[w].i1 > vocab) g_mt_job[w].i1 = vocab;
        if (g_mt_job[w].i0 >= vocab) { used = w; break; }
    }
    if (used <= 1) return 0;
    tb_mt_dispatch(used);
    return 1;
}

int tb_mt_try_attention_f32(const float *Q, const float *K, const float *V,
                               float *out, float *attn_w_out,
                               int seq, int n_heads, int d_head,
                               int causal, float scale)
{
    int nw, w, chunk, used;
    float *scratch[TB_MT_MAX_WORKERS];
    size_t row_bytes;

    if (!g_mt_ready) tb_mt_init();
    nw = g_mt_nworkers;
    if (nw <= 1 || n_heads < 2 || seq < TB_MT_MIN_ATTN_SEQ) return 0;

    row_bytes = (size_t)seq * (size_t)seq * sizeof(float);
    for (w = 0; w < nw; w++) scratch[w] = NULL;
    for (w = 0; w < nw; w++) {
        scratch[w] = (float*)malloc(row_bytes);
        if (!scratch[w]) {
            while (w > 0) free(scratch[--w]);
            return 0;
        }
    }

    memset(out, 0, (size_t)seq * (size_t)n_heads * (size_t)d_head * sizeof(float));

    chunk = (n_heads + nw - 1) / nw;
    used = nw;
    for (w = 0; w < nw; w++) {
        g_mt_job[w].kind       = TB_MT_JOB_ATTN;
        g_mt_job[w].Q          = Q;
        g_mt_job[w].K          = K;
        g_mt_job[w].V          = V;
        g_mt_job[w].out        = out;
        g_mt_job[w].attn_w_out = attn_w_out;
        g_mt_job[w].attn_row   = scratch[w];
        g_mt_job[w].seq        = seq;
        g_mt_job[w].n_heads    = n_heads;
        g_mt_job[w].d_head     = d_head;
        g_mt_job[w].causal     = causal;
        g_mt_job[w].scale      = scale;
        g_mt_job[w].i0         = w * chunk;
        g_mt_job[w].i1         = g_mt_job[w].i0 + chunk;
        if (g_mt_job[w].i1 > n_heads) g_mt_job[w].i1 = n_heads;
        if (g_mt_job[w].i0 >= n_heads) { used = w; break; }
    }
    if (used <= 1) {
        for (w = 0; w < nw; w++) free(scratch[w]);
        return 0;
    }
    tb_mt_dispatch(used);
    for (w = 0; w < nw; w++) free(scratch[w]);
    return 1;
}

#else /* !_WIN32 */

void tb_mt_init(void) {}
void tb_mt_shutdown(void) {}
int  tb_mt_nworkers(void) { return 1; }
int tb_mt_try_matmul_t_f32(const float *A, const float *B, float *C,
                              int M, int K, int N)
{ (void)A;(void)B;(void)C;(void)M;(void)K;(void)N; return 0; }
int tb_mt_try_linear_f32(const float *x, const float *W, float *out,
                          int d_model, int vocab)
{ (void)x;(void)W;(void)out;(void)d_model;(void)vocab; return 0; }
int tb_mt_try_attention_f32(const float *Q, const float *K, const float *V,
                               float *out, float *attn_w_out,
                               int seq, int n_heads, int d_head,
                               int causal, float scale)
{ (void)Q;(void)K;(void)V;(void)out;(void)attn_w_out;(void)seq;
  (void)n_heads;(void)d_head;(void)causal;(void)scale; return 0; }

#endif /* _WIN32 */

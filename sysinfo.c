#include "sysinfo.h"
#include <stdarg.h>
#include <string.h>
#include <time.h>

static CRITICAL_SECTION g_blog_cs;
static FILE            *g_blog_fp   = NULL;
static int              g_blog_init = 0;

extern int g_override_tier;

/* ── sysinfo_probe ─────────────────────────────────────────── */
void sysinfo_probe(SysInfo *out)
{
    MEMORYSTATUS ms;
    DWORD total_mb, avail_mb;

    memset(out, 0, sizeof(SysInfo));
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = sizeof(ms);

    /*
     * GlobalMemoryStatus is available on Windows 95/98/NT/2000+.
     * It saturates at 4 GB (DWORD overflow) — sufficient for v13.
     */
    GlobalMemoryStatus(&ms);

    total_mb = (DWORD)(ms.dwTotalPhys / (1024UL * 1024UL));
    avail_mb = (DWORD)(ms.dwAvailPhys / (1024UL * 1024UL));

    out->total_mb = total_mb;
    out->avail_mb = avail_mb;

    if (g_override_tier >= 0 && g_override_tier <= 4) {
        out->tier = g_override_tier;
    } else {
        /* Determine tier from TOTAL physical RAM */
        if      (total_mb < 256)  out->tier = RAM_TIER_TINY;
        else if (total_mb < 512)  out->tier = RAM_TIER_SMALL;
        else if (total_mb < 1024) out->tier = RAM_TIER_MEDIUM;
        else if (total_mb < 4096) out->tier = RAM_TIER_LARGE;
        else                      out->tier = RAM_TIER_XLARGE;
    }

    /*
     * MaxEmbeds: each embedding is CFG_D_MODEL floats + filename + label.
     * Rough budget: use 25% of available RAM for embedding DB.
     * d_model varies by tier; use worst-case 512 floats = 2 KB/embed.
     */
    {
        DWORD embed_budget_mb = avail_mb / 4;
        DWORD bytes_per_embed = 512 * sizeof(float) + 256 + 8;
        DWORD max_e = (embed_budget_mb * 1024UL * 1024UL) / bytes_per_embed;
        if (max_e < 1000)   max_e = 1000;
        if (max_e > 500000) max_e = 500000;
        out->max_embeds = (int)max_e;
    }

    /*
     * Arena: use 40% of available RAM for tensor arena.
     * Min 32 MB, max 2048 MB.
     */
    {
        DWORD arena = avail_mb * 40 / 100;
        if (arena < 32)   arena = 32;
        if (arena > 2048) arena = 2048;
        out->arena_mb = (int)arena;
    }

    BLOG_INFO("SysInfo: total=%luMB avail=%luMB tier=%d "
              "max_embeds=%d arena=%dMB",
              (unsigned long)total_mb, (unsigned long)avail_mb,
              out->tier, out->max_embeds, out->arena_mb);
}

/* ── sysinfo_make_cfg ──────────────────────────────────────── */
/*
 * RAM tier -> transformer architecture:
 *
 *  TINY   (<256 MB) : 2 layers, 2 heads, d=128,  ff=256,  ctx=256
 *  SMALL  (256-512) : 4 layers, 4 heads, d=256,  ff=512,  ctx=384
 *  MEDIUM (512-1GB) : 6 layers, 4 heads, d=256,  ff=1024, ctx=512
 *  LARGE  (1-4 GB)  : 8 layers, 8 heads, d=512,  ff=2048, ctx=1024
 *  XLARGE (>4 GB)   : 12 layers,8 heads, d=768,  ff=3072, ctx=2048
 *
 * These are safe upper bounds; the user can override in brain.conf.
 */
DynModelCfg sysinfo_make_cfg(const SysInfo *si,
                               int use_swiglu,
                               int use_rmsnorm,
                               int tie_embeddings)
{
    DynModelCfg c;
    memset(&c, 0, sizeof(c));

    c.rms_eps      = 1e-5f;
    c.use_swiglu   = use_swiglu;
    c.use_rmsnorm  = use_rmsnorm;
    c.tie_embeddings = tie_embeddings;
    c.n_classes    = 16;
    c.n_lang       = 4;
    c.vocab_size   = 32768;

    switch (si->tier) {
    case RAM_TIER_TINY:
        c.n_layers=2; c.n_heads=2; c.d_model=128;
        c.d_ff=256;   c.ctx_len=256;
        break;
    case RAM_TIER_SMALL:
        c.n_layers=4; c.n_heads=4; c.d_model=256;
        c.d_ff=512;   c.ctx_len=384;
        break;
    case RAM_TIER_MEDIUM:
        c.n_layers=6; c.n_heads=4; c.d_model=256;
        c.d_ff=1024;  c.ctx_len=512;
        break;
    case RAM_TIER_LARGE:
        c.n_layers=8; c.n_heads=8; c.d_model=512;
        c.d_ff=2048;  c.ctx_len=1024;
        break;
    case RAM_TIER_XLARGE:
    default:
        c.n_layers=12; c.n_heads=8; c.d_model=768;
        c.d_ff=3072;   c.ctx_len=2048;
        break;
    }

    BLOG_INFO("DynModelCfg: tier=%d layers=%d heads=%d "
              "d=%d ff=%d ctx=%d vocab=%d",
              si->tier, c.n_layers, c.n_heads,
              c.d_model, c.d_ff, c.ctx_len, c.vocab_size);
    return c;
}

/* ── CPU yield helpers ─────────────────────────────────────── */
void tb_yield(void)
{
    Sleep(0);   /* relinquish timeslice; no sleep                    */
}

void tb_yield_bg(void)
{
    Sleep(1);   /* 1 ms sleep for background tasks                   */
}

void tb_thread_set_bg(HANDLE hThread)
{
    if (hThread && hThread != INVALID_HANDLE_VALUE)
        SetThreadPriority(hThread, THREAD_PRIORITY_BELOW_NORMAL);
}

void tb_pump_messages(void)
{
#ifdef _WIN32
    extern HWND g_hMain;
    if (g_hMain && GetCurrentThreadId() == GetWindowThreadProcessId(g_hMain, NULL)) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }
#endif
}

/* ── brain.log sink ────────────────────────────────────────── */
void blog_init(const char *path)
{
    if (g_blog_init) return;
    InitializeCriticalSection(&g_blog_cs);
    g_blog_fp = fopen(path, "a");
    g_blog_init = 1;
}

void blog_write(int severity, const char *fmt, ...)
{
    char     buf[2048];
    va_list  ap;
    SYSTEMTIME st;
    static const char *sevs[] = {"DBG","INF","WRN","ERR","CRT"};
    int idx = severity;

    if (!g_blog_init || !g_blog_fp) return;
    if (idx < 0 || idx > 4) idx = 4;

    GetLocalTime(&st);
    va_start(ap, fmt);
    _vsnprintf(buf, sizeof(buf)-1, fmt, ap);
    buf[sizeof(buf)-1] = '\0';
    va_end(ap);

    EnterCriticalSection(&g_blog_cs);
    fprintf(g_blog_fp,
            "[%04d-%02d-%02d %02d:%02d:%02d][%s] %s\n",
            st.wYear, st.wMonth, st.wDay,
            st.wHour, st.wMinute, st.wSecond,
            sevs[idx], buf);
    fflush(g_blog_fp);
    LeaveCriticalSection(&g_blog_cs);
}

void blog_close(void)
{
    if (!g_blog_init) return;
    if (g_blog_fp) { fclose(g_blog_fp); g_blog_fp = NULL; }
    DeleteCriticalSection(&g_blog_cs);
    g_blog_init = 0;
}

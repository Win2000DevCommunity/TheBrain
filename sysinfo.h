#ifndef SYSINFO_H
#define SYSINFO_H

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0500
#define WINVER       0x0500
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── RAM tier constants ── */
#define RAM_TIER_TINY    0   /* < 256 MB  -> minimal model            */
#define RAM_TIER_SMALL   1   /* 256-511MB -> default model            */
#define RAM_TIER_MEDIUM  2   /* 512MB-1GB -> medium model             */
#define RAM_TIER_LARGE   3   /* 1-4 GB    -> large model              */
#define RAM_TIER_XLARGE  4   /* > 4 GB    -> extra-large model        */

/* ── Log severity (mirrors brain.h, kept self-contained) ── */
#ifndef LOG_INFO
#  define LOG_INFO  1
#  define LOG_WARN  2
#  define LOG_ERROR 3
#endif

/* ── Probe result ── */
typedef struct {
    DWORD total_mb;        /* total physical RAM in MB               */
    DWORD avail_mb;        /* available physical RAM in MB           */
    int   tier;            /* RAM_TIER_* constant                    */
    int   max_embeds;      /* suggested MAX_EMBEDS                   */
    int   arena_mb;        /* suggested tensor arena in MB           */
} SysInfo;

/* ── Dynamic ModelConfig snapshot (mirrors model.h layout) ── */
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
} DynModelCfg;

#ifdef __cplusplus
extern "C" {
#endif

/* Probe the system and populate a SysInfo struct.
 * Uses GlobalMemoryStatus (Win2000+, no IntelliSense needed). */
void      sysinfo_probe      (SysInfo *out);

/* Map a SysInfo to a DynModelCfg. */
DynModelCfg sysinfo_make_cfg (const SysInfo *si,
                               int use_swiglu,
                               int use_rmsnorm,
                               int tie_embeddings);

/* CPU yield: call in long inner loops to keep UI responsive.
 * Calls Sleep(0) on Windows (gives up timeslice without delay). */
void      tb_yield           (void);

/* Throttled yield: Sleep(1) for background tasks. */
void      tb_yield_bg        (void);

/* Set worker thread to below-normal priority. */
void      tb_thread_set_bg   (HANDLE hThread);

/* Pump message loop on the main GUI thread. */
void      tb_pump_messages   (void);

/* Brain.log error sink (thread-safe via CriticalSection).
 * Initialise once; all modules write here. */
void      blog_init          (const char *path);
void      blog_write         (int severity, const char *fmt, ...);
void      blog_close         (void);

/* Convenience macros */
#define BLOG_INFO(...)  blog_write(LOG_INFO,  __VA_ARGS__)
#define BLOG_WARN(...)  blog_write(LOG_WARN,  __VA_ARGS__)
#define BLOG_ERROR(...) blog_write(LOG_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif
#endif /* SYSINFO_H */

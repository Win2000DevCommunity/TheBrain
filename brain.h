#ifndef BRAIN_H
#define BRAIN_H

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT  0x0500
#define WINVER        0x0500

#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#include <shellapi.h>
#include <richedit.h>
#include <wininet.h>
#include <commctrl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <stdarg.h>

/* All subsystem headers */
#include "sysinfo.h"
#include "tensor.h"
#include "graph.h"
#include "ops.h"
#include "ops_mt.h"
#include "model.h"
#include "tokenizer.h"
#include "train.h"
#include "converse.h"  /* NEW v13 */

/* ── Version ── */
#define BRAIN_VERSION_STR  "13.0"
#define BRAIN_BUILD_DATE   __DATE__

/* ── Memory budget ── */
#ifndef MEM_BUDGET_MB
#  define MEM_BUDGET_MB 256
#endif

/* ── ML subsystem limits ── */
#define MAX_FEATURES     32
#define MLP_HIDDEN       32
#define MLP_OUT          2
#define MAX_SAMPLES      3000
#define MAX_K            32
#define KMEANS_MAX_ITER  200
#define ISO_TREES        64
#define ISO_NODES        256
#define ISO_SUBSAMPLE    128
#define OCSVM_D          64
#define SELFTRAIN_ROUNDS 5
#define WEAK_SCORE_POS   3
#define WEAK_SCORE_NEG   (-2)
#define HIST_SIZE        128
#define CONTEXT_WINDOW   8
#define FEAT_CACHE_MAX   2048
#define UNDO_DEPTH       32
#define MAX_EMBEDS       100000  /* overridden by SysInfo.max_embeds */
#define MAX_PATTERNS     512
#define PATTERN_MAX_TOKS 64
#define MAX_CURRICULUM   64
#define FORUM_HASH_RING  1024
#define LZW_DICT_SIZE    4096
#define EXPLAIN_TOP_N    8
#define DEAD_LETTER_MAX  (1024u*1024u)
#define GH_MAX_FILESZ    (512*1024)
#define GH_MAX_TREE      4096
#define GH_TIMEOUT_MS    15000
#define GH_UA            "TheBrain/13.0"
#define GH_API_HOST      "api.github.com"
#define GH_RAW_HOST      "raw.githubusercontent.com"
#define PE_MAX_MAP_SIZE  (64u*1024u*1024u)
#define BPAK_MAGIC       "BPAK"
#define BPAK_VERSION     4
#define CHECKPOINT_DIR   "checkpoints"
#define DEAD_LETTER_FILE "dead_letter.txt"
#define FEAT_CACHE_FILE  "feat_cache.bin"
#define EMBEDS_FILE      "embeds.bin"
#define CONTEXT_FILE     "context.bin"
#define PATTERNS_FILE    "patterns.dat"
#define N_CLASSES        16
#define N_LANG_CLASSES   4

/* ── Model downloader ── */
#define MODEL_DL_MAX_URL   512
#define MODEL_DL_CHUNK     (64*1024)  /* 64 KB per read */

/* ── GUI IDs ── */
#define CHAT_ID         1001
#define INPUT_ID        1002
#define BTN_SEND        1003
#define BTN_TRAIN       1004
#define BTN_STATS       1005
#define BTN_SCAN        1006
#define BTN_HELP        1007
#define BTN_UNDO        1008
#define BTN_CANCEL      1009
#define PROG_ID         1010
#define BTN_GENERATE    1011
#define BTN_SIMILAR     1012
#define BTN_SUMMARIZE   1013
#define BTN_EXPLAIN     1014
#define BTN_REPORT      1015
#define BTN_FULLTRAIN   1016
#define BTN_CONVERSE    1017   /* NEW v13 */

/* ── Windows messages ── */
#define WM_APP_DONE     (WM_APP+1)
#define WM_APP_PROGRESS (WM_APP+2)
#define WM_APP_LOG      (WM_APP+3)
#define WM_APP_WATCHDOG (WM_APP+5)
#define WM_APP_TOKEN    (WM_APP+6)   /* NEW v13: streamed token text */
#define WM_APP_GUARD    (WM_APP+7)   /* NEW v13: file guard alert    */

/* ── Colours ── */
#define COL_DEFAULT  RGB(220,220,220)
#define COL_DANGER   RGB(255, 70, 70)
#define COL_SAFE     RGB( 70,220,100)
#define COL_WARN     RGB(255,200, 40)
#define COL_INFO     RGB( 80,200,255)
#define COL_PROMPT   RGB(200,200, 70)
#define COL_GREY     RGB(130,130,130)
#define COL_CYAN     RGB(  0,240,230)
#define COL_ORANGE   RGB(255,165,  0)
#define COL_PURPLE   RGB(180,100,255)
#define COL_GREEN    RGB( 80,255,120)
#define COL_GENERATE RGB(255,215,  0)
#define COL_REASON   RGB(160,120,255)
#define COL_EXPLAIN  RGB(255,180, 50)
#define COL_REPORT   RGB( 50,220,255)
#define COL_TRAIN    RGB(100,255,160)
#define COL_PINK     RGB(255,120,180)
#define COL_YELLOW   RGB(255,255, 80)
#define COL_CONVERSE RGB(180,255,220)  /* NEW v13: conversational reply */
#define COL_FACT     RGB(255,230,100)  /* NEW v13: fact annotation      */
#define COL_GUARD    RGB(255, 80,200)  /* NEW v13: file guard alert     */

/* ── Log severity ── */
#define LOG_DEBUG    0
#define LOG_INFO     1
#define LOG_WARN     2
#define LOG_ERROR    3
#define LOG_CRITICAL 4

/* ── Error codes ── */
#define TB_OK              0
#define TB_ERR_OOM         1
#define TB_ERR_IO          2
#define TB_ERR_CORRUPT     3
#define TB_ERR_HMAC        4
#define TB_ERR_RANGE       5

/* ─────────────────────────────────────────────────────────────
 * ML data structures  (unchanged from v12)
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    double features[MAX_FEATURES];
    int    label;
    float  weight;
    char   filename[256];
} Sample;

typedef struct {
    double w1[MLP_HIDDEN][MAX_FEATURES], b1[MLP_HIDDEN];
    double w2[MLP_OUT][MLP_HIDDEN],      b2[MLP_OUT];
    double mw1[MLP_HIDDEN][MAX_FEATURES],vw1[MLP_HIDDEN][MAX_FEATURES];
    double mb1[MLP_HIDDEN],              vb1[MLP_HIDDEN];
    double mw2[MLP_OUT][MLP_HIDDEN],     vw2[MLP_OUT][MLP_HIDDEN];
    double mb2[MLP_OUT],                 vb2[MLP_OUT];
    long   adam_t;
} MLP;

typedef struct {
    double mean[2][MAX_FEATURES];
    double var [2][MAX_FEATURES];
    double prior[2];
    int    trained;
} NaiveBayes;

typedef struct {
    double mean[MAX_FEATURES];
    double std [MAX_FEATURES];
} NormStats;

typedef struct {
    double centroid[MAX_FEATURES];
    int    label, count;
    double radius;
} KCluster;

typedef struct {
    KCluster clusters[MAX_K];
    int      k, trained;
    double   inertia;
} KMeansModel;

typedef struct {
    int    feature;
    double split;
    int    left, right, size;
} IsoNode;

typedef struct {
    IsoNode nodes[ISO_NODES];
    int     n_nodes, root;
} IsoTree;

typedef struct {
    IsoTree trees[ISO_TREES];
    int     trained;
    double  threshold, avg_path;
} IsoForest;

typedef struct {
    double omega[OCSVM_D][MAX_FEATURES];
    double bias_b[OCSVM_D];
    double w[OCSVM_D];
    double rho, gamma;
    int    trained;
} OCSVM;

typedef struct {
    DWORD  path_crc;
    DWORD  mtime_lo, mtime_hi;
    double features[MAX_FEATURES];
    DWORD  clock_bit;
    DWORD  lru_stamp;
} FeatCacheEntry;

typedef struct {
    float vec[CFG_D_MODEL];
    char  filename[256];
    int   label;
    DWORD crc;
} FileEmbed;

typedef struct {
    char   tokens[PATTERN_MAX_TOKS][64];
    int    n_tokens;
    int    is_wildcard[PATTERN_MAX_TOKS];
    char   name[64];
    int    label;
    float  confidence;
} CodePattern;

typedef struct {
    CodePattern patterns[MAX_PATTERNS];
    int n_patterns;
} PatternLib;

typedef struct {
    char repo[256];
    int  label, done, n_files;
} CurriculumEntry;

typedef struct {
    CurriculumEntry entries[MAX_CURRICULUM];
    int n_entries, current, running;
} Curriculum;

typedef struct {
    char user_msg[512];
    char bot_reply[512];
    char last_file[512];
    DWORD timestamp;
} ContextTurn;

typedef struct {
    Sample  samples[MAX_SAMPLES];
    int     nsamples;
} UndoSnapshot;

typedef struct {
    UndoSnapshot ring[UNDO_DEPTH];
    int          head, count;
} UndoStack;

typedef struct {
    long inferences_total;
    long train_cycles;
    long http_requests;
    long http_failures;
    long files_scanned;
    long transformer_steps;
    long full_backward_steps;
    long conv_turns;           /* NEW v13 */
    long tokens_generated;     /* NEW v13 */
    DWORD session_start;
} PerfCounters;

typedef struct {
    COLORREF col;
    int      severity;
    char     text[1];
} AppLogMsg;

/* NEW v13: real-time file guard state */
typedef struct {
    HANDLE  hDir;           /* directory handle for RDCW              */
    HANDLE  hThread;        /* guard worker thread                    */
    char    watch_dir[520]; /* watched directory path                 */
    int     running;        /* 1 = guard active                       */
    long    n_alerts;       /* total alerts fired                     */
} FileGuardState;

/* NEW v13: model downloader request */
typedef struct {
    char url[MODEL_DL_MAX_URL];
    char save_path[MAX_PATH];
    int  verify_hmac;
} ModelDLRequest;

/* ─────────────────────────────────────────────────────────────
 * BrainConfig  (v13 extended)
 * ───────────────────────────────────────────────────────────── */
typedef struct {
    /* ML classic */
    double lr, dropout;
    int    epochs;
    double iso_thresh_min, iso_thresh_max;
    double nu, conf_thresh;
    int    max_checkpoints;
    char   proxy[256];
    int    async_ops, ssl_verify;
    int    log_level;
    int    watchdog_timeout_ms;
    /* Transformer */
    float  t_lr_max, t_lr_min;
    long   t_warmup, t_total;
    float  t_wd, t_grad_clip;
    int    t_batch, t_ctx;
    int    use_swiglu, use_rmsnorm, tie_embeddings;
    /* Generation */
    float  temperature;
    int    top_k;
    /* CoT */
    int    cot_think_tokens, cot_answer_tokens;
    /* Early stopping */
    int    early_stop_patience;
    /* NEW v13: conversation */
    int    conv_max_tokens;    /* max tokens per converse turn         */
    int    conv_use_facts;     /* 1 = prepend facts                   */
    int    conv_stream;        /* 1 = stream tokens to UI             */
    int    conv_history_turns; /* how many turns to keep in context   */
    /* NEW v13: system info */
    int    sysinfo_tier;       /* RAM_TIER_* detected at startup      */
    int    dyn_max_embeds;     /* actual MAX_EMBEDS from sysinfo      */
    /* NEW v13: file guard */
    int    guard_enabled;      /* 1 = real-time guard on at startup   */
    char   guard_dir[520];     /* directory to watch                  */
    /* NEW v13: training corpus */
    int    train_use_conv;     /* 1 = include .conv in training       */
    int    train_use_text;     /* 1 = include .txt/.md in training    */
    /* Files */
    char   vocab_file[256];
    char   model_file[256];
    char   embeds_file[256];
    char   corpus_file[256];
    DWORD  conf_crc;
} BrainConfig;

/* ─────────────────────────────────────────────────────────────
 * Intent / NLU  (v13 extended)
 * ───────────────────────────────────────────────────────────── */
typedef enum {
    INTENT_UNKNOWN=0, INTENT_PREDICT, INTENT_TRAIN_DANGER,
    INTENT_TRAIN_SAFE, INTENT_SCAN_DIR, INTENT_PE_HEADER,
    INTENT_PE_IMPORTS, INTENT_DISASM, INTENT_ENTROPY,
    INTENT_STATS, INTENT_HELP, INTENT_UNDO, INTENT_FORUM_TRAIN,
    INTENT_GITRAIN, INTENT_GITFILE, INTENT_GITSHOW,
    INTENT_DECRYPT_SMART, INTENT_DECRYPT_XOR, INTENT_DECRYPT_B64,
    INTENT_DECRYPT_ROT13, INTENT_DECRYPT_CAESAR,
    INTENT_LIKE, INTENT_DISLIKE, INTENT_GREET, INTENT_JOKE,
    INTENT_TRIVIA, INTENT_GITSTATUS, INTENT_KMEANS,
    INTENT_KSELECT, INTENT_CLUSTERMAP, INTENT_ANOMALY,
    INTENT_ANOMALYSCAN, INTENT_ANOMALYTRAIN, INTENT_SELFTRAIN,
    INTENT_WEAKLABEL, INTENT_WEAKLABELDIR, INTENT_OCSVM_TRAIN,
    INTENT_OCSVM_PREDICT, INTENT_VERSION, INTENT_CLEARLOG,
    INTENT_IMPORTANCE, INTENT_RETRY, INTENT_DEADLETTER,
    INTENT_ROLLBACK, INTENT_CHECKPOINTS, INTENT_CONFIG,
    INTENT_GENERATE, INTENT_SUMMARIZE, INTENT_SIMILAR,
    INTENT_EMBEDSCAN, INTENT_SCANPATTERNS, INTENT_BPE_TRAIN,
    INTENT_PRETRAIN, INTENT_FULLTRAIN, INTENT_TRAINSTATUS,
    INTENT_QUANTISE, INTENT_EXPLAIN, INTENT_REASON,
    INTENT_REPORT,
    INTENT_CONVERSE,       /* NEW v13: natural conversation           */
    INTENT_CONV_RESET,     /* NEW v13: clear conversation history     */
    INTENT_CONV_STATS,     /* NEW v13: show conversation stats        */
    INTENT_FACTS,          /* NEW v13: show/search facts              */
    INTENT_GUARD_ON,       /* NEW v13: start real-time file guard     */
    INTENT_GUARD_OFF,      /* NEW v13: stop  real-time file guard     */
    INTENT_MODEL_DOWNLOAD, /* NEW v13: download pre-trained model     */
    INTENT_EASYTRAIN,      /* NEW v13: one-shot chat training         */
    INTENT_CHAT            /* fallback: always route to cmd_converse  */
} Intent;

typedef struct {
    Intent intent;
    char   file[512];
    char   url[1024];
    char   key[256];
    char   val[256];
    int    label;
    int    number;
    double fnum;
} NLU_Result;

/* ─────────────────────────────────────────────────────────────
 * Task types for async worker thread  (v13 extended)
 * ───────────────────────────────────────────────────────────── */
typedef enum {
    TASK_NONE=0, TASK_GITRAIN, TASK_SELFTRAIN,
    TASK_ANOMALYSCAN, TASK_BATCHTRAIN, TASK_FORUM_TRAIN,
    TASK_RETRY, TASK_PRETRAIN, TASK_FULLTRAIN,
    TASK_GENERATE, TASK_BPE_TRAIN, TASK_CURRICULUM,
    TASK_REPORT,
    TASK_CONVERSE,      /* NEW v13 */
    TASK_GUARD,         /* NEW v13 */
    TASK_DOWNLOAD,      /* NEW v13 */
    TASK_TRAIN          /* Async classic training */
} TaskType;

typedef struct {
    char  prompt[4096];
    int   max_new_tokens;
    float temperature;
    int   top_k, use_rag;
} GenRequest;

typedef struct {
    TaskType type;
    char     arg1[512], arg2[512];
    int      int1;
    double   dbl1;
    GenRequest gen;
    ModelDLRequest dl;   /* NEW v13 */
} WorkerTask;

/* ─────────────────────────────────────────────────────────────
 * PE structures  (unchanged from v12)
 * ───────────────────────────────────────────────────────────── */
#define MY_IMAGE_DOS_SIGNATURE 0x5A4D
#define MY_IMAGE_NT_SIGNATURE  0x00004550
#pragma pack(push,1)
typedef struct {WORD e_magic;WORD _p[29];LONG e_lfanew;}MY_DOS;
typedef struct {WORD Machine;WORD NumSecs;DWORD TS;DWORD PtrSym;
                DWORD NumSym;WORD SzOpt;WORD Chars;}MY_FILEHDR;
typedef struct {WORD Magic;BYTE MajLink,MinLink;DWORD SzCode,
                SzInitData,SzUninitData;DWORD EP,BaseCode,BaseData;
                DWORD ImageBase,SecAlign,FileAlign;
                WORD MajOS,MinOS,MajImg,MinImg,MajSub,MinSub;
                DWORD Win32Ver;DWORD SzImage,SzHeaders,CheckSum;
                WORD Subsys,DllChars;
                DWORD SzStackRes,SzStackCom,SzHeapRes,SzHeapCom;
                DWORD LoaderFlags,NumRva;}MY_OPTHDR32;
typedef struct {DWORD Sig;MY_FILEHDR File;MY_OPTHDR32 Opt;}MY_NTHDRS;
typedef struct {BYTE Name[8];DWORD VirtSz,VirtAddr,RawSz,RawPtr;
                DWORD PtrReloc,PtrLineNo;WORD NumReloc,NumLineNo;
                DWORD Chars;}MY_SECHDR;
typedef struct {DWORD OFT,TS,FwdChain,Name,FT;}MY_IMPORT_DESC;
#pragma pack(pop)

/* ─────────────────────────────────────────────────────────────
 * Global state (extern declarations)
 * ───────────────────────────────────────────────────────────── */
extern Sample          *g_samples;
extern int              g_nsamples;
extern MLP              g_mlp;
extern NaiveBayes       g_nb;
extern NormStats        g_norm;
extern KMeansModel      g_kmeans;
extern IsoForest        g_isoforest;
extern OCSVM            g_ocsvm;
extern BrainConfig      g_cfg;
extern FeatCacheEntry  *g_feat_cache;
extern int              g_n_feat_cache;
extern FileEmbed       *g_embeds;
extern int              g_n_embeds;
extern int              g_dyn_max_embeds;  /* NEW v13 */
extern PatternLib       g_patterns;
extern Curriculum       g_curriculum;
extern UndoStack        g_undo;
extern PerfCounters     g_perf;
extern ContextTurn      g_ctx[CONTEXT_WINDOW];
extern int              g_ctx_count;
extern char             g_last_file[512];
extern char             g_last_input[1024];
extern Model           *g_model;
extern BPETokenizer     g_tokenizer;
extern TrainState       g_train_state;
extern SysInfo          g_sysinfo;         /* NEW v13 */
extern FileGuardState   g_guard;           /* NEW v13 */

extern HWND             g_hMain, g_hChat, g_hInput;
extern HWND             g_hCancel, g_hProgress;
extern HINSTANCE        g_hInst;
extern volatile LONG    g_cancel_flag;
extern volatile LONG    g_worker_busy;
extern volatile DWORD   g_worker_ping_ms;
extern WorkerTask       g_worker_task;

extern CRITICAL_SECTION g_cs_samples;
extern CRITICAL_SECTION g_cs_log;
extern CRITICAL_SECTION g_cs_cache;
extern CRITICAL_SECTION g_cs_embeds;
extern CRITICAL_SECTION g_cs_model;

/* ─────────────────────────────────────────────────────────────
 * Function declarations
 * ───────────────────────────────────────────────────────────── */
void  app_colored (const char *t, COLORREF c);
void  brain_get_input_utf8 (char *out, int out_sz);
void  brain_clear_input    (void);
void  app         (const char *t);
void  app_info    (const char *t);
void  app_warn    (const char *t);
void  app_danger  (const char *t);
void  app_safe    (const char *t);
void  app_cyan    (const char *t);
void  app_orange  (const char *t);
void  report_progress(int pct);
void  tb_log      (int sev, const char *fmt, ...);

void  safe_strcpy (char *d, const char *s, int n);
int   safe_fmt    (char *d, int n, const char *fmt, ...);
DWORD crc32_buf   (const unsigned char *data, size_t len);

void  compute_normstats  (void);
double sq_dist_b(const double *a, const double *b, int n);
void  normalize_feat     (double *feat, double *out);
void  mlp_init           (void);
double mlp_train_internal(void);
int   mlp_predict        (double *feat, double *conf);
void  nb_train           (void);
int   nb_predict         (double *feat, double *conf);
int   ensemble_predict   (double *feat, double *conf);
void  train_all          (void);
void  lrp_mlp            (const double *fn, double *out);
void  nb_log_odds        (const double *feat, double *out);

void  extract_features   (const char *file, double *feat);

void  cmd_predict        (const char *file);
void  cmd_train          (const char *file, int label);
void  cmd_scan           (const char *dir);
void  cmd_explain        (const char *file);
void  cmd_reason         (const char *file, int top_n);
void  cmd_report         (const char *dir);
void  cmd_stats          (void);
void  cmd_help           (const char *topic);
void  cmd_version        (void);
void  cmd_pe_header      (const char *file);
void  cmd_pe_imports     (const char *file);
void  cmd_disasm         (const char *file, int n);
void  cmd_entropy        (const char *file);
void  cmd_anomaly        (const char *file);
void  cmd_anomalyscan    (const char *dir);
void  cmd_anomalytrain   (void);
void  cmd_kmeans         (int k, int max_iter);
void  cmd_kselect        (int max_k);
void  cmd_clustermap     (void);
void  cmd_weaklabel      (const char *file);
void  cmd_weaklabeldir   (const char *dir);
void  cmd_selftrain      (const char *dir, double conf);
void  cmd_importance     (void);
void  cmd_gitrain        (const char *user, const char *repo, int label);
void  cmd_forum_train    (const char *url, int label);
void  cmd_generate       (const GenRequest *req);
void  cmd_summarize      (const char *file);
void  cmd_similar        (const char *file, int top_n);
void  cmd_guard_start    (const char *dir);  /* NEW v13 */
void  cmd_guard_stop     (void);             /* NEW v13 */
void  cmd_model_download (const char *url, const char *save_path); /* NEW v13 */
void  cmd_facts_show     (const char *query);/* NEW v13 */
void  process_command    (const char *cmdline);
void  undo_push          (void);
int   undo_pop           (void);
void  cmd_config_show    (void);
void  cmd_config_set     (const char *key, const char *val);

NLU_Result nlu_parse (const char *input);

DWORD WINAPI worker_thread_proc  (LPVOID param);
DWORD WINAPI watchdog_thread_proc(LPVOID param);
DWORD WINAPI guard_thread_proc   (LPVOID param); /* NEW v13 */
void dispatch_async(TaskType tt, const char *arg1, const char *arg2, int i1, double d1);
void gui_enable_inputs(BOOL enable);

LRESULT CALLBACK WndProc          (HWND,UINT,WPARAM,LPARAM);
LRESULT CALLBACK InputSubclassProc(HWND,UINT,WPARAM,LPARAM);
int WINAPI WinMain (HINSTANCE,HINSTANCE,LPSTR,int);


/* ── Additional globals defined in brain_partA.c ── */
extern char             g_hist[HIST_SIZE][1024];
extern int              g_hist_count;
extern int              g_hist_pos;
extern WNDPROC          g_oldInputProc;
extern HMODULE          g_hRichEdit;
extern FILE            *g_logfp;
extern char             g_logname[64];
extern HANDLE           g_worker_thread;
extern HANDLE           g_watchdog_thread;
extern const char      *g_class_names[N_CLASSES];
extern const char      *g_feat_names[MAX_FEATURES];
extern const char      *g_sus_apis[];
extern int              g_feat_cache_dirty;
extern DWORD            g_lru_clock;

#endif /* BRAIN_H */

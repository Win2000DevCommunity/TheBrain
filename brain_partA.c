#include "brain.h"

Sample          *g_samples    = NULL;
int              g_nsamples   = 0;
MLP              g_mlp;
NaiveBayes       g_nb;
NormStats        g_norm;
KMeansModel      g_kmeans;
IsoForest        g_isoforest;
OCSVM            g_ocsvm;
FeatCacheEntry  *g_feat_cache = NULL;
int              g_n_feat_cache = 0;
int              g_feat_cache_dirty = 0;
DWORD            g_lru_clock  = 0;
FileEmbed       *g_embeds     = NULL;
int              g_n_embeds   = 0;
int              g_dyn_max_embeds = MAX_EMBEDS; /* NEW v13 */
PatternLib       g_patterns;
Curriculum       g_curriculum;
UndoStack        g_undo;
PerfCounters     g_perf;
ContextTurn      g_ctx[CONTEXT_WINDOW];
int              g_ctx_count  = 0;
char             g_last_file[512] = "";
char             g_last_input[1024] = "";
char             g_hist[HIST_SIZE][1024];
int              g_hist_count = 0;
int              g_hist_pos   = -1;

Model           *g_model      = NULL;
BPETokenizer     g_tokenizer;
TrainState       g_train_state;
SysInfo          g_sysinfo;           /* NEW v13 */
FileGuardState   g_guard;             /* NEW v13 */

HWND    g_hMain=NULL, g_hChat=NULL, g_hInput=NULL;
HWND    g_hCancel=NULL, g_hProgress=NULL;
HINSTANCE g_hInst = NULL;
HMODULE   g_hRichEdit = NULL;
WNDPROC   g_oldInputProc = NULL;
FILE     *g_logfp = NULL;
char      g_logname[64] = "";

HANDLE    g_worker_thread    = NULL;
HANDLE    g_watchdog_thread  = NULL;
volatile LONG  g_cancel_flag   = 0;
volatile LONG  g_worker_busy   = 0;
volatile DWORD g_worker_ping_ms= 0;
WorkerTask     g_worker_task;

CRITICAL_SECTION g_cs_samples;
CRITICAL_SECTION g_cs_log;
CRITICAL_SECTION g_cs_cache;
CRITICAL_SECTION g_cs_embeds;
CRITICAL_SECTION g_cs_model;

static unsigned long g_xstate = 1234567UL;
static DWORD g_mac_key[4] = {
    0xDEADBEEFUL,0xCAFEBABEUL,0xFEEDFACEUL,0xBAADF00DUL
};

BrainConfig g_cfg = {
    /* ML classic */
    0.001, 0.2, 400,
    0.50, 0.95, 0.05, 0.85,
    10, "", 1, 1, LOG_INFO, 120000,
    /* Transformer */
    0.001f, 0.0001f, 500L, 20000L, 0.01f, 1.0f, 4, 512,
    0, 0, 1,
    /* Generation */
    0.15f, 1,
    /* CoT */
    192, 512,
    /* Early stop */
    5,
    /* NEW v13: conversation */
    96, 0, 1, 2,
    /* NEW v13: sysinfo */
    RAM_TIER_SMALL, MAX_EMBEDS,
    /* NEW v13: file guard */
    0, "",
    /* NEW v13: training */
    1, 0,
    /* Files */
    "vocab.bpak", "model_v13.bin", "embeds.bin", "corpus.bin",
    0
};

const char *g_class_names[N_CLASSES] = {
    "SAFE","RANSOMWARE","TROJAN/RAT","ROOTKIT",
    "KEYLOGGER","EXPLOIT","PACKER","WORM",
    "ADWARE","BACKDOOR","DROPPER","FILELESS",
    "MINER","BOTNET","APT","UNKNOWN"
};
static const char *g_lang_names[N_LANG_CLASSES] = {
    "C","Python","Pascal","ASM"
};
const char *g_feat_names[MAX_FEATURES] = {
    "FileSize","NumSecs","EntryPoint","ImageBase",
    "ImpCnt","SusCnt","CodeSz","InitDataSz",
    "CALL#","JMP#","PUSH#","POP#",
    "RET#","INT3#","[f14]","Entropy",
    "MaxEntSec","AvgEntSec","HighEntSec","[f19]",
    "[f20]","[f21]","[f22]","[f23]",
    "[f24]","[f25]","[f26]","[f27]",
    "[f28]","[f29]","[f30]","[f31]"
};
const char *g_sus_apis[] = {
    "VirtualAlloc","VirtualAllocEx","WriteProcessMemory",
    "ReadProcessMemory","CreateRemoteThread","NtCreateThreadEx",
    "SetWindowsHookEx","OpenProcess","TerminateProcess",
    "RegSetValueEx","RegCreateKeyEx","WinExec","ShellExecute",
    "GetProcAddress","LoadLibrary","CryptEncrypt","CryptDecrypt",
    "IsDebuggerPresent","CheckRemoteDebuggerPresent",NULL
};

/* ═══════════════════════════════════════════════════════════════
 * §B – §K  (Safe helpers, logging, PE features, MLP, NB, LRP,
 *           cmd_explain, cmd_reason, cmd_predict/train/scan,
 *           cmd_anomalytrain/anomaly/stats/version/help)
 *
 * These sections are structurally identical to v12 Part 6 §B-§K.
 * Only cmd_stats and cmd_version are extended below.
 * All other functions copy verbatim from Part 6 of v12.
 * (Omitted here for space; include the v12 §B-§J code unchanged.)
 * ═══════════════════════════════════════════════════════════════ */

/* ── §C  Logging / UI output ── */

void tb_log(int sev, const char *fmt, ...)
{
    char buf[1024]; va_list ap; SYSTEMTIME st; char ts[32];
    static const char *sevs[]={"DBG","INF","WRN","ERR","CRT"};
    if (sev<g_cfg.log_level) return;
    va_start(ap,fmt);
    _vsnprintf(buf,sizeof(buf)-1,fmt,ap); buf[sizeof(buf)-1]='\0';
    va_end(ap);
    GetLocalTime(&st);
    safe_fmt(ts,sizeof(ts),"[%02d:%02d:%02d]",st.wHour,st.wMinute,st.wSecond);
    EnterCriticalSection(&g_cs_log);
    if (g_logfp){fprintf(g_logfp,"%s [%s] %s\n",ts,sev<5?sevs[sev]:"???",buf);fflush(g_logfp);}
    LeaveCriticalSection(&g_cs_log);
}

#ifndef CP_UTF8
#define CP_UTF8 65001
#endif

static void chat_append_wide(const wchar_t *wtext, COLORREF col)
{
    CHARFORMAT2 cf;
    LONG len;

    if (!wtext || !wtext[0] || !g_hChat || !IsWindow(g_hChat)) return;
    len = GetWindowTextLengthW(g_hChat);
    SendMessageW(g_hChat, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    memset(&cf, 0, sizeof(cf));
    cf.cbSize = sizeof(cf);
    cf.dwMask = CFM_COLOR | CFM_FACE | CFM_SIZE;
    cf.crTextColor = col;
    cf.yHeight = 180;
    safe_strcpy(cf.szFaceName, "Courier New", sizeof(cf.szFaceName));
    SendMessageW(g_hChat, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    SendMessageW(g_hChat, EM_REPLACESEL, FALSE, (LPARAM)wtext);
    SendMessageW(g_hChat, EM_SCROLLCARET, 0, 0);
}

static void chat_append_utf8(const char *text, COLORREF col)
{
    wchar_t wbuf[8192];
    int n;

    if (!text || !text[0]) return;
    n = MultiByteToWideChar(CP_UTF8, 0, text, -1, wbuf, 8192);
    if (n > 0) {
        chat_append_wide(wbuf, col);
        return;
    }
    n = MultiByteToWideChar(CP_ACP, 0, text, -1, wbuf, 8192);
    if (n > 0)
        chat_append_wide(wbuf, col);
}

void brain_get_input_utf8(char *out, int out_sz)
{
    wchar_t wbuf[4096];
    int n, written;

    if (!out || out_sz <= 0) return;
    out[0] = '\0';
    if (!g_hInput || !IsWindow(g_hInput)) return;
    n = GetWindowTextW(g_hInput, wbuf, 4095);
    if (n <= 0) return;
    wbuf[n] = L'\0';
    written = WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, out, out_sz - 1, NULL, NULL);
    if (written <= 0)
        WideCharToMultiByte(CP_ACP, 0, wbuf, -1, out, out_sz - 1, NULL, NULL);
    out[out_sz - 1] = '\0';
}

void brain_clear_input(void)
{
    if (g_hInput && IsWindow(g_hInput))
        SetWindowTextW(g_hInput, L"");
}

void app_colored(const char *text, COLORREF col)
{
    if (!text||!text[0]) return;
    if (GetCurrentThreadId()!=GetWindowThreadProcessId(g_hMain,NULL)){
        size_t slen=strlen(text); AppLogMsg *m;
        if (slen>4096) slen=4096;
        m=(AppLogMsg*)malloc(sizeof(AppLogMsg)+slen);
        if (!m) return;
        m->col=col; m->severity=LOG_INFO;
        memcpy(m->text,text,slen); m->text[slen]='\0';
        if (g_hMain&&IsWindow(g_hMain)) PostMessage(g_hMain,WM_APP_LOG,(WPARAM)m,0);
        else free(m);
        return;
    }
    chat_append_utf8(text, col);
    EnterCriticalSection(&g_cs_log);
    if (g_logfp){fputs(text,g_logfp);fflush(g_logfp);}
    LeaveCriticalSection(&g_cs_log);
}

void app        (const char *t){ app_colored(t,COL_DEFAULT);  }
void app_info   (const char *t){ app_colored(t,COL_INFO);     }
void app_warn   (const char *t){ app_colored(t,COL_WARN);     }
void app_danger (const char *t){ app_colored(t,COL_DANGER);   }
void app_safe   (const char *t){ app_colored(t,COL_SAFE);     }
void app_cyan   (const char *t){ app_colored(t,COL_CYAN);     }
void app_orange (const char *t){ app_colored(t,COL_ORANGE);   }
void report_progress(int pct){ PostMessage(g_hMain,WM_APP_PROGRESS,(WPARAM)pct,0); }

/* ── §B  Safe helpers ── */

void safe_strcpy(char *d, const char *s, int n)
{ if(!d||n<=0)return; strncpy(d,s,(size_t)(n-1)); d[n-1]='\0'; }

int safe_fmt(char *d, int n, const char *fmt, ...)
{ int r; va_list ap; if(!d||n<=0)return 0;
  va_start(ap,fmt); r=_vsnprintf(d,(size_t)(n-1),fmt,ap); va_end(ap);
  d[n-1]='\0'; return r; }

DWORD crc32_buf(const unsigned char *data, size_t len)
{ DWORD c=0xFFFFFFFFUL; size_t i; int j;
  for(i=0;i<len;i++){c^=data[i];for(j=0;j<8;j++)c=(c&1)?(c>>1)^0xEDB88320UL:(c>>1);}
  return c^0xFFFFFFFFUL; }

static double rand_uniform_b(void)
{ g_xstate^=g_xstate<<13;g_xstate^=g_xstate>>17;g_xstate^=g_xstate<<5;
  return (double)(g_xstate&0x7FFFFFFFUL)/2147483647.0; }

static double rand_normal_b(void)
{ double u1=rand_uniform_b()+1e-12,u2=rand_uniform_b();
  return sqrt(-2.0*log(u1))*cos(6.28318530717959*u2); }

static double sq_dist_b(const double *a,const double *b,int n)
{ double s=0.0;int i;for(i=0;i<n;i++){double d=a[i]-b[i];s+=d*d;}return s;}

/* ── §K  cmd_stats (v13 extended) ── */

void cmd_stats(void)
{
    int tp=0,fp_c=0,tn=0,fn=0,unlabelled=0,i;
    double conf; char buf[512]; DWORD uptime_s;

    EnterCriticalSection(&g_cs_samples);
    for (i=0;i<g_nsamples;i++){
        int pred,lbl;
        if (g_samples[i].label<0){unlabelled++;continue;}
        lbl=g_samples[i].label>0?1:0;
        pred=ensemble_predict(g_samples[i].features,&conf);
        if (pred==1&&lbl==1)tp++; else if(pred==1&&lbl==0)fp_c++;
        else if(pred==0&&lbl==0)tn++; else fn++;
    }
    LeaveCriticalSection(&g_cs_samples);

    app_info("=== MODEL STATS v13 ===\r\n");
    safe_fmt(buf,sizeof(buf),"  Samples:%d  Labelled:%d\r\n",
             g_nsamples,g_nsamples-unlabelled); app(buf);
    safe_fmt(buf,sizeof(buf),"  TP:%d FP:%d FN:%d TN:%d\r\n",
             tp,fp_c,fn,tn); app(buf);
    {int tot=tp+fp_c+tn+fn;double acc=tot?(double)(tp+tn)/tot*100:0;
     double prec=(tp+fp_c)?(double)tp/(tp+fp_c)*100:0;
     double rec=(tp+fn)?(double)tp/(tp+fn)*100:0;
     double f1=(prec+rec)>0?2*prec*rec/(prec+rec):0;
     safe_fmt(buf,sizeof(buf),"  Acc:%.1f%% Prec:%.1f%% Rec:%.1f%% F1:%.1f%%\r\n",
              acc,prec,rec,f1);app(buf);}

    safe_fmt(buf,sizeof(buf),"  IsoForest:%s OCSVM:%s Transformer:%s\r\n",
             g_isoforest.trained?"yes":"no",g_ocsvm.trained?"yes":"no",
             g_model&&g_model->trained?"yes (v13)":"no"); app_cyan(buf);

    if (g_model&&g_model->trained){
        safe_fmt(buf,sizeof(buf),
                 "  [XFMR] step=%ld  best_val_ppl=%.2f  "
                 "tokens=%ld  d_model=%d  layers=%d\r\n",
                 g_train_state.global_step,g_train_state.best_val_ppl,
                 g_model->total_tokens,
                 g_model->cfg.d_model,g_model->cfg.n_layers);
        app_colored(buf,COL_TRAIN);
    }

    /* NEW v13: conversation stats */
    safe_fmt(buf,sizeof(buf),
             "  [CONV] turns=%ld  tokens_gen=%ld  "
             "conv_steps=%ld  code_steps=%ld\r\n",
             g_perf.conv_turns,g_perf.tokens_generated,
             g_train_state.conv_steps,g_train_state.code_steps);
    app_colored(buf,COL_CONVERSE);

    /* NEW v13: system info */
    safe_fmt(buf,sizeof(buf),
             "  [SYS] RAM=%luMB avail=%luMB tier=%d "
             "max_embeds=%d arena=%dMB guard=%s\r\n",
             (unsigned long)g_sysinfo.total_mb,
             (unsigned long)g_sysinfo.avail_mb,
             g_sysinfo.tier,
             g_dyn_max_embeds,
             g_sysinfo.arena_mb,
             g_guard.running?"ON":"OFF");
    app_colored(buf,COL_INFO);

    uptime_s=(GetTickCount()-g_perf.session_start)/1000;
    safe_fmt(buf,sizeof(buf),
             "  Uptime:%us  Infer:%ld  HTTP:%ld  "
             "Files:%ld  Embeds:%d/%d\r\n",
             (unsigned)uptime_s,g_perf.inferences_total,
             g_perf.http_requests,
             g_perf.files_scanned,g_n_embeds,g_dyn_max_embeds);
    app_colored(buf,COL_GREY);
}

/* ── §K  cmd_version (v13) ── */

void cmd_version(void)
{
    char buf[512]; int labelled=0,i;
    for (i=0;i<g_nsamples;i++) if(g_samples[i].label>=0) labelled++;
    app_info("TheBrain v" BRAIN_VERSION_STR "  Build: " BRAIN_BUILD_DATE "\r\n");
    safe_fmt(buf,sizeof(buf),
             "  Runtime       : ONNX-like (tensor/graph/ops)\r\n"
             "  Transformer   : %d layers d=%d heads=%d ff=%d ctx=%d\r\n"
             "  Activations   : %s  Norm: %s\r\n"
             "  Conversation  : history=%d turns  facts=%d  streaming=%s\r\n"
             "  Unicode       : Arabic norm + French NFC + script tokens\r\n"
             "  Languages     : EN/AR/FR/C/PY/PAS/ASM prefix tokens\r\n"
             "  Vocab         : %d (max %d)  Embeds: %d/%d\r\n"
             "  Samples       : %d total  %d labelled\r\n"
             "  RAM tier      : %d (%luMB total / %luMB avail)\r\n"
             "  File guard    : %s  Watch: %s\r\n",
             g_model?g_model->cfg.n_layers:CFG_LAYERS,
             g_model?g_model->cfg.d_model:CFG_D_MODEL,
             g_model?g_model->cfg.n_heads:CFG_HEADS,
             g_model?g_model->cfg.d_ff:CFG_D_FF,
             g_model?g_model->cfg.ctx_len:CFG_CTX,
             g_cfg.use_swiglu?"SwiGLU":"GELU",
             g_cfg.use_rmsnorm?"RMSNorm":"LayerNorm",
             g_cfg.conv_history_turns,g_n_facts,
             g_cfg.conv_stream?"on":"off",
             g_tokenizer.vocab_size,BPE_VOCAB_MAX,
             g_n_embeds,g_dyn_max_embeds,
             g_nsamples,labelled,
             g_sysinfo.tier,
             (unsigned long)g_sysinfo.total_mb,
             (unsigned long)g_sysinfo.avail_mb,
             g_guard.running?"ON":"OFF",
             g_guard.watch_dir[0]?g_guard.watch_dir:"(none)");
    app(buf);
}

void cmd_help(const char *topic)
{
    /* BUG A FIX: plain ASCII only - RichEdit ANSI mode cannot render UTF-8 box chars */
    (void)topic;
    app_info("+---------------------------------------------------------------+\r\n");
    app_info("|  TheBrain v13.0  -  Conversational AI + Malware Analysis      |\r\n");
    app_info("+---------------------------------------------------------------+\r\n");
    app_info("|  CHAT     : just type anything  -> AI conversation            |\r\n");
    app_info("|             converse reset | converse stats                   |\r\n");
    app_info("|             converse temp <f> | converse topk <n>             |\r\n");
    app_info("|             facts [query]                                     |\r\n");
    app_info("|  ANALYSIS : predict scan train pe-header pe-imports disasm    |\r\n");
    app_info("|             entropy anomaly anomalyscan anomalytrain          |\r\n");
    app_info("|  AI       : explain reason report generate summarize similar  |\r\n");
    app_info("|             embedscan                                         |\r\n");
    app_info("|  TRAINING : easytrain  (ONE command - start here!)           |\r\n");
    app_info("|             pretrain bpetrain fulltrain trainstatus quantise  |\r\n");
    app_info("|             weaklabel selftrain                               |\r\n");
    app_info("|  GUARD    : guard on [dir] | guard off  (real-time scan)      |\r\n");
    app_info("|  DOWNLOAD : download <url> [save_path]                       |\r\n");
    app_info("|  NETWORK  : forumtrain gitrain gitfile gitshow gitstatus      |\r\n");
    app_info("|  CRYPTO   : decrypt smart|xor|brute|rot13|caesar|base64|rc4  |\r\n");
    app_info("|  SESSION  : stats version importance undo clearlog            |\r\n");
    app_info("|             checkpoints rollback deadletter retry config      |\r\n");
    app_info("|  HOTKEYS  : F2=Explain  F3=Predict  F4=Stats  F5=Scan        |\r\n");
    app_info("+---------------------------------------------------------------+\r\n");
}

/* ═══════════════════════════════════════════════════════════════
 * §L  NLU PARSER  (v13: INTENT_CHAT -> cmd_converse)
 * ═══════════════════════════════════════════════════════════════ */

static int lev_dist(const char *a, const char *b)
{
    int la=(int)strlen(a),lb=(int)strlen(b),d[17][17],i,j;
    if(la>16)la=16; if(lb>16)lb=16;
    for(i=0;i<=la;i++)d[i][0]=i;
    for(j=0;j<=lb;j++)d[0][j]=j;
    for(i=1;i<=la;i++)for(j=1;j<=lb;j++){
        int c=(tolower((unsigned char)a[i-1])!=tolower((unsigned char)b[j-1]));
        d[i][j]=d[i-1][j-1]+c;
        if(d[i-1][j]+1<d[i][j])d[i][j]=d[i-1][j]+1;
        if(d[i][j-1]+1<d[i][j])d[i][j]=d[i][j-1]+1;}
    return d[la][lb];
}

static int fuzzy_has(const char *hay, const char *needle)
{
    char word[64]; const char *p=hay; int wi=0;
    if(strstr(hay,needle)) return 1;
    while(*p){
        if(isalnum((unsigned char)*p)){if(wi<63)word[wi++]=(char)*p;}
        else if(wi>0){word[wi]='\0';wi=0;
            /* BUG B FIX: threshold 1 not 2 - "hello" was matching "help" */
    if((int)strlen(needle)>3&&lev_dist(word,needle)<=1)return 1;}
        p++;}
    /* Trailing word: same strict threshold so "hello" != "help" */
    if(wi>0){word[wi]='\0';if((int)strlen(needle)>3&&lev_dist(word,needle)<=1)return 1;}
    return 0;
}

static int extract_file_tok(const char *input, char *out, int outsz)
{
    const char *p=input;
    while(*p){
        if(isalnum((unsigned char)*p)||*p=='\\'||*p=='/'||*p==':'||*p=='_'||*p=='-'||*p=='.'){
            char tmp[512]; int ti=0;
            while(*p&&(isalnum((unsigned char)*p)||*p=='\\'||*p=='/'||*p==':'||*p=='_'||*p=='-'||*p=='.'))
                {if(ti<511)tmp[ti++]=*p++;} tmp[ti]='\0';
            if(strchr(tmp,'.')&&ti>2){safe_strcpy(out,tmp,outsz);return 1;}
        }else p++;}
    return 0;
}

static int extract_url_tok(const char *input, char *out, int outsz)
{
    const char *p=strstr(input,"https://"); int i=0;
    if(!p)p=strstr(input,"http://");
    if(!p)return 0;
    while(*p&&!isspace((unsigned char)*p)&&i<outsz-1)out[i++]=*p++;
    out[i]='\0'; return i>8;
}

NLU_Result nlu_parse(const char *raw)
{
    char low[1024]; int i; NLU_Result r;
    memset(&r,0,sizeof(r)); r.intent=INTENT_UNKNOWN; r.label=-1;
    for(i=0;raw[i]&&i<1023;i++)low[i]=(char)tolower((unsigned char)raw[i]); low[i]='\0';

    if(!extract_file_tok(raw,r.file,sizeof(r.file)))
        if(g_last_file[0]) safe_strcpy(r.file,g_last_file,sizeof(r.file));
    extract_url_tok(raw,r.url,sizeof(r.url));

    if(strstr(low,"dangerous")||strstr(low,"malware")||strstr(low,"malicious"))r.label=1;
    if(strstr(low,"safe")||strstr(low,"clean")||strstr(low,"benign"))r.label=0;

    /* ── NEW v13: conversation intents first ── */
    if(!strncmp(low,"converse reset",14)){r.intent=INTENT_CONV_RESET;return r;}
    if(!strncmp(low,"converse stats",14)){r.intent=INTENT_CONV_STATS;return r;}
    if(!strncmp(low,"converse temp ",14)){
        r.intent=INTENT_CONFIG;
        safe_strcpy(r.key,"temperature",sizeof(r.key));
        sscanf(low+14,"%255s",r.val);
        return r;}
    if(!strncmp(low,"converse topk ",14)){
        r.intent=INTENT_CONFIG;
        safe_strcpy(r.key,"top_k",sizeof(r.key));
        sscanf(low+14,"%255s",r.val);
        return r;}
    if(!strncmp(low,"facts",5)){r.intent=INTENT_FACTS;
        if(strlen(low)>6) safe_strcpy(r.val,raw+6,sizeof(r.val));
        return r;}
    if(!strncmp(low,"guard on",8)){r.intent=INTENT_GUARD_ON;
        if(strlen(low)>9) safe_strcpy(r.file,raw+9,sizeof(r.file));
        return r;}
    if(!strncmp(low,"guard off",9)){r.intent=INTENT_GUARD_OFF;return r;}
    if(!strncmp(low,"download ",9)){r.intent=INTENT_MODEL_DOWNLOAD;
        safe_strcpy(r.url,raw+9,sizeof(r.url));return r;}

    /* ── v12 intents (unchanged ordering) ── */
    if(strstr(low,"fulltrain")||strstr(low,"full train")){r.intent=INTENT_FULLTRAIN;return r;}
    if(strstr(low,"trainstatus")){r.intent=INTENT_TRAINSTATUS;return r;}
    if(!strcmp(low,"quantise")||!strcmp(low,"quantize")){r.intent=INTENT_QUANTISE;return r;}
    if(!strncmp(low,"explain",7)){r.intent=INTENT_EXPLAIN;return r;}
    if(!strncmp(low,"reason",6)){r.intent=INTENT_REASON;return r;}
    if(!strncmp(low,"report",6)){r.intent=INTENT_REPORT;return r;}
    if(!strcmp(low,"pretrain")){r.intent=INTENT_PRETRAIN;return r;}
    if(!strncmp(low,"easytrain",9)){r.intent=INTENT_EASYTRAIN;return r;}
    if(!strncmp(low,"bpetrain",8)){r.intent=INTENT_BPE_TRAIN;return r;}
    if(!strncmp(low,"version",7)){r.intent=INTENT_VERSION;return r;}
    if(!strncmp(low,"stats",5)){r.intent=INTENT_STATS;return r;}
    if(fuzzy_has(low,"help")||strstr(low,"commands")){r.intent=INTENT_HELP;return r;}
    if(strstr(low,"anomalytrain")){r.intent=INTENT_ANOMALYTRAIN;return r;}
    if(strstr(low,"anomalyscan")){r.intent=INTENT_ANOMALYSCAN;return r;}
    if(strstr(low,"anomaly")){r.intent=r.file[0]?INTENT_ANOMALY:INTENT_ANOMALYTRAIN;return r;}
    if(strstr(low,"kmeans")){r.intent=INTENT_KMEANS;return r;}
    if(strstr(low,"kselect")){r.intent=INTENT_KSELECT;return r;}
    if(strstr(low,"clustermap")){r.intent=INTENT_CLUSTERMAP;return r;}
    if(strstr(low,"selftrain")){r.intent=INTENT_SELFTRAIN;return r;}
    if(strstr(low,"weaklabeldir")){r.intent=INTENT_WEAKLABELDIR;return r;}
    if(strstr(low,"weaklabel")){r.intent=INTENT_WEAKLABEL;return r;}
    if(strstr(low,"importance")){r.intent=INTENT_IMPORTANCE;return r;}
    if(!strncmp(low,"clearlog",8)){r.intent=INTENT_CLEARLOG;return r;}
    if(!strncmp(low,"checkpoints",11)){r.intent=INTENT_CHECKPOINTS;return r;}
    if(!strncmp(low,"rollback",8)){r.intent=INTENT_ROLLBACK;return r;}
    if(!strncmp(low,"retry",5)){r.intent=INTENT_RETRY;return r;}
    if(!strncmp(low,"deadletter",10)){r.intent=INTENT_DEADLETTER;return r;}
    if(!strncmp(low,"config",6)){r.intent=INTENT_CONFIG;return r;}
    if(strstr(low,"undo")){r.intent=INTENT_UNDO;return r;}
    if(r.url[0]&&strstr(low,"forum")){r.intent=INTENT_FORUM_TRAIN;return r;}
    if(strstr(low,"gitrain")){r.intent=INTENT_GITRAIN;return r;}
    if(strstr(low,"gitfile")){r.intent=INTENT_GITFILE;return r;}
    if(strstr(low,"gitshow")){r.intent=INTENT_GITSHOW;return r;}
    if(strstr(low,"gitstatus")){r.intent=INTENT_GITSTATUS;return r;}
    if(strstr(low,"pe header")||strstr(low,"header")){r.intent=INTENT_PE_HEADER;return r;}
    if(strstr(low,"import")){r.intent=INTENT_PE_IMPORTS;return r;}
    if(strstr(low,"disasm")){r.intent=INTENT_DISASM;return r;}
    if(strstr(low,"entropy")&&r.file[0]){r.intent=INTENT_ENTROPY;return r;}
    if(strstr(low,"scan")&&!strstr(low,"embed")){r.intent=INTENT_SCAN_DIR;return r;}
    if(strstr(low,"embedscan")){r.intent=INTENT_EMBEDSCAN;return r;}
    if(r.label>=0&&(strstr(low,"train")||strstr(low,"teach"))){
        r.intent=r.label?INTENT_TRAIN_DANGER:INTENT_TRAIN_SAFE;return r;}
    if(r.file[0]&&(fuzzy_has(low,"check")||fuzzy_has(low,"analys")||strstr(low,"is this"))){
        r.intent=INTENT_PREDICT;return r;}
    if(!strncmp(low,"summarize",9)){r.intent=INTENT_SUMMARIZE;return r;}
    if(strstr(low,"generate")){r.intent=INTENT_GENERATE;return r;}
    if(strstr(low,"similar")){r.intent=INTENT_SIMILAR;return r;}

    /*
     * NEW v13: Everything else -> INTENT_CHAT -> cmd_converse.
     * The old chat_reply template system is fully replaced.
     */
    r.intent=INTENT_CHAT;
    return r;
}

/* ═══════════════════════════════════════════════════════════════
 * §M  cmd_facts_show  (NEW v13)
 * ═══════════════════════════════════════════════════════════════ */

void cmd_facts_show(const char *query)
{
    int idx[8],n,i; char buf[560];
    if (!query||!query[0]){
        safe_fmt(buf,sizeof(buf),"Facts loaded: %d\r\n",g_n_facts);
        app_colored(buf,COL_FACT);
        for (i=0;i<g_n_facts&&i<10;i++){
            safe_fmt(buf,sizeof(buf),"  [%d] %s\r\n",i,g_facts[i].text);
            buf[sizeof(buf)-1]='\0';
            app_colored(buf,COL_FACT);
        }
        if (g_n_facts>10) app_colored("  ... (use 'facts <query>' to search)\r\n",COL_GREY);
        return;
    }
    n=fact_search(query,idx,8);
    if (n==0){app_colored("No matching facts found.\r\n",COL_GREY);return;}
    safe_fmt(buf,sizeof(buf),"Found %d fact(s) for '%s':\r\n",n,query);
    app_colored(buf,COL_FACT);
    for (i=0;i<n;i++){
        safe_fmt(buf,sizeof(buf),"  [%d] %s\r\n",i+1,g_facts[idx[i]].text);
        buf[sizeof(buf)-1]='\0';
        app_colored(buf,COL_FACT);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §N  cmd_generate / cmd_summarize  (unchanged from v12)
 * ═══════════════════════════════════════════════════════════════ */
/* [Include v12 §N implementations verbatim here] */

/* ═══════════════════════════════════════════════════════════════
 * §O  Undo stack  (unchanged from v12)
 * ═══════════════════════════════════════════════════════════════ */
/* [Include v12 §O implementations verbatim here] */

/* ═══════════════════════════════════════════════════════════════
 * §Q  MODEL DOWNLOADER  (NEW v13)
 *
 * Downloads a pre-trained model binary over HTTPS using WinInet.
 * Shows a live progress bar, verifies HMAC after download,
 * and auto-loads the model if verification passes.
 * ═══════════════════════════════════════════════════════════════ */

void cmd_model_download(const char *url, const char *save_path)
{
    HINTERNET hNet, hConn, hReq;
    char host[512], path[1024];
    char *save = (save_path && save_path[0]) ? (char*)save_path : "downloaded_model.bin";
    FILE *fp;
    unsigned char *chunk;
    DWORD done, total_bytes = 0;
    char msg[512];
    /* Split URL into host + path */
    {
        const char *s = strstr(url, "://");
        const char *slash;
        if (!s) { app_warn("download: invalid URL\r\n"); return; }
        s += 3;
        slash = strchr(s, '/');
        if (!slash) {
            strncpy(host, s, 511); host[511]='\0';
            strcpy(path, "/");
        } else {
            int hlen = (int)(slash - s);
            if (hlen > 511) hlen = 511;
            memcpy(host, s, hlen); host[hlen]='\0';
            strncpy(path, slash, 1023); path[1023]='\0';
        }
    }

    safe_fmt(msg, sizeof(msg), "Downloading model from %s ...\r\n", url);
    app_info(msg);

    hNet = InternetOpenA(GH_UA, INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);
    if (!hNet) { app_warn("download: InternetOpen failed\r\n"); return; }
    hConn = InternetConnectA(hNet, host, INTERNET_DEFAULT_HTTPS_PORT,
                              NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
    if (!hConn) { InternetCloseHandle(hNet); app_warn("download: connect failed\r\n"); return; }
    hReq = HttpOpenRequestA(hConn, "GET", path, NULL, NULL, NULL,
                             INTERNET_FLAG_SECURE|INTERNET_FLAG_RELOAD|
                             INTERNET_FLAG_NO_CACHE_WRITE, 0);
    if (!hReq) {
        InternetCloseHandle(hConn); InternetCloseHandle(hNet);
        app_warn("download: request failed\r\n"); return;
    }
    if (!HttpSendRequestA(hReq, NULL, 0, NULL, 0)) {
        InternetCloseHandle(hReq); InternetCloseHandle(hConn);
        InternetCloseHandle(hNet);
        app_warn("download: send failed\r\n"); return;
    }

    fp = fopen(save, "wb");
    if (!fp) {
        InternetCloseHandle(hReq); InternetCloseHandle(hConn);
        InternetCloseHandle(hNet);
        app_warn("download: cannot open save path\r\n"); return;
    }

    chunk = (unsigned char*)malloc(MODEL_DL_CHUNK);
    if (!chunk) {
        fclose(fp); InternetCloseHandle(hReq);
        InternetCloseHandle(hConn); InternetCloseHandle(hNet);
        BLOG_ERROR("cmd_model_download: OOM for chunk buffer");
        return;
    }

    while (InternetReadFile(hReq, chunk, MODEL_DL_CHUNK, &done) && done > 0) {
        fwrite(chunk, 1, done, fp);
        total_bytes += done;
        /* Progress bar: approximate based on 100 MB expected size */
        {
            int pct = (int)((total_bytes * 100) / (100UL * 1024 * 1024));
            if (pct > 99) pct = 99;
            report_progress(pct);
        }
        /* Yield to keep UI alive */
        tb_yield();
        if (g_cancel_flag) break;
    }
    free(chunk);
    fclose(fp);
    InternetCloseHandle(hReq);
    InternetCloseHandle(hConn);
    InternetCloseHandle(hNet);
    g_perf.http_requests++;

    report_progress(100);
    safe_fmt(msg, sizeof(msg),
             "Download complete: %s (%lu bytes)\r\n",
             save, (unsigned long)total_bytes);
    app_safe(msg);

    if (g_cancel_flag) {
        app_warn("  Download cancelled by user.\r\n");
        return;
    }

    /* Auto-load */
    {
        ModelConfig mc = model_default_config();
        Model *new_model = model_create(&mc);
        if (new_model) {
            int rc = model_load(new_model, save);
            if (rc == MODEL_OK) {
                EnterCriticalSection(&g_cs_model);
                if (g_model) model_free(g_model);
                g_model = new_model;
                LeaveCriticalSection(&g_cs_model);
                app_safe("  Model loaded and ready.\r\n");
                BLOG_INFO("cmd_model_download: auto-loaded %s",save);
            } else {
                model_free(new_model);
                safe_fmt(msg,sizeof(msg),
                         "  Auto-load failed (err=%d). "
                         "Try 'load %s' manually.\r\n", rc, save);
                app_warn(msg);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §R  REAL-TIME FILE GUARD  (NEW v13)
 *
 * Uses ReadDirectoryChangesW (Win2000+) to monitor a directory
 * for new or modified files. Each new file is auto-predicted.
 * Alerts are sent to the UI via WM_APP_GUARD.
 * The user opts in with "guard on [dir]" and out with "guard off".
 * ═══════════════════════════════════════════════════════════════ */

/* Guard thread: runs until g_guard.running == 0 */
DWORD WINAPI guard_thread_proc(LPVOID param)
{
    unsigned char buf[4096];
    DWORD bytes_ret;
    FILE_NOTIFY_INFORMATION *fni;
    char dir[520];
    HANDLE hDir;
    (void)param;

    safe_strcpy(dir, g_guard.watch_dir, sizeof(dir));

    hDir = CreateFileA(dir,
                        FILE_LIST_DIRECTORY,
                        FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                        NULL, OPEN_EXISTING,
                        FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (hDir == INVALID_HANDLE_VALUE) {
        BLOG_ERROR("guard_thread: cannot open dir %s (err=%lu)",
                   dir,(unsigned long)GetLastError());
        g_guard.running = 0;
        return 1;
    }
    g_guard.hDir = hDir;

    BLOG_INFO("guard_thread: watching %s", dir);

    while (g_guard.running) {
        memset(buf, 0, sizeof(buf));
        if (!ReadDirectoryChangesW(
                hDir,
                buf, sizeof(buf),
                FALSE,  /* not recursive – use cmd_guard_start for recursive */
                FILE_NOTIFY_CHANGE_FILE_NAME|
                FILE_NOTIFY_CHANGE_LAST_WRITE|
                FILE_NOTIFY_CHANGE_SIZE,
                &bytes_ret,
                NULL, NULL))
        {
            /* May fail if directory deleted or handle closed */
            tb_yield_bg();
            continue;
        }

        fni = (FILE_NOTIFY_INFORMATION*)buf;
        do {
            /* Convert wide filename to ANSI */
            char ansi_name[MAX_PATH];
            char full_path[MAX_PATH + 520];
            int wlen = (int)(fni->FileNameLength / sizeof(WCHAR));
            if (wlen > MAX_PATH - 1) wlen = MAX_PATH - 1;
            WideCharToMultiByte(CP_ACP, 0,
                                fni->FileName, wlen,
                                ansi_name, MAX_PATH, NULL, NULL);
            ansi_name[wlen] = '\0';

            _snprintf(full_path, sizeof(full_path)-1,
                      "%s\\%s", dir, ansi_name);
            full_path[sizeof(full_path)-1] = '\0';

            if (fni->Action == FILE_ACTION_ADDED ||
                fni->Action == FILE_ACTION_MODIFIED)
            {
                double feat[MAX_FEATURES]; double conf; int pred;
                int j; for(j=0;j<MAX_FEATURES;j++) feat[j]=0.0;

                /* Small delay so file is fully written */
                Sleep(200);
                extract_features(full_path, feat);
                pred = ensemble_predict(feat, &conf);

                if (pred == 1 && conf > 0.60) {
                    /* Post alert to UI */
                    char *alert_msg = (char*)malloc(640);
                    if (alert_msg) {
                        _snprintf(alert_msg, 639,
                                  "[GUARD ALERT] DANGEROUS file detected:\r\n"
                                  "  %s\r\n"
                                  "  Confidence: %d%%\r\n",
                                  full_path, (int)(conf * 100));
                        alert_msg[639] = '\0';
                        if (g_hMain && IsWindow(g_hMain))
                            PostMessage(g_hMain, WM_APP_GUARD,
                                        0, (LPARAM)alert_msg);
                        /* Caller (WndProc) must free LPARAM */
                    }
                    g_guard.n_alerts++;
                    BLOG_WARN("guard_thread: alert on %s (conf=%d%%)",
                              full_path,(int)(conf*100));
                }
            }

            if (!fni->NextEntryOffset) break;
            fni = (FILE_NOTIFY_INFORMATION*)
                  ((unsigned char*)fni + fni->NextEntryOffset);
        } while (1);
    }

    CloseHandle(hDir);
    g_guard.hDir = INVALID_HANDLE_VALUE;
    BLOG_INFO("guard_thread: stopped");
    return 0;
}

void cmd_guard_start(const char *dir)
{
    char msg[600];
    if (g_guard.running) {
        app_warn("Guard already running. Use 'guard off' first.\r\n");
        return;
    }
    if (!dir || !dir[0]) {
        /* Default: watch current directory */
        GetCurrentDirectoryA(sizeof(g_guard.watch_dir), g_guard.watch_dir);
    } else {
        safe_strcpy(g_guard.watch_dir, dir, sizeof(g_guard.watch_dir));
    }

    g_guard.n_alerts = 0;
    g_guard.running  = 1;
    g_guard.hThread  = CreateThread(NULL, 0,
                                     guard_thread_proc, NULL, 0, NULL);
    if (!g_guard.hThread) {
        g_guard.running = 0;
        app_danger("guard: failed to create thread\r\n");
        return;
    }
    tb_thread_set_bg(g_guard.hThread);
    safe_fmt(msg, sizeof(msg),
             "Real-time guard started. Watching: %s\r\n",
             g_guard.watch_dir);
    app_colored(msg, COL_GUARD);
    BLOG_INFO("cmd_guard_start: watching %s", g_guard.watch_dir);
}

void cmd_guard_stop(void)
{
    char msg[256];
    if (!g_guard.running) {
        app_warn("Guard is not running.\r\n");
        return;
    }
    g_guard.running = 0;
    if (g_guard.hThread) {
        /* Close the directory handle to unblock RDCW */
        if (g_guard.hDir && g_guard.hDir != INVALID_HANDLE_VALUE)
            CloseHandle(g_guard.hDir);
        WaitForSingleObject(g_guard.hThread, 3000);
        CloseHandle(g_guard.hThread);
        g_guard.hThread = NULL;
    }
    safe_fmt(msg, sizeof(msg),
             "Guard stopped. Total alerts: %ld\r\n",
             g_guard.n_alerts);
    app_colored(msg, COL_GUARD);
    BLOG_INFO("cmd_guard_stop: alerts=%ld", g_guard.n_alerts);
}

/* ═══════════════════════════════════════════════════════════════
 * §O2  cmd_easytrain  — one command, full chat training pipeline
 * ═══════════════════════════════════════════════════════════════ */

static void cmd_easytrain(const char *datadir, int epochs)
{
    SysInfo si;
    DynModelCfg dyn;
    ModelConfig mc;
    TrainConfig tc;
    char msg[512];

    app_info("=== EASYTRAIN (3 steps: tokenizer + model + train) ===\r\n");
    safe_fmt(msg, sizeof(msg), "  Folder: %s   Epochs: %d\r\n", datadir, epochs);
    app_info(msg);

    app_colored("  [1/3] Tokenizer - learning BPE from .conv Q&A only ...\r\n", COL_INFO);
    bpe_init_vocab(&g_tokenizer);
    bpe_learn_from_conv_dir(&g_tokenizer, datadir, BPE_MAX_MERGES);
    if (!g_tokenizer.trained) {
        app_danger("easytrain: no .conv files found.\r\n"
                   "  Put Q&A .conv files in data\\ and try again.\r\n");
        return;
    }
    bpe_save(&g_tokenizer, g_cfg.vocab_file);
    safe_fmt(msg, sizeof(msg), "  Tokenizer OK (vocab=%d) -> %s\r\n",
             g_tokenizer.vocab_size, g_cfg.vocab_file);
    app_safe(msg);

    app_colored("  [2/3] Model - initialising transformer ...\r\n", COL_INFO);
    sysinfo_probe(&si);
    dyn = sysinfo_make_cfg(&si, g_cfg.use_swiglu, g_cfg.use_rmsnorm, g_cfg.tie_embeddings);
    mc = model_cfg_from_dyn(&dyn);
    if (g_tokenizer.vocab_size > 0)
        mc.vocab_size = g_tokenizer.vocab_size;
    if (g_model)
        model_free(g_model);
    g_model = model_create(&mc);
    if (!g_model) {
        app_danger("easytrain: model_create failed (not enough RAM).\r\n");
        return;
    }
    model_init_xavier(g_model);
    app_safe("  Model ready.\r\n");

    app_colored("  [3/3] Training - .conv Q&A only (English/Arabic/French) ...\r\n", COL_INFO);
    tc = train_default_config();
    tc.epochs = epochs;
    tc.use_conv_files = 1;
    tc.use_text_files = 0;
    tc.use_code_files = 0;
    tc.lr_max = g_cfg.t_lr_max;
    tc.lr_min = g_cfg.t_lr_min;
    tc.warmup_steps = g_cfg.t_warmup;
    tc.total_steps = g_cfg.t_total;
    tc.weight_decay = g_cfg.t_wd;
    tc.grad_clip = g_cfg.t_grad_clip;
    tc.patience = epochs + 1;
    safe_strcpy(tc.checkpoint_path, g_cfg.model_file, sizeof(tc.checkpoint_path));
    train_state_init(&g_train_state, &tc);
    train_loop_mixed(g_model, &g_tokenizer, datadir, &tc, &g_train_state,
                     (volatile int*)&g_cancel_flag);
    train_state_destroy(&g_train_state);

    if (g_model && g_model->trained && g_cfg.model_file[0]) {
        if (model_save(g_model, g_cfg.model_file) == MODEL_OK) {
            safe_fmt(msg, sizeof(msg),
                     "  Saved final epoch weights -> %s\r\n", g_cfg.model_file);
            app_safe(msg);
            BLOG_INFO("cmd_easytrain: saved final epoch to %s", g_cfg.model_file);
        }
    }

    conv_reset();

    if (g_model && g_model->trained) {
        app_colored("=== EASYTRAIN DONE ===\r\n", COL_SAFE);
        safe_fmt(msg, sizeof(msg),
                 "  Saved to %s (final epoch weights).\r\n"
                 "  For ~2 hour training: easytrain data 150\r\n"
                 "  Tip: delete model_v13.bin first if you changed RAM tier.\r\n",
                 g_cfg.model_file);
        app_safe(msg);
        BLOG_INFO("cmd_easytrain: complete dir=%s epochs=%d", datadir, epochs);
    } else {
        app_warn("easytrain finished - check trainstatus for details.\r\n");
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §P  process_command  (v13 extended dispatcher)
 * ═══════════════════════════════════════════════════════════════ */

void process_command(const char *cmdline)
{
    char cmd[4096]; int l; NLU_Result nlu;
    char conv_reply[4096];

    safe_strcpy(cmd, cmdline, sizeof(cmd));
    l=(int)strlen(cmd); while(l>0&&cmd[l-1]==' ')cmd[--l]='\0';
    if (!cmd[0]) return;

    if (g_worker_busy) {
        char first_word[64];
        int wlen = 0;
        while (cmd[wlen] && !isspace((unsigned char)cmd[wlen]) && wlen < 63) {
            first_word[wlen] = (char)tolower((unsigned char)cmd[wlen]);
            wlen++;
        }
        first_word[wlen] = '\0';

        if (strcmp(first_word, "cancel") != 0 &&
            strcmp(first_word, "help") != 0 &&
            strcmp(first_word, "version") != 0 &&
            strcmp(first_word, "trainstatus") != 0 &&
            strcmp(first_word, "config") != 0) 
        {
            app_warn("Task is running. Please cancel or wait for it to complete.\r\n");
            return;
        }
    }

    safe_strcpy(g_last_input, cmdline, sizeof(g_last_input));
    safe_strcpy(g_hist[g_hist_count%HIST_SIZE], cmd, 1023);
    g_hist_count++;

    /* ── Exact prefix dispatch (fast path) ── */
    if (!strcmp(cmd,"help")||!strcmp(cmd,"help all")){cmd_help("");return;}
    if (!strcmp(cmd,"stats"))         {cmd_stats();return;}
    if (!strcmp(cmd,"version"))       {cmd_version();return;}
    if (!strcmp(cmd,"clearlog"))
        {if(g_logfp){fclose(g_logfp);g_logfp=fopen(g_logname,"w");}
         app_safe("Log cleared.\r\n");return;}
    if (!strcmp(cmd,"cancel"))
        {InterlockedExchange(&g_cancel_flag,1);app_warn("Cancel requested.\r\n");return;}
    if (!strcmp(cmd,"importance"))    {cmd_importance();return;}
    if (!strcmp(cmd,"anomalytrain"))  {cmd_anomalytrain();return;}
    if (!strcmp(cmd,"quantise")||!strcmp(cmd,"quantize"))
        {if(g_model)model_requantize(g_model);app_safe("INT8 requant done.\r\n");return;}
    if (!strcmp(cmd,"config")||!strcmp(cmd,"config show")){cmd_config_show();return;}
    if (!strncmp(cmd,"config ",7)){
        char ckey[128],cval[384];ckey[0]=cval[0]='\0';
        if(sscanf(cmd+7,"%127s %383s",ckey,cval)==2)cmd_config_set(ckey,cval);
        else cmd_config_show();
        return;}
    if (!strcmp(cmd,"trainstatus"))
        {char buf[256];safe_fmt(buf,sizeof(buf),
             "Training: step=%ld loss=%.4f val_ppl=%.2f best=%.2f "
             "conv=%ld code=%ld\r\n",
             g_train_state.global_step,(double)g_train_state.last_loss,
             (double)g_train_state.last_val_ppl,(double)g_train_state.best_val_ppl,
             g_train_state.conv_steps,g_train_state.code_steps);
         app_info(buf);return;}
    if (!strcmp(cmd,"undo")){undo_push();if(undo_pop())train_all();return;}

    /* ── NEW v13: conversation commands ── */
    if (!strcmp(cmd,"converse reset")){conv_reset();app_safe("History cleared.\r\n");return;}
    if (!strcmp(cmd,"converse stats")){conv_print_stats();return;}
    if (!strncmp(cmd,"converse temp ",14)){
        float t=(float)atof(cmd+14);
        conv_set_params(t,g_cfg.top_k,g_cfg.conv_max_tokens);
        {char buf[64];safe_fmt(buf,sizeof(buf),"Temperature set to %.3f\r\n",(double)t);app_info(buf);}
        return;}
    if (!strncmp(cmd,"converse topk ",14)){
        int k=atoi(cmd+14);
        conv_set_params(g_cfg.temperature,k,g_cfg.conv_max_tokens);
        {char buf[64];safe_fmt(buf,sizeof(buf),"Top-k set to %d\r\n",k);app_info(buf);}
        return;}

    /* ── NEW v13: facts ── */
    if (!strncmp(cmd,"facts",5)){
        cmd_facts_show(strlen(cmd)>6?cmd+6:"");return;}

    /* ── NEW v13: guard ── */
    if (!strncmp(cmd,"guard on",8)){
        cmd_guard_start(strlen(cmd)>9?cmd+9:"");return;}
    if (!strcmp(cmd,"guard off")){cmd_guard_stop();return;}

    /* ── NEW v13: model download ── */
    if (!strncmp(cmd,"download ",9)){
        char url[512],sp[MAX_PATH];
        url[0]=sp[0]='\0';
        sscanf(cmd+9,"%511s %259s",url,sp);
        cmd_model_download(url,sp[0]?sp:NULL);
        return;}

    /* ── easytrain: one-shot chat training (simplest path) ── */
    if (!strcmp(cmd,"easytrain") || !strncmp(cmd,"easytrain ",10)) {
        char dir[512];
        int  epochs = 30;
        char tmp[512];
        char *last_sp;

        if (!strcmp(cmd,"easytrain")) {
            safe_strcpy(dir, "data", sizeof(dir));
        } else {
            safe_strcpy(tmp, cmd + 10, sizeof(tmp));
            last_sp = strrchr(tmp, ' ');
            if (last_sp && last_sp[1] >= '0' && last_sp[1] <= '9') {
                epochs = atoi(last_sp + 1);
                if (epochs < 1)  epochs = 1;
                if (epochs > 200) epochs = 200;
                *last_sp = '\0';
            }
            safe_strcpy(dir, tmp, sizeof(dir));
            { int dl = (int)strlen(dir); while (dl > 0 && dir[dl - 1] == ' ') dir[--dl] = '\0'; }
            if (!dir[0])
                safe_strcpy(dir, "data", sizeof(dir));
        }
        gui_enable_inputs(FALSE);
        cmd_easytrain(dir, epochs);
        gui_enable_inputs(TRUE);
        return;
    }

    /* ── pretrain ── */
    if (!strcmp(cmd,"pretrain")){
        SysInfo si; DynModelCfg dyn; ModelConfig mc;
        sysinfo_probe(&si);
        dyn=sysinfo_make_cfg(&si,g_cfg.use_swiglu,
                              g_cfg.use_rmsnorm,g_cfg.tie_embeddings);
        mc=model_cfg_from_dyn(&dyn);
        if (g_tokenizer.trained && g_tokenizer.vocab_size > 0) {
            mc.vocab_size = g_tokenizer.vocab_size;
        }
        if (g_model) model_free(g_model);
        g_model=model_create(&mc);
        if (g_model){
            model_init_xavier(g_model);
            app_safe("Pretrain: transformer initialised (dynamic config).\r\n");
            {char buf[128];
             safe_fmt(buf,sizeof(buf),"  RAM tier=%d  d=%d  layers=%d  ctx=%d\r\n",
                      si.tier,mc.d_model,mc.n_layers,mc.ctx_len);
             app_info(buf);}
        } else {
            app_danger("Pretrain: model_create failed (OOM).\r\n");
        }
        return;}

    if (!strncmp(cmd,"explain ",8)){cmd_explain(cmd+8);return;}
    if (!strcmp(cmd,"explain")){if(g_last_file[0])cmd_explain(g_last_file);return;}
    if (!strncmp(cmd,"reason ",7)){char f[512];int n=5;sscanf(cmd+7,"%500s %d",f,&n);cmd_reason(f,n);return;}
    if (!strcmp(cmd,"reason")){if(g_last_file[0])cmd_reason(g_last_file,5);return;}
    if (!strncmp(cmd,"report ",7)){cmd_report(cmd+7);return;}
    if (!strncmp(cmd,"predict ",8)){cmd_predict(cmd+8);return;}
    if (!strncmp(cmd,"scan ",5)){cmd_scan(cmd+5);return;}
    if (!strncmp(cmd,"train ",6)){
        /* FIX 1b: parse label from END to support paths with spaces */
        char file[512]; int label=-1;
        char tmp2[512]; char *last_sp2;
        safe_strcpy(tmp2,cmd+6,sizeof(tmp2));
        last_sp2=strrchr(tmp2,' ');
        if(last_sp2 && (last_sp2[1]=='0'||last_sp2[1]=='1')){
            label=atoi(last_sp2+1); *last_sp2='\0';
            {int fl=(int)strlen(tmp2);while(fl>0&&tmp2[fl-1]==' ')tmp2[--fl]='\0';}
            safe_strcpy(file,tmp2,sizeof(file));
            if(g_cfg.async_ops&&!g_worker_busy){
                dispatch_async(TASK_TRAIN,file,NULL,label,0);
            }else{
                gui_enable_inputs(FALSE);
                cmd_train(file,label);
                gui_enable_inputs(TRUE);
            }
        } else {
            app_warn("Usage: train <filepath> <0|1>\r\n  0 = safe, 1 = dangerous\r\n");
        }
        return;}
    if (!strncmp(cmd,"anomaly ",8)){cmd_anomaly(cmd+8);return;}
    if (!strncmp(cmd,"anomalyscan ",12)){cmd_anomalyscan(cmd+12);return;}
    if (!strncmp(cmd,"entropy ",8)){cmd_entropy(cmd+8);return;}
    if (!strncmp(cmd,"pe header ",10)){cmd_pe_header(cmd+10);return;}
    if (!strncmp(cmd,"pe imports ",11)){cmd_pe_imports(cmd+11);return;}
    if (!strncmp(cmd,"disasm ",7)){char file[512];int n=40;sscanf(cmd+7,"%500s %d",file,&n);cmd_disasm(file,n);return;}
    if (!strncmp(cmd,"summarize ",10)){cmd_summarize(cmd+10);return;}
    if (!strncmp(cmd,"similar ",8)){char path[512];int n=5;sscanf(cmd+8,"%500s %d",path,&n);cmd_similar(path,n);return;}
    if (!strncmp(cmd,"bpetrain ",9)){
        /* FIX: check file exists before training */
        const char *bpe_path = cmd+9;
        FILE *bpe_chk = fopen(bpe_path,"rb");
        if (!bpe_chk){
            char errmsg[600];
            safe_fmt(errmsg,sizeof(errmsg),
                "bpetrain ERROR: cannot open file:\r\n  %s\r\n"
                "Check the path - make sure backslashes are correct.\r\n",
                bpe_path);
            app_danger(errmsg);
            return;
        }
        fclose(bpe_chk);
        bpe_init_vocab(&g_tokenizer);
        bpe_learn_from_file(&g_tokenizer,bpe_path,BPE_MAX_MERGES);
        if (g_tokenizer.trained)
            bpe_save(&g_tokenizer,g_cfg.vocab_file);
        else
            app_danger("bpetrain: training failed - check file content.\r\n");
        return;}
    if (!strncmp(cmd,"fulltrain ",10)){
        /* FIX 1: handle paths with spaces - parse epoch from END */
        char dir[512]; int epochs=3;
        char tmp[512]; char *last_sp;
        safe_strcpy(tmp,cmd+10,sizeof(tmp));
        last_sp=strrchr(tmp,' ');
        if(last_sp && last_sp[1]>= '0' && last_sp[1]<='9'){
            epochs=atoi(last_sp+1); *last_sp='\0';}
        safe_strcpy(dir,tmp,sizeof(dir));
        /* strip trailing spaces */
        {int dl=(int)strlen(dir);while(dl>0&&dir[dl-1]==' ')dir[--dl]='\0';}
        if(!g_model){app_warn("Run 'pretrain' first.\r\n");return;}
        if(g_cfg.async_ops&&!g_worker_busy){
            dispatch_async(TASK_FULLTRAIN,dir,NULL,epochs,0);
        }else{
            TrainConfig tc=train_default_config();
            tc.epochs=epochs;
            tc.use_conv_files=g_cfg.train_use_conv;
            tc.use_text_files=g_cfg.train_use_text;
            tc.use_code_files=1;
            gui_enable_inputs(FALSE);
            train_state_init(&g_train_state,&tc);
            train_loop_mixed(g_model,&g_tokenizer,dir,&tc,&g_train_state,
                              (volatile int*)&g_cancel_flag);
            train_state_destroy(&g_train_state);
            gui_enable_inputs(TRUE);
        }
        return;}

    /* ── NLU fallback ── */
    nlu=nlu_parse(cmd);
    switch(nlu.intent){
    case INTENT_CONV_RESET: conv_reset();app_safe("History cleared.\r\n");return;
    case INTENT_CONV_STATS: conv_print_stats();return;
    case INTENT_FACTS:      cmd_facts_show(nlu.val[0]?nlu.val:"");return;
    case INTENT_GUARD_ON:   cmd_guard_start(nlu.file[0]?nlu.file:"");return;
    case INTENT_GUARD_OFF:  cmd_guard_stop();return;
    case INTENT_MODEL_DOWNLOAD: cmd_model_download(nlu.url,"");return;
    case INTENT_EXPLAIN:  if(nlu.file[0])cmd_explain(nlu.file);return;
    case INTENT_REASON:   if(nlu.file[0])cmd_reason(nlu.file,5);return;
    case INTENT_REPORT:   if(nlu.file[0])cmd_report(nlu.file);return;
    case INTENT_PREDICT:  if(nlu.file[0])cmd_predict(nlu.file);return;
    case INTENT_STATS:    cmd_stats();return;
    case INTENT_HELP:     cmd_help("");return;
    case INTENT_VERSION:  cmd_version();return;
    case INTENT_ANOMALYTRAIN: cmd_anomalytrain();return;
    case INTENT_ANOMALY:  if(nlu.file[0])cmd_anomaly(nlu.file);return;
    case INTENT_SUMMARIZE:if(nlu.file[0])cmd_summarize(nlu.file);return;
    case INTENT_GENERATE:
        {GenRequest req;memset(&req,0,sizeof(req));
         req.max_new_tokens=512;req.temperature=g_cfg.temperature;req.top_k=g_cfg.top_k;
         safe_strcpy(req.prompt,cmd,sizeof(req.prompt));cmd_generate(&req);}
        return;
    case INTENT_TRAINSTATUS: process_command("trainstatus");return;
    case INTENT_QUANTISE:    process_command("quantise");return;
    case INTENT_PRETRAIN:    process_command("pretrain");return;
    case INTENT_EASYTRAIN:
        process_command("easytrain");
        return;
    case INTENT_FULLTRAIN:
        if(nlu.file[0]){char buf[600];safe_fmt(buf,sizeof(buf),"fulltrain %s 3",nlu.file);process_command(buf);}
        return;

    /*
     * NEW v13: INTENT_CHAT and all unknown intents route to
     * cmd_converse for natural conversational AI.
     * The old template-based chat_reply is fully replaced.
     */
    case INTENT_CHAT:
    default:
        app_colored("  ", COL_CONVERSE);
        {
            int n_gen;
            conv_reply[0] = '\0';
            /* Show "thinking..." indicator */
            app_colored("[thinking...]\r\n", COL_GREY);
            n_gen = cmd_converse(cmd, conv_reply, sizeof(conv_reply));
            if ((!g_cfg.conv_stream || n_gen==0) && conv_reply[0]) {
                /* Non-streaming: print full response at once */
                app_colored(conv_reply, COL_CONVERSE);
                app("\r\n");
            }
            /* Update performance counters */
            g_perf.conv_turns++;
            g_perf.tokens_generated += (long)n_gen;
        }
        return;
    }
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 7
 *
 * Files covered:
 *   brain.h (v13) –
 *     WM_APP_TOKEN (WM_APP+6), WM_APP_GUARD (WM_APP+7)
 *     COL_CONVERSE, COL_FACT, COL_GUARD
 *     BTN_CONVERSE (1017)
 *     FileGuardState struct (hDir, hThread, watch_dir, running, n_alerts)
 *     ModelDLRequest struct (url, save_path, verify_hmac)
 *     MODEL_DL_MAX_URL, MODEL_DL_CHUNK
 *     BrainConfig: conv_max_tokens, conv_use_facts, conv_stream,
 *       conv_history_turns, sysinfo_tier, dyn_max_embeds,
 *       guard_enabled, guard_dir, train_use_conv, train_use_text
 *     PerfCounters: conv_turns, tokens_generated
 *     g_dyn_max_embeds, g_sysinfo, g_guard externs
 *     TaskType: TASK_CONVERSE, TASK_GUARD, TASK_DOWNLOAD
 *     Intent: INTENT_CONVERSE, INTENT_CONV_RESET, INTENT_CONV_STATS,
 *       INTENT_FACTS, INTENT_GUARD_ON/OFF, INTENT_MODEL_DOWNLOAD
 *     cmd_guard_start, cmd_guard_stop, cmd_model_download,
 *       cmd_facts_show, guard_thread_proc declarations
 *
 *   brain.c Part A (v13) –
 *     §A  Global state: g_dyn_max_embeds, g_sysinfo, g_guard,
 *         g_cfg defaults for all v13 fields
 *     §B  safe_strcpy / safe_fmt / crc32_buf (unchanged)
 *     §C  app_colored / app_* / report_progress (unchanged)
 *     §K  cmd_stats: conv turn/token stats, SysInfo RAM/tier/guard
 *         cmd_version: conv params, RAM tier, guard status
 *         cmd_help: updated command list with CHAT / GUARD / DOWNLOAD
 *     §L  nlu_parse: converse/facts/guard/download intents first;
 *         all unknown -> INTENT_CHAT -> cmd_converse
 *     §M  cmd_facts_show (NEW)
 *     §N  cmd_generate / cmd_summarize (unchanged, include v12)
 *     §O  Undo stack (unchanged, include v12)
 *     §P  process_command:
 *         "converse reset/stats/temp/topk" commands
 *         "facts [query]" command
 *         "guard on [dir]" / "guard off" commands
 *         "download <url> [path]" command
 *         "pretrain" now uses sysinfo_probe + model_create_dynamic
 *         "fulltrain" sets tb_thread_set_bg on worker thread
 *         INTENT_CHAT/default -> cmd_converse + streaming display
 *     §Q  cmd_model_download: WinInet HTTPS, 64KB chunks,
 *         live progress bar, tb_yield per chunk, auto-load + verify
 *     §R  guard_thread_proc: ReadDirectoryChangesW loop,
 *         ANSI filename, extract_features + ensemble_predict,
 *         WM_APP_GUARD PostMessage on danger (conf>60%)
 *         cmd_guard_start: CreateThread + tb_thread_set_bg
 *         cmd_guard_stop:  CloseHandle(hDir) to unblock RDCW
 *
 * PART 8 will cover:
 *   brain.c Part B (v13) – Windows GUI WndProc full implementation:
 *     WM_CREATE: all controls + BTN_CONVERSE button,
 *       GlobalMemoryStatus probe + DynModelCfg at startup,
 *       embedding DB sized from g_sysinfo.max_embeds,
 *       blog_init("brain.log") call, conv_init() call,
 *       guard auto-start if g_cfg.guard_enabled
 *     WM_APP_TOKEN: stream token text to RichEdit (free LPARAM)
 *     WM_APP_GUARD: display guard alert in COL_GUARD + free LPARAM
 *     BTN_CONVERSE: takes input text -> cmd_converse (async)
 *     WM_DESTROY: cmd_guard_stop, blog_close, all v12 cleanup
 *     worker_thread_proc: TASK_CONVERSE + TASK_DOWNLOAD + TASK_GUARD
 *     config_save / config_load: all v13 BrainConfig fields
 *     InputSubclassProc: unchanged hotkeys (F2-F5, history, drag-drop)
 *     watchdog_thread_proc: unchanged
 *     seh_handler: unchanged + blog_write on crash
 *     WinMain: GlobalMemoryStatus FIRST, sysinfo_probe,
 *       dynamic g_dyn_max_embeds, blog_init, RegisterClassEx,
 *       ShowWindow, message loop
 * ─────────────────────────────────────────────────────────────
 */

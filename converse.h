#include "sysinfo.h"
#include "model.h"
#include "tokenizer.h"
/* ============================================================
 * FILE: converse.h
 * ============================================================ */
#ifndef CONVERSE_H
#define CONVERSE_H

/* sysinfo.h, tokenizer.h, model.h assumed included via brain.h */

/* ── History ring buffer size ── */
#define CONV_HISTORY_MAX   32    /* turns kept in RAM             */
#define CONV_FACT_MAX      256   /* max facts in knowledge base   */
#define CONV_FACT_TEXT_MAX 512   /* bytes per fact                */
#define CONV_FACT_KEY_MAX  8     /* keywords per fact             */
#define CONV_FACT_KLEN     32    /* max keyword length            */
#define CONV_GEN_MAX_TOKS  512   /* max tokens generated per turn */
#define CONV_STREAM_CHUNK  4     /* tokens per UI PostMessage     */
#define CONV_MIN_GEN_TOKENS 4    /* block EOS until this many out */

/* NEW Windows message for streaming tokens to UI */
#ifndef WM_APP_TOKEN
#  define WM_APP_TOKEN  (WM_APP+6)
#endif

/* ── Fact entry ── */
typedef struct {
    char text[CONV_FACT_TEXT_MAX];
    char keys[CONV_FACT_KEY_MAX][CONV_FACT_KLEN];
    int  n_keys;
    int  category;   /* 0=general, 1=malware, 2=network, 3=crypto */
} FactEntry;

/* ── Conversation history ── */
typedef struct {
    ConvTurn turns[CONV_HISTORY_MAX];
    int      n_turns;
    int      head;          /* ring-buffer write head             */
    int      total_turns;   /* total turns ever (for stats)       */
} ConvHistory;

/* ── Generation parameters ── */
typedef struct {
    float temperature;
    int   top_k;
    int   max_new_tokens;
    int   use_facts;        /* legacy; chat path ignores facts    */
    int   stream;           /* 1 = stream tokens to UI live       */
    int   history_turns;    /* prior turns fed into encode window */
} ConvParams;

/* ── API ── */
#ifdef __cplusplus
extern "C" {
#endif

/* FIX 4: exposed for brain.c */
extern FactEntry g_facts[CONV_FACT_MAX];
extern int       g_n_facts;

/* Lifecycle */
void conv_init          (void);
void conv_reset         (void);
void conv_set_params    (float temperature, int top_k,
                          int max_new_tokens);
void conv_apply_config  (int use_facts, int stream, int max_new_tokens,
                          float temperature, int top_k, int history_turns);

/* Facts store */
void fact_add           (const char *text, const char **keys, int n_keys,
                          int category);
void fact_seed_security (void);           /* load built-in 50+ facts */
int  fact_search        (const char *query, int *out_idx,
                          int max_results);

/* Core conversational engine */
/* Returns number of tokens generated; fills out_text (decoded). */
int  cmd_converse       (const char *user_input,
                          char *out_text, int out_sz);

/* Add a turn manually (e.g. from a loaded .conv file) */
void conv_history_push  (int role, const char *text);

/* Retrieve history for display or saving */
int  conv_history_get   (ConvTurn *out_buf, int max_turns);

/* Stats */
void conv_print_stats   (void);

#ifdef __cplusplus
}
#endif
#endif /* CONVERSE_H */

#include "model.h"
/* ============================================================
 * FILE: tokenizer.h
 * ============================================================ */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

/* ── BPE limits ── */
#define BPE_VOCAB_SIZE    32768
#define BPE_VOCAB_MAX     65536   /* NEW v13: expanded ceiling       */
#define BPE_MAX_MERGES    65000   /* NEW v13: matches expanded vocab  */
#define BPE_MAX_WORD      512
#define BPE_MAX_SEQ       8192

/* ── Special token IDs (match model.h) ── */
#ifndef TOKEN_PAD
#  define TOKEN_PAD           0
#  define TOKEN_BOS           1
#  define TOKEN_EOS           2
#  define TOKEN_UNK           3
#  define TOKEN_LANG_C        4
#  define TOKEN_LANG_PY       5
#  define TOKEN_LANG_PAS      6
#  define TOKEN_LANG_ASM      7
#  define TOKEN_LANG_EN       8   /* NEW v13 */
#  define TOKEN_LANG_AR       9   /* NEW v13 */
#  define TOKEN_LANG_FR      10   /* NEW v13 */
#  define TOKEN_SCRIPT_LATIN  11
#  define TOKEN_SCRIPT_ARABIC 12
#  define TOKEN_SCRIPT_CJK    13
#  define TOKEN_SCRIPT_DEVA   14
#  define TOKEN_SCRIPT_CYRIL  15
#  define TOKEN_SPECIAL_END   16
#endif

/* NEW v13: conversation role tokens */
#define TOKEN_ROLE_USER   (TOKEN_SPECIAL_END + 256 + 0)   /* virtual */
#define TOKEN_ROLE_ASST   (TOKEN_SPECIAL_END + 256 + 1)   /* virtual */
/* These are encoded as byte sequences "<USER>" and "<ASST>" in vocab */

/* ── Script IDs ── */
#define SCRIPT_UNKNOWN -1
#define SCRIPT_LATIN    0
#define SCRIPT_ARABIC   1
#define SCRIPT_CJK      2
#define SCRIPT_DEVA     3
#define SCRIPT_CYRIL    4
#define SCRIPT_HANGUL   5
#define SCRIPT_THAI     6
#define SCRIPT_HEBREW   7
#define SCRIPT_GREEK    8

/* ── Language IDs ── */
#define LANG_C       0
#define LANG_PYTHON  1
#define LANG_PASCAL  2
#define LANG_ASM     3
#define LANG_EN      4   /* NEW v13 */
#define LANG_AR      5   /* NEW v13 */
#define LANG_FR      6   /* NEW v13 */

/* ── Conversation turn ── */
#define CONV_MAX_TURN_BYTES 2048
#define CONV_MAX_TURNS      128

typedef struct {
    int  role;          /* 0=user, 1=assistant */
    char text[CONV_MAX_TURN_BYTES];
} ConvTurn;

typedef struct {
    ConvTurn turns[CONV_MAX_TURNS];
    int      n_turns;
} Conversation;

/* ── BPE Tokenizer ── */
typedef struct {
    /* Merges – expanded for v13 */
    int  merge_a[BPE_MAX_MERGES];
    int  merge_b[BPE_MAX_MERGES];
    int  merge_c[BPE_MAX_MERGES];
    int  n_merges;

    /* Vocabulary – expanded ceiling */
    char vocab    [BPE_VOCAB_MAX][BPE_MAX_WORD];
    int  vocab_len[BPE_VOCAB_MAX];
    int  vocab_size;

    int  trained;
    int  needs_rebuild;   /* NEW v13: set when vocab_size nears cap */
} BPETokenizer;

/* ── API ── */
#ifdef __cplusplus
extern "C" {
#endif

/* UTF-8 helpers */
int          utf8_char_len    (unsigned char lead);
unsigned int utf8_decode_one  (const unsigned char *p, int *len_out);
int          utf8_encode_one  (unsigned int cp, char *out);

/* Unicode normalisation */
unsigned int arabic_pres_to_base(unsigned int cp);
unsigned int combine_accent      (unsigned int base, unsigned int accent);
void         normalize_unicode   (char *text, int max_bytes);

/* Script detection */
int  detect_script   (unsigned int cp);
int  script_to_token (int script);

/* Sentence / chunk */
int  is_sentence_end (unsigned int cp);
int  find_cjk_split  (const unsigned char *buf, int max_bytes);

/* Language detection */
int  detect_lang_token (const char *filepath);
int  detect_lang_class (const char *filepath);
int  detect_lang_token_from_file (const char *filepath);

/* Natural-language detection heuristic (NEW v13) */
int  detect_nl_lang_token (const char *text, int sample_bytes);

/* Tokenizer lifecycle */
void bpe_init_vocab      (BPETokenizer *t);
void bpe_learn_from_file (BPETokenizer *t, const char *path, int max_merges);
void bpe_learn_from_dir      (BPETokenizer *t, const char *dir, int max_merges);
void bpe_learn_from_conv_dir (BPETokenizer *t, const char *dir, int max_merges);
void bpe_save            (const BPETokenizer *t, const char *path);
int  bpe_load            (BPETokenizer *t,       const char *path);

/* NEW v13: auto-scale */
int  tokenizer_needs_rebuild (const BPETokenizer *t);
int  bpe_rebuild_larger      (BPETokenizer *t,
                               const char *corpus_path,
                               int additional_merges);

/* Encode (plain / with lang / multilingual) */
int  bpe_encode            (const BPETokenizer *t,
                             const char *text,
                             int *out_ids, int max_ids);
int  bpe_encode_with_lang  (const BPETokenizer *t,
                             const char *text,
                             int lang_token,
                             int *out_ids, int max_ids);
int  bpe_encode_multilingual(const BPETokenizer *t,
                              const char *text,
                              int lang_token,
                              int *out_ids, int max_ids);

/* NEW v13: conversation encoding */
int  bpe_parse_conv_file   (const char *path, Conversation *out);
int  bpe_encode_conv_turn  (const BPETokenizer *t,
                             const ConvTurn *turn,
                             int lang_token,
                             int *out_ids, int max_ids);
int  bpe_encode_conv       (const BPETokenizer *t,
                             const Conversation *conv,
                             int lang_token,
                             int *out_ids, int max_ids);

/* NEW v13: encode with conversation history prefix */
int  bpe_encode_with_history(const BPETokenizer *t,
                              const char *current_input,
                              const ConvTurn *history,
                              int n_history,
                              int lang_token,
                              int *out_ids, int max_ids);

/* Decode */
void bpe_decode (const BPETokenizer *t,
                  const int *ids, int n,
                  char *out, int outsz);

#ifdef __cplusplus
}
#endif
#endif /* TOKENIZER_H */

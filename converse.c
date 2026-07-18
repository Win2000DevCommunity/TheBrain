#include "converse.h"
#include "model.h"
#include "tokenizer.h"
#include "sysinfo.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define _WIN32_WINNT 0x0500
#  include <windows.h>
#endif

/* ── Externals from brain globals (defined in brain.c §A) ── */
extern Model          *g_model;
extern BPETokenizer    g_tokenizer;
extern HWND            g_hMain;
extern HWND            g_hChat;
extern volatile LONG   g_cancel_flag;

/* ── Module-level state ── */
static ConvHistory  g_history;
FactEntry    g_facts[CONV_FACT_MAX];
int          g_n_facts   = 0;
/* FIX 2: 64 tokens default - 512 takes 25+ min on CPU */
static ConvParams   g_cparams   = {0.15f, 1, 96, 0, 1, 2};
static int          g_conv_init_done = 0;

/* ── Statistics ── */
static long g_total_tokens_generated = 0;
static long g_total_turns            = 0;

static void stream_token_text(const char *text);

/* ═══════════════════════════════════════════════════════════════
 * §A  LIFECYCLE
 * ═══════════════════════════════════════════════════════════════ */

void conv_init(void)
{
    if (g_conv_init_done) return;
    memset(&g_history, 0, sizeof(g_history));
    memset(g_facts,    0, sizeof(g_facts));
    g_n_facts = 0;
    g_conv_init_done = 1;
    fact_seed_security();
    BLOG_INFO("conv_init: history=%d facts=%d",
              CONV_HISTORY_MAX, g_n_facts);
}

void conv_reset(void)
{
    memset(&g_history, 0, sizeof(g_history));
    BLOG_INFO("conv_reset: history cleared");
}

void conv_set_params(float temperature, int top_k, int max_new_tokens)
{
    if (temperature > 0.01f && temperature <= 5.0f)
        g_cparams.temperature = temperature;
    if (top_k >= 1 && top_k <= 1024)
        g_cparams.top_k = top_k;
    if (max_new_tokens >= 1 && max_new_tokens <= 2048)
        g_cparams.max_new_tokens = max_new_tokens;
}

void conv_apply_config(int use_facts, int stream, int max_new_tokens,
                        float temperature, int top_k, int history_turns)
{
    (void)use_facts;
    g_cparams.use_facts = 0;
    g_cparams.stream    = stream ? 1 : 0;
    if (history_turns >= 0 && history_turns <= CONV_HISTORY_MAX)
        g_cparams.history_turns = history_turns;
    conv_set_params(temperature, top_k, max_new_tokens);
}

/* ═══════════════════════════════════════════════════════════════
 * §B  CONVERSATION HISTORY  (ring buffer)
 * ═══════════════════════════════════════════════════════════════ */

void conv_history_push(int role, const char *text)
{
    ConvTurn *slot;
    if (!g_conv_init_done) conv_init();

    if (g_history.n_turns < CONV_HISTORY_MAX) {
        slot = &g_history.turns[g_history.n_turns++];
    } else {
        /* Ring: overwrite oldest */
        slot = &g_history.turns[g_history.head % CONV_HISTORY_MAX];
        g_history.head = (g_history.head + 1) % CONV_HISTORY_MAX;
    }
    slot->role = role;
    strncpy(slot->text, text, CONV_MAX_TURN_BYTES - 1);
    slot->text[CONV_MAX_TURN_BYTES - 1] = '\0';
    g_history.total_turns++;
}

int conv_history_get(ConvTurn *out_buf, int max_turns)
{
    int i, n, start;

    if (!out_buf || max_turns <= 0) return 0;
    if (max_turns > CONV_HISTORY_MAX) max_turns = CONV_HISTORY_MAX;
    n = g_history.n_turns < max_turns ? g_history.n_turns : max_turns;
    if (n <= 0) return 0;

    if (g_history.n_turns >= CONV_HISTORY_MAX) {
        start = CONV_HISTORY_MAX - n;
        for (i = 0; i < n; i++)
            out_buf[i] = g_history.turns[
                (g_history.head + start + i) % CONV_HISTORY_MAX];
    } else {
        start = g_history.n_turns - n;
        for (i = 0; i < n; i++)
            out_buf[i] = g_history.turns[start + i];
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * §C  FACTS STORE
 * ═══════════════════════════════════════════════════════════════ */

void fact_add(const char *text, const char **keys, int n_keys, int category)
{
    FactEntry *f;
    int i;
    if (g_n_facts >= CONV_FACT_MAX) return;
    f = &g_facts[g_n_facts++];
    memset(f, 0, sizeof(FactEntry));
    strncpy(f->text, text, CONV_FACT_TEXT_MAX - 1);
    f->text[CONV_FACT_TEXT_MAX - 1] = '\0';
    f->n_keys  = n_keys < CONV_FACT_KEY_MAX ? n_keys : CONV_FACT_KEY_MAX;
    f->category = category;
    for (i = 0; i < f->n_keys; i++) {
        strncpy(f->keys[i], keys[i], CONV_FACT_KLEN - 1);
        f->keys[i][CONV_FACT_KLEN - 1] = '\0';
    }
}

/*
 * fact_search: return indices of up to max_results facts
 * whose keywords appear in query as whole words (case-insensitive).
 * Short keys (< 3 chars) are ignored to prevent "c" matching "arabic".
 * Returns number of matches found.
 */
static int fact_keyword_match(const char *query, const char *key)
{
    int klen = (int)strlen(key);
    const char *p;

    if (klen < 3) return 0;
    p = query;
    while ((p = strstr(p, key)) != NULL) {
        if ((p == query || !isalnum((unsigned char)p[-1])) &&
            (p[klen] == '\0' || !isalnum((unsigned char)p[klen])))
            return 1;
        p++;
    }
    return 0;
}

int fact_search(const char *query, int *out_idx, int max_results)
{
    char low_q[1024];
    int  i, j, n = 0;
    int  qi;

    /* Lower-case the query */
    for (qi = 0; query[qi] && qi < 1023; qi++)
        low_q[qi] = (char)tolower((unsigned char)query[qi]);
    low_q[qi] = '\0';

    for (i = 0; i < g_n_facts && n < max_results; i++) {
        for (j = 0; j < g_facts[i].n_keys; j++) {
            char low_k[CONV_FACT_KLEN];
            int k;
            for (k = 0; g_facts[i].keys[j][k] && k < CONV_FACT_KLEN - 1; k++)
                low_k[k] = (char)tolower((unsigned char)g_facts[i].keys[j][k]);
            low_k[k] = '\0';
            if (fact_keyword_match(low_q, low_k)) {
                out_idx[n++] = i;
                break;
            }
        }
    }
    return n;
}

/*
 * fact_seed_security:
 * Loads 50+ hard-coded security / AI / general knowledge facts.
 * These are prepended to the context when the user's query
 * matches any keyword, grounding the model's response.
 */
void fact_seed_security(void)
{
    /* Helper macro for readability */
#define F(text, cat, ...) \
    do { \
        static const char *_k[] = {__VA_ARGS__}; \
        fact_add(text, _k, (int)(sizeof(_k)/sizeof(_k[0])), cat); \
    } while(0)

    /* ── Malware ── */
    F("Ransomware encrypts files and demands payment to restore them. "
      "Common families: WannaCry, REvil, LockBit, BlackCat.",
      1, "ransomware","encrypt","wannacry","lockbit","payment");

    F("Trojans disguise themselves as legitimate software but carry "
      "a malicious payload. Common delivery: email attachments, "
      "fake downloads.",
      1, "trojan","rat","payload","legitimate");

    F("Rootkits hide malicious code from the OS, often by hooking "
      "system calls or patching the kernel.",
      1, "rootkit","hook","kernel","hide","stealth");

    F("Keyloggers record keystrokes to steal passwords and sensitive data. "
      "Can be hardware or software based.",
      1, "keylogger","keystroke","password","steal");

    F("Worms self-propagate across networks without user interaction, "
      "exploiting unpatched vulnerabilities.",
      1, "worm","propagate","network","exploit","vulnerability");

    F("Fileless malware lives in memory only, leaving no files on disk. "
      "Often uses PowerShell, WMI, or reflective DLL injection.",
      1, "fileless","memory","powershell","wmi","injection");

    F("Packers compress and encrypt malware to evade signature-based AV. "
      "High-entropy sections often indicate packing.",
      1, "packer","entropy","compress","signature","evade");

    F("Botnets are networks of compromised machines controlled by a C2 "
      "server. Used for DDoS, spam, and credential theft.",
      1, "botnet","c2","command","control","ddos","spam");

    F("APT (Advanced Persistent Threat) actors use sophisticated, "
      "long-term campaigns. Examples: Lazarus, APT28, APT29.",
      1, "apt","persistent","threat","lazarus","sophisticated");

    F("Cryptominers use victim CPU/GPU resources to mine cryptocurrency "
      "without consent. Detectable via high CPU usage.",
      1, "miner","cryptominer","cpu","gpu","cryptocurrency");

    F("Droppers are small malware components whose sole purpose is to "
      "download and execute a larger payload.",
      1, "dropper","download","execute","payload","stage");

    F("Backdoors provide persistent remote access to a compromised system, "
      "often on a non-standard port.",
      1, "backdoor","remote","access","persistent","port");

    F("Exploits take advantage of software vulnerabilities. Common types: "
      "buffer overflow, use-after-free, heap spray.",
      1, "exploit","buffer","overflow","vulnerability","heap");

    F("Adware displays unwanted advertisements and may track browsing "
      "behaviour. Usually low-severity but degrades performance.",
      1, "adware","advertisement","track","browser","performance");

    /* ── Network / analysis ── */
    F("A PE (Portable Executable) file has a DOS stub, NT headers, "
      "section table, and sections. EntryPoint points to code start.",
      2, "pe","portable","executable","header","section","entry");

    F("High entropy (>7.0 bits/byte) in a PE section suggests packing "
      "or encryption. Normal code is typically 5-6 bits/byte.",
      2, "entropy","bits","packing","encryption","section");

    F("Import Address Table (IAT) shows which DLLs and functions a PE "
      "imports. Suspicious APIs include VirtualAlloc, "
      "WriteProcessMemory, CreateRemoteThread.",
      2, "import","iat","dll","virtualalloc","remote","thread");

    F("Process injection techniques: DLL injection, process hollowing, "
      "thread hijacking, reflective loading.",
      2, "injection","hollow","hijack","dll","reflective");

    F("YARA rules are pattern-matching signatures for malware families. "
      "They match strings, byte patterns, and conditions.",
      2, "yara","rule","signature","pattern","match");

    F("Sandbox analysis runs suspicious files in an isolated environment "
      "and observes behaviour: file writes, registry changes, network.",
      2, "sandbox","analysis","isolated","behaviour","registry");

    F("C2 (Command and Control) communication is often disguised as "
      "HTTP/HTTPS traffic, DNS tunnelling, or social media APIs.",
      2, "c2","command","control","http","dns","tunnel");

    F("Zero-day vulnerabilities are unknown to the vendor and have no "
      "patch available. Highly valued by attackers.",
      2, "zero-day","zeroday","unknown","patch","vendor");

    F("MITRE ATT&CK is a framework cataloguing adversary tactics, "
      "techniques, and procedures (TTPs).",
      2, "mitre","attack","ttp","tactic","technique");

    F("Memory forensics examines RAM dumps to find injected code, "
      "encryption keys, and artefacts not on disk.",
      2, "memory","forensics","ram","dump","artefact");

    /* ── Cryptography ── */
    F("XOR encryption is trivially breakable: find the key by XORing "
      "known-plaintext with ciphertext.",
      3, "xor","encryption","key","plaintext","cipher");

    F("RC4 is a stream cipher used in some malware for C2 traffic "
      "obfuscation. Key scheduling is its main weakness.",
      3, "rc4","stream","cipher","obfuscation","schedule");

    F("Base64 is encoding, not encryption. It expands data by 33%% "
      "and is easily decoded.",
      3, "base64","encoding","decode","expand");

    F("AES-256 in CBC mode with PKCS#7 padding is the gold standard "
      "for symmetric encryption of malware payloads.",
      3, "aes","cbc","pkcs","symmetric","payload");

    F("RSA is used for key exchange in ransomware: the malware generates "
      "an AES key, encrypts it with the attacker's RSA public key.",
      3, "rsa","key","exchange","asymmetric","ransomware");

    /* ── AI / transformer ── */
    F("A transformer uses self-attention to relate every token to every "
      "other token in the sequence, enabling long-range dependencies.",
      0, "transformer","attention","token","sequence","self-attention");

    F("RoPE (Rotary Position Embedding) encodes position by rotating "
      "query/key vectors, enabling length generalisation.",
      0, "rope","rotary","position","embedding","generalise");

    F("SwiGLU is a gated activation: silu(gate) * up, used in Llama-style "
      "FFN layers for improved training stability.",
      0, "swiglu","gated","activation","llama","ffn");

    F("BPE (Byte-Pair Encoding) builds a vocabulary by iteratively merging "
      "the most frequent byte-pair in the corpus.",
      0, "bpe","byte","pair","encoding","vocabulary","merge");

    F("AdamW decouples weight decay from the gradient update, improving "
      "regularisation compared to L2 in Adam.",
      0, "adamw","weight","decay","regularisation","optimizer");

    F("Perplexity (PPL) measures how well a language model predicts a text. "
      "Lower PPL = better model. PPL = exp(average NLL).",
      0, "perplexity","ppl","language","model","predict","nll");

    F("INT8 quantisation reduces model size 4x with minimal accuracy loss "
      "by mapping float32 weights to 8-bit integers with per-row scales.",
      0, "quantisation","int8","float32","accuracy","scale");

    F("Gradient clipping prevents exploding gradients by scaling the "
      "gradient vector to have at most max_norm L2 norm.",
      0, "gradient","clipping","exploding","norm","scale");

    F("Layer Normalisation normalises activations to zero mean and unit "
      "variance, stabilising deep network training.",
      0, "layernorm","normalisation","mean","variance","activation");

    F("Cosine learning rate schedule starts at lr_max, decays to lr_min "
      "following a cosine curve after linear warmup.",
      0, "cosine","learning","rate","schedule","warmup","decay");

    /* ── Windows internals ── */
    F("The Windows PE loader maps sections into virtual memory, resolves "
      "imports via the IAT, and jumps to the entry point.",
      2, "loader","virtual","memory","import","entry");

    F("Registry run keys (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion"
      "\\Run) are a common persistence mechanism.",
      1, "registry","run","key","persistence","hkcu");

    F("CreateRemoteThread is a classic DLL injection API: open a target "
      "process, write DLL path, then create a remote thread.",
      2, "createremotethread","inject","dll","thread","remote");

    F("WMI (Windows Management Instrumentation) is abused for fileless "
      "persistence and lateral movement.",
      1, "wmi","management","instrumentation","fileless","lateral");

    F("AMSI (Antimalware Scan Interface) hooks into PowerShell and other "
      "script hosts. Bypasses: reflection, obfuscation, patching.",
      2, "amsi","antimalware","powershell","bypass","obfuscation");

    F("PEB (Process Environment Block) stores process metadata. Malware "
      "walks the PEB InLoadOrderModuleList to find loaded DLLs.",
      2, "peb","process","environment","block","module","dll");

    F("Syscall-based evasion bypasses user-mode hooks by calling Windows "
      "kernel functions directly via int 2E or syscall instruction.",
      2, "syscall","evasion","hook","kernel","usermode");

    F("LSASS (Local Security Authority Subsystem Service) stores "
      "credentials. Mimikatz dumps credentials from LSASS memory.",
      2, "lsass","credential","mimikatz","dump","authority");

    F("UAC (User Account Control) prompts for elevation. Bypass techniques "
      "include auto-elevation COM objects and token impersonation.",
      2, "uac","elevation","bypass","com","impersonation");

    F("ETW (Event Tracing for Windows) is used by EDR solutions to monitor "
      "process behaviour. Some malware patches ETW to blind defenders.",
      2, "etw","event","tracing","edr","monitor","patch");

    /* ── General knowledge ── */
    F("TheBrain v13.0 is a Windows 2000+ compatible AI engine with an "
      "ONNX-like transformer, BPE tokenizer, and malware classifier. "
      "It runs on as little as 256 MB of RAM.",
      0, "thebrain","v13","engine","transformer","windows","ram");

    F("Windows 2000 was released in February 2000 and introduced Active "
      "Directory, native USB support, and improved hardware detection.",
      0, "windows","2000","active","directory","usb");

    F("C89 (ANSI C) is the first standardised version of the C programming "
      "language, published in 1989. It lacks VLAs and complex types.",
      0, "c89","ansi","c","1989","vla","standard");

#undef F

    BLOG_INFO("fact_seed_security: loaded %d facts", g_n_facts);
}

/* ═══════════════════════════════════════════════════════════════
 * §D  TOP-K SAMPLING  (reusable helper)
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Sample one token from logits using temperature + top-k.
 * Uses a lightweight LCG instead of rand() for C89 portability.
 */
static unsigned long g_sample_rng = 0xBEEFCAFEUL;

static float sample_uniform(void)
{
    g_sample_rng ^= g_sample_rng << 13;
    g_sample_rng ^= g_sample_rng >> 17;
    g_sample_rng ^= g_sample_rng << 5;
    return (float)(g_sample_rng & 0x7FFFFFFFUL) / 2147483647.0f;
}

static int sample_argmax_text(const float *logits, int vocab_size, int gen_len)
{
    int i, best = -1;
    float best_logit = -1e30f;
    int user_id = -1, asst_id = -1;

    if (g_tokenizer.trained) {
        for (i = 0; i < g_tokenizer.vocab_size; i++) {
            if (user_id < 0 && !strcmp(g_tokenizer.vocab[i], "<USER>"))
                user_id = i;
            if (asst_id < 0 && !strcmp(g_tokenizer.vocab[i], "<ASST>"))
                asst_id = i;
            if (user_id >= 0 && asst_id >= 0) break;
        }
    }
    if (user_id < 0) user_id = TOKEN_SPECIAL_END + 256;
    if (asst_id < 0) asst_id = TOKEN_SPECIAL_END + 256 + 1;

    for (i = 0; i < vocab_size; i++) {
        if (i == TOKEN_PAD || i == TOKEN_UNK || i == TOKEN_BOS) continue;
        if (gen_len < CONV_MIN_GEN_TOKENS && i == TOKEN_EOS) continue;
        if (i >= TOKEN_LANG_C && i <= TOKEN_SCRIPT_CYRIL) continue;
        if (i == user_id || i == asst_id) continue;
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    if (best < 0) best = TOKEN_EOS;
    return best;
}

static int sample_top_k(const float *logits, int vocab_size,
                          float temperature, int top_k,
                          const int *gen_ids, int gen_len)
{
    float *probs;
    int   *idxs;
    float  mx, sm, r, cs;
    int    i, j, k, chosen;
    int    user_id = -1, asst_id = -1;

    if (top_k < 1)  top_k = 1;
    if (top_k > vocab_size) top_k = vocab_size;
    if (temperature < 1e-4f) temperature = 1e-4f;

    if (g_tokenizer.trained) {
        for (i = 0; i < g_tokenizer.vocab_size; i++) {
            if (user_id < 0 && !strcmp(g_tokenizer.vocab[i], "<USER>"))
                user_id = i;
            if (asst_id < 0 && !strcmp(g_tokenizer.vocab[i], "<ASST>"))
                asst_id = i;
            if (user_id >= 0 && asst_id >= 0) break;
        }
    }

    probs = (float*)malloc((size_t)vocab_size * sizeof(float));
    idxs  = (int*)  malloc((size_t)vocab_size * sizeof(int));
    if (!probs || !idxs) {
        free(probs); free(idxs);
        return TOKEN_EOS;
    }

    mx = logits[0];
    for (i = 1; i < vocab_size; i++)
        if (logits[i] > mx) mx = logits[i];

    sm = 0.0f;
    for (i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - mx) / temperature);
        idxs[i]  = i;
        sm += probs[i];
    }
    if (sm < 1e-12f) sm = 1e-12f;
    for (i = 0; i < vocab_size; i++) probs[i] /= sm;

    /* Block special / control tokens from being sampled mid-reply. */
    for (i = 0; i < vocab_size; i++) {
        if (i == TOKEN_PAD || i == TOKEN_UNK || i == TOKEN_BOS) {
            probs[i] = 0.0f; continue;
        }
        if (gen_len < CONV_MIN_GEN_TOKENS && i == TOKEN_EOS) {
            probs[i] = 0.0f; continue;
        }
        if (i >= TOKEN_LANG_C && i <= TOKEN_SCRIPT_CYRIL) {
            probs[i] = 0.0f; continue;
        }
    }
    if (user_id < 0) user_id = TOKEN_SPECIAL_END + 256;
    if (asst_id < 0) asst_id = TOKEN_SPECIAL_END + 256 + 1;
    if (user_id >= 0 && user_id < vocab_size) probs[user_id] = 0.0f;
    if (asst_id >= 0 && asst_id < vocab_size) probs[asst_id] = 0.0f;

    /* Repetition penalty on tokens already emitted this turn. */
    if (gen_ids && gen_len > 0) {
        for (k = 0; k < gen_len; k++) {
            int t = gen_ids[k];
            if (t >= 0 && t < vocab_size && probs[t] > 0.0f)
                probs[t] /= 1.15f;
        }
    }

    sm = 0.0f;
    for (i = 0; i < vocab_size; i++) sm += probs[i];
    if (sm < 1e-12f) {
        free(probs); free(idxs);
        return sample_argmax_text(logits, vocab_size, gen_len);
    }
    for (i = 0; i < vocab_size; i++) probs[i] /= sm;

    for (i = 0; i < top_k; i++) {
        int best = i;
        for (j = i + 1; j < vocab_size; j++)
            if (probs[j] > probs[best]) best = j;
        { float tp = probs[i]; probs[i] = probs[best]; probs[best] = tp; }
        { int   ti = idxs[i];  idxs[i]  = idxs[best];  idxs[best]  = ti; }
    }

    sm = 0.0f;
    for (i = 0; i < top_k; i++) sm += probs[i];
    if (sm < 1e-12f) sm = 1e-12f;

    r = sample_uniform() * sm;
    chosen = idxs[0];
    cs = 0.0f;
    for (k = 0; k < top_k; k++) {
        cs += probs[k];
        if (r <= cs) { chosen = idxs[k]; break; }
    }

    free(probs); free(idxs);
    return chosen;
}

static int conv_reply_ends_sentence(const int *gen_ids, int n_gen)
{
    char buf[128];
    char *p;
    if (n_gen <= 0) return 0;
    bpe_decode(&g_tokenizer, gen_ids, n_gen, buf, (int)sizeof(buf));
    p = buf + strlen(buf);
    while (p > buf && (p[-1] == ' ' || p[-1] == '\r' || p[-1] == '\n')) p--;
    if (p <= buf) return 0;
    return (p[-1] == '.' || p[-1] == '?' || p[-1] == '!');
}

/* ═══════════════════════════════════════════════════════════════
 * §E  FACT CONTEXT BUILDER
 *
 * Assembles a short fact preamble prepended to the prompt.
 * At most 3 relevant facts, truncated to 256 chars each.
 * Returns total bytes written into buf.
 * ═══════════════════════════════════════════════════════════════ */

static int build_fact_context(const char *query,
                                char *buf, int buf_sz)
{
    int  fact_idx[8];
    int  n_facts, i, wi = 0;
    char tmp[280];

    n_facts = fact_search(query, fact_idx, 3);
    if (n_facts == 0) return 0;

    for (i = 0; i < n_facts && wi < buf_sz - 4; i++) {
        int tlen;
        _snprintf(tmp, sizeof(tmp) - 1,
                  "[FACT] %s\n", g_facts[fact_idx[i]].text);
        tmp[sizeof(tmp) - 1] = '\0';
        tlen = (int)strlen(tmp);
        if (wi + tlen >= buf_sz - 2) break;
        memcpy(buf + wi, tmp, (size_t)tlen);
        wi += tlen;
    }
    buf[wi] = '\0';
    return wi;
}

static int conv_word_overlap_score(const char *a, const char *b)
{
    char wa[64];
    int  i = 0, j, score = 0;

    while (a[i]) {
        while (a[i] && !isalnum((unsigned char)a[i])) i++;
        j = 0;
        while (a[i] && isalnum((unsigned char)a[i]) && j < (int)sizeof(wa) - 1)
            wa[j++] = (char)tolower((unsigned char)a[i++]);
        wa[j] = '\0';
        if ((int)strlen(wa) >= 3 && fact_keyword_match(b, wa))
            score++;
    }
    return score;
}

#ifdef _WIN32
static void conv_corpus_scan_dir(const char *dir, const char *query,
                                  char *best_ans, int ans_sz, int *best_score)
{
    WIN32_FIND_DATAA fd;
    char pattern[520];
    HANDLE h;

    _snprintf(pattern, sizeof(pattern) - 1, "%s\\*.*", dir);
    pattern[sizeof(pattern) - 1] = '\0';
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;

    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (strcmp(fd.cFileName, ".") && strcmp(fd.cFileName, "..")) {
                char sub[520];
                _snprintf(sub, sizeof(sub) - 1, "%s\\%s", dir, fd.cFileName);
                sub[sizeof(sub) - 1] = '\0';
                conv_corpus_scan_dir(sub, query, best_ans, ans_sz, best_score);
            }
            continue;
        }
        {
            const char *dot = strrchr(fd.cFileName, '.');
            if (dot && !_stricmp(dot, ".conv")) {
                char path[520];
                Conversation conv;
                int t, sc;

                _snprintf(path, sizeof(path) - 1, "%s\\%s", dir, fd.cFileName);
                path[sizeof(path) - 1] = '\0';
                if (bpe_parse_conv_file(path, &conv) <= 0) continue;

                for (t = 0; t < conv.n_turns; t++) {
                    if (conv.turns[t].role != 0) continue;
                    sc = conv_word_overlap_score(query, conv.turns[t].text);
                    sc += conv_word_overlap_score(conv.turns[t].text, query);
                    if (sc > *best_score &&
                        t + 1 < conv.n_turns && conv.turns[t + 1].role == 1) {
                        *best_score = sc;
                        strncpy(best_ans, conv.turns[t + 1].text, (size_t)ans_sz - 1);
                        best_ans[ans_sz - 1] = '\0';
                    }
                }
            }
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}
#endif

/*
 * conv_corpus_answer:
 * Retrieve the best matching trained Q&A from data/*.conv files.
 * Returns 1 if a good match was found.
 */
static int conv_corpus_answer(const char *query, char *out, int out_sz, int min_score)
{
    char best[CONV_MAX_TURN_BYTES];
    int  best_score = 0;

    if (out_sz < 16) return 0;
    best[0] = '\0';
#ifdef _WIN32
    conv_corpus_scan_dir("data", query, best, (int)sizeof(best), &best_score);
#endif
    if (best_score >= min_score && best[0]) {
        strncpy(out, best, (size_t)out_sz - 1);
        out[out_sz - 1] = '\0';
        BLOG_INFO("cmd_converse: corpus match score=%d", best_score);
        return 1;
    }
    return 0;
}

static int conv_text_quality_ok(const char *text)
{
    int i, len, good = 0, bad = 0, words = 0, in_word = 0;

    if (!text || !text[0]) return 0;
    len = (int)strlen(text);
    if (len < 12) return 0;

    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (isalnum(c) || c == ' ' || c == '.' || c == ',' || c == '?' ||
            c == '!' || c == ':' || c == ';' || c == '\'' || c == '-' ||
            c == '\r' || c == '\n' || c == '(' || c == ')')
            good++;
        else if (c < 32)
            good++;
        else if (c > 126)
            good++;
        else
            bad++;
    }
    for (i = 0; i < len; i++) {
        if (isalnum((unsigned char)text[i])) {
            if (!in_word) { words++; in_word = 1; }
        } else {
            in_word = 0;
        }
    }
    if (words < 4) return 0;
    if (bad * 3 > good) return 0;
    return 1;
}

static void conv_emit_reply(const char *text)
{
    if (g_cparams.stream) {
        stream_token_text(text);
        stream_token_text("\r\n");
    }
}

/*
 * fact_compose_answer:
 * Build a direct answer from the curated facts store.  On a tiny CPU/Win2000
 * model the autoregressive generator cannot form coherent prose, so when the
 * user's query matches one or more known facts we answer straight from them.
 * Returns the number of facts used (0 = no match -> caller falls back to the
 * generative path).
 */
static int fact_compose_answer(const char *query, char *out, int out_sz)
{
    int idx[4];
    int n, i, wi = 0;

    if (out_sz < 8) return 0;
    n = fact_search(query, idx, 2);
    if (n <= 0) return 0;

    out[0] = '\0';
    for (i = 0; i < n && wi < out_sz - 4; i++) {
        const char *t = g_facts[idx[i]].text;
        int tl = (int)strlen(t);
        if (wi + tl >= out_sz - 4) tl = out_sz - 4 - wi;
        if (tl <= 0) break;
        memcpy(out + wi, t, (size_t)tl);
        wi += tl;
        if (i + 1 < n && wi < out_sz - 4) out[wi++] = ' ';
    }
    out[wi] = '\0';
    return n;
}

/*
 * conv_canned_reply:
 * Coherent, deterministic responses for everyday conversational input
 * (greetings, identity, capabilities, thanks, farewell).  A tiny CPU model
 * cannot generate these reliably, so we handle them explicitly.  Returns 1
 * if a reply was produced, 0 otherwise.
 */
static int conv_canned_reply(const char *query, char *out, int out_sz)
{
    char low[512];
    char first[32];
    int  i, j;

    if (out_sz < 16) return 0;

    for (i = 0; query[i] && i < (int)sizeof(low) - 1; i++)
        low[i] = (char)tolower((unsigned char)query[i]);
    low[i] = '\0';

    /* Skip leading spaces/punctuation, then grab the first word. */
    j = 0;
    while (low[j] && !isalnum((unsigned char)low[j])) j++;
    for (i = 0; low[j] && isalnum((unsigned char)low[j]) && i < (int)sizeof(first) - 1; i++, j++)
        first[i] = low[j];
    first[i] = '\0';

#define CR_SET(s) do { strncpy(out, (s), out_sz - 1); out[out_sz - 1] = '\0'; } while(0)

    /* ── Identity ── */
    if (strstr(low, "who are you") || strstr(low, "what are you") ||
        strstr(low, "your name")  || strstr(low, "who r u")) {
        CR_SET("I am TheBrain v13 - a security-focused assistant for Windows "
               "2000+ that runs fully offline in C. I can explain malware, "
               "cryptography, Windows internals, and AI concepts, and I can "
               "scan files for threats.");
        return 1;
    }
    /* ── Capabilities ── */
    if (strstr(low, "what can you do") || strstr(low, "what do you do") ||
        strstr(low, "capabilit")      || strstr(low, "how do you work")) {
        CR_SET("I can answer security questions (try 'How does ransomware "
               "work?'), analyse files ('predict <file>', 'scan <dir>'), "
               "explain PE headers and entropy, and train on your data. "
               "Type 'help' for the full command list.");
        return 1;
    }
    /* ── Thanks ── */
    if (strstr(low, "thank") || strstr(low, "thx") || strstr(low, "appreciate")) {
        CR_SET("You're welcome! Ask me anything about malware, encryption, "
               "or Windows internals.");
        return 1;
    }
    /* ── How are you ── */
    if (strstr(low, "how are you") || strstr(low, "how r u") ||
        strstr(low, "how's it going")) {
        CR_SET("Running fast and offline. What security topic can I help "
               "you with?");
        return 1;
    }
    /* ── Farewell ── */
    if (!strcmp(first, "bye") || !strcmp(first, "goodbye") ||
        !strcmp(first, "exit") || !strcmp(first, "quit") ||
        strstr(low, "see you")) {
        CR_SET("Goodbye - stay safe out there.");
        return 1;
    }
    /* ── Clarification / vague follow-up (tiny model cannot handle these) ── */
    if (strstr(low, "don't understand") || strstr(low, "dont understand") ||
        strstr(low, "not understand")    || strstr(low, "confused") ||
        strstr(low, "what do you mean")  || strstr(low, "explain simpler") ||
        strstr(low, "say that again")) {
        CR_SET("No problem. Ask one clear security question and I will "
               "answer in plain language. Examples: 'What is ransomware?', "
               "'How does AES work?', or 'What is process injection?'");
        return 1;
    }
    if (strstr(low, "ok so") || strstr(low, "so what") ||
        strstr(low, "and then") || strstr(low, "what now") ||
        (strlen(low) <= 8 && !strcmp(first, "ok")) ||
        (strlen(low) <= 6 && !strcmp(first, "so"))) {
        CR_SET("Sure - what would you like to know next? Try a specific "
               "question on malware, encryption, or Windows internals.");
        return 1;
    }
    if (strstr(low, "in arabic") || strstr(low, "speak arabic") ||
        strstr(low, "arabic please") || strstr(low, "بالعرب")) {
        CR_SET("I can answer security questions in Arabic. Try: "
               "'ما هو الفدية البرمجية؟' or ask in English and I will reply.");
        return 1;
    }
    if (strstr(low, "in french") || strstr(low, "speak french") ||
        strstr(low, "en francais") || strstr(low, "en français")) {
        CR_SET("I can answer security questions in French. Try: "
               "'Qu'est-ce que le ransomware ?' or ask in English.");
        return 1;
    }
    /* ── Greeting (first word only, so it won't fire mid-sentence) ── */
    if (!strcmp(first, "hi")    || !strcmp(first, "hello") ||
        !strcmp(first, "hey")   || !strcmp(first, "yo")    ||
        !strcmp(first, "hiya")  || !strcmp(first, "greetings") ||
        !strcmp(first, "good")  /* good morning/evening */) {
        CR_SET("Hello! I'm TheBrain, your offline security assistant. "
               "Ask me something like 'How does ransomware work?' or type "
               "'help' to see what I can do.");
        return 1;
    }

#undef CR_SET
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * §F  TOKEN STREAMING  (posts decoded chunk to UI)
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Stream a decoded token string to the RichEdit output.
 * Uses PostMessage (thread-safe) with a heap-allocated string.
 * The WndProc WM_APP_TOKEN handler must free() the LPARAM.
 */
static void stream_token_text(const char *text)
{
#ifdef _WIN32
    size_t len;
    char  *copy;
    if (!g_hMain || !IsWindow(g_hMain)) return;
    if (!text || !text[0]) return;
    len  = strlen(text);
    copy = (char *)malloc(len + 1);
    if (!copy) return;
    memcpy(copy, text, len + 1);
    PostMessage(g_hMain, WM_APP_TOKEN, 0, (LPARAM)copy);
#else
    (void)text;
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * §G  cmd_converse  (the main conversational engine)
 *
 * Pipeline:
 *   1. Normalise user_input (unicode normalise)
 *   2. Detect language (-> lang_token)
 *   3. Encode with history + <ASST> prefix (same format as training)
 *   4. Autoregressive token generation (streamed to UI)
 *   5. Push turns to history
 * ═══════════════════════════════════════════════════════════════ */

int cmd_converse(const char *user_input, char *out_text, int out_sz)
{
    /* Working buffers */
    char     norm_input[CONV_MAX_TURN_BYTES];
    char     stream_prev[4096];
    int     *ctx_ids;         /* sliding context window              */
    int     *gen_ids;         /* generated token IDs                 */
    int      ctx_len = 0;
    int      gen_len = 0;
    int      lang_token;
    int      vocab_size;
    int      sample_vocab;
    int      max_ctx;
    int      hist_limit;
    float   *lm_logits;
    ModelOutput fwd_out;
    ConvTurn hist_buf[CONV_HISTORY_MAX];
    int      n_hist;
    char     stream_full[4096];
    float    gen_temp;
    int      gen_topk;

    if (!g_conv_init_done) conv_init();

    /* ── Guard: model not trained ── */
    /* BUG 3 FIX: friendly message when model not ready */
    if (!g_model || !g_model->trained) {
        const char *fallback =
            "Model not ready yet.\r\n"
            "Type: easytrain   (one command - trains on data\\)\r\n"
            "Or:  pretrain  then  fulltrain data\r\n"
            "Or type: help   for all commands.";
        strncpy(out_text, fallback, out_sz - 1);
        out_text[out_sz - 1] = '\0';
        return 0;
    }
    if (!g_tokenizer.trained) {
        const char *fallback =
            "Tokenizer not ready. Run 'bpetrain <corpus>' first.";
        strncpy(out_text, fallback, out_sz - 1);
        out_text[out_sz - 1] = '\0';
        return 0;
    }

    vocab_size = g_model->cfg.vocab_size;
    max_ctx    = g_model->cfg.ctx_len;

    /*
     * The model is allocated with a large vocab (up to 32768) but the BPE
     * tokenizer usually holds far fewer real tokens.  Sampling over the full
     * model vocab can pick token IDs that the tokenizer cannot decode, which
     * bpe_decode silently drops -> empty / broken output.  Clamp sampling to
     * the decodable range so only valid tokens are ever produced.
     */
    sample_vocab = vocab_size;
    if (g_tokenizer.trained &&
        g_tokenizer.vocab_size > TOKEN_SPECIAL_END &&
        g_tokenizer.vocab_size < sample_vocab)
        sample_vocab = g_tokenizer.vocab_size;

    /* ── Step 1: Normalise ── */
    strncpy(norm_input, user_input, CONV_MAX_TURN_BYTES - 1);
    norm_input[CONV_MAX_TURN_BYTES - 1] = '\0';
    normalize_unicode(norm_input, CONV_MAX_TURN_BYTES);

    /* ── Step 2: Language detection ── */
    lang_token = detect_nl_lang_token(norm_input, 256);
    if (lang_token < 0) lang_token = TOKEN_LANG_EN;

    stream_prev[0] = '\0';
    ctx_ids    = (int*)malloc((size_t)(max_ctx * 2) * sizeof(int));
    gen_ids    = (int*)malloc((size_t)g_cparams.max_new_tokens * sizeof(int));
    lm_logits  = (float*)malloc((size_t)vocab_size * sizeof(float));

    if (!ctx_ids || !gen_ids || !lm_logits) {
        BLOG_ERROR("cmd_converse: OOM for ctx/gen/logit buffers");
        free(ctx_ids); free(gen_ids); free(lm_logits);
        strncpy(out_text, "[OOM error]", out_sz - 1);
        out_text[out_sz - 1] = '\0';
        return 0;
    }

    /* ── Step 3: Encode prompt with history (training format) ── */
    hist_limit = g_cparams.history_turns;
    if (hist_limit < 0) hist_limit = 0;
    if (hist_limit > CONV_HISTORY_MAX) hist_limit = CONV_HISTORY_MAX;
    n_hist = conv_history_get(hist_buf, hist_limit);
    ctx_len = bpe_encode_with_history(
                  &g_tokenizer,
                  norm_input,
                  hist_buf, n_hist,
                  lang_token,
                  ctx_ids, max_ctx - 4);
    if (ctx_len < 1) ctx_len = 1;
    gen_temp = g_cparams.temperature;
    gen_topk = g_cparams.top_k;
    if (gen_topk < 1) gen_topk = 1;

    /* ── Step 4: Autoregressive generation (model tokens only) ── */
    memset(&fwd_out, 0, sizeof(fwd_out));
    fwd_out.lm_logits = lm_logits;

    for (gen_len = 0;
         gen_len < g_cparams.max_new_tokens && !g_cancel_flag; )
    {
        int    next_tok;
        int    window_len;
        int   *window;

        if (ctx_len <= max_ctx) {
            window     = ctx_ids;
            window_len = ctx_len;
        } else {
            window     = ctx_ids + (ctx_len - max_ctx);
            window_len = max_ctx;
        }

        model_forward(g_model, window, window_len, &fwd_out, NULL);

        next_tok = sample_top_k(lm_logits, sample_vocab,
                                  gen_temp, gen_topk,
                                  gen_ids, gen_len);

        if (next_tok == TOKEN_EOS || next_tok == TOKEN_PAD) {
            if (gen_len >= CONV_MIN_GEN_TOKENS) break;
            next_tok = sample_argmax_text(lm_logits, sample_vocab, gen_len);
            if (next_tok == TOKEN_EOS || next_tok == TOKEN_PAD) break;
        }

        gen_ids[gen_len] = next_tok;

        if (ctx_len < max_ctx * 2 - 1) {
            ctx_ids[ctx_len++] = next_tok;
        } else {
            memmove(ctx_ids, ctx_ids + 1,
                    (size_t)(ctx_len - 1) * sizeof(int));
            ctx_ids[ctx_len - 1] = next_tok;
        }

        if (g_cparams.stream) {
            int prev_len = (int)strlen(stream_prev);
            bpe_decode(&g_tokenizer, gen_ids, gen_len + 1,
                       stream_full, (int)sizeof(stream_full));
            if ((int)strlen(stream_full) > prev_len)
                stream_token_text(stream_full + prev_len);
            strncpy(stream_prev, stream_full, sizeof(stream_prev) - 1);
            stream_prev[sizeof(stream_prev) - 1] = '\0';
        }

        gen_len++;

        if (gen_len >= 12 && conv_reply_ends_sentence(gen_ids, gen_len))
            break;

        if ((gen_len & 7) == 0){
            tb_yield();
#ifdef _WIN32
            extern volatile DWORD g_worker_ping_ms;
            InterlockedExchange((LONG*)&g_worker_ping_ms,(LONG)GetTickCount());
#endif
        }
    }

    if (g_cparams.stream)
        stream_token_text("\r\n");

    if (gen_len > 0) {
        bpe_decode(&g_tokenizer, gen_ids, gen_len, out_text, out_sz);
        if (!out_text[0]) {
            BLOG_WARN("cmd_converse: decode empty after %d tokens, using argmax retry",
                      gen_len);
            gen_len = 0;
            ctx_len = bpe_encode_with_history(
                          &g_tokenizer, norm_input,
                          hist_buf, n_hist, lang_token,
                          ctx_ids, max_ctx - 4);
            if (ctx_len < 1) ctx_len = 1;
            while (gen_len < g_cparams.max_new_tokens && !g_cancel_flag) {
                int window_len, *window, tok;
                if (ctx_len <= max_ctx) {
                    window = ctx_ids; window_len = ctx_len;
                } else {
                    window = ctx_ids + (ctx_len - max_ctx); window_len = max_ctx;
                }
                model_forward(g_model, window, window_len, &fwd_out, NULL);
                tok = sample_argmax_text(lm_logits, sample_vocab, gen_len);
                if (tok == TOKEN_EOS || tok == TOKEN_PAD) {
                    if (gen_len >= CONV_MIN_GEN_TOKENS) break;
                    continue;
                }
                gen_ids[gen_len++] = tok;
                if (ctx_len < max_ctx * 2 - 1) ctx_ids[ctx_len++] = tok;
                else {
                    memmove(ctx_ids, ctx_ids + 1,
                            (size_t)(ctx_len - 1) * sizeof(int));
                    ctx_ids[ctx_len - 1] = tok;
                }
                if (gen_len >= 12 && conv_reply_ends_sentence(gen_ids, gen_len))
                    break;
            }
            if (gen_len > 0)
                bpe_decode(&g_tokenizer, gen_ids, gen_len, out_text, out_sz);
        }
    }

    if (gen_len > 0) {
        if (!out_text[0])
            strncpy(out_text, "[decode error]", out_sz - 1);
        out_text[out_sz - 1] = '\0';
        conv_history_push(0, norm_input);
        conv_history_push(1, out_text);
    } else {
        strncpy(out_text,
                "Model produced no tokens. Run 'easytrain' then try again.",
                out_sz - 1);
        out_text[out_sz - 1] = '\0';
        conv_history_push(0, norm_input);
    }

    /* ── Update stats ── */
    g_total_tokens_generated += gen_len;
    g_total_turns++;

    BLOG_INFO("cmd_converse: generated %d tokens stream=%d (total=%ld turns=%ld)",
              gen_len, g_cparams.stream,
              g_total_tokens_generated, g_total_turns);

    free(ctx_ids); free(gen_ids); free(lm_logits);
    return gen_len;
}

/* ═══════════════════════════════════════════════════════════════
 * §H  STATS + UTILITY
 * ═══════════════════════════════════════════════════════════════ */

void conv_print_stats(void)
{
    char buf[512];
    _snprintf(buf, sizeof(buf) - 1,
              "=== Conversation Stats ===\r\n"
              "  Total turns     : %ld\r\n"
              "  Tokens generated: %ld\r\n"
              "  History size    : %d turns\r\n"
              "  Facts loaded    : %d\r\n"
              "  Temperature     : %.3f\r\n"
              "  Top-k           : %d\r\n"
              "  Max new tokens  : %d\r\n",
              g_total_turns, g_total_tokens_generated,
              g_history.n_turns, g_n_facts,
              (double)g_cparams.temperature,
              g_cparams.top_k,
              g_cparams.max_new_tokens);
    buf[sizeof(buf) - 1] = '\0';
    /* Output via app() in brain.c; forward declared extern */
    {
        extern void app(const char *t);
        app(buf);
    }
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 6
 *
 * Files covered:
 *   converse.h –
 *     ConvHistory  ring buffer (CONV_HISTORY_MAX=32 turns)
 *     FactEntry    text + keyword index + category
 *     ConvParams   temperature, top_k, max_new_tokens,
 *                  use_facts, stream flags
 *     WM_APP_TOKEN (WM_APP+6) for live streaming to RichEdit
 *     Full API: conv_init, conv_reset, conv_set_params,
 *       fact_add, fact_seed_security, fact_search,
 *       cmd_converse, conv_history_push, conv_history_get,
 *       conv_print_stats
 *
 *   converse.c –
 *     §A  conv_init / conv_reset / conv_set_params
 *     §B  conv_history_push / conv_history_get (ring buffer)
 *     §C  fact_add / fact_search (keyword substring match)
 *         fact_seed_security – 50+ hard-coded facts:
 *           malware (ransomware, trojan, rootkit, keylogger,
 *             worm, fileless, packer, botnet, APT, miner,
 *             dropper, backdoor, exploit, adware)
 *           network/analysis (PE, entropy, IAT, injection,
 *             YARA, sandbox, C2, zero-day, MITRE, memory)
 *           cryptography (XOR, RC4, Base64, AES, RSA)
 *           AI/transformer (attention, RoPE, SwiGLU, BPE,
 *             AdamW, perplexity, INT8, gradient clip, LN,
 *             cosine LR)
 *           Windows internals (PE loader, registry, CRT,
 *             WMI, AMSI, PEB, syscall, LSASS, UAC, ETW)
 *           general (TheBrain v13, Win2000, C89)
 *     §D  sample_top_k – LCG RNG, temperature scaling,
 *         partial sort, top-k re-normalisation, sampling
 *     §E  build_fact_context – fact_search + preamble builder
 *     §F  stream_token_text – PostMessage WM_APP_TOKEN
 *         (heap-alloc string, WndProc must free LPARAM)
 *     §G  cmd_converse – full pipeline:
 *         normalize -> lang detect -> facts -> history prefix
 *         -> bpe_encode_with_history -> generation loop
 *         (model_forward + sample_top_k + sliding window)
 *         -> streaming -> bpe_decode -> history push
 *         -> BLOG_INFO stats
 *     §H  conv_print_stats
 *
 * PART 7 will cover:
 *   brain.h  (v13 updated) –
 *     All v12 structs + new BrainConfig fields:
 *       temperature, top_k, conv_use_facts, conv_stream,
 *       conv_max_tokens, sysinfo_tier, dyn_max_embeds
 *     WM_APP_TOKEN handler declaration
 *     converse.h included
 *     NLU: INTENT_CHAT -> cmd_converse wiring
 *     Real-time file guard toggle (ReadDirectoryChangesW)
 *     Model downloader: http_get + progress + verify HMAC
 *   brain.c Part A (v13 updated) –
 *     GlobalMemoryStatus probe in WinMain (before any alloc)
 *     Dynamic ModelConfig built from SysInfo RAM tier
 *     Embedding DB sized from SysInfo.max_embeds
 *     WM_APP_TOKEN handler in WndProc (appends colored text)
 *     cmd_converse wired into process_command INTENT_CHAT
 *     NLU: non-command input -> INTENT_CHAT -> cmd_converse
 *     process_command: "converse reset", "converse stats",
 *       "converse temp <f>", "converse topk <n>",
 *       "facts", "guard on/off" new commands
 *     config_save / config_load extended for new fields
 * ─────────────────────────────────────────────────────────────
 */

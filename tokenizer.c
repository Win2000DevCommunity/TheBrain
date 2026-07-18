#include "tokenizer.h"
#include "sysinfo.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

#define BPE_HT_SZ    (1<<20)
#define BPE_HT_MASK  ((1<<20)-1)

typedef struct {
    int  a;
    int  b;
    long freq;
} BPEPairEntry;

/* ═══════════════════════════════════════════════════════════════
 * §A  UTF-8 HELPERS
 * ═══════════════════════════════════════════════════════════════ */

int utf8_char_len(unsigned char lead)
{
    if (lead<0x80) return 1;
    if ((lead&0xE0)==0xC0) return 2;
    if ((lead&0xF0)==0xE0) return 3;
    if ((lead&0xF8)==0xF0) return 4;
    return 1;
}

unsigned int utf8_decode_one(const unsigned char *p, int *len_out)
{
    unsigned int cp; int len=utf8_char_len(p[0]); *len_out=len;
    switch(len){
    case 1: return (unsigned int)p[0];
    case 2:
        if ((p[1]&0xC0)!=0x80){*len_out=1;return 0xFFFD;}
        cp=((unsigned int)(p[0]&0x1F)<<6)|(unsigned int)(p[1]&0x3F);
        return cp<0x80?0xFFFD:cp;
    case 3:
        if ((p[1]&0xC0)!=0x80||(p[2]&0xC0)!=0x80){*len_out=1;return 0xFFFD;}
        cp=((unsigned int)(p[0]&0x0F)<<12)|((unsigned int)(p[1]&0x3F)<<6)|(unsigned int)(p[2]&0x3F);
        return (cp<0x800||(cp>=0xD800&&cp<=0xDFFF))?0xFFFD:cp;
    case 4:
        if ((p[1]&0xC0)!=0x80||(p[2]&0xC0)!=0x80||(p[3]&0xC0)!=0x80){*len_out=1;return 0xFFFD;}
        cp=((unsigned int)(p[0]&0x07)<<18)|((unsigned int)(p[1]&0x3F)<<12)
          |((unsigned int)(p[2]&0x3F)<<6)|(unsigned int)(p[3]&0x3F);
        return (cp<0x10000||cp>0x10FFFF)?0xFFFD:cp;
    default: return 0xFFFD;
    }
}

int utf8_encode_one(unsigned int cp, char *out)
{
    if (cp<0x80){out[0]=(char)cp;return 1;}
    if (cp<0x800){out[0]=(char)(0xC0|(cp>>6));out[1]=(char)(0x80|(cp&0x3F));return 2;}
    if (cp<0x10000){
        out[0]=(char)(0xE0|(cp>>12));out[1]=(char)(0x80|((cp>>6)&0x3F));
        out[2]=(char)(0x80|(cp&0x3F));return 3;}
    out[0]=(char)(0xF0|(cp>>18));out[1]=(char)(0x80|((cp>>12)&0x3F));
    out[2]=(char)(0x80|((cp>>6)&0x3F));out[3]=(char)(0x80|(cp&0x3F));return 4;
}

/* ═══════════════════════════════════════════════════════════════
 * §B  ARABIC PRESENTATION FORMS
 * ═══════════════════════════════════════════════════════════════ */

static const unsigned short pres_b_base[128]={
    0x0621,0x0622,0x0622,0x0623,0x0623,0x0624,0x0624,0x0625,
    0x0625,0x0626,0x0626,0x0626,0x0626,0x0627,0x0627,0x0628,
    0x0628,0x0628,0x0628,0x0629,0x0629,0x062A,0x062A,0x062A,
    0x062A,0x062B,0x062B,0x062B,0x062B,0x062C,0x062C,0x062C,
    0x062C,0x062D,0x062D,0x062D,0x062D,0x062E,0x062E,0x062E,
    0x062E,0x062F,0x062F,0x0630,0x0630,0x0631,0x0631,0x0632,
    0x0632,0x0633,0x0633,0x0633,0x0633,0x0634,0x0634,0x0634,
    0x0634,0x0635,0x0635,0x0635,0x0635,0x0636,0x0636,0x0636,
    0x0636,0x0637,0x0637,0x0637,0x0637,0x0638,0x0638,0x0638,
    0x0638,0x0639,0x0639,0x0639,0x0639,0x063A,0x063A,0x063A,
    0x063A,0x0641,0x0641,0x0641,0x0641,0x0642,0x0642,0x0642,
    0x0642,0x0643,0x0643,0x0643,0x0643,0x0644,0x0644,0x0644,
    0x0644,0x0645,0x0645,0x0645,0x0645,0x0646,0x0646,0x0646,
    0x0646,0x0647,0x0647,0x0647,0x0647,0x0648,0x0648,0x0649,
    0x0649,0x064A,0x064A,0x064A,0x064A,0x0644,0x0644,0x0644,
    0x0644,0x0644,0x0644,0x0644,0x0644,0x0644,0x0644,0x0644
};

static const unsigned short pres_a_base[96]={
    0x0671,0x0671,0x067B,0x067B,0x067B,0x067B,0x067E,0x067E,
    0x067E,0x067E,0x0680,0x0680,0x0680,0x0680,0x067A,0x067A,
    0x067A,0x067A,0x067F,0x067F,0x067F,0x067F,0x0679,0x0679,
    0x0679,0x0679,0x06A4,0x06A4,0x06A4,0x06A4,0x06A6,0x06A6,
    0x06A6,0x06A6,0x0684,0x0684,0x0684,0x0684,0x0683,0x0683,
    0x0683,0x0683,0x0686,0x0686,0x0686,0x0686,0x0687,0x0687,
    0x0687,0x0687,0x068D,0x068D,0x068C,0x068C,0x068E,0x068E,
    0x0688,0x0688,0x0698,0x0698,0x0691,0x0691,0x06A9,0x06A9,
    0x06A9,0x06A9,0x06AF,0x06AF,0x06AF,0x06AF,0x06B3,0x06B3,
    0x06B3,0x06B3,0x06B1,0x06B1,0x06B1,0x06B1,0x06BA,0x06BA,
    0x06BB,0x06BB,0x06BB,0x06BB,0x06BE,0x06BE,0x06BE,0x06BE,
    0x06C1,0x06C1,0x06C1,0x06C1,0x06BE,0x06BE,0x06BE,0x06BE
};

unsigned int arabic_pres_to_base(unsigned int cp)
{
    unsigned int idx;
    if (cp>=0xFE80&&cp<=0xFEFF){idx=cp-0xFE80;if(idx<128)return(unsigned int)pres_b_base[idx];}
    if (cp>=0xFB50&&cp<=0xFBAF){idx=cp-0xFB50;if(idx<96) return(unsigned int)pres_a_base[idx];}
    if (cp>=0xFEF5&&cp<=0xFEFC) return 0x0644;
    if (cp>=0x0660&&cp<=0x0669) return cp-0x0660+'0';
    if (cp>=0x06F0&&cp<=0x06F9) return cp-0x06F0+'0';
    return cp;
}

/* ═══════════════════════════════════════════════════════════════
 * §C  ACCENT COMBINING
 * ═══════════════════════════════════════════════════════════════ */

unsigned int combine_accent(unsigned int base, unsigned int accent)
{
    if (accent==0x0301){
        switch(base){case 0x65:return 0x00E9;case 0x45:return 0x00C9;
        case 0x61:return 0x00E1;case 0x41:return 0x00C1;case 0x75:return 0x00FA;
        case 0x55:return 0x00DA;case 0x69:return 0x00ED;case 0x49:return 0x00CD;
        case 0x6F:return 0x00F3;case 0x4F:return 0x00D3;}}
    if (accent==0x0300){
        switch(base){case 0x65:return 0x00E8;case 0x45:return 0x00C8;
        case 0x61:return 0x00E0;case 0x41:return 0x00C0;case 0x75:return 0x00F9;
        case 0x55:return 0x00D9;case 0x6F:return 0x00F2;case 0x4F:return 0x00D2;}}
    if (accent==0x0302){
        switch(base){case 0x65:return 0x00EA;case 0x61:return 0x00E2;
        case 0x69:return 0x00EE;case 0x6F:return 0x00F4;case 0x75:return 0x00FB;}}
    if (accent==0x0308){
        switch(base){case 0x65:return 0x00EB;case 0x69:return 0x00EF;
        case 0x75:return 0x00FC;case 0x61:return 0x00E4;case 0x6F:return 0x00F6;}}
    if (accent==0x0303){
        switch(base){case 0x6E:return 0x00F1;case 0x4E:return 0x00D1;
        case 0x61:return 0x00E3;case 0x6F:return 0x00F5;}}
    if (accent==0x0327){
        switch(base){case 0x63:return 0x00E7;case 0x43:return 0x00C7;}}
    if (accent==0x030A){
        switch(base){case 0x61:return 0x00E5;case 0x41:return 0x00C5;}}
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * §D  UNICODE NORMALISATION
 * ═══════════════════════════════════════════════════════════════ */

void normalize_unicode(char *text, int max_bytes)
{
    unsigned char *src=(unsigned char*)text;
    unsigned char *dst=(unsigned char*)text;
    unsigned int base_cp=0;
    char base_utf8[4];
    int base_utf8_len=0, len;
    unsigned int cp;
    (void)max_bytes;
    while (*src){
        cp=utf8_decode_one(src,&len);
        if (cp==0x200B||cp==0x200C||cp==0x200D||cp==0xFEFF||cp==0x00AD){src+=len;continue;}
        if (cp>=0xFF01&&cp<=0xFF5E) cp=cp-0xFF01+0x21;
        {unsigned int norm=arabic_pres_to_base(cp);if(norm!=cp)cp=norm;}
        if (cp>=0x0300&&cp<=0x036F){
            if (base_cp){
                unsigned int composed=combine_accent(base_cp,cp);
                if (composed){
                    dst-=base_utf8_len;
                    base_utf8_len=utf8_encode_one(composed,(char*)dst);
                    dst+=base_utf8_len; base_cp=composed;
                    src+=len; continue;}}
            base_utf8_len=utf8_encode_one(cp,(char*)dst);
            dst+=base_utf8_len; base_cp=0; src+=len; continue;}
        base_utf8_len=utf8_encode_one(cp,(char*)dst);
        dst+=base_utf8_len;
        if ((cp>=0x41&&cp<=0x7A)||(cp>=0x00C0&&cp<=0x024F)||
            (cp>=0x0600&&cp<=0x06FF)||(cp>=0x0400&&cp<=0x04FF))
            base_cp=cp;
        else base_cp=0;
        (void)base_utf8;
        src+=len;
    }
    *dst='\0';
}

/* ═══════════════════════════════════════════════════════════════
 * §E  SCRIPT DETECTION
 * ═══════════════════════════════════════════════════════════════ */

int detect_script(unsigned int cp)
{
    if ((cp>=0x0041&&cp<=0x007A)||(cp>=0x00C0&&cp<=0x024F)||(cp>=0x1E00&&cp<=0x1EFF)) return SCRIPT_LATIN;
    if ((cp>=0x0600&&cp<=0x06FF)||(cp>=0xFB50&&cp<=0xFDFF)||(cp>=0xFE70&&cp<=0xFEFF)) return SCRIPT_ARABIC;
    if ((cp>=0x3040&&cp<=0x30FF)||(cp>=0x4E00&&cp<=0x9FFF)||(cp>=0x3400&&cp<=0x4DBF)) return SCRIPT_CJK;
    if (cp>=0x0900&&cp<=0x097F) return SCRIPT_DEVA;
    if ((cp>=0x0400&&cp<=0x04FF)||(cp>=0x0500&&cp<=0x052F)) return SCRIPT_CYRIL;
    if ((cp>=0xAC00&&cp<=0xD7AF)||(cp>=0x1100&&cp<=0x11FF)) return SCRIPT_HANGUL;
    if (cp>=0x0E00&&cp<=0x0E7F) return SCRIPT_THAI;
    if (cp>=0x0590&&cp<=0x05FF) return SCRIPT_HEBREW;
    if (cp>=0x0370&&cp<=0x03FF) return SCRIPT_GREEK;
    return SCRIPT_UNKNOWN;
}

int script_to_token(int script)
{
    switch(script){
    case SCRIPT_LATIN:  return TOKEN_SCRIPT_LATIN;
    case SCRIPT_ARABIC: return TOKEN_SCRIPT_ARABIC;
    case SCRIPT_CJK:    return TOKEN_SCRIPT_CJK;
    case SCRIPT_DEVA:   return TOKEN_SCRIPT_DEVA;
    case SCRIPT_CYRIL:  return TOKEN_SCRIPT_CYRIL;
    default:            return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * §F  SENTENCE BOUNDARY + CJK SPLIT
 * ═══════════════════════════════════════════════════════════════ */

int is_sentence_end(unsigned int cp)
{
    switch(cp){
    case '.':case '!':case '?':
    case 0xFF0E:case 0xFF01:case 0xFF1F:
    case 0x061F:case 0x06D4:case 0x061B:
    case 0x3002:case 0xFF61:
    case 0x0964:case 0x0965:
    case '\n':case 0x2029:case 0x2028:
        return 1;
    default: return 0;
    }
}

int find_cjk_split(const unsigned char *buf, int max_bytes)
{
    int pos=0,prev=0;
    while (pos<max_bytes){
        int len=utf8_char_len(buf[pos]);
        if (pos+len>max_bytes) break;
        prev=pos; pos+=len;
    }
    return prev>0?prev:max_bytes;
}

/* ═══════════════════════════════════════════════════════════════
 * §G  LANGUAGE DETECTION
 * ═══════════════════════════════════════════════════════════════ */

typedef struct{const char *frag;int tok;int cls;}LangMapEntry;
static const LangMapEntry g_lang_map[]={
    {"\\C\\",     TOKEN_LANG_C,   LANG_C},
    {"/C/",       TOKEN_LANG_C,   LANG_C},
    {"\\Python\\",TOKEN_LANG_PY,  LANG_PYTHON},
    {"/Python/",  TOKEN_LANG_PY,  LANG_PYTHON},
    {"\\Pascal\\",TOKEN_LANG_PAS, LANG_PASCAL},
    {"/Pascal/",  TOKEN_LANG_PAS, LANG_PASCAL},
    {"\\ASM\\",   TOKEN_LANG_ASM, LANG_ASM},
    {"/ASM/",     TOKEN_LANG_ASM, LANG_ASM},
    {"\\ar\\",    TOKEN_LANG_AR,  LANG_AR},
    {"/ar/",      TOKEN_LANG_AR,  LANG_AR},
    {"\\fr\\",    TOKEN_LANG_FR,  LANG_FR},
    {"/fr/",      TOKEN_LANG_FR,  LANG_FR},
    {NULL,0,0}
};

int detect_lang_token(const char *filepath)
{
    int i; const char *dot;
    for (i=0;g_lang_map[i].frag;i++)
        if (strstr(filepath,g_lang_map[i].frag)) return g_lang_map[i].tok;
    dot=strrchr(filepath,'.');
    if (dot){
#ifdef _WIN32
#  define ICMP _stricmp
#else
#  define ICMP strcasecmp
#endif
        if (!ICMP(dot,".c")||!ICMP(dot,".h")||!ICMP(dot,".cpp")||!ICMP(dot,".cc")) return TOKEN_LANG_C;
        if (!ICMP(dot,".py"))  return TOKEN_LANG_PY;
        if (!ICMP(dot,".pas")) return TOKEN_LANG_PAS;
        if (!ICMP(dot,".asm")||!ICMP(dot,".s")) return TOKEN_LANG_ASM;
        if (!ICMP(dot,".conv")||!ICMP(dot,".txt")||!ICMP(dot,".md")) return TOKEN_LANG_EN;
#undef ICMP
    }
    return -1;
}

int detect_lang_class(const char *filepath)
{
    int i; const char *dot;
    for (i=0;g_lang_map[i].frag;i++)
        if (strstr(filepath,g_lang_map[i].frag)) return g_lang_map[i].cls;
    dot=strrchr(filepath,'.');
    if (dot){
#ifdef _WIN32
#  define ICMP _stricmp
#else
#  define ICMP strcasecmp
#endif
        if (!ICMP(dot,".c")||!ICMP(dot,".h")||!ICMP(dot,".cpp")) return LANG_C;
        if (!ICMP(dot,".py"))  return LANG_PYTHON;
        if (!ICMP(dot,".pas")) return LANG_PASCAL;
        if (!ICMP(dot,".asm")||!ICMP(dot,".s")) return LANG_ASM;
        if (!ICMP(dot,".conv")||!ICMP(dot,".txt")) return LANG_EN;
#undef ICMP
    }
    return LANG_EN;
}

int detect_lang_token_from_file(const char *filepath)
{
    FILE *f;
    char  buf[512];
    int   n;
    int   tok;

    tok = detect_lang_token(filepath);
    if (tok == TOKEN_LANG_AR || tok == TOKEN_LANG_FR ||
        tok == TOKEN_LANG_C || tok == TOKEN_LANG_PY ||
        tok == TOKEN_LANG_PAS || tok == TOKEN_LANG_ASM)
        return tok;

    f = fopen(filepath, "rb");
    if (!f) return tok >= 0 ? tok : TOKEN_LANG_EN;
    n = (int)fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n <= 0) return tok >= 0 ? tok : TOKEN_LANG_EN;
    buf[n] = '\0';
    return detect_nl_lang_token(buf, n);
}

int detect_nl_lang_token(const char *text, int sample_bytes)
{
    const unsigned char *p=(const unsigned char*)text;
    int latin=0,arabic=0,cyril=0,cjk=0,total=0,len;
    unsigned int cp;
    while (*p && total<sample_bytes){
        cp=utf8_decode_one(p,&len);
        switch(detect_script(cp)){
        case SCRIPT_LATIN:  latin++;  break;
        case SCRIPT_ARABIC: arabic++; break;
        case SCRIPT_CYRIL:  cyril++;  break;
        case SCRIPT_CJK:    cjk++;    break;
        default: break;
        }
        total++; p+=len;
    }
    if (arabic>latin && arabic>cyril && arabic>cjk) return TOKEN_LANG_AR;
    {
        const unsigned char *q=(const unsigned char*)text;
        int accented=0,qlen;
        while (*q){
            unsigned int qcp=utf8_decode_one(q,&qlen);
            if ((qcp>=0x00C0&&qcp<=0x00FF)||qcp==0x0152||qcp==0x0153) accented++;
            q+=qlen; if (q-(const unsigned char*)text>sample_bytes) break;
        }
        if (accented*4 > latin) return TOKEN_LANG_FR;
    }
    return TOKEN_LANG_EN;
}

/* ═══════════════════════════════════════════════════════════════
 * §H  BPE VOCAB INIT
 * ═══════════════════════════════════════════════════════════════ */

void bpe_init_vocab(BPETokenizer *t)
{
    int i;
    memset(t,0,sizeof(BPETokenizer));
    strcpy(t->vocab[TOKEN_PAD],  "<pad>");
    strcpy(t->vocab[TOKEN_BOS],  "<bos>");
    strcpy(t->vocab[TOKEN_EOS],  "<eos>");
    strcpy(t->vocab[TOKEN_UNK],  "<unk>");
    strcpy(t->vocab[TOKEN_LANG_C],   "<LANG_C>");
    strcpy(t->vocab[TOKEN_LANG_PY],  "<LANG_PY>");
    strcpy(t->vocab[TOKEN_LANG_PAS], "<LANG_PAS>");
    strcpy(t->vocab[TOKEN_LANG_ASM], "<LANG_ASM>");
    strcpy(t->vocab[TOKEN_LANG_EN],  "<LANG_EN>");
    strcpy(t->vocab[TOKEN_LANG_AR],  "<LANG_AR>");
    strcpy(t->vocab[TOKEN_LANG_FR],  "<LANG_FR>");
    strcpy(t->vocab[TOKEN_SCRIPT_LATIN],  "<SCR_LAT>");
    strcpy(t->vocab[TOKEN_SCRIPT_ARABIC], "<SCR_ARA>");
    strcpy(t->vocab[TOKEN_SCRIPT_CJK],    "<SCR_CJK>");
    strcpy(t->vocab[TOKEN_SCRIPT_DEVA],   "<SCR_DEV>");
    strcpy(t->vocab[TOKEN_SCRIPT_CYRIL],  "<SCR_CYR>");
    for (i=0;i<TOKEN_SPECIAL_END;i++)
        t->vocab_len[i]=(int)strlen(t->vocab[i]);
    for (i=0;i<256;i++){
        t->vocab[TOKEN_SPECIAL_END+i][0]=(char)(unsigned char)i;
        t->vocab[TOKEN_SPECIAL_END+i][1]='\0';
        t->vocab_len[TOKEN_SPECIAL_END+i]=1;
    }
    {
        int uid=TOKEN_SPECIAL_END+256;
        strcpy(t->vocab[uid], "<USER>"); t->vocab_len[uid]=6; uid++;
        strcpy(t->vocab[uid], "<ASST>"); t->vocab_len[uid]=6;
    }
    t->vocab_size=TOKEN_SPECIAL_END+256+2;
    t->needs_rebuild=0;
}

/* ─── BPE merge apply ────────────────────────────────────────── */
static int apply_merge_once(int *seq, int seq_len, int a, int b, int c)
{
    int ri=0,wi=0;
    while (ri<seq_len){
        if (ri<seq_len-1&&seq[ri]==a&&seq[ri+1]==b){seq[wi++]=c;ri+=2;}
        else seq[wi++]=seq[ri++];
    }
    return wi;
}

static void bpe_do_merges(BPETokenizer *t, int *seq, long seq_len, int max_merges)
{
    int iter;
    for (iter=0;iter<max_merges&&t->vocab_size<BPE_VOCAB_MAX;iter++){
        int  best_a=-1, best_b=-1, new_id;
        long best_freq=0, i;
        int  vs=t->vocab_size;
        int  s;
        BPEPairEntry *ht=(BPEPairEntry*)calloc(BPE_HT_SZ,sizeof(BPEPairEntry));
        if (!ht) break;

        for (i=0;i<seq_len-1;i++){
            int a=seq[i], b=seq[i+1];
            unsigned int slot;
            if (a<0||b<0||a>=vs||b>=vs) continue;
            slot=((unsigned int)(a*1000003)^(unsigned int)b)&(unsigned int)BPE_HT_MASK;
            for(;;){
                if (!ht[slot].freq){ht[slot].a=a;ht[slot].b=b;ht[slot].freq=1;break;}
                if (ht[slot].a==a&&ht[slot].b==b){ht[slot].freq++;break;}
                slot=(slot+1)&(unsigned int)BPE_HT_MASK;
            }
        }
        for (s=0;s<BPE_HT_SZ;s++){
            if (ht[s].freq>best_freq){
                best_freq=ht[s].freq; best_a=ht[s].a; best_b=ht[s].b;
            }
        }
        free(ht);

        if (best_freq<2) break;

        new_id=t->vocab_size;
        {
            int la=t->vocab_len[best_a],lb=t->vocab_len[best_b],nl=la+lb;
            if (nl>=BPE_MAX_WORD) nl=BPE_MAX_WORD-1;
            memcpy(t->vocab[new_id],t->vocab[best_a],(size_t)la);
            memcpy(t->vocab[new_id]+la,t->vocab[best_b],(size_t)lb);
            t->vocab[new_id][nl]='\0'; t->vocab_len[new_id]=nl;
        }
        t->merge_a[t->n_merges]=best_a;
        t->merge_b[t->n_merges]=best_b;
        t->merge_c[t->n_merges]=new_id;
        t->n_merges++; t->vocab_size++;
        seq_len=apply_merge_once(seq,(int)seq_len,best_a,best_b,new_id);

        if ((iter+1)%500==0)
            printf("  BPE iter %d: vocab=%d seq=%ld\n",iter+1,t->vocab_size,seq_len);
        tb_yield_bg();
    }
}

static int bpe_is_text_ext(const char *name)
{
    const char *dot=strrchr(name,'.');
    if (!dot) return 0;
#ifdef _WIN32
#  define ICMP _stricmp
#else
#  define ICMP strcasecmp
#endif
    if (!ICMP(dot,".conv")||!ICMP(dot,".txt")||!ICMP(dot,".md")) return 1;
#undef ICMP
    return 0;
}

static int bpe_is_conv_ext(const char *name)
{
    const char *dot=strrchr(name,'.');
    if (!dot) return 0;
#ifdef _WIN32
    return !_stricmp(dot,".conv");
#else
    return !strcasecmp(dot,".conv");
#endif
}

#ifdef _WIN32
static int bpe_collect_conv_files(const char *dir, char paths[][512], int max_n, int n)
{
    WIN32_FIND_DATAA fd;
    char pattern[520];
    HANDLE h;

    _snprintf(pattern,sizeof(pattern)-1,"%s\\*.*",dir);
    pattern[sizeof(pattern)-1]='\0';
    h=FindFirstFileA(pattern,&fd);
    if (h==INVALID_HANDLE_VALUE) return n;
    do {
        if (fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){
            if (strcmp(fd.cFileName,".")&&strcmp(fd.cFileName,"..")&&n<max_n){
                char sub[520];
                _snprintf(sub,sizeof(sub)-1,"%s\\%s",dir,fd.cFileName);
                sub[sizeof(sub)-1]='\0';
                n=bpe_collect_conv_files(sub,paths,max_n,n);
            }
            continue;
        }
        if (n>=max_n) break;
        if (bpe_is_conv_ext(fd.cFileName)){
            _snprintf(paths[n],511,"%s\\%s",dir,fd.cFileName);
            paths[n][511]='\0';
            n++;
        }
    } while (FindNextFileA(h,&fd));
    FindClose(h);
    return n;
}

static int bpe_collect_text_files(const char *dir, char paths[][512], int max_n, int n)
{
    WIN32_FIND_DATAA fd;
    char pattern[520];
    HANDLE h;

    _snprintf(pattern,sizeof(pattern)-1,"%s\\*.*",dir);
    pattern[sizeof(pattern)-1]='\0';
    h=FindFirstFileA(pattern,&fd);
    if (h==INVALID_HANDLE_VALUE) return n;
    do {
        if (fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){
            if (strcmp(fd.cFileName,".")&&strcmp(fd.cFileName,"..")&&n<max_n){
                char sub[520];
                _snprintf(sub,sizeof(sub)-1,"%s\\%s",dir,fd.cFileName);
                sub[sizeof(sub)-1]='\0';
                n=bpe_collect_text_files(sub,paths,max_n,n);
            }
            continue;
        }
        if (n>=max_n) break;
        if (bpe_is_text_ext(fd.cFileName)){
            _snprintf(paths[n],511,"%s\\%s",dir,fd.cFileName);
            paths[n][511]='\0';
            n++;
        }
    } while (FindNextFileA(h,&fd));
    FindClose(h);
    return n;
}
#endif

static void bpe_append_byte(int **seq, long *seq_len, long *seq_cap, unsigned char c)
{
    if (*seq_len>=*seq_cap){
        int *ns; *seq_cap*=2;
        ns=(int*)realloc(*seq,(size_t)(*seq_cap)*sizeof(int));
        if (!ns) return;
        *seq=ns;
    }
    (*seq)[(*seq_len)++] = TOKEN_SPECIAL_END + c;
}

static long bpe_append_file(int **seq, long *seq_len, long *seq_cap, const char *path)
{
    FILE *fp;
    int c;
    fp=fopen(path,"rb");
    if (!fp) return *seq_len;
    while ((c=fgetc(fp))!=EOF)
        bpe_append_byte(seq, seq_len, seq_cap, (unsigned char)c);
    fclose(fp);
    return *seq_len;
}

static long bpe_append_conv_file(int **seq, long *seq_len, long *seq_cap,
                                   const char *path)
{
    Conversation conv;
    int t;
    const char *p;

    if (bpe_parse_conv_file(path, &conv) <= 0) return *seq_len;
    for (t = 0; t < conv.n_turns; t++) {
        p = conv.turns[t].text;
        while (*p)
            bpe_append_byte(seq, seq_len, seq_cap, (unsigned char)*p++);
        bpe_append_byte(seq, seq_len, seq_cap, (unsigned char)'\n');
    }
    return *seq_len;
}

/* ═══════════════════════════════════════════════════════════════
 * §I  BPE LEARN  — FIX 2 APPLIED
 *     BPEPairEntry now at file scope (above), not inside function
 * ═══════════════════════════════════════════════════════════════ */

void bpe_learn_from_file(BPETokenizer *t, const char *path, int max_merges)
{
    int  *seq=NULL;
    long  seq_len=0, seq_cap=0;
    FILE *fp;

    fp=fopen(path,"rb");
    if (!fp){ BLOG_WARN("bpe_learn: cannot open %s",path); return; }
    fclose(fp);

    seq_cap=1024*1024;
    seq=(int*)malloc((size_t)seq_cap*sizeof(int));
    if (!seq){ BLOG_ERROR("bpe_learn: OOM for seq buffer"); return; }

    bpe_append_file(&seq,&seq_len,&seq_cap,path);
    if (seq_len < 2){ free(seq); BLOG_WARN("bpe_learn: file too small %s",path); return; }

    bpe_do_merges(t, seq, seq_len, max_merges);
    free(seq);
    t->trained=1;
    if (t->vocab_size > BPE_VOCAB_SIZE-512)
        t->needs_rebuild=1;
    BLOG_INFO("bpe_learn: vocab=%d merges=%d needs_rebuild=%d",
              t->vocab_size,t->n_merges,t->needs_rebuild);
}

void bpe_learn_from_dir(BPETokenizer *t, const char *dir, int max_merges)
{
#ifdef _WIN32
    char paths[256][512];
    int  nfiles, i;
    int *seq=NULL;
    long seq_len=0, seq_cap=0;

    nfiles=bpe_collect_text_files(dir,paths,256,0);
    if (nfiles<=0){
        BLOG_WARN("bpe_learn_dir: no .conv/.txt/.md in %s",dir);
        return;
    }

    seq_cap=1024*1024;
    seq=(int*)malloc((size_t)seq_cap*sizeof(int));
    if (!seq){ BLOG_ERROR("bpe_learn_dir: OOM"); return; }

    for (i=0;i<nfiles;i++)
        bpe_append_file(&seq,&seq_len,&seq_cap,paths[i]);

    if (seq_len < 2){ free(seq); BLOG_WARN("bpe_learn_dir: corpus too small"); return; }

    BLOG_INFO("bpe_learn_dir: %d files, %ld bytes",nfiles,seq_len);
    bpe_do_merges(t, seq, seq_len, max_merges);
    free(seq);
    t->trained=1;
    if (t->vocab_size > BPE_VOCAB_SIZE-512)
        t->needs_rebuild=1;
    BLOG_INFO("bpe_learn: vocab=%d merges=%d needs_rebuild=%d",
              t->vocab_size,t->n_merges,t->needs_rebuild);
#else
    (void)t;(void)dir;(void)max_merges;
#endif
}

void bpe_learn_from_conv_dir(BPETokenizer *t, const char *dir, int max_merges)
{
#ifdef _WIN32
    char paths[256][512];
    int  nfiles, i;
    int *seq=NULL;
    long seq_len=0, seq_cap=0;

    nfiles=bpe_collect_conv_files(dir,paths,256,0);
    if (nfiles<=0){
        BLOG_WARN("bpe_learn_conv_dir: no .conv in %s",dir);
        return;
    }

    seq_cap=1024*1024;
    seq=(int*)malloc((size_t)seq_cap*sizeof(int));
    if (!seq){ BLOG_ERROR("bpe_learn_conv_dir: OOM"); return; }

    for (i=0;i<nfiles;i++)
        bpe_append_conv_file(&seq,&seq_len,&seq_cap,paths[i]);

    if (seq_len < 2){ free(seq); BLOG_WARN("bpe_learn_conv_dir: corpus too small"); return; }

    BLOG_INFO("bpe_learn_conv_dir: %d conv files, %ld bytes",nfiles,seq_len);
    bpe_do_merges(t, seq, seq_len, max_merges);
    free(seq);
    t->trained=1;
    if (t->vocab_size > BPE_VOCAB_SIZE-512)
        t->needs_rebuild=1;
    BLOG_INFO("bpe_learn_conv: vocab=%d merges=%d",t->vocab_size,t->n_merges);
#else
    (void)t;(void)dir;(void)max_merges;
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * §J  SAVE / LOAD
 * ═══════════════════════════════════════════════════════════════ */

void bpe_save(const BPETokenizer *t, const char *path)
{
    FILE *fp=fopen(path,"wb"); if (!fp) return;
    fwrite(&t->vocab_size,sizeof(int),1,fp);
    fwrite(&t->n_merges,  sizeof(int),1,fp);
    fwrite(t->vocab,     BPE_MAX_WORD*(size_t)t->vocab_size,1,fp);
    fwrite(t->vocab_len, sizeof(int),(size_t)t->vocab_size,fp);
    fwrite(t->merge_a,sizeof(int),(size_t)t->n_merges,fp);
    fwrite(t->merge_b,sizeof(int),(size_t)t->n_merges,fp);
    fwrite(t->merge_c,sizeof(int),(size_t)t->n_merges,fp);
    fclose(fp);
    BLOG_INFO("bpe_save: vocab=%d -> %s",t->vocab_size,path);
}

int bpe_load(BPETokenizer *t, const char *path)
{
    FILE *fp=fopen(path,"rb"); if (!fp) return 0;
    fread(&t->vocab_size,sizeof(int),1,fp);
    fread(&t->n_merges,  sizeof(int),1,fp);
    if (t->vocab_size<0||t->vocab_size>BPE_VOCAB_MAX||
        t->n_merges<0  ||t->n_merges>BPE_MAX_MERGES){
        fclose(fp);
        BLOG_ERROR("bpe_load: corrupt vocab (size=%d merges=%d) in %s",
                   t->vocab_size,t->n_merges,path);
        bpe_init_vocab(t); return 0;
    }
    fread(t->vocab,     BPE_MAX_WORD*(size_t)t->vocab_size,1,fp);
    fread(t->vocab_len, sizeof(int),(size_t)t->vocab_size,fp);
    fread(t->merge_a,sizeof(int),(size_t)t->n_merges,fp);
    fread(t->merge_b,sizeof(int),(size_t)t->n_merges,fp);
    fread(t->merge_c,sizeof(int),(size_t)t->n_merges,fp);
    fclose(fp);
    t->trained=1;
    t->needs_rebuild=(t->vocab_size>BPE_VOCAB_SIZE-512)?1:0;
    BLOG_INFO("bpe_load: vocab=%d merges=%d from %s",t->vocab_size,t->n_merges,path);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * §K  ENCODE
 * ═══════════════════════════════════════════════════════════════ */

static int encode_raw(const BPETokenizer *t, const char *text,
                       int *out_ids, int max_ids)
{
    int *seq; int seq_len=0,n_ids,i;
    const char *p=text;
    size_t alloc=strlen(text)+1;
    if (alloc>(size_t)max_ids*2+1) alloc=(size_t)max_ids*2+1;
    seq=(int*)malloc(alloc*sizeof(int)); if (!seq) return 0;
    while (*p&&seq_len<(int)alloc-1)
        seq[seq_len++]=TOKEN_SPECIAL_END+(unsigned char)*p++;
    for (i=0;i<t->n_merges&&seq_len>1;i++)
        seq_len=apply_merge_once(seq,seq_len,t->merge_a[i],t->merge_b[i],t->merge_c[i]);
    n_ids=seq_len<max_ids?seq_len:max_ids;
    memcpy(out_ids,seq,(size_t)n_ids*sizeof(int));
    free(seq); return n_ids;
}

int bpe_encode(const BPETokenizer *t,const char *text,int *out_ids,int max_ids)
{
    size_t tlen=strlen(text); char *tmp=(char*)malloc(tlen+4); int n;
    if (!tmp) return 0;
    memcpy(tmp,text,tlen+1);
    normalize_unicode(tmp,(int)(tlen+4));
    n=encode_raw(t,tmp,out_ids,max_ids);
    free(tmp); return n;
}

int bpe_encode_with_lang(const BPETokenizer *t,const char *text,
                          int lang_token,int *out_ids,int max_ids)
{
    int offset=0,n;
    if (lang_token>=0&&max_ids>1){out_ids[0]=lang_token;offset=1;}
    n=bpe_encode(t,text,out_ids+offset,max_ids-offset);
    return n+offset;
}

int bpe_encode_multilingual(const BPETokenizer *t,const char *text,
                              int lang_token,int *out_ids,int max_ids)
{
    char *tmp; size_t tlen; const unsigned char *src;
    int pos=0,last_script=-1,seg_start=0,seg_byte=0;

    tlen=strlen(text); tmp=(char*)malloc(tlen+4); if (!tmp) return 0;
    memcpy(tmp,text,tlen+1);
    normalize_unicode(tmp,(int)(tlen+4));
    tlen=strlen(tmp);

    if (lang_token>=0&&pos<max_ids) out_ids[pos++]=lang_token;
    src=(const unsigned char*)tmp;

    while (src[seg_byte]&&pos<max_ids-4){
        int char_len; unsigned int cp=utf8_decode_one(src+seg_byte,&char_len);
        int script=detect_script(cp);
        if (script!=SCRIPT_UNKNOWN&&script!=last_script){
            if (seg_byte>seg_start&&pos<max_ids-2){
                int n; char seg_buf[BPE_MAX_WORD*4];
                int seg_len=seg_byte-seg_start;
                if (seg_len>=(int)sizeof(seg_buf)) seg_len=(int)sizeof(seg_buf)-1;
                memcpy(seg_buf,tmp+seg_start,(size_t)seg_len);
                seg_buf[seg_len]='\0';
                n=encode_raw(t,seg_buf,out_ids+pos,max_ids-pos-1);
                pos+=n;
            }
            {int st=script_to_token(script);if(st>=0&&pos<max_ids-1)out_ids[pos++]=st;}
            last_script=script; seg_start=seg_byte;
        }
        seg_byte+=char_len;
    }
    if (seg_byte>seg_start&&pos<max_ids-1){
        int n; char seg_buf[BPE_MAX_WORD*4];
        int seg_len=seg_byte-seg_start;
        if (seg_len>=(int)sizeof(seg_buf)) seg_len=(int)sizeof(seg_buf)-1;
        memcpy(seg_buf,tmp+seg_start,(size_t)seg_len);
        seg_buf[seg_len]='\0';
        n=encode_raw(t,seg_buf,out_ids+pos,max_ids-pos-1);
        pos+=n;
    }
    free(tmp); return pos;
}

/* ═══════════════════════════════════════════════════════════════
 * §L  CONVERSATION ENCODING
 * ═══════════════════════════════════════════════════════════════ */

int bpe_parse_conv_file(const char *path, Conversation *out)
{
    FILE *fp=fopen(path,"r"); char line[CONV_MAX_TURN_BYTES+8];
    if (!fp) return 0;
    memset(out,0,sizeof(Conversation));
    while (fgets(line,sizeof(line),fp) && out->n_turns<CONV_MAX_TURNS){
        int len=(int)strlen(line); char *p;
        while (len>0&&(line[len-1]=='\r'||line[len-1]=='\n'))line[--len]='\0';
        if (!len||line[0]=='#') continue;
        p=line+2;
        if (line[0]=='U'&&line[1]==':'){
            out->turns[out->n_turns].role=0;
            strncpy(out->turns[out->n_turns].text,p[0]==' '?p+1:p,CONV_MAX_TURN_BYTES-1);
            out->turns[out->n_turns].text[CONV_MAX_TURN_BYTES-1]='\0';
            out->n_turns++;
        } else if (line[0]=='A'&&line[1]==':'){
            out->turns[out->n_turns].role=1;
            strncpy(out->turns[out->n_turns].text,p[0]==' '?p+1:p,CONV_MAX_TURN_BYTES-1);
            out->turns[out->n_turns].text[CONV_MAX_TURN_BYTES-1]='\0';
            out->n_turns++;
        }
    }
    fclose(fp);
    return out->n_turns;
}

int bpe_encode_conv_turn(const BPETokenizer *t, const ConvTurn *turn,
                          int lang_token, int *out_ids, int max_ids)
{
    int pos=0, n, i;
    const char *marker=(turn->role==0)?"<USER>":"<ASST>";
    for (i=0;i<t->vocab_size&&pos<max_ids;i++){
        if (strcmp(t->vocab[i],marker)==0){out_ids[pos++]=i;break;}
    }
    n=bpe_encode_multilingual(t,turn->text,
                               lang_token>=0?lang_token:TOKEN_LANG_EN,
                               out_ids+pos,max_ids-pos);
    return pos+n;
}

int bpe_encode_conv(const BPETokenizer *t, const Conversation *conv,
                     int lang_token, int *out_ids, int max_ids)
{
    int pos=0,i,n;
    for (i=0;i<conv->n_turns&&pos<max_ids-4;i++){
        n=bpe_encode_conv_turn(t,&conv->turns[i],lang_token,out_ids+pos,max_ids-pos-2);
        pos+=n;
    }
    if (pos<max_ids) out_ids[pos++]=TOKEN_EOS;
    return pos;
}

int bpe_encode_with_history(const BPETokenizer *t,
                              const char *current_input,
                              const ConvTurn *history, int n_history,
                              int lang_token,
                              int *out_ids, int max_ids)
{
    Conversation tmp_conv;
    int i, copy_from, n;

    memset(&tmp_conv,0,sizeof(tmp_conv));
    copy_from=n_history-(CONV_MAX_TURNS-1);
    if (copy_from<0) copy_from=0;
    for (i=copy_from;i<n_history&&tmp_conv.n_turns<CONV_MAX_TURNS-1;i++){
        tmp_conv.turns[tmp_conv.n_turns++]=history[i];
    }
    tmp_conv.turns[tmp_conv.n_turns].role=0;
    strncpy(tmp_conv.turns[tmp_conv.n_turns].text,current_input,CONV_MAX_TURN_BYTES-1);
    tmp_conv.turns[tmp_conv.n_turns].text[CONV_MAX_TURN_BYTES-1]='\0';
    tmp_conv.n_turns++;

    n=bpe_encode_conv(t,&tmp_conv,lang_token,out_ids,max_ids);
    if (n > 0 && out_ids[n-1] == TOKEN_EOS) {
        int i, asst_id = TOKEN_EOS;
        for (i = 0; i < t->vocab_size; i++) {
            if (!strcmp(t->vocab[i], "<ASST>")) { asst_id = i; break; }
        }
        if (asst_id != TOKEN_EOS)
            out_ids[n-1] = asst_id;
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * §M  AUTO-SCALE
 * ═══════════════════════════════════════════════════════════════ */

int tokenizer_needs_rebuild(const BPETokenizer *t)
{
    return t->needs_rebuild || (t->vocab_size > BPE_VOCAB_SIZE-256);
}

int bpe_rebuild_larger(BPETokenizer *t, const char *corpus_path,
                        int additional_merges)
{
    int old_vs=t->vocab_size, old_nm=t->n_merges;
    int target=old_nm+additional_merges;
    if (target>BPE_MAX_MERGES) target=BPE_MAX_MERGES;
    if (t->vocab_size>=BPE_VOCAB_MAX){
        BLOG_WARN("bpe_rebuild_larger: already at max vocab=%d",t->vocab_size);
        return 0;
    }
    BLOG_INFO("bpe_rebuild_larger: vocab %d -> target %d",old_vs,old_vs+additional_merges);
    bpe_learn_from_file(t,corpus_path,target-old_nm);
    t->needs_rebuild=0;
    BLOG_INFO("bpe_rebuild_larger: done, vocab=%d",t->vocab_size);
    return t->vocab_size-old_vs;
}

/* ═══════════════════════════════════════════════════════════════
 * §N  DECODE
 * ═══════════════════════════════════════════════════════════════ */

void bpe_decode(const BPETokenizer *t, const int *ids, int n,
                 char *out, int outsz)
{
    int i,ol=0; out[0]='\0';
    for (i=0;i<n;i++){
        int id=ids[i],tl;
        if (id<TOKEN_SPECIAL_END) continue;
        if (id<0||id>=t->vocab_size) continue;
        tl=t->vocab_len[id];
        if (ol+tl<outsz-1){memcpy(out+ol,t->vocab[id],(size_t)tl);ol+=tl;}
    }
    out[ol]='\0';
}

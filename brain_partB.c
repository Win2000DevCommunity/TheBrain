#include "brain.h"

int g_override_tier = -1;

static void config_save(void);
static void config_load(void);
static void checkpoint_save(void);
static void checkpoint_load_latest(void);
static void dead_letter_append(const char *text);
static void rollback_to_checkpoint(int idx);
static void cmd_checkpoints(void);
void cmd_config_show(void);
void cmd_config_set(const char *key, const char *val);
static void cmd_deadletter(void);
static void cmd_rollback(int idx);
static void cmd_retry(void);
void dispatch_async(TaskType tt, const char *arg1,
                            const char *arg2, int i1, double d1);

/* Undo helpers (defined in Part A §O, declared here for §N) */

/* ═══════════════════════════════════════════════════════════════
 * §B  PE ENTROPY / HEADER / IMPORTS / x86 DISASM
 * ═══════════════════════════════════════════════════════════════ */

static double buf_entropy(const unsigned char *buf, size_t n)
{
    double freq[256], e = 0.0; size_t i;
    if (!n) return 0.0;
    memset(freq,0,sizeof(freq));
    for (i=0;i<n;i++) freq[buf[i]]+=1.0;
    for (i=0;i<256;i++){double p=freq[i]/(double)n;if(p>1e-12)e-=p*log(p)/0.693147180559945;}
    return e;
}

void cmd_entropy(const char *file)
{
    HANDLE hf; DWORD fsz,done,i,n_secs;
    unsigned char *buf; MY_DOS *dos; MY_NTHDRS *nt; MY_SECHDR *sec;
    char msg[512]; double whole_ent;
    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){app_warn("entropy: cannot open file\r\n");return;}
    fsz=GetFileSize(hf,NULL);
    if(!fsz||fsz>PE_MAX_MAP_SIZE){CloseHandle(hf);app_warn("entropy: file too large\r\n");return;}
    buf=(unsigned char*)malloc(fsz);
    if(!buf){CloseHandle(hf);app_warn("entropy: OOM\r\n");return;}
    ReadFile(hf,buf,fsz,&done,NULL); CloseHandle(hf);
    whole_ent=buf_entropy(buf,(size_t)done);
    safe_fmt(msg,sizeof(msg),"=== ENTROPY: %s ===\r\n  File: %lu bytes\r\n  Whole: %.4f bits/byte  %s\r\n",
             file,(unsigned long)done,whole_ent,
             whole_ent>7.2?"[HIGH - packer/crypto]":whole_ent>6.0?"[MEDIUM]":"[NORMAL]");
    app_info(msg);
    dos=(MY_DOS*)buf;
    if(done>sizeof(MY_DOS)&&dos->e_magic==MY_IMAGE_DOS_SIGNATURE&&
       (DWORD)dos->e_lfanew+sizeof(MY_NTHDRS)<=done){
        nt=(MY_NTHDRS*)(buf+dos->e_lfanew); n_secs=nt->File.NumSecs;
        sec=(MY_SECHDR*)((unsigned char*)nt+sizeof(DWORD)+sizeof(MY_FILEHDR)+nt->File.SzOpt);
        app_info("  Sections:\r\n");
        for(i=0;i<n_secs;i++){
            char name[10]; double se;
            DWORD rp=sec[i].RawPtr,rs=sec[i].RawSz;
            if(!rs||rp+rs>done) continue;
            memset(name,0,sizeof(name)); memcpy(name,sec[i].Name,8);
            se=buf_entropy(buf+rp,rs);
            safe_fmt(msg,sizeof(msg),"    %-10s  %7lu bytes  %.4f  %s\r\n",
                     name,(unsigned long)rs,se,
                     se>7.2?"[PACKED/CRYPTO]":se>6.0?"[COMPRESSED?]":"");
            if(se>7.2)app_danger(msg); else if(se>6.0)app_warn(msg); else app(msg);
        }
    }
    free(buf);
}

void cmd_pe_header(const char *file)
{
    HANDLE hf; DWORD done; unsigned char buf[4096];
    MY_DOS *dos; MY_NTHDRS *nt; MY_SECHDR *sec; DWORD i,n_secs; char msg[512];
    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){app_warn("pe header: cannot open\r\n");return;}
    ReadFile(hf,buf,sizeof(buf),&done,NULL); CloseHandle(hf);
    dos=(MY_DOS*)buf;
    if(done<sizeof(MY_DOS)||dos->e_magic!=MY_IMAGE_DOS_SIGNATURE){app_warn("pe header: not PE\r\n");return;}
    if((DWORD)dos->e_lfanew+sizeof(MY_NTHDRS)>done){app_warn("pe header: truncated\r\n");return;}
    nt=(MY_NTHDRS*)(buf+dos->e_lfanew);
    if(nt->Sig!=MY_IMAGE_NT_SIGNATURE){app_warn("pe header: bad signature\r\n");return;}
    safe_fmt(msg,sizeof(msg),
             "=== PE HEADER: %s ===\r\n"
             "  Machine    : 0x%04X (%s)\r\n"
             "  Sections   : %u\r\n"
             "  Timestamp  : 0x%08lX\r\n"
             "  Chars      : 0x%04X\r\n",
             file,nt->File.Machine,
             nt->File.Machine==0x014C?"x86":nt->File.Machine==0x8664?"x64":
             nt->File.Machine==0x01C0?"ARM":"Unknown",
             (unsigned)nt->File.NumSecs,(unsigned long)nt->File.TS,
             (unsigned)nt->File.Chars);
    app_info(msg);
    safe_fmt(msg,sizeof(msg),
             "  EntryPoint : 0x%08lX\r\n"
             "  ImageBase  : 0x%08lX\r\n"
             "  SizeImage  : 0x%08lX\r\n"
             "  Subsystem  : %u (%s)\r\n"
             "  CheckSum   : 0x%08lX\r\n",
             (unsigned long)nt->Opt.EP,(unsigned long)nt->Opt.ImageBase,
             (unsigned long)nt->Opt.SzImage,(unsigned)nt->Opt.Subsys,
             nt->Opt.Subsys==2?"GUI":nt->Opt.Subsys==3?"CUI":
             nt->Opt.Subsys==1?"Native":"Other",
             (unsigned long)nt->Opt.CheckSum);
    app(msg);
    n_secs=nt->File.NumSecs;
    sec=(MY_SECHDR*)((unsigned char*)nt+sizeof(DWORD)+sizeof(MY_FILEHDR)+nt->File.SzOpt);
    app_info("  Name       VirtAddr   VirtSz     RawSz      Chars\r\n");
    for(i=0;i<n_secs;i++){
        char name[10]; memset(name,0,sizeof(name)); memcpy(name,sec[i].Name,8);
        safe_fmt(msg,sizeof(msg),"  %-8s   %08lX   %08lX   %08lX   %08lX\r\n",
                 name,(unsigned long)sec[i].VirtAddr,(unsigned long)sec[i].VirtSz,
                 (unsigned long)sec[i].RawSz,(unsigned long)sec[i].Chars);
        app(msg);
    }
}

void cmd_pe_imports(const char *file)
{
    HANDLE hf; DWORD fsz,done,i,n_secs,imp_rva,imp_sz;
    unsigned char *buf; MY_DOS *dos; MY_NTHDRS *nt; MY_SECHDR *sec;
    DWORD sec_va,sec_raw,sec_rsz; MY_IMPORT_DESC *imd;
    char msg[512]; int n_sus=0;
    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){app_warn("pe imports: cannot open\r\n");return;}
    fsz=GetFileSize(hf,NULL);
    if(!fsz||fsz>PE_MAX_MAP_SIZE){CloseHandle(hf);return;}
    buf=(unsigned char*)malloc(fsz);
    if(!buf){CloseHandle(hf);app_warn("pe imports: OOM\r\n");return;}
    ReadFile(hf,buf,fsz,&done,NULL); CloseHandle(hf);
    dos=(MY_DOS*)buf;
    if(done<sizeof(MY_DOS)||dos->e_magic!=MY_IMAGE_DOS_SIGNATURE){free(buf);app_warn("pe imports: not PE\r\n");return;}
    nt=(MY_NTHDRS*)(buf+dos->e_lfanew);
    if(nt->Sig!=MY_IMAGE_NT_SIGNATURE){free(buf);app_warn("pe imports: invalid PE\r\n");return;}
    if(nt->Opt.NumRva<2){free(buf);app_warn("pe imports: no import dir\r\n");return;}
    {DWORD *dd=(DWORD*)((unsigned char*)&nt->Opt+96);imp_rva=dd[0];imp_sz=dd[1];}
    if(!imp_rva){free(buf);app_info("pe imports: none\r\n");return;}
    n_secs=nt->File.NumSecs;
    sec=(MY_SECHDR*)((unsigned char*)nt+sizeof(DWORD)+sizeof(MY_FILEHDR)+nt->File.SzOpt);
    sec_va=sec_raw=sec_rsz=0;
    for(i=0;i<n_secs;i++){
        if(imp_rva>=sec[i].VirtAddr&&imp_rva<sec[i].VirtAddr+sec[i].VirtSz){
            sec_va=sec[i].VirtAddr;sec_raw=sec[i].RawPtr;sec_rsz=sec[i].RawSz;break;}}
    if(!sec_raw){free(buf);app_warn("pe imports: cannot map RVA\r\n");return;}
#define RVA2OFF(rva) ((DWORD)((rva)-sec_va+sec_raw))
    safe_fmt(msg,sizeof(msg),"=== PE IMPORTS: %s ===\r\n",file); app_info(msg);
    imd=(MY_IMPORT_DESC*)(buf+RVA2OFF(imp_rva));
    while(imd->Name&&RVA2OFF(imd->Name)+2<done){
        char *dll_name=(char*)(buf+RVA2OFF(imd->Name)); DWORD *thunk;
        safe_fmt(msg,sizeof(msg),"  DLL: %s\r\n",dll_name); app_orange(msg);
        if(imd->OFT&&RVA2OFF(imd->OFT)<done){
            thunk=(DWORD*)(buf+RVA2OFF(imd->OFT));
            while(*thunk&&!(*thunk&0x80000000UL)){
                DWORD nrva=*thunk,noff; const char *fn; int j,sus=0;
                if(RVA2OFF(nrva)+2>=done){thunk++;continue;}
                noff=RVA2OFF(nrva)+2; fn=(const char*)(buf+noff);
                for(j=0;g_sus_apis[j];j++) if(!strcmp(fn,g_sus_apis[j])){sus=1;n_sus++;break;}
                safe_fmt(msg,sizeof(msg),"    %s%s\r\n",fn,sus?"  [!]":"");
                if(sus)app_danger(msg); else app(msg);
                thunk++;}}
        imd++;}
    if(n_sus>0){safe_fmt(msg,sizeof(msg),"  ** %d suspicious API(s) **\r\n",n_sus);app_danger(msg);}
    free(buf);
#undef RVA2OFF
}

/* x86 length decoder */
static int x86_instr_len(const unsigned char *p, int remain)
{
    int plen=0,has66=0; unsigned char op; int len;
    while(remain>0){
        unsigned char b=*p;
        if(b==0x66||b==0x67||b==0xF2||b==0xF3||b==0xF0||
           b==0x26||b==0x2E||b==0x36||b==0x3E||b==0x64||b==0x65){
            if(b==0x66)has66=1; p++;remain--;plen++;
        }else break;}
    if(remain<=0)return plen+1;
    op=*p;
    if(op==0x0F){
        if(remain<2)return plen+2;
        {unsigned char op2=p[1];
         if(op2>=0x80&&op2<=0x8F)return plen+6;
         if(op2==0xC8)return plen+2;}
        return plen+3;}
    switch(op){
        case 0x90:case 0xC3:case 0xC9:case 0xCB:case 0xCC:case 0xF4:
        case 0xF8:case 0xF9:case 0xFA:case 0xFB:case 0xFC:case 0xFD:
        case 0x9C:case 0x9D:case 0x60:case 0x61:case 0x98:case 0x99:
        case 0x50:case 0x51:case 0x52:case 0x53:case 0x54:case 0x55:
        case 0x56:case 0x57:case 0x58:case 0x59:case 0x5A:case 0x5B:
        case 0x5C:case 0x5D:case 0x5E:case 0x5F:
        case 0x40:case 0x41:case 0x42:case 0x43:case 0x44:case 0x45:
        case 0x46:case 0x47:case 0x48:case 0x49:case 0x4A:case 0x4B:
        case 0x4C:case 0x4D:case 0x4E:case 0x4F:
            return plen+1;
        case 0xE8:case 0xE9: return plen+5;
        case 0xEB:
        case 0x70:case 0x71:case 0x72:case 0x73:case 0x74:case 0x75:
        case 0x76:case 0x77:case 0x78:case 0x79:case 0x7A:case 0x7B:
        case 0x7C:case 0x7D:case 0x7E:case 0x7F: return plen+2;
        case 0x6A:case 0xA8:case 0xB0:case 0xB1:case 0xB2:case 0xB3:
        case 0xB4:case 0xB5:case 0xB6:case 0xB7:case 0xCD: return plen+2;
        case 0x68:case 0xB8:case 0xB9:case 0xBA:case 0xBB:
        case 0xBC:case 0xBD:case 0xBE:case 0xBF: return plen+(has66?3:5);
        case 0xC2:case 0xCA: return plen+3;
        case 0xA0:case 0xA1:case 0xA2:case 0xA3: return plen+5;}
    if(remain<2)return plen+2;
    {unsigned char modrm=p[1],mod=(modrm>>6)&3,rm=modrm&7;
     len=2;
     if(mod==1)len+=1; else if(mod==2)len+=4;
     else if(mod==0&&rm==5)len+=4;
     if(rm==4&&mod!=3)len+=1;
     if(op==0x81)len+=has66?2:4;
     else if(op==0x83||op==0x80)len+=1;
     return plen+len;}
}

static const char *x86_mnemonic(const unsigned char *p)
{
    static char tmp[32]; unsigned char op=*p;
    switch(op){
        case 0x90:return"nop"; case 0xC3:return"ret"; case 0xC9:return"leave";
        case 0xCC:return"int3"; case 0xE8:return"call"; case 0xE9:return"jmp";
        case 0xEB:return"jmp short"; case 0x68:return"push imm32";
        case 0x6A:return"push imm8"; case 0x89:return"mov r/m,r";
        case 0x8B:return"mov r,r/m"; case 0x8D:return"lea";
        case 0x85:return"test"; case 0x39:return"cmp";
        case 0x83:return"alu r/m,imm8"; case 0x81:return"alu r/m,imm32";
        case 0xFF:return"call/jmp r/m"; case 0x74:return"jz"; case 0x75:return"jnz";
        case 0x7C:return"jl"; case 0x7D:return"jge"; case 0x7E:return"jle";
        case 0x7F:return"jg"; case 0x31:return"xor r/m,r"; case 0x33:return"xor r,r/m";
        case 0x01:return"add r/m,r"; case 0x03:return"add r,r/m";
        case 0x29:return"sub r/m,r"; case 0x2B:return"sub r,r/m";
        case 0x55:return"push ebp"; case 0x5D:return"pop ebp";
        case 0x0F:return"[0F...]";
        default:safe_fmt(tmp,sizeof(tmp),"db 0x%02X",op);return tmp;}
}

void cmd_disasm(const char *file, int n_insns)
{
    HANDLE hf; DWORD fsz,done,offset; unsigned char *buf;
    MY_DOS *dos; MY_NTHDRS *nt; MY_SECHDR *sec;
    DWORD i,n_secs,code_raw,code_sz,ep; char msg[256]; int count=0;
    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){app_warn("disasm: cannot open\r\n");return;}
    fsz=GetFileSize(hf,NULL);
    if(!fsz||fsz>PE_MAX_MAP_SIZE){CloseHandle(hf);return;}
    buf=(unsigned char*)malloc(fsz);
    if(!buf){CloseHandle(hf);app_warn("disasm: OOM\r\n");return;}
    ReadFile(hf,buf,fsz,&done,NULL); CloseHandle(hf);
    dos=(MY_DOS*)buf;
    if(done<sizeof(MY_DOS)||dos->e_magic!=MY_IMAGE_DOS_SIGNATURE){free(buf);app_warn("disasm: not PE\r\n");return;}
    nt=(MY_NTHDRS*)(buf+dos->e_lfanew);
    if(nt->Sig!=MY_IMAGE_NT_SIGNATURE){free(buf);app_warn("disasm: invalid PE\r\n");return;}
    ep=nt->Opt.EP; n_secs=nt->File.NumSecs;
    sec=(MY_SECHDR*)((unsigned char*)nt+sizeof(DWORD)+sizeof(MY_FILEHDR)+nt->File.SzOpt);
    code_raw=code_sz=0;
    for(i=0;i<n_secs;i++){
        if(ep>=sec[i].VirtAddr&&ep<sec[i].VirtAddr+sec[i].VirtSz){
            code_raw=sec[i].RawPtr+(ep-sec[i].VirtAddr);
            code_sz=sec[i].RawSz-(ep-sec[i].VirtAddr); break;}}
    if(!code_raw||code_raw>=done){code_raw=0;code_sz=done;}
    safe_fmt(msg,sizeof(msg),"=== DISASM: %s (EP=0x%08lX, %d insns) ===\r\n",
             file,(unsigned long)(nt->Opt.ImageBase+ep),n_insns);
    app_info(msg);
    offset=code_raw;
    while(count<n_insns&&offset<code_raw+code_sz&&offset<done){
        int remain=(int)(done-offset),ilen,j; unsigned char *p=buf+offset;
        DWORD vaddr=nt->Opt.ImageBase+ep+(offset-code_raw);
        char hexbuf[32]; int hw=0;
        ilen=x86_instr_len(p,remain); if(ilen<1)ilen=1; if(ilen>15)ilen=15;
        for(j=0;j<ilen&&j<8;j++)hw+=_snprintf(hexbuf+hw,sizeof(hexbuf)-hw-1,"%02X ",p[j]);
        hexbuf[hw]='\0';
        safe_fmt(msg,sizeof(msg),"  %08lX  %-24s  %s\r\n",(unsigned long)vaddr,hexbuf,x86_mnemonic(p));
        app(msg); offset+=(DWORD)ilen; count++;}
    free(buf);
}

/* ═══════════════════════════════════════════════════════════════
 * §C  K-MEANS / KSELECT / CLUSTERMAP
 * ═══════════════════════════════════════════════════════════════ */

void cmd_kmeans(int k, int max_iter)
{
    int i,j,it,changed,best; double best_d,d; char msg[256];
    if(g_nsamples<k){app_warn("kmeans: not enough samples\r\n");return;}
    if(k<1||k>MAX_K){app_warn("kmeans: k out of [1,32]\r\n");return;}
    g_kmeans.k=k;
    for(i=0;i<k;i++){
        int idx=(int)(((unsigned long)(i*1664525UL+1013904223UL))%(unsigned long)g_nsamples);
        memcpy(g_kmeans.clusters[i].centroid,g_samples[idx].features,MAX_FEATURES*sizeof(double));
        g_kmeans.clusters[i].count=0; g_kmeans.clusters[i].label=i; g_kmeans.clusters[i].radius=0.0;}
    for(it=0;it<max_iter;it++){
        double tmp[MAX_K][MAX_FEATURES]; int cnt[MAX_K];
        changed=0;
        for(j=0;j<k;j++) g_kmeans.clusters[j].count=0;
        for(i=0;i<g_nsamples;i++){
            double fn[MAX_FEATURES]; normalize_feat(g_samples[i].features,fn);
            best=0;best_d=1e308;
            for(j=0;j<k;j++){d=sq_dist_b(fn,g_kmeans.clusters[j].centroid,MAX_FEATURES);if(d<best_d){best_d=d;best=j;}}
            if(g_samples[i].label!=best)changed++;
            g_samples[i].label=best; g_kmeans.clusters[best].count++;}
        memset(tmp,0,sizeof(tmp)); memset(cnt,0,sizeof(cnt));
        for(i=0;i<g_nsamples;i++){
            int c=g_samples[i].label; double fn[MAX_FEATURES]; int f;
            normalize_feat(g_samples[i].features,fn);
            for(f=0;f<MAX_FEATURES;f++) tmp[c][f]+=fn[f]; cnt[c]++;}
        for(j=0;j<k;j++) if(cnt[j]>0){
            int f; for(f=0;f<MAX_FEATURES;f++) g_kmeans.clusters[j].centroid[f]=tmp[j][f]/(double)cnt[j];}
        if(!changed)break;}
    g_kmeans.inertia=0.0;
    for(j=0;j<k;j++) g_kmeans.clusters[j].radius=0.0;
    for(i=0;i<g_nsamples;i++){
        int c=g_samples[i].label; double fn[MAX_FEATURES];
        normalize_feat(g_samples[i].features,fn);
        d=sq_dist_b(fn,g_kmeans.clusters[c].centroid,MAX_FEATURES);
        g_kmeans.inertia+=d;
        if(d>g_kmeans.clusters[c].radius)g_kmeans.clusters[c].radius=d;}
    for(j=0;j<k;j++) g_kmeans.clusters[j].radius=sqrt(g_kmeans.clusters[j].radius);
    g_kmeans.trained=1;
    safe_fmt(msg,sizeof(msg),"KMeans: k=%d iters=%d inertia=%.4f\r\n",k,it,g_kmeans.inertia); app_safe(msg);
    for(j=0;j<k;j++){safe_fmt(msg,sizeof(msg),"  Cluster %d: n=%d radius=%.4f\r\n",j,g_kmeans.clusters[j].count,g_kmeans.clusters[j].radius);app(msg);}
}

void cmd_kselect(int max_k)
{
    int k,best_k=2; double prev_inertia=1e308,best_drop=0.0; char msg[256];
    if(max_k<2)max_k=2; if(max_k>MAX_K)max_k=MAX_K;
    app_info("=== AUTO k-SELECT (elbow) ===\r\n");
    for(k=2;k<=max_k;k++){
        double inertia,drop;
        cmd_kmeans(k,KMEANS_MAX_ITER); inertia=g_kmeans.inertia;
        drop=(prev_inertia<1e307)?prev_inertia-inertia:0.0;
        safe_fmt(msg,sizeof(msg),"  k=%d inertia=%.4f drop=%.4f\r\n",k,inertia,drop); app(msg);
        if(prev_inertia<1e307&&drop>best_drop){best_drop=drop;best_k=k-1;}
        prev_inertia=inertia;}
    safe_fmt(msg,sizeof(msg),"  => Suggested k=%d\r\n",best_k); app_safe(msg);
    cmd_kmeans(best_k,KMEANS_MAX_ITER);
}

void cmd_clustermap(void)
{
    int i,k,row,col; char row_buf[82],msg[256];
    double xmin=1e30,xmax=-1e30,ymin=1e30,ymax=-1e30;
    char grid[20][40];
    static const char *symbols="0123456789ABCDEFGHIJKLMNOPQRSTU";
    if(!g_kmeans.trained){app_warn("clustermap: run kmeans first\r\n");return;}
    k=g_kmeans.k;
    for(i=0;i<g_nsamples;i++){
        double x=g_samples[i].features[0],y=g_samples[i].features[15];
        if(x<xmin)xmin=x; if(x>xmax)xmax=x;
        if(y<ymin)ymin=y; if(y>ymax)ymax=y;}
    if(xmax<=xmin)xmax=xmin+1.0; if(ymax<=ymin)ymax=ymin+1.0;
    memset(grid,'.',sizeof(grid));
    for(i=0;i<g_nsamples;i++){
        double x=(g_samples[i].features[0]-xmin)/(xmax-xmin);
        double y=(g_samples[i].features[15]-ymin)/(ymax-ymin);
        row=(int)(y*19.0); col=(int)(x*39.0);
        if(row<0)row=0; if(row>19)row=19; if(col<0)col=0; if(col>39)col=39;
        grid[row][col]=symbols[g_samples[i].label%31];}
    app_info("=== CLUSTER MAP (x=FileSize y=Entropy) ===\r\n");
    for(row=19;row>=0;row--){
        int c; for(c=0;c<40;c++)row_buf[c]=grid[row][c]; row_buf[40]='\0';
        safe_fmt(msg,sizeof(msg),"  |%s|\r\n",row_buf); app(msg);}
    app_info("  +----------------------------------------+\r\n");
    safe_fmt(msg,sizeof(msg),"  Clusters:%d Samples:%d\r\n",k,g_nsamples); app(msg);
    for(i=0;i<k&&i<31;i++){safe_fmt(msg,sizeof(msg),"  %c cluster %-2d n=%d\r\n",symbols[i],i,g_kmeans.clusters[i].count);app(msg);}
}

/* ═══════════════════════════════════════════════════════════════
 * §D  WEAK LABEL / SELF-TRAIN
 * ═══════════════════════════════════════════════════════════════ */

static int weak_label_score(double *feat, float *weight_out)
{
    double conf=0.0; int pred=ensemble_predict(feat,&conf);
    float score=(float)(pred==1?WEAK_SCORE_POS:WEAK_SCORE_NEG); int j;
    if(g_kmeans.trained){
        double fn[MAX_FEATURES]; double best_d=1e308; int best_c=0;
        normalize_feat(feat,fn);
        for(j=0;j<g_kmeans.k;j++){
            double d=sq_dist_b(fn,g_kmeans.clusters[j].centroid,MAX_FEATURES);
            if(d<best_d){best_d=d;best_c=j;}}
        score+=(float)(g_kmeans.clusters[best_c].label==1?2:-1);}
    *weight_out=(float)conf;
    return score>=2.0f?1:0;
}

void cmd_weaklabel(const char *file)
{
    double feat[MAX_FEATURES]; float w; int label; char msg[512];
    extract_features(file,feat); label=weak_label_score(feat,&w);
    safe_fmt(msg,sizeof(msg),"WeakLabel: %s => label=%d weight=%.3f\r\n",file,label,(double)w);
    if(label==1)app_danger(msg); else app_safe(msg);
    EnterCriticalSection(&g_cs_samples);
    if(w>=(float)g_cfg.conf_thresh&&g_nsamples<MAX_SAMPLES){
        memcpy(g_samples[g_nsamples].features,feat,MAX_FEATURES*sizeof(double));
        g_samples[g_nsamples].label=label; g_samples[g_nsamples].weight=w*0.7f;
        safe_strcpy(g_samples[g_nsamples].filename,file,sizeof(g_samples[0].filename));
        g_nsamples++;}
    LeaveCriticalSection(&g_cs_samples);
}

void cmd_weaklabeldir(const char *dir)
{
    WIN32_FIND_DATAA fd; char pattern[520],full[520]; HANDLE h; int n=0; char msg[256];
    safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("weaklabeldir: cannot open\r\n");return;}
    do{if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
       safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
       cmd_weaklabel(full);n++;}
    while(FindNextFileA(h,&fd));
    FindClose(h);
    safe_fmt(msg,sizeof(msg),"WeakLabelDir: %d files in '%s'\r\n",n,dir); app_safe(msg);
    if(n>0)train_all();
}

void cmd_selftrain(const char *dir, double conf_threshold)
{
    int round,n_added_total=0; char msg[256];
    app_info("=== SELF-TRAINING ===\r\n");
    for(round=0;round<SELFTRAIN_ROUNDS;round++){
        WIN32_FIND_DATAA fd; char pattern[520],full[520]; HANDLE h; int n_added=0;
        safe_fmt(msg,sizeof(msg),"  Round %d/%d (thresh=%.3f)...\r\n",round+1,SELFTRAIN_ROUNDS,conf_threshold);app(msg);
        safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
        h=FindFirstFileA(pattern,&fd); if(h==INVALID_HANDLE_VALUE)break;
        do{double feat[MAX_FEATURES],conf=0.0; int label;
           if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
           safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
           extract_features(full,feat); label=ensemble_predict(feat,&conf);
           if(conf>=conf_threshold){
               EnterCriticalSection(&g_cs_samples);
               if(g_nsamples<MAX_SAMPLES){
                   memcpy(g_samples[g_nsamples].features,feat,MAX_FEATURES*sizeof(double));
                   g_samples[g_nsamples].label=label; g_samples[g_nsamples].weight=(float)conf*0.8f;
                   safe_strcpy(g_samples[g_nsamples].filename,full,sizeof(g_samples[0].filename));
                   g_nsamples++;n_added++;}
               LeaveCriticalSection(&g_cs_samples);}
           tb_yield_bg();}
        while(FindNextFileA(h,&fd));
        FindClose(h);
        n_added_total+=n_added;
        safe_fmt(msg,sizeof(msg),"    Added %d pseudo-labels (total=%d)\r\n",n_added,g_nsamples);app(msg);
        if(n_added==0)break;
        train_all();
        conf_threshold=(conf_threshold+0.05<0.95)?conf_threshold+0.05:0.95;}
    safe_fmt(msg,sizeof(msg),"Self-train done. Total added: %d\r\n",n_added_total); app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §E  FEATURE IMPORTANCE  (permutation on MLP)
 * ═══════════════════════════════════════════════════════════════ */

void cmd_importance(void)
{
    double base_acc,acc_drop[MAX_FEATURES]; int i,j; double conf; char msg[256];
    int correct_base=0,correct_perm=0; double feat[MAX_FEATURES],fn[MAX_FEATURES];
    if(g_nsamples<2){app_warn("importance: need >=2 samples\r\n");return;}
    for(i=0;i<g_nsamples;i++){
        normalize_feat(g_samples[i].features,fn);
        if(mlp_predict(fn,&conf)==g_samples[i].label)correct_base++;}
    base_acc=(double)correct_base/(double)g_nsamples;
    app_info("=== FEATURE IMPORTANCE ===\r\n");
    safe_fmt(msg,sizeof(msg),"  Baseline accuracy: %.4f\r\n",base_acc); app(msg);
    for(j=0;j<MAX_FEATURES;j++){
        correct_perm=0;
        for(i=0;i<g_nsamples;i++){
            int prev=(i+1)%g_nsamples;
            memcpy(feat,g_samples[i].features,MAX_FEATURES*sizeof(double));
            feat[j]=g_samples[prev].features[j];
            normalize_feat(feat,fn);
            if(mlp_predict(fn,&conf)==g_samples[i].label)correct_perm++;}
        acc_drop[j]=base_acc-(double)correct_perm/(double)g_nsamples;}
    {int order[MAX_FEATURES]; double sd[MAX_FEATURES];
     for(i=0;i<MAX_FEATURES;i++){order[i]=i;sd[i]=acc_drop[i];}
     for(i=0;i<MAX_FEATURES-1;i++)for(j=i+1;j<MAX_FEATURES;j++)
         if(sd[j]>sd[i]){double td=sd[i];sd[i]=sd[j];sd[j]=td;int ti=order[i];order[i]=order[j];order[j]=ti;}
     for(i=0;i<MAX_FEATURES;i++){
         int bl=(int)(sd[i]*40.0/(base_acc+1e-9)); char bar[48]; int b;
         if(bl<0)bl=0; if(bl>40)bl=40;
         for(b=0;b<bl;b++)bar[b]='#'; bar[bl]='\0';
         safe_fmt(msg,sizeof(msg),"  [%2d] %-16s drop=%+.4f |%s\r\n",order[i]+1,g_feat_names[order[i]],sd[i],bar);
         if(sd[i]>0.05)app_danger(msg); else app(msg);}}
    (void)conf;
}

/* ═══════════════════════════════════════════════════════════════
 * §F  ANOMALY SCAN  (directory)
 * ═══════════════════════════════════════════════════════════════ */

void cmd_anomalyscan(const char *dir)
{
    WIN32_FIND_DATAA fd; char pattern[520],full[520]; HANDLE h; int n=0; char msg[256];
    safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("anomalyscan: cannot open\r\n");return;}
    safe_fmt(msg,sizeof(msg),"=== ANOMALY SCAN: %s ===\r\n",dir); app_info(msg);
    do{if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){
           if(strcmp(fd.cFileName,".")&&strcmp(fd.cFileName,"..")){
               safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
               cmd_anomalyscan(full);}
           continue;}
       safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
       cmd_anomaly(full); n++;
       tb_yield_bg();
       InterlockedExchange(&g_worker_ping_ms, GetTickCount());
       if(g_cancel_flag)break;}
    while(FindNextFileA(h,&fd));
    FindClose(h);
    safe_fmt(msg,sizeof(msg),"Anomaly scan done: %d files\r\n",n); app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §G  SIMILAR FILE SEARCH
 * ═══════════════════════════════════════════════════════════════ */

void cmd_similar(const char *file, int top_n)
{
    float hidden[CFG_D_MODEL]; float q_norm; float best_sims[8]; int best_idx[8];
    ModelOutput out2; int i,j,k; int *toks; int nt; FILE *fp; char *text; long fsz; char msg[512];
    if(!g_model||!g_model->trained){app_warn("similar: not trained\r\n");return;}
    if(!g_embeds||g_n_embeds==0){app_warn("similar: no embeddings (run embedscan)\r\n");return;}
    fp=fopen(file,"rb"); if(!fp){app_warn("similar: cannot open\r\n");return;}
    fseek(fp,0,SEEK_END);fsz=ftell(fp);rewind(fp);
    if(fsz<=0||fsz>(long)PE_MAX_MAP_SIZE){fclose(fp);return;}
    text=(char*)malloc((size_t)fsz+1); if(!text){fclose(fp);return;}
    fread(text,1,(size_t)fsz,fp);fclose(fp);text[fsz]='\0';
    toks=(int*)malloc((size_t)g_model->cfg.ctx_len*sizeof(int)); if(!toks){free(text);return;}
    nt=bpe_encode_multilingual(&g_tokenizer,text,detect_lang_token(file),toks,g_model->cfg.ctx_len);
    free(text);
    memset(hidden,0,sizeof(hidden)); memset(&out2,0,sizeof(out2)); out2.hidden=hidden;
    EnterCriticalSection(&g_cs_model);model_forward(g_model,toks,nt,&out2,NULL);LeaveCriticalSection(&g_cs_model);
    free(toks);
    q_norm=0.0f; for(i=0;i<g_model->cfg.d_model;i++)q_norm+=hidden[i]*hidden[i];
    q_norm=(float)sqrt((double)q_norm)+1e-9f;
    if(top_n>8)top_n=8;
    for(i=0;i<top_n;i++){best_sims[i]=-2.0f;best_idx[i]=-1;}
    EnterCriticalSection(&g_cs_embeds);
    for(j=0;j<g_n_embeds;j++){
        float dot=0.0f,dnorm=0.0f;
        for(i=0;i<g_model->cfg.d_model;i++){dot+=(hidden[i]/q_norm)*g_embeds[j].vec[i];dnorm+=g_embeds[j].vec[i]*g_embeds[j].vec[i];}
        {float sim=dot/((float)sqrt((double)dnorm)+1e-9f);
         for(k=0;k<top_n;k++){if(sim>best_sims[k]){int m;for(m=top_n-1;m>k;m--){best_sims[m]=best_sims[m-1];best_idx[m]=best_idx[m-1];}best_sims[k]=sim;best_idx[k]=j;break;}}}}
    LeaveCriticalSection(&g_cs_embeds);
    safe_fmt(msg,sizeof(msg),"=== SIMILAR TO: %s ===\r\n",file); app_info(msg);
    for(k=0;k<top_n&&best_idx[k]>=0;k++){
        int idx=best_idx[k];
        safe_fmt(msg,sizeof(msg),"  #%d sim=%.4f label=%s %s\r\n",k+1,(double)best_sims[k],
                 g_embeds[idx].label<N_CLASSES?g_class_names[g_embeds[idx].label]:"?",
                 g_embeds[idx].filename);
        if(g_embeds[idx].label!=0)app_danger(msg); else app_safe(msg);}
}

/* ═══════════════════════════════════════════════════════════════
 * §H  CHAIN-OF-THOUGHT REPORT
 * ═══════════════════════════════════════════════════════════════ */

void cmd_report(const char *dir)
{
    WIN32_FIND_DATAA fd; char pattern[520],full[520]; HANDLE h; char msg[512];
    int n_dangerous=0,n_safe=0,n_total=0; char dangerous_files[32][520]; int n_df=0;
    safe_fmt(msg,sizeof(msg),"=== SECURITY REPORT: %s ===\r\n",dir); app_colored(msg,COL_REPORT);
    safe_fmt(pattern,sizeof(pattern),"%s\\*.*",dir);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("report: cannot open dir\r\n");return;}
    app_colored("  Step 1: Triage\r\n",COL_REPORT);
    do{double feat[MAX_FEATURES]; double conf=0.0; int label;
       if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
       safe_fmt(full,sizeof(full),"%s\\%s",dir,fd.cFileName);
       extract_features(full,feat); label=ensemble_predict(feat,&conf); n_total++;
       if(label==1){n_dangerous++;if(n_df<32)safe_strcpy(dangerous_files[n_df++],full,520);}
       else n_safe++;}
    while(FindNextFileA(h,&fd)); FindClose(h);
    safe_fmt(msg,sizeof(msg),"    %d files: %d dangerous %d safe\r\n",n_total,n_dangerous,n_safe); app(msg);
    if(n_dangerous==0){app_safe("  Clean. No dangerous files found.\r\n");return;}
    app_colored("  Step 2: Deep analysis\r\n",COL_REPORT);
    {int i; for(i=0;i<n_df&&!g_cancel_flag;i++){
        safe_fmt(msg,sizeof(msg),"\r\n  [%d/%d] %s\r\n",i+1,n_df,dangerous_files[i]); app_danger(msg);
        cmd_entropy(dangerous_files[i]); cmd_explain(dangerous_files[i]); tb_yield_bg(); InterlockedExchange(&g_worker_ping_ms, GetTickCount());}}
    app_colored("\r\n  Step 3: Summary\r\n",COL_REPORT);
    safe_fmt(msg,sizeof(msg),"    Total:%d Dangerous:%d(%.1f%%) Safe:%d(%.1f%%)\r\n",
             n_total,n_dangerous,n_total?100.0*(double)n_dangerous/n_total:0.0,
             n_safe,n_total?100.0*(double)n_safe/n_total:0.0); app_danger(msg);
    app_colored("  Step 4: Recommendation\r\n",COL_REPORT);
    if((double)n_dangerous/(n_total+1)>0.5)app_danger("    HIGH RISK: Quarantine immediately.\r\n");
    else if(n_dangerous>0)app_warn("    MEDIUM RISK: Investigate flagged files.\r\n");
    else app_safe("    LOW RISK: Monitor.\r\n");
}

/* ═══════════════════════════════════════════════════════════════
 * §I  GITHUB TRAINING
 * ═══════════════════════════════════════════════════════════════ */

static char *http_get(const char *host, const char *path,
                       const char *ua, DWORD *out_sz)
{
    HINTERNET hNet,hConn,hReq; char *buf=NULL; DWORD cap=0,used=0,done;
    char tmp[4096];
    hNet=InternetOpenA(ua,INTERNET_OPEN_TYPE_PRECONFIG,NULL,NULL,0);
    if(!hNet)return NULL;
    hConn=InternetConnectA(hNet,host,INTERNET_DEFAULT_HTTPS_PORT,NULL,NULL,INTERNET_SERVICE_HTTP,0,0);
    if(!hConn){InternetCloseHandle(hNet);return NULL;}
    hReq=HttpOpenRequestA(hConn,"GET",path,NULL,NULL,NULL,
                           INTERNET_FLAG_SECURE|INTERNET_FLAG_NO_CACHE_WRITE|INTERNET_FLAG_RELOAD,0);
    if(!hReq){InternetCloseHandle(hConn);InternetCloseHandle(hNet);return NULL;}
    if(!HttpSendRequestA(hReq,NULL,0,NULL,0)){InternetCloseHandle(hReq);InternetCloseHandle(hConn);InternetCloseHandle(hNet);return NULL;}
    while(InternetReadFile(hReq,tmp,sizeof(tmp),&done)&&done>0){
        if(used+done+1>cap){DWORD nc=(cap+done+1)*2;char *nb=(char*)realloc(buf,nc);if(!nb){free(buf);buf=NULL;break;}buf=nb;cap=nc;}
        memcpy(buf+used,tmp,done);used+=done;}
    if(buf){buf[used]='\0';if(out_sz)*out_sz=used;}
    InternetCloseHandle(hReq);InternetCloseHandle(hConn);InternetCloseHandle(hNet);
    g_perf.http_requests++;
    return buf;
}

static int gh_parse_tree_paths(const char *json, char paths[][512], int max_paths)
{
    const char *p=json; int n=0;
    while((p=strstr(p,"\"path\""))!=NULL&&n<max_paths){
        const char *s,*e; char ext[8]; const char *dot;
        p+=6; s=strchr(p,'"'); if(!s)break; s++;
        e=strchr(s,'"'); if(!e)break;
        if(e-s>0&&e-s<511){
            memcpy(paths[n],s,(size_t)(e-s)); paths[n][e-s]='\0';
            dot=strrchr(paths[n],'.');
            if(dot){strncpy(ext,dot,7);ext[7]='\0';
                if(!_stricmp(ext,".c")||!_stricmp(ext,".h")||!_stricmp(ext,".cpp")||
                   !_stricmp(ext,".py")||!_stricmp(ext,".asm")||!_stricmp(ext,".pas"))
                    n++; else paths[n][0]='\0';}
        }
        p=e+1;}
    return n;
}

void cmd_gitrain(const char *user, const char *repo, int label)
{
    char api_path[512],raw_path[512],msg[512]; char *tree_json; DWORD tree_sz;
    char (*paths)[512]; int n_paths,i,n_trained=0;
    safe_fmt(api_path,sizeof(api_path),"/repos/%s/%s/git/trees/HEAD?recursive=1",user,repo);
    safe_fmt(msg,sizeof(msg),"GitTrain: %s/%s (label=%d)...\r\n",user,repo,label); app_info(msg);
    tree_json=http_get(GH_API_HOST,api_path,GH_UA,&tree_sz);
    if(!tree_json){g_perf.http_failures++;app_warn("gitrain: fetch failed\r\n");return;}
    paths=(char(*)[512])malloc((size_t)GH_MAX_TREE*sizeof(*paths));
    if(!paths){free(tree_json);return;}
    n_paths=gh_parse_tree_paths(tree_json,paths,GH_MAX_TREE); free(tree_json);
    safe_fmt(msg,sizeof(msg),"  Found %d source files\r\n",n_paths); app(msg);
    for(i=0;i<n_paths&&!g_cancel_flag;i++){
        char *content; DWORD csz; double feat[MAX_FEATURES];
        safe_fmt(raw_path,sizeof(raw_path),"/%s/%s/HEAD/%s",user,repo,paths[i]);
        content=http_get(GH_RAW_HOST,raw_path,GH_UA,&csz);
        if(!content||csz==0||csz>GH_MAX_FILESZ){if(content)free(content);g_perf.http_failures++;continue;}
        {char tmpf[MAX_PATH]; FILE *tf;
         GetTempPathA(MAX_PATH-16,tmpf); strncat(tmpf,"tb_gh_tmp.bin",13);
         tf=fopen(tmpf,"wb");
         if(tf){fwrite(content,1,csz,tf);fclose(tf);
             extract_features(tmpf,feat);
             undo_push();
             EnterCriticalSection(&g_cs_samples);
             if(g_nsamples<MAX_SAMPLES){
                 memcpy(g_samples[g_nsamples].features,feat,MAX_FEATURES*sizeof(double));
                 g_samples[g_nsamples].label=label;g_samples[g_nsamples].weight=1.0f;
                 safe_strcpy(g_samples[g_nsamples].filename,paths[i],sizeof(g_samples[0].filename));
                 g_nsamples++;n_trained++;}
             LeaveCriticalSection(&g_cs_samples);
             DeleteFileA(tmpf);}
        }
        free(content);
        if(i%20==0){safe_fmt(msg,sizeof(msg),"  [%d/%d] trained=%d\r\n",i+1,n_paths,n_trained);app(msg);report_progress((i*100)/(n_paths+1));}
        tb_yield_bg();
        InterlockedExchange(&g_worker_ping_ms,GetTickCount());
    }
    free(paths); train_all();
    safe_fmt(msg,sizeof(msg),"GitTrain done: %d/%d (label=%d)\r\n",n_trained,n_paths,label); app_safe(msg);
    report_progress(100);
}

/* ═══════════════════════════════════════════════════════════════
 * §J  FORUM TRAINING
 * ═══════════════════════════════════════════════════════════════ */

static void html_strip(const char *html, char *out, int max_out)
{
    const char *p=html; int wi=0,in_tag=0;
    while(*p&&wi<max_out-1){
        if(*p=='<'){in_tag=1;p++;continue;}
        if(*p=='>'){in_tag=0;p++;continue;}
        if(in_tag){p++;continue;}
        if(*p=='&'){
            if(!strncmp(p,"&amp;",5)){out[wi++]='&';p+=5;continue;}
            if(!strncmp(p,"&lt;",4)){out[wi++]='<';p+=4;continue;}
            if(!strncmp(p,"&gt;",4)){out[wi++]='>';p+=4;continue;}
            if(!strncmp(p,"&quot;",6)){out[wi++]='"';p+=6;continue;}
            if(!strncmp(p,"&#39;",5)){out[wi++]='\'';p+=5;continue;}}
        out[wi++]=*p++;}
    out[wi]='\0';
}

static int url_split(const char *url,char *host,int hmax,char *path,int pmax)
{
    const char *s=strstr(url,"://"),*slash;
    if(!s)return 0; s+=3; slash=strchr(s,'/');
    if(!slash){strncpy(host,s,hmax-1);host[hmax-1]='\0';strncpy(path,"/",pmax-1);path[pmax-1]='\0';}
    else{int hlen=(int)(slash-s);if(hlen>=hmax)hlen=hmax-1;memcpy(host,s,hlen);host[hlen]='\0';strncpy(path,slash,pmax-1);path[pmax-1]='\0';}
    return 1;
}

static DWORD s_forum_hashes[FORUM_HASH_RING];
static int   s_forum_ring_init=0;

void cmd_forum_train(const char *url, int label)
{
    char host[512],path[1024],msg[256]; char *html; DWORD hsz; char *text; DWORD url_hash;
    if(!url_split(url,host,sizeof(host),path,sizeof(path))){app_warn("forumtrain: invalid URL\r\n");return;}
    url_hash=crc32_buf((const unsigned char*)url,strlen(url));
    if(!s_forum_ring_init){memset(s_forum_hashes,0,sizeof(s_forum_hashes));s_forum_ring_init=1;}
    {int slot=(int)(url_hash%FORUM_HASH_RING);
     if(s_forum_hashes[slot]==url_hash){app_warn("forumtrain: already processed (dedup)\r\n");return;}
     s_forum_hashes[slot]=url_hash;}
    safe_fmt(msg,sizeof(msg),"ForumTrain: %s (label=%d)...\r\n",url,label); app_info(msg);
    html=http_get(host,path,GH_UA,&hsz);
    if(!html||hsz==0){
        g_perf.http_failures++; app_warn("forumtrain: HTTP fetch failed\r\n");
        if(html)free(html);
        {char dl[1024];safe_fmt(dl,sizeof(dl),"forumtrain %s %d\n",url,label);dead_letter_append(dl);}
        return;}
    text=(char*)malloc(hsz+1); if(!text){free(html);return;}
    html_strip(html,text,(int)(hsz+1)); free(html);
    {char tmpf[MAX_PATH]; FILE *tf;
     GetTempPathA(MAX_PATH-16,tmpf);strncat(tmpf,"tb_forum.txt",12);
     tf=fopen(tmpf,"wb");
     if(tf){double feat[MAX_FEATURES]; size_t tsz=strlen(text);
         fwrite(text,1,tsz,tf);fclose(tf);
         extract_features(tmpf,feat);
         undo_push();
         EnterCriticalSection(&g_cs_samples);
         if(g_nsamples<MAX_SAMPLES){
             memcpy(g_samples[g_nsamples].features,feat,MAX_FEATURES*sizeof(double));
             g_samples[g_nsamples].label=(label>=0?label:0);
             g_samples[g_nsamples].weight=(label>=0?1.0f:0.5f);
             safe_strcpy(g_samples[g_nsamples].filename,url,sizeof(g_samples[0].filename));
             g_nsamples++;}
         LeaveCriticalSection(&g_cs_samples);
         DeleteFileA(tmpf);}
    }
    free(text); train_all();
    safe_fmt(msg,sizeof(msg),"ForumTrain done: %s label=%d total=%d\r\n",url,label,g_nsamples); app_safe(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §K  DECRYPT TOOLKIT
 * ═══════════════════════════════════════════════════════════════ */

static void do_xor(const unsigned char *in,size_t n,unsigned char key,unsigned char *out)
{size_t i;for(i=0;i<n;i++)out[i]=in[i]^key;}

static const signed char B64_DEC[256]={
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,0,-1,-1,
    -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

static size_t b64_decode(const char *in,size_t in_len,unsigned char *out)
{
    size_t wi=0,i=0;
    while(i+3<in_len){
        int a=B64_DEC[(unsigned char)in[i]],b=B64_DEC[(unsigned char)in[i+1]],
            c=B64_DEC[(unsigned char)in[i+2]],d=B64_DEC[(unsigned char)in[i+3]];
        if(a<0||b<0)break;
        out[wi++]=(unsigned char)((a<<2)|(b>>4));
        if(c>=0)out[wi++]=(unsigned char)(((b&0xF)<<4)|(c>>2));
        if(d>=0)out[wi++]=(unsigned char)(((c&3)<<6)|d);
        i+=4;}
    return wi;
}

static void do_rot13(const char *in,char *out,size_t n)
{size_t i;for(i=0;i<n;i++){char c=in[i];if(c>='A'&&c<='Z')out[i]=(char)('A'+(c-'A'+13)%26);else if(c>='a'&&c<='z')out[i]=(char)('a'+(c-'a'+13)%26);else out[i]=c;}}

static void do_caesar(const char *in,char *out,size_t n,int shift)
{size_t i;shift=((shift%26)+26)%26;for(i=0;i<n;i++){char c=in[i];if(c>='A'&&c<='Z')out[i]=(char)('A'+(c-'A'+shift)%26);else if(c>='a'&&c<='z')out[i]=(char)('a'+(c-'a'+shift)%26);else out[i]=c;}}

static void do_rc4(const unsigned char *in,size_t n,const unsigned char *key,size_t klen,unsigned char *out)
{
    unsigned char S[256]; int i,j=0,a,b; size_t k;
    for(i=0;i<256;i++)S[i]=(unsigned char)i;
    for(i=0;i<256;i++){j=(j+S[i]+key[i%klen])&255;a=S[i];S[i]=S[j];S[j]=(unsigned char)a;}
    i=j=0;
    for(k=0;k<n;k++){i=(i+1)&255;j=(j+S[i])&255;b=S[i];S[i]=S[j];S[j]=(unsigned char)b;out[k]=in[k]^S[(S[i]+S[j])&255];}
}

static void cmd_decrypt(const char *method,const char *file,const char *key_arg)
{
    HANDLE hf; DWORD fsz,done; unsigned char *buf,*out; char msg[256]; size_t out_sz; int printable; size_t i;
    hf=CreateFileA(file,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){app_warn("decrypt: cannot open\r\n");return;}
    fsz=GetFileSize(hf,NULL);
    if(!fsz||fsz>4*1024*1024){CloseHandle(hf);app_warn("decrypt: too large (max 4MB)\r\n");return;}
    buf=(unsigned char*)malloc(fsz+1); out=(unsigned char*)malloc(fsz+16);
    if(!buf||!out){free(buf);free(out);CloseHandle(hf);app_warn("decrypt: OOM\r\n");return;}
    ReadFile(hf,buf,fsz,&done,NULL); CloseHandle(hf); buf[done]='\0'; out_sz=0;
    safe_fmt(msg,sizeof(msg),"=== DECRYPT (%s): %s ===\r\n",method,file); app_info(msg);
    if(!_stricmp(method,"xor")){
        unsigned char xkey=key_arg?(unsigned char)strtoul(key_arg,NULL,0):0xFF;
        do_xor(buf,done,xkey,out); out_sz=done;
        safe_fmt(msg,sizeof(msg),"  XOR key: 0x%02X\r\n",(unsigned)xkey); app(msg);}
    else if(!_stricmp(method,"b64")||!_stricmp(method,"base64")){
        out_sz=b64_decode((char*)buf,done,out);
        safe_fmt(msg,sizeof(msg),"  Decoded %lu->%lu bytes\r\n",(unsigned long)done,(unsigned long)out_sz);app(msg);}
    else if(!_stricmp(method,"rot13")){do_rot13((char*)buf,(char*)out,done);out_sz=done;}
    else if(!_stricmp(method,"caesar")){
        int shift=key_arg?atoi(key_arg):3;
        do_caesar((char*)buf,(char*)out,done,shift);out_sz=done;
        safe_fmt(msg,sizeof(msg),"  Caesar shift=%d\r\n",shift);app(msg);}
    else if(!_stricmp(method,"rc4")){
        const unsigned char *rc4key=key_arg?(const unsigned char*)key_arg:(const unsigned char*)"TheBrain13";
        size_t klen=key_arg?strlen(key_arg):10;
        do_rc4(buf,done,rc4key,klen,out);out_sz=done;
        safe_fmt(msg,sizeof(msg),"  RC4 key: \"%s\"\r\n",key_arg?key_arg:"TheBrain13");app(msg);}
    else if(!_stricmp(method,"smart")){
        unsigned char best_key=0; double best_ratio=0.0; unsigned char xk;
        for(xk=1;xk!=0;xk++){
            double pr=0.0; DWORD si;
            for(si=0;si<done&&si<512;si++){unsigned char c=buf[si]^xk;if((c>=0x20&&c<=0x7E)||c==0x09||c==0x0A||c==0x0D)pr+=1.0;}
            pr/=(double)(done<512?done:512); if(pr>best_ratio){best_ratio=pr;best_key=xk;}}
        safe_fmt(msg,sizeof(msg),"  Smart: best XOR=0x%02X (printable=%.1f%%)\r\n",(unsigned)best_key,best_ratio*100.0);app(msg);
        do_xor(buf,done,best_key,out);out_sz=done;}
    else{app_warn("decrypt: unknown method\r\n");free(buf);free(out);return;}
    printable=1;
    for(i=0;i<out_sz&&i<512;i++) if(out[i]<0x09||(out[i]>0x0D&&out[i]<0x20&&out[i]!=0x1B)){printable=0;break;}
    if(printable&&out_sz>0){
        size_t show=out_sz<2048?out_sz:2048; char *preview=(char*)malloc(show+4);
        if(preview){memcpy(preview,out,show);preview[show]='\0';
            for(i=0;i<show;i++)if((unsigned char)preview[i]<0x20&&preview[i]!='\r'&&preview[i]!='\n'&&preview[i]!='\t')preview[i]='.';
            app_cyan(preview);app("\r\n");free(preview);}
    }else{safe_fmt(msg,sizeof(msg),"  Output: %lu bytes (binary)\r\n",(unsigned long)out_sz);app(msg);}
    {char outf[520];FILE *tf;
     safe_fmt(outf,sizeof(outf),"%s.dec",file);
     tf=fopen(outf,"wb");if(tf){fwrite(out,1,out_sz,tf);fclose(tf);safe_fmt(msg,sizeof(msg),"  Saved: %s\r\n",outf);app_safe(msg);}}
    free(buf);free(out);
}

/* ═══════════════════════════════════════════════════════════════
 * §L  CONFIG SAVE / LOAD  (v13 — all fields)
 * ═══════════════════════════════════════════════════════════════ */

static void config_apply_chat_defaults(void)
{
    if (g_cfg.temperature > 0.25f)
        g_cfg.temperature = 0.15f;
    if (g_cfg.top_k > 4)
        g_cfg.top_k = 1;
    g_cfg.conv_use_facts = 0;
    if (g_cfg.conv_history_turns > 4)
        g_cfg.conv_history_turns = 2;
    g_cfg.train_use_text = 0;
    /* -1 = auto RAM tier (use full model on 1GB+ machines) */
    if (g_cfg.sysinfo_tier >= 0 && g_cfg.sysinfo_tier <= 1)
        g_cfg.sysinfo_tier = -1;
    g_override_tier = (g_cfg.sysinfo_tier >= 0 && g_cfg.sysinfo_tier <= 4)
                      ? g_cfg.sysinfo_tier : -1;
    if (!g_cfg.tie_embeddings)
        g_cfg.tie_embeddings = 1;
    conv_apply_config(g_cfg.conv_use_facts, g_cfg.conv_stream,
                      g_cfg.conv_max_tokens, g_cfg.temperature, g_cfg.top_k,
                      g_cfg.conv_history_turns);
}

static void config_save(void)
{
    FILE *f=fopen("brain.conf","w");
    if(!f){app_warn("config: cannot write brain.conf\r\n");return;}
    fprintf(f,"# TheBrain v13.0 config\n");
    /* ML */
    fprintf(f,"lr=%.6f\ndropout=%.4f\nepochs=%d\n",g_cfg.lr,g_cfg.dropout,g_cfg.epochs);
    fprintf(f,"iso_thresh_min=%.4f\niso_thresh_max=%.4f\n",g_cfg.iso_thresh_min,g_cfg.iso_thresh_max);
    fprintf(f,"nu=%.4f\nconf_thresh=%.4f\nmax_checkpoints=%d\n",g_cfg.nu,g_cfg.conf_thresh,g_cfg.max_checkpoints);
    fprintf(f,"async_ops=%d\nlog_level=%d\nwatchdog_ms=%d\n",g_cfg.async_ops,g_cfg.log_level,g_cfg.watchdog_timeout_ms);
    /* Transformer */
    fprintf(f,"t_lr_max=%.6f\nt_lr_min=%.6f\n",(double)g_cfg.t_lr_max,(double)g_cfg.t_lr_min);
    fprintf(f,"t_warmup=%ld\nt_total=%ld\n",g_cfg.t_warmup,g_cfg.t_total);
    fprintf(f,"t_wd=%.6f\nt_grad_clip=%.4f\n",(double)g_cfg.t_wd,(double)g_cfg.t_grad_clip);
    fprintf(f,"t_batch=%d\nt_ctx=%d\n",g_cfg.t_batch,g_cfg.t_ctx);
    fprintf(f,"use_swiglu=%d\nuse_rmsnorm=%d\ntie_embeddings=%d\n",g_cfg.use_swiglu,g_cfg.use_rmsnorm,g_cfg.tie_embeddings);
    /* Generation */
    fprintf(f,"temperature=%.4f\ntop_k=%d\n",(double)g_cfg.temperature,g_cfg.top_k);
    fprintf(f,"cot_think=%d\ncot_answer=%d\nearly_stop=%d\n",g_cfg.cot_think_tokens,g_cfg.cot_answer_tokens,g_cfg.early_stop_patience);
    /* NEW v13 */
    fprintf(f,"conv_max_tokens=%d\nconv_use_facts=%d\nconv_stream=%d\nconv_history_turns=%d\n",
            g_cfg.conv_max_tokens,g_cfg.conv_use_facts,g_cfg.conv_stream,g_cfg.conv_history_turns);
    fprintf(f,"guard_enabled=%d\nguard_dir=%s\n",g_cfg.guard_enabled,g_cfg.guard_dir);
    fprintf(f,"train_use_conv=%d\ntrain_use_text=%d\n",g_cfg.train_use_conv,g_cfg.train_use_text);
    fprintf(f,"sysinfo_tier=%d\n",g_cfg.sysinfo_tier);
    /* Files */
    fprintf(f,"vocab_file=%s\nmodel_file=%s\nembeds_file=%s\ncorpus_file=%s\n",
            g_cfg.vocab_file,g_cfg.model_file,g_cfg.embeds_file,g_cfg.corpus_file);
    fclose(f);
    app_safe("Config saved to brain.conf\r\n");
    BLOG_INFO("config_save: done");
}

static void config_load(void)
{
    FILE *f=fopen("brain.conf","r"); char line[512],key[128],val[384];
    if(!f)return;
    while(fgets(line,sizeof(line),f)){
        int vl;
        if(line[0]=='#'||line[0]=='\n')continue;
        if(sscanf(line,"%127[^=]=%383s",key,val)!=2)continue;
        vl=(int)strlen(val); while(vl>0&&(val[vl-1]=='\r'||val[vl-1]=='\n'||val[vl-1]==' '))val[--vl]='\0';
        cmd_config_set(key,val);}
    fclose(f);
    config_apply_chat_defaults();
    app_info("Config loaded from brain.conf\r\n");
    BLOG_INFO("config_load: done");
}

void cmd_config_show(void)
{
    char msg[512];
    app_info("=== BrainConfig v13 ===\r\n");
    safe_fmt(msg,sizeof(msg),"  lr=%.6f dropout=%.4f epochs=%d\r\n",g_cfg.lr,g_cfg.dropout,g_cfg.epochs);app(msg);
    safe_fmt(msg,sizeof(msg),"  t_lr=[%.6f,%.6f] warmup=%ld total=%ld\r\n",(double)g_cfg.t_lr_max,(double)g_cfg.t_lr_min,g_cfg.t_warmup,g_cfg.t_total);app(msg);
    safe_fmt(msg,sizeof(msg),"  temperature=%.4f top_k=%d swiglu=%d rmsnorm=%d\r\n",(double)g_cfg.temperature,g_cfg.top_k,g_cfg.use_swiglu,g_cfg.use_rmsnorm);app(msg);
    safe_fmt(msg,sizeof(msg),"  conv_max=%d facts=%d stream=%d hist=%d\r\n",g_cfg.conv_max_tokens,g_cfg.conv_use_facts,g_cfg.conv_stream,g_cfg.conv_history_turns);app(msg);
    safe_fmt(msg,sizeof(msg),"  guard_enabled=%d guard_dir=%s\r\n",g_cfg.guard_enabled,g_cfg.guard_dir[0]?g_cfg.guard_dir:"(none)");app(msg);
    safe_fmt(msg,sizeof(msg),"  train_use_conv=%d train_use_text=%d\r\n",g_cfg.train_use_conv,g_cfg.train_use_text);app(msg);
    safe_fmt(msg,sizeof(msg),"  sysinfo_tier=%d\r\n",g_cfg.sysinfo_tier);app(msg);
    safe_fmt(msg,sizeof(msg),"  vocab=%s model=%s\r\n",g_cfg.vocab_file,g_cfg.model_file);app(msg);
}

void cmd_config_set(const char *key, const char *val)
{
    char msg[256];
    if      (!strcmp(key,"lr"))              g_cfg.lr               =atof(val);
    else if (!strcmp(key,"dropout"))         g_cfg.dropout           =atof(val);
    else if (!strcmp(key,"epochs"))          g_cfg.epochs            =atoi(val);
    else if (!strcmp(key,"iso_thresh_min"))  g_cfg.iso_thresh_min    =atof(val);
    else if (!strcmp(key,"iso_thresh_max"))  g_cfg.iso_thresh_max    =atof(val);
    else if (!strcmp(key,"nu"))              g_cfg.nu                =atof(val);
    else if (!strcmp(key,"conf_thresh"))     g_cfg.conf_thresh       =atof(val);
    else if (!strcmp(key,"max_checkpoints")) g_cfg.max_checkpoints   =atoi(val);
    else if (!strcmp(key,"async_ops"))       g_cfg.async_ops         =atoi(val);
    else if (!strcmp(key,"log_level"))       g_cfg.log_level         =atoi(val);
    else if (!strcmp(key,"watchdog_ms"))     g_cfg.watchdog_timeout_ms=atoi(val);
    else if (!strcmp(key,"t_lr_max"))        g_cfg.t_lr_max          =(float)atof(val);
    else if (!strcmp(key,"t_lr_min"))        g_cfg.t_lr_min          =(float)atof(val);
    else if (!strcmp(key,"t_warmup"))        g_cfg.t_warmup           =atol(val);
    else if (!strcmp(key,"t_total"))         g_cfg.t_total            =atol(val);
    else if (!strcmp(key,"t_wd"))            g_cfg.t_wd              =(float)atof(val);
    else if (!strcmp(key,"t_grad_clip"))     g_cfg.t_grad_clip       =(float)atof(val);
    else if (!strcmp(key,"t_batch"))         g_cfg.t_batch            =atoi(val);
    else if (!strcmp(key,"t_ctx"))           g_cfg.t_ctx              =atoi(val);
    else if (!strcmp(key,"use_swiglu"))      g_cfg.use_swiglu         =atoi(val);
    else if (!strcmp(key,"use_rmsnorm"))     g_cfg.use_rmsnorm        =atoi(val);
    else if (!strcmp(key,"tie_embeddings"))  g_cfg.tie_embeddings     =atoi(val);
    else if (!strcmp(key,"temperature"))     g_cfg.temperature       =(float)atof(val);
    else if (!strcmp(key,"top_k"))           g_cfg.top_k              =atoi(val);
    else if (!strcmp(key,"cot_think"))       g_cfg.cot_think_tokens   =atoi(val);
    else if (!strcmp(key,"cot_answer"))      g_cfg.cot_answer_tokens  =atoi(val);
    else if (!strcmp(key,"early_stop"))      g_cfg.early_stop_patience=atoi(val);
    /* v13 */
    else if (!strcmp(key,"conv_max_tokens"))    g_cfg.conv_max_tokens    =atoi(val);
    else if (!strcmp(key,"conv_use_facts"))     g_cfg.conv_use_facts     =atoi(val);
    else if (!strcmp(key,"conv_stream"))        g_cfg.conv_stream        =atoi(val);
    else if (!strcmp(key,"conv_history_turns")) g_cfg.conv_history_turns =atoi(val);
    else if (!strcmp(key,"guard_enabled"))      g_cfg.guard_enabled      =atoi(val);
    else if (!strcmp(key,"guard_dir"))          safe_strcpy(g_cfg.guard_dir,val,sizeof(g_cfg.guard_dir));
    else if (!strcmp(key,"train_use_conv"))     g_cfg.train_use_conv     =atoi(val);
    else if (!strcmp(key,"train_use_text"))     g_cfg.train_use_text     =atoi(val);
    else if (!strcmp(key,"sysinfo_tier")) {
        g_cfg.sysinfo_tier = atoi(val);
        g_override_tier = (g_cfg.sysinfo_tier >= 0 && g_cfg.sysinfo_tier <= 4)
                          ? g_cfg.sysinfo_tier : -1;
    }
    /* Files */
    else if (!strcmp(key,"vocab_file"))  safe_strcpy(g_cfg.vocab_file, val,256);
    else if (!strcmp(key,"model_file"))  safe_strcpy(g_cfg.model_file, val,256);
    else if (!strcmp(key,"embeds_file")) safe_strcpy(g_cfg.embeds_file,val,256);
    else if (!strcmp(key,"corpus_file")) safe_strcpy(g_cfg.corpus_file,val,256);
    else{safe_fmt(msg,sizeof(msg),"config: unknown key '%s'\r\n",key);app_warn(msg);return;}
    /* Propagate live */
    conv_apply_config(g_cfg.conv_use_facts, g_cfg.conv_stream,
                      g_cfg.conv_max_tokens, g_cfg.temperature, g_cfg.top_k,
                      g_cfg.conv_history_turns);
    safe_fmt(msg,sizeof(msg),"  %s = %s\r\n",key,val); app(msg);
}

/* ═══════════════════════════════════════════════════════════════
 * §M  CHECKPOINT / DEAD-LETTER / RETRY
 * ═══════════════════════════════════════════════════════════════ */

static void checkpoint_save(void)
{
    char path[320],name[64]; SYSTEMTIME st; GetLocalTime(&st);
    safe_fmt(name,sizeof(name),"ckpt_%04d%02d%02d_%02d%02d%02d.bin",
             st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,st.wSecond);
    CreateDirectoryA(CHECKPOINT_DIR,NULL);
    safe_fmt(path,sizeof(path),"%s\\%s",CHECKPOINT_DIR,name);
    if(g_model){model_save(g_model,path);
        {char msg[384];safe_fmt(msg,sizeof(msg),"Checkpoint: %s\r\n",path);app_safe(msg);}
    }else app_warn("checkpoint: no model\r\n");
}

static void checkpoint_load_latest(void)
{
    WIN32_FIND_DATAA fd; char pattern[320],best[320]; HANDLE h; FILETIME best_ft;
    CreateDirectoryA(CHECKPOINT_DIR,NULL);
    safe_fmt(pattern,sizeof(pattern),"%s\\ckpt_*.bin",CHECKPOINT_DIR);
    best[0]='\0'; best_ft.dwHighDateTime=best_ft.dwLowDateTime=0;
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("checkpoint: none found\r\n");return;}
    do{if(CompareFileTime(&fd.ftLastWriteTime,&best_ft)>0){best_ft=fd.ftLastWriteTime;safe_fmt(best,sizeof(best),"%s\\%s",CHECKPOINT_DIR,fd.cFileName);}}
    while(FindNextFileA(h,&fd)); FindClose(h);
    if(best[0]&&g_model){char msg[384];model_load(g_model,best);safe_fmt(msg,sizeof(msg),"Checkpoint loaded: %s\r\n",best);app_safe(msg);}
}

static void cmd_checkpoints(void)
{
    WIN32_FIND_DATAA fd; char pattern[320],msg[400]; HANDLE h; int n=0;
    CreateDirectoryA(CHECKPOINT_DIR,NULL);
    safe_fmt(pattern,sizeof(pattern),"%s\\ckpt_*.bin",CHECKPOINT_DIR);
    app_info("=== CHECKPOINTS ===\r\n");
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("  None.\r\n");return;}
    do{SYSTEMTIME st;FileTimeToSystemTime(&fd.ftLastWriteTime,&st);
       safe_fmt(msg,sizeof(msg),"  [%2d] %s  %04d-%02d-%02d %02d:%02d  %lu KB\r\n",
                n,fd.cFileName,st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,
                (unsigned long)(fd.nFileSizeLow/1024));
       app(msg);n++;}
    while(FindNextFileA(h,&fd)); FindClose(h);
    {char msg2[64];safe_fmt(msg2,sizeof(msg2),"  Total: %d\r\n",n);app(msg2);}
}

static void rollback_to_checkpoint(int idx)
{
    WIN32_FIND_DATAA fd; char pattern[320],full[320],msg[400]; HANDLE h; int n=0;
    safe_fmt(pattern,sizeof(pattern),"%s\\ckpt_*.bin",CHECKPOINT_DIR);
    h=FindFirstFileA(pattern,&fd);
    if(h==INVALID_HANDLE_VALUE){app_warn("rollback: none\r\n");return;}
    do{if(n==idx){FindClose(h);safe_fmt(full,sizeof(full),"%s\\%s",CHECKPOINT_DIR,fd.cFileName);
           if(g_model){model_load(g_model,full);safe_fmt(msg,sizeof(msg),"Rolled back: %s\r\n",full);app_safe(msg);}
           return;}n++;}
    while(FindNextFileA(h,&fd)); FindClose(h);
    app_warn("rollback: index out of range\r\n");
}

static void cmd_rollback(int idx){rollback_to_checkpoint(idx);}

static void dead_letter_append(const char *text)
{FILE *f;EnterCriticalSection(&g_cs_log);f=fopen(DEAD_LETTER_FILE,"a");if(f){fprintf(f,"%s",text);fclose(f);}LeaveCriticalSection(&g_cs_log);}

static void cmd_deadletter(void)
{FILE *f;char line[1024],msg[256];int n=0;app_info("=== DEAD LETTER ===\r\n");f=fopen(DEAD_LETTER_FILE,"r");if(!f){app_warn("  Empty.\r\n");return;}while(fgets(line,sizeof(line),f)){app_warn(line);n++;}fclose(f);safe_fmt(msg,sizeof(msg),"  Total: %d\r\n",n);app(msg);}

static void cmd_retry(void)
{
    FILE *f; char line[1024],newdl[1024*64]; int wi=0,n_ok=0,n_fail=0;
    app_info("=== RETRY ===\r\n");
    f=fopen(DEAD_LETTER_FILE,"r"); if(!f){app_warn("  Nothing.\r\n");return;}
    while(fgets(line,sizeof(line),f)){
        int len=(int)strlen(line);
        while(len>0&&(line[len-1]=='\r'||line[len-1]=='\n'))line[--len]='\0';
        if(!len)continue;
        {int prev=g_nsamples;process_command(line);
         if(g_nsamples>prev)n_ok++;
         else{if(wi+len+2<(int)sizeof(newdl)){memcpy(newdl+wi,line,len);wi+=len;newdl[wi++]='\n';}n_fail++;}}}
    fclose(f);
    f=fopen(DEAD_LETTER_FILE,"w"); if(f){fwrite(newdl,1,wi,f);fclose(f);}
    {char msg[256];safe_fmt(msg,sizeof(msg),"Retry: %d ok %d fail\r\n",n_ok,n_fail);app_safe(msg);}
}

/* ═══════════════════════════════════════════════════════════════
 * §N  WORKER THREAD  (v13: all tasks including CONVERSE/DOWNLOAD/GUARD)
 * ═══════════════════════════════════════════════════════════════ */

DWORD WINAPI worker_thread_proc(LPVOID param)
{
    WorkerTask *task=(WorkerTask*)param; char msg[512];
    tb_thread_set_bg(GetCurrentThread()); /* NEW v13 */
    InterlockedExchange(&g_worker_ping_ms,GetTickCount());

    switch(task->type){

    case TASK_GITRAIN:
        {char *sl=strchr(task->arg1,'/');
         if(sl){*sl='\0';cmd_gitrain(task->arg1,sl+1,task->int1);*sl='/';}
         else app_warn("worker: GITRAIN malformed\r\n");}
        break;

    case TASK_SELFTRAIN:
        cmd_selftrain(task->arg1,task->dbl1>0.0?task->dbl1:g_cfg.conf_thresh);
        break;

    case TASK_ANOMALYSCAN:
        cmd_anomalyscan(task->arg1);
        break;

    case TASK_BATCHTRAIN:
        {WIN32_FIND_DATAA fd;char pattern[520],full[520];HANDLE h;int n=0;
         safe_fmt(pattern,sizeof(pattern),"%s\\*.*",task->arg1);
         h=FindFirstFileA(pattern,&fd);
         if(h!=INVALID_HANDLE_VALUE){
             do{if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
                if(g_cancel_flag)break;
                safe_fmt(full,sizeof(full),"%s\\%s",task->arg1,fd.cFileName);
                cmd_train(full,task->int1);n++;
                InterlockedExchange(&g_worker_ping_ms,GetTickCount());
                tb_yield_bg();}
             while(FindNextFileA(h,&fd));FindClose(h);}
         safe_fmt(msg,sizeof(msg),"BatchTrain: %d files (label=%d)\r\n",n,task->int1);app_safe(msg);}
        break;

    case TASK_FORUM_TRAIN:
        cmd_forum_train(task->arg1,task->int1);
        break;

    case TASK_RETRY:
        cmd_retry();
        break;

    case TASK_PRETRAIN:
        {SysInfo si;DynModelCfg dyn;ModelConfig mc;
         sysinfo_probe(&si);
         dyn=sysinfo_make_cfg(&si,g_cfg.use_swiglu,g_cfg.use_rmsnorm,g_cfg.tie_embeddings);
         mc=model_cfg_from_dyn(&dyn);
         if(g_model)model_free(g_model);
         g_model=model_create(&mc);
         if(g_model){model_init_xavier(g_model);app_safe("Pretrain done.\r\n");}
         else{BLOG_ERROR("TASK_PRETRAIN: model_create OOM");app_warn("Pretrain: OOM\r\n");}}
        break;

    case TASK_FULLTRAIN:
        if(!g_model){app_warn("worker: need pretrained model\r\n");}
        else{
            TrainConfig tc=train_default_config();
            tc.epochs        =task->int1>0?task->int1:3;
            tc.lr_max        =g_cfg.t_lr_max;
            tc.lr_min        =g_cfg.t_lr_min;
            tc.warmup_steps  =g_cfg.t_warmup;
            tc.total_steps   =g_cfg.t_total;
            tc.weight_decay  =(float)g_cfg.t_wd;
            tc.grad_clip     =g_cfg.t_grad_clip;
            tc.patience      =g_cfg.early_stop_patience;
            tc.use_conv_files=g_cfg.train_use_conv;
            tc.use_text_files=g_cfg.train_use_text;
            tc.use_code_files=1;
            safe_strcpy(tc.checkpoint_path,g_cfg.model_file,sizeof(tc.checkpoint_path));
            if(!train_config_validate(&tc))
                app_warn("worker: training config issues, using defaults\r\n");
            train_state_init(&g_train_state,&tc);
            train_loop_mixed(g_model,&g_tokenizer,task->arg1,&tc,&g_train_state,
                              (volatile int*)&g_cancel_flag);
            train_state_destroy(&g_train_state);
            model_load(g_model, g_cfg.model_file);
        }
        break;

    case TASK_GENERATE:
        cmd_generate(&task->gen);
        break;

    case TASK_BPE_TRAIN:
        bpe_init_vocab(&g_tokenizer);
        bpe_learn_from_file(&g_tokenizer,task->arg1,BPE_MAX_MERGES);
        bpe_save(&g_tokenizer,g_cfg.vocab_file);
        app_safe("BPE vocab trained.\r\n");
        break;

    case TASK_CURRICULUM:
        {Curriculum *cur=&g_curriculum;int i;
         app_info("=== CURRICULUM ===\r\n"); cur->running=1;
         for(i=cur->current;i<cur->n_entries&&!g_cancel_flag;i++){
             char *sl=strchr(cur->entries[i].repo,'/');
             safe_fmt(msg,sizeof(msg),"  [%d/%d] %s label=%d\r\n",i+1,cur->n_entries,cur->entries[i].repo,cur->entries[i].label);
             app_info(msg);
             if(sl){char u[128],r[128];int ul=(int)(sl-cur->entries[i].repo);if(ul>127)ul=127;memcpy(u,cur->entries[i].repo,ul);u[ul]='\0';safe_strcpy(r,sl+1,128);cmd_gitrain(u,r,cur->entries[i].label);cur->entries[i].done=1;}
             cur->current=i+1;InterlockedExchange(&g_worker_ping_ms,GetTickCount());
             report_progress((int)((i+1)*100/cur->n_entries));tb_yield_bg();}
         cur->running=0;app_safe("Curriculum done.\r\n");}
        break;

    case TASK_REPORT:
        cmd_report(task->arg1);
        break;

    /* ── NEW v13 tasks ── */
    case TASK_CONVERSE:
        {char reply[4096];int n_gen;reply[0]='\0';
         n_gen=cmd_converse(task->gen.prompt,reply,sizeof(reply));
         g_perf.conv_turns++;g_perf.tokens_generated+=(long)n_gen;
         BLOG_INFO("TASK_CONVERSE: %d tokens",n_gen);}
        break;

    case TASK_DOWNLOAD:
        cmd_model_download(task->dl.url,task->dl.save_path[0]?task->dl.save_path:NULL);
        break;

    case TASK_GUARD:
        cmd_guard_start(task->arg1[0]?task->arg1:NULL);
        break;

    case TASK_TRAIN:
        cmd_train(task->arg1, task->int1);
        break;

    default:
        app_warn("worker: unknown task\r\n");
        break;
    }

    if(g_hMain&&IsWindow(g_hMain)) PostMessage(g_hMain,WM_APP_DONE,0,0);
    InterlockedExchange(&g_worker_busy,0);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * §O  WATCHDOG THREAD
 * ═══════════════════════════════════════════════════════════════ */

/* FIX 4: watchdog uses cancel_flag to exit cleanly */
static volatile LONG g_watchdog_shutdown = 0;

DWORD WINAPI watchdog_thread_proc(LPVOID param)
{
    (void)param;
    while(!g_watchdog_shutdown){
        Sleep(1000); /* check every 1s for shutdown */
        if(g_watchdog_shutdown) break;
        if(g_worker_busy){
            DWORD now=GetTickCount(),last=(DWORD)g_worker_ping_ms,elapsed=now-last;
            if(elapsed>(DWORD)g_cfg.watchdog_timeout_ms){
                if(g_hMain&&IsWindow(g_hMain))PostMessage(g_hMain,WM_APP_WATCHDOG,0,(LPARAM)elapsed);
                InterlockedExchange(&g_cancel_flag,1);}}}
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * §P  SEH CRASH HANDLER  (v13: BLOG_ERROR)
 * ═══════════════════════════════════════════════════════════════ */

static LONG WINAPI seh_handler(EXCEPTION_POINTERS *ep)
{
    char msg[512]; FILE *f; SYSTEMTIME st;
    DWORD code=ep->ExceptionRecord->ExceptionCode;
    GetLocalTime(&st);
    safe_fmt(msg,sizeof(msg),"CRASH %04d-%02d-%02d %02d:%02d:%02d Code=0x%08lX Addr=%p\r\n",
             st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,st.wSecond,
             (unsigned long)code,ep->ExceptionRecord->ExceptionAddress);
    f=fopen("crash.log","a"); if(f){fprintf(f,"%s",msg);fclose(f);}
    BLOG_ERROR("SEH: code=0x%08lX addr=%p",(unsigned long)code,ep->ExceptionRecord->ExceptionAddress);
    if(g_model)model_save(g_model,"crash_autosave.bin");
    if(g_hMain&&IsWindow(g_hMain))MessageBoxA(g_hMain,msg,"Fatal Error",MB_OK|MB_ICONERROR);
    return EXCEPTION_EXECUTE_HANDLER;
}

/* ═══════════════════════════════════════════════════════════════
 * §Q  INPUT SUBCLASS PROC  (F2-F5, history, drag-drop)
 * ═══════════════════════════════════════════════════════════════ */

LRESULT CALLBACK InputSubclassProc(HWND hwnd,UINT msg,WPARAM wParam,LPARAM lParam)
{
    switch(msg){
    case WM_KEYDOWN:
        switch(wParam){
        case VK_F2: if(g_last_file[0])cmd_explain(g_last_file);else app_warn("F2: no file\r\n");return 0;
        case VK_F3: if(g_last_file[0])cmd_predict(g_last_file);else app_warn("F3: no file\r\n");return 0;
        case VK_F4: cmd_stats();return 0;
        case VK_F5: {char dir[MAX_PATH];if(GetCurrentDirectoryA(MAX_PATH,dir))cmd_scan(dir);}return 0;
        case VK_UP:
            if(g_hist_count>0){if(g_hist_pos<0)g_hist_pos=g_hist_count-1;else if(g_hist_pos>0)g_hist_pos--;
             SetWindowTextA(hwnd,g_hist[g_hist_pos%HIST_SIZE]);SendMessage(hwnd,EM_SETSEL,0x7FFF,0x7FFF);}return 0;
        case VK_DOWN:
            if(g_hist_pos>=0&&g_hist_pos<g_hist_count-1){g_hist_pos++;SetWindowTextA(hwnd,g_hist[g_hist_pos%HIST_SIZE]);SendMessage(hwnd,EM_SETSEL,0x7FFF,0x7FFF);}
            else if(g_hist_pos==g_hist_count-1){g_hist_pos=-1;SetWindowTextA(hwnd,"");}return 0;
        case VK_RETURN:
            /* BUG D FIX: plain Enter submits - removed Ctrl requirement */
            SendMessage(GetParent(hwnd),WM_COMMAND,MAKEWPARAM(BTN_SEND,BN_CLICKED),(LPARAM)hwnd);
            return 0;}
        break;
    case WM_DROPFILES:
        {HDROP hDrop=(HDROP)wParam;char path[MAX_PATH];UINT n=DragQueryFileA(hDrop,0xFFFFFFFF,NULL,0),i;
         for(i=0;i<n;i++){
             if(DragQueryFileA(hDrop,i,path,MAX_PATH)){
                 const char *ext=strrchr(path,'.');
                 DWORD attr=GetFileAttributesA(path);
                 char cmd_buf[MAX_PATH+32];
                 if(attr!=INVALID_FILE_ATTRIBUTES&&(attr&FILE_ATTRIBUTE_DIRECTORY)){
                     safe_fmt(cmd_buf,sizeof(cmd_buf),"fulltrain %s 3",path);
                     SetWindowTextA(hwnd,cmd_buf);
                 } else if(ext&&_stricmp(ext,".conv")==0){
                     safe_fmt(cmd_buf,sizeof(cmd_buf),"bpetrain %s",path);
                     SetWindowTextA(hwnd,cmd_buf);
                 } else {
                     SetWindowTextA(hwnd,path);
                     safe_strcpy(g_last_file,path,sizeof(g_last_file));
                 }
             }
         }
         DragFinish(hDrop);}return 0;}
    return CallWindowProc(g_oldInputProc,hwnd,msg,wParam,lParam);
}

/* ═══════════════════════════════════════════════════════════════
 * §R  WndProc  (v13: WM_APP_TOKEN, WM_APP_GUARD, BTN_CONVERSE)
 * ═══════════════════════════════════════════════════════════════ */

void gui_enable_inputs(BOOL enable)
{
    if (g_hInput) EnableWindow(g_hInput, enable);
    {
        int ids[] = {
            BTN_SEND, BTN_TRAIN, BTN_SCAN, BTN_STATS, BTN_HELP, BTN_UNDO,
            BTN_GENERATE, BTN_SIMILAR, BTN_SUMMARIZE, BTN_EXPLAIN, BTN_REPORT,
            BTN_FULLTRAIN, BTN_CONVERSE
        };
        int i;
        for (i = 0; i < (int)(sizeof(ids) / sizeof(ids[0])); i++) {
            HWND h = GetDlgItem(g_hMain, ids[i]);
            if (h) EnableWindow(h, enable);
        }
    }
    if (g_hCancel) EnableWindow(g_hCancel, !enable);
}

void dispatch_async(TaskType tt,const char *arg1,const char *arg2,int i1,double d1)
{
    if(g_worker_busy){app_warn("Task running. Cancel first.\r\n");return;}
    memset(&g_worker_task,0,sizeof(g_worker_task));
    g_worker_task.type=tt;
    if(arg1)safe_strcpy(g_worker_task.arg1,arg1,sizeof(g_worker_task.arg1));
    if(arg2)safe_strcpy(g_worker_task.arg2,arg2,sizeof(g_worker_task.arg2));
    g_worker_task.int1=i1; g_worker_task.dbl1=d1;
    InterlockedExchange(&g_cancel_flag,0);
    InterlockedExchange(&g_worker_busy,1);
    InterlockedExchange(&g_worker_ping_ms,GetTickCount());
    gui_enable_inputs(FALSE);
    if(g_hProgress)SendMessage(g_hProgress,PBM_SETPOS,0,0);
    g_worker_thread=CreateThread(NULL,0,worker_thread_proc,&g_worker_task,0,NULL);
    tb_thread_set_bg(g_worker_thread); /* v13 */
}

LRESULT CALLBACK WndProc(HWND hwnd,UINT msg,WPARAM wParam,LPARAM lParam)
{
    switch(msg){

    case WM_CREATE:
    {
        HFONT hFont; RECT rc; int W,H;
        g_hMain=hwnd;
        /* v13 STEP 1: probe RAM before any alloc */
        sysinfo_probe(&g_sysinfo);
        g_dyn_max_embeds=g_sysinfo.max_embeds;
        g_cfg.sysinfo_tier=g_sysinfo.tier;
        g_cfg.dyn_max_embeds=g_dyn_max_embeds;
        /* v13 STEP 2: open brain.log */
        blog_init("brain.log");
        tb_cpu_init();
        BLOG_INFO("TheBrain v13.0 starting (RAM=%luMB tier=%d embeds=%d)",
                  (unsigned long)g_sysinfo.total_mb,g_sysinfo.tier,g_dyn_max_embeds);

        GetClientRect(hwnd,&rc); W=rc.right; H=rc.bottom;

        /* Unicode RichEdit + input (Arabic/French UTF-8) */
        g_hChat=CreateWindowExW(WS_EX_CLIENTEDGE,L"RichEdit20W",NULL,
            WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_READONLY|ES_AUTOVSCROLL,
            0,0,W,H-140,hwnd,(HMENU)(UINT_PTR)CHAT_ID,g_hInst,NULL);
        g_hInput=CreateWindowExW(WS_EX_CLIENTEDGE,L"Edit",NULL,
            WS_CHILD|WS_VISIBLE|ES_AUTOHSCROLL,
            0,H-138,W-80,26,hwnd,(HMENU)(UINT_PTR)INPUT_ID,g_hInst,NULL);
        CreateWindowA("BUTTON","Send",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
            W-78,H-138,78,26,hwnd,(HMENU)(UINT_PTR)BTN_SEND,g_hInst,NULL);
        /* Row 1 buttons */
        CreateWindowA("BUTTON","Train",  WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,0,   H-108,70,24,hwnd,(HMENU)(UINT_PTR)BTN_TRAIN,   g_hInst,NULL);
        CreateWindowA("BUTTON","Scan",   WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,74,  H-108,70,24,hwnd,(HMENU)(UINT_PTR)BTN_SCAN,    g_hInst,NULL);
        CreateWindowA("BUTTON","Stats",  WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,148, H-108,70,24,hwnd,(HMENU)(UINT_PTR)BTN_STATS,   g_hInst,NULL);
        CreateWindowA("BUTTON","Help",   WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,222, H-108,70,24,hwnd,(HMENU)(UINT_PTR)BTN_HELP,    g_hInst,NULL);
        CreateWindowA("BUTTON","Undo",   WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,296, H-108,70,24,hwnd,(HMENU)(UINT_PTR)BTN_UNDO,    g_hInst,NULL);
        /* Row 2 buttons */
        CreateWindowA("BUTTON","Generate", WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,0,  H-80,70,24,hwnd,(HMENU)(UINT_PTR)BTN_GENERATE,g_hInst,NULL);
        CreateWindowA("BUTTON","Similar",  WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,74, H-80,70,24,hwnd,(HMENU)(UINT_PTR)BTN_SIMILAR, g_hInst,NULL);
        CreateWindowA("BUTTON","Summarize",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,148,H-80,70,24,hwnd,(HMENU)(UINT_PTR)BTN_SUMMARIZE,g_hInst,NULL);
        CreateWindowA("BUTTON","Explain",  WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,222,H-80,70,24,hwnd,(HMENU)(UINT_PTR)BTN_EXPLAIN, g_hInst,NULL);
        CreateWindowA("BUTTON","Report",   WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,296,H-80,72,24,hwnd,(HMENU)(UINT_PTR)BTN_REPORT,  g_hInst,NULL);
        CreateWindowA("BUTTON","FullTrain",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,372,H-80,72,24,hwnd,(HMENU)(UINT_PTR)BTN_FULLTRAIN,g_hInst,NULL);
        /* v13: Converse button */
        CreateWindowA("BUTTON","Converse", WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,448,H-80,72,24,hwnd,(HMENU)(UINT_PTR)BTN_CONVERSE,g_hInst,NULL);
        /* Cancel + Progress */
        g_hCancel=CreateWindowA("BUTTON","Cancel",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON|WS_DISABLED,
            W-78,H-108,78,24,hwnd,(HMENU)(UINT_PTR)BTN_CANCEL,g_hInst,NULL);
        g_hProgress=CreateWindowExA(0,PROGRESS_CLASS,NULL,WS_CHILD|WS_VISIBLE|PBS_SMOOTH,
            0,H-50,W,24,hwnd,(HMENU)(UINT_PTR)PROG_ID,g_hInst,NULL);
        SendMessage(g_hProgress,PBM_SETRANGE,0,MAKELPARAM(0,100));
        /* Font */
        hFont=CreateFontW(-12,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS,DEFAULT_QUALITY,FIXED_PITCH|FF_MODERN,L"Courier New");
        if(hFont){SendMessage(g_hChat,WM_SETFONT,(WPARAM)hFont,TRUE);SendMessage(g_hInput,WM_SETFONT,(WPARAM)hFont,TRUE);}
        SendMessage(g_hChat,EM_SETBKGNDCOLOR,0,(LPARAM)RGB(15,15,20));
        SendMessage(g_hChat,EM_SETLIMITTEXT,0x7FFFFFFF,0);
        /* Subclass input */
        g_oldInputProc=(WNDPROC)(LONG_PTR)SetWindowLongA(g_hInput,GWL_WNDPROC,(LONG)(LONG_PTR)InputSubclassProc);
        DragAcceptFiles(g_hInput,TRUE); DragAcceptFiles(hwnd,TRUE);
        /* Critical sections */
        InitializeCriticalSection(&g_cs_samples); InitializeCriticalSection(&g_cs_log);
        InitializeCriticalSection(&g_cs_cache);   InitializeCriticalSection(&g_cs_embeds);
        InitializeCriticalSection(&g_cs_model);
        /* ML buffers */
        g_samples   =(Sample*)       calloc(MAX_SAMPLES,sizeof(Sample));
        g_feat_cache=(FeatCacheEntry*)calloc(FEAT_CACHE_MAX,sizeof(FeatCacheEntry));
        /* FIX 5: cap embeds at 10000 to avoid 535MB allocation */
        if(g_dyn_max_embeds>10000) g_dyn_max_embeds=10000;
        g_embeds=(FileEmbed*)calloc((size_t)g_dyn_max_embeds,sizeof(FileEmbed));
        if(!g_embeds){
            BLOG_WARN("WM_CREATE: embeds OOM at %d, fallback 1000",g_dyn_max_embeds);
            g_dyn_max_embeds=1000;
            g_embeds=(FileEmbed*)calloc(1000,sizeof(FileEmbed));}
        /* Log file */
        {SYSTEMTIME st;char logn[64];GetLocalTime(&st);
         safe_fmt(logn,sizeof(logn),"brain_%04d%02d%02d.log",st.wYear,st.wMonth,st.wDay);
         safe_strcpy(g_logname,logn,sizeof(g_logname));g_logfp=fopen(logn,"a");}
        /* Load config + vocab */
        config_load();
        conv_apply_config(g_cfg.conv_use_facts, g_cfg.conv_stream,
                      g_cfg.conv_max_tokens, g_cfg.temperature, g_cfg.top_k,
                      g_cfg.conv_history_turns);
        bpe_init_vocab(&g_tokenizer);
        if(g_cfg.vocab_file[0])bpe_load(&g_tokenizer,g_cfg.vocab_file);
        /* Auto-load model: try model_v13.bin first, then latest checkpoint */
        {
            SysInfo si; DynModelCfg dyn; ModelConfig mc;
            sysinfo_probe(&si);
            dyn=sysinfo_make_cfg(&si,g_cfg.use_swiglu,g_cfg.use_rmsnorm,g_cfg.tie_embeddings);
            mc=model_cfg_from_dyn(&dyn);
            if (g_tokenizer.trained && g_tokenizer.vocab_size > 0) {
                mc.vocab_size = g_tokenizer.vocab_size;
            }
            g_model=model_create(&mc);
            if(g_model){
                int load_ok=0;
                /* Try model_v13.bin */
                if(g_cfg.model_file[0]){
                    load_ok=(model_load(g_model,g_cfg.model_file)==MODEL_OK);
                    if(load_ok){
                        BLOG_INFO("Auto-loaded %s",g_cfg.model_file);
                        app_colored("  Model loaded: ",COL_SAFE);
                        app_colored(g_cfg.model_file,COL_SAFE);
                        app("\r\n");
                    }
                }
                /* Fallback: load latest checkpoint */
                if(!load_ok){
                    WIN32_FIND_DATAA fd2; char pat2[320],best2[320]; HANDLE h2;
                    FILETIME bft2; bft2.dwHighDateTime=bft2.dwLowDateTime=0;
                    best2[0]='\0';
                    safe_fmt(pat2,sizeof(pat2),"%s\\ckpt_*.bin",CHECKPOINT_DIR);
                    h2=FindFirstFileA(pat2,&fd2);
                    if(h2!=INVALID_HANDLE_VALUE){
                        do{
                            if(CompareFileTime(&fd2.ftLastWriteTime,&bft2)>0){
                                bft2=fd2.ftLastWriteTime;
                                safe_fmt(best2,sizeof(best2),"%s\\%s",CHECKPOINT_DIR,fd2.cFileName);
                            }
                        }while(FindNextFileA(h2,&fd2));
                        FindClose(h2);
                    }
                    if(best2[0]){
                        load_ok=(model_load(g_model,best2)==MODEL_OK);
                        if(load_ok){
                            BLOG_INFO("Auto-loaded checkpoint: %s",best2);
                            app_colored("  Checkpoint auto-loaded: ",COL_SAFE);
                            app_colored(best2,COL_SAFE);
                            app("\r\n");
                        }
                    }
                }
                if(!load_ok){model_free(g_model);g_model=NULL;}
            }
            if(g_model && g_model->trained && g_tokenizer.trained &&
               g_tokenizer.vocab_size != g_model->cfg.vocab_size){
                char vmsg[256];
                safe_fmt(vmsg,sizeof(vmsg),
                         "  Warning: vocab mismatch (tokenizer=%d model=%d). "
                         "Run easytrain to resync.\r\n",
                         g_tokenizer.vocab_size,g_model->cfg.vocab_size);
                app_warn(vmsg);
                BLOG_WARN("Auto-load vocab mismatch tokenizer=%d model=%d",
                          g_tokenizer.vocab_size,g_model->cfg.vocab_size);
            }
            if(g_model && g_model->trained){
                app_colored("  Ready to chat - just type a question "
                            "(no retraining needed).\r\n",COL_INFO);
            } else {
                app_warn("  No saved model found. Type: easytrain\r\n"
                         "  (trains on data\\ folder - one simple command).\r\n");
            }
        }
        /* v13: init conversation engine */
        conv_init();
        /* Watchdog */
        g_watchdog_thread=CreateThread(NULL,0,watchdog_thread_proc,NULL,0,NULL);
        /* v13: auto-start guard */
        if(g_cfg.guard_enabled&&g_cfg.guard_dir[0])cmd_guard_start(g_cfg.guard_dir);
        g_perf.session_start=GetTickCount();
        /* Welcome */
        app_info("TheBrain v13.0  |  Conversational AI + Malware Analysis\r\n");
        {char wb[256];char cpuf[32];tb_cpu_features_str(cpuf,sizeof(cpuf));
         safe_fmt(wb,sizeof(wb),"  RAM: %luMB (tier %d)  Embeds: %d  CPU: %s  Guard: %s\r\n",
             (unsigned long)g_sysinfo.total_mb,g_sysinfo.tier,g_dyn_max_embeds,cpuf,g_cfg.guard_enabled?"AUTO":"off");
         app_colored(wb,COL_INFO);}
        /* Show working directory so user knows where files are saved */
        {char cwd[MAX_PATH]; char cwd_msg[MAX_PATH+64];
         GetCurrentDirectoryA(MAX_PATH,cwd);
         safe_fmt(cwd_msg,sizeof(cwd_msg),"  Dir : %s\r\n",cwd);
         app_colored(cwd_msg,COL_GREY);}
        app("  Tip : Type easytrain  (one command - trains on data\\)\r\n");
        app("        Drag a .conv file -> auto bpetrain\r\n");
        app("        Drag a folder  -> auto fulltrain\r\n");
        app_safe("Ready.\r\n");
    }
    return 0;

    case WM_SIZE:
    {
        int W=LOWORD(lParam),H=HIWORD(lParam); HWND h;
        if(g_hChat)     MoveWindow(g_hChat,    0,0,W,H-140,TRUE);
        if(g_hInput)    MoveWindow(g_hInput,   0,H-138,W-80,26,TRUE);
        if(g_hProgress) MoveWindow(g_hProgress,0,H-50,W,24,TRUE);
        h=GetDlgItem(hwnd,BTN_SEND);     if(h)MoveWindow(h,W-78,H-138,78,26,TRUE);
        h=GetDlgItem(hwnd,BTN_TRAIN);    if(h)MoveWindow(h,0,   H-108,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_SCAN);     if(h)MoveWindow(h,74,  H-108,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_STATS);    if(h)MoveWindow(h,148, H-108,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_HELP);     if(h)MoveWindow(h,222, H-108,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_UNDO);     if(h)MoveWindow(h,296, H-108,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_CANCEL);   if(h)MoveWindow(h,W-78,H-108,78,24,TRUE);
        h=GetDlgItem(hwnd,BTN_GENERATE); if(h)MoveWindow(h,0,   H-80,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_SIMILAR);  if(h)MoveWindow(h,74,  H-80,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_SUMMARIZE);if(h)MoveWindow(h,148, H-80,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_EXPLAIN);  if(h)MoveWindow(h,222, H-80,70,24,TRUE);
        h=GetDlgItem(hwnd,BTN_REPORT);   if(h)MoveWindow(h,296, H-80,72,24,TRUE);
        h=GetDlgItem(hwnd,BTN_FULLTRAIN);if(h)MoveWindow(h,372, H-80,72,24,TRUE);
        h=GetDlgItem(hwnd,BTN_CONVERSE); if(h)MoveWindow(h,448, H-80,72,24,TRUE);
    }
    return 0;

    case WM_COMMAND:
    {
        int ctrl_id=LOWORD(wParam);
        if(ctrl_id==BTN_SEND){ /* INPUT_ID Enter handled by InputSubclassProc */
            char cmd[4096];
            brain_get_input_utf8(cmd, (int)sizeof(cmd));
            if(cmd[0]){brain_clear_input();g_hist_pos=-1;
                app_colored("> ",COL_PROMPT);app_colored(cmd,COL_PROMPT);app("\r\n");
                process_command(cmd);}
            return 0;}
        switch(ctrl_id){
        case BTN_TRAIN:
            {OPENFILENAMEA ofn;char path[MAX_PATH];memset(&ofn,0,sizeof(ofn));ofn.lStructSize=sizeof(ofn);
             ofn.hwndOwner=hwnd;ofn.lpstrFile=path;ofn.nMaxFile=MAX_PATH;
             ofn.lpstrFilter="All Files\0*.*\0\0";ofn.Flags=OFN_FILEMUSTEXIST;path[0]='\0';
             if(GetOpenFileNameA(&ofn)){int label=(MessageBoxA(hwnd,"Is this DANGEROUS?","Label",MB_YESNO|MB_ICONQUESTION)==IDYES)?1:0;
                                        dispatch_async(TASK_TRAIN,path,NULL,label,0);}}
            break;
        case BTN_SCAN:
            {char dir[MAX_PATH];BROWSEINFOA bi;LPITEMIDLIST pidl;memset(&bi,0,sizeof(bi));
             bi.hwndOwner=hwnd;bi.lpszTitle="Select directory";bi.ulFlags=BIF_RETURNONLYFSDIRS|BIF_NEWDIALOGSTYLE;
             pidl=SHBrowseForFolderA(&bi);
             if(pidl&&SHGetPathFromIDListA(pidl,dir)){dispatch_async(TASK_ANOMALYSCAN,dir,NULL,0,0);CoTaskMemFree(pidl);}}
            break;
        case BTN_STATS:    cmd_stats();break;
        case BTN_HELP:     cmd_help("");break;
        case BTN_UNDO:     undo_push();break;
        case BTN_CANCEL:   InterlockedExchange(&g_cancel_flag,1);app_warn("Cancelling...\r\n");break;
        case BTN_GENERATE:
            {char prompt[4096];GenRequest req;
             brain_get_input_utf8(prompt, (int)sizeof(prompt));
             memset(&req,0,sizeof(req));req.max_new_tokens=512;req.temperature=g_cfg.temperature;req.top_k=g_cfg.top_k;
             safe_strcpy(req.prompt,prompt,sizeof(req.prompt));brain_clear_input();
             memcpy(&g_worker_task.gen,&req,sizeof(req));dispatch_async(TASK_GENERATE,NULL,NULL,0,0);}
            break;
        case BTN_SIMILAR:
            if(g_last_file[0])cmd_similar(g_last_file,5);else app_warn("Predict a file first.\r\n");
            break;
        case BTN_SUMMARIZE:
            {OPENFILENAMEA ofn;char path[MAX_PATH];memset(&ofn,0,sizeof(ofn));ofn.lStructSize=sizeof(ofn);
             ofn.hwndOwner=hwnd;ofn.lpstrFile=path;ofn.nMaxFile=MAX_PATH;ofn.lpstrFilter="All Files\0*.*\0\0";ofn.Flags=OFN_FILEMUSTEXIST;path[0]='\0';
             if(GetOpenFileNameA(&ofn))cmd_summarize(path);}
            break;
        case BTN_EXPLAIN:
            if(g_last_file[0])cmd_explain(g_last_file);else app_warn("Predict a file first.\r\n");
            break;
        case BTN_REPORT:
            {char dir[MAX_PATH];BROWSEINFOA bi;LPITEMIDLIST pidl;memset(&bi,0,sizeof(bi));
             bi.hwndOwner=hwnd;bi.lpszTitle="Report directory";bi.ulFlags=BIF_RETURNONLYFSDIRS|BIF_NEWDIALOGSTYLE;
             pidl=SHBrowseForFolderA(&bi);
             if(pidl&&SHGetPathFromIDListA(pidl,dir)){dispatch_async(TASK_REPORT,dir,NULL,0,0);CoTaskMemFree(pidl);}}
            break;
        case BTN_FULLTRAIN:
            {char dir[MAX_PATH];BROWSEINFOA bi;LPITEMIDLIST pidl;memset(&bi,0,sizeof(bi));
             bi.hwndOwner=hwnd;bi.lpszTitle="Corpus directory";bi.ulFlags=BIF_RETURNONLYFSDIRS|BIF_NEWDIALOGSTYLE;
             pidl=SHBrowseForFolderA(&bi);
             if(pidl&&SHGetPathFromIDListA(pidl,dir)){
                 if(!g_model)app_warn("Run 'pretrain' first.\r\n");
                 else dispatch_async(TASK_FULLTRAIN,dir,NULL,3,0);
                 CoTaskMemFree(pidl);}}
            break;
        /* NEW v13 */
        case BTN_CONVERSE:
            {char prompt[4096];
             brain_get_input_utf8(prompt, (int)sizeof(prompt));
             if(!prompt[0]){app_warn("Type something first.\r\n");break;}
             brain_clear_input();
             app_colored("> ",COL_PROMPT);app_colored(prompt,COL_PROMPT);app("\r\n");
             app_colored("[thinking...]\r\n",COL_GREY);
             safe_strcpy(g_worker_task.gen.prompt,prompt,sizeof(g_worker_task.gen.prompt));
             dispatch_async(TASK_CONVERSE,NULL,NULL,0,0);}
            break;
        }
    }
    return 0;

    case WM_APP_DONE:
        gui_enable_inputs(TRUE);
        if(g_hProgress)SendMessage(g_hProgress,PBM_SETPOS,100,0);
        if(g_worker_task.type==TASK_FULLTRAIN||g_worker_task.type==TASK_BATCHTRAIN||g_worker_task.type==TASK_GITRAIN)
            checkpoint_save();
        /* Suppress "Task complete." for conversational tasks */
        if(g_worker_task.type!=TASK_CONVERSE)
            app_safe("Task complete.\r\n");
        return 0;

    case WM_APP_PROGRESS:
        if(g_hProgress)SendMessage(g_hProgress,PBM_SETPOS,wParam,0);
        return 0;

    case WM_APP_LOG:
        {AppLogMsg *m=(AppLogMsg*)wParam;if(m){app_colored(m->text,m->col);free(m);}}
        return 0;

    /* NEW v13: streamed token from cmd_converse */
    case WM_APP_TOKEN:
        {char *tok=(char*)lParam;if(tok){app_colored(tok,COL_CONVERSE);free(tok);}}
        return 0;

    /* NEW v13: real-time guard alert */
    case WM_APP_GUARD:
        {char *alert=(char*)lParam;if(alert){app_colored(alert,COL_GUARD);free(alert);}FlashWindow(hwnd,TRUE);}
        return 0;

    case WM_APP_WATCHDOG:
        {char msg[256];safe_fmt(msg,sizeof(msg),"WARNING: worker hung for %lu ms. Cancelling.\r\n",(unsigned long)(DWORD)lParam);app_warn(msg);gui_enable_inputs(TRUE);}
        return 0;

    case WM_DROPFILES:
        {
        HDROP hDrop=(HDROP)wParam;
        char path[MAX_PATH];
        DWORD attr;
        char cmd_buf[MAX_PATH+32];
        if(DragQueryFileA(hDrop,0,path,MAX_PATH)){
            attr=GetFileAttributesA(path);
            if(attr!=INVALID_FILE_ATTRIBUTES && (attr&FILE_ATTRIBUTE_DIRECTORY)){
                /* FOLDER dropped -> suggest fulltrain */
                safe_fmt(cmd_buf,sizeof(cmd_buf),"fulltrain %s 3",path);
                SetWindowTextA(g_hInput,cmd_buf);
                app_colored("Folder dropped - press Enter to start fulltrain\r\n",COL_INFO);
            } else {
                const char *ext=strrchr(path,'.');
                if(ext && _stricmp(ext,".conv")==0){
                    /* .conv file dropped -> suggest bpetrain */
                    safe_fmt(cmd_buf,sizeof(cmd_buf),"bpetrain %s",path);
                    SetWindowTextA(g_hInput,cmd_buf);
                    app_colored(".conv file dropped - press Enter to train tokenizer\r\n",COL_INFO);
                } else {
                    /* other file -> just fill path */
                    SetWindowTextA(g_hInput,path);
                    app_colored("> ",COL_PROMPT);app_colored(path,COL_INFO);app(" (dropped)\r\n");
                }
                safe_strcpy(g_last_file,path,sizeof(g_last_file));
            }
        }
        DragFinish(hDrop);}
        return 0;

    case WM_DESTROY:
        InterlockedExchange(&g_cancel_flag,1);
        /* v13: stop guard first */
        cmd_guard_stop();
        if(g_worker_thread){WaitForSingleObject(g_worker_thread,3000);CloseHandle(g_worker_thread);}
        /* FIX 4: signal watchdog to exit cleanly */
        InterlockedExchange(&g_watchdog_shutdown,1);
        if(g_watchdog_thread){
            WaitForSingleObject(g_watchdog_thread,3000);
            CloseHandle(g_watchdog_thread);
        }
        config_save();
        if(g_model&&g_cfg.model_file[0])model_save(g_model,g_cfg.model_file);
        if(g_cfg.vocab_file[0])bpe_save(&g_tokenizer,g_cfg.vocab_file);
        if(g_model)    {model_free(g_model);g_model=NULL;}
        if(g_samples)  {free(g_samples);   g_samples=NULL;}
        if(g_feat_cache){free(g_feat_cache);g_feat_cache=NULL;}
        if(g_embeds)   {free(g_embeds);    g_embeds=NULL;}
        if(g_logfp)    {fclose(g_logfp);   g_logfp=NULL;}
        if(g_hRichEdit){FreeLibrary(g_hRichEdit);}
        DeleteCriticalSection(&g_cs_samples);DeleteCriticalSection(&g_cs_log);
        DeleteCriticalSection(&g_cs_cache);  DeleteCriticalSection(&g_cs_embeds);
        DeleteCriticalSection(&g_cs_model);
        /* v13: close brain.log last */
        BLOG_INFO("TheBrain v13.0 shutdown");
        tb_mt_shutdown();
        blog_close();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd,msg,wParam,lParam);
}

/* ═══════════════════════════════════════════════════════════════
 * §S  WinMain  (v13: GlobalMemoryStatus FIRST, dynamic title)
 * ═══════════════════════════════════════════════════════════════ */

int WINAPI WinMain(HINSTANCE hInst,HINSTANCE hPrev,LPSTR lpCmdLine,int nCmdShow)
{
    WNDCLASSEXA wc; HWND hwnd; MSG msg;
    char app_title[128]; int scr_w,scr_h,win_w,win_h;
    SysInfo early_si;
    static const char *tier_names[]={"TINY","SMALL","MEDIUM","LARGE","XLARGE"};
    int t;

    (void)hPrev;(void)lpCmdLine;
    SetUnhandledExceptionFilter(seh_handler);
    g_hInst=hInst;

    /* PORTABILITY FIX: set CWD to the exe's own directory.
     * This means model_v13.bin, checkpoints\, vocab.bpak, brain.conf
     * are always found relative to the exe, regardless of how it was launched. */
    {
        char exe_path[MAX_PATH];
        char *last_slash;
        GetModuleFileNameA(NULL, exe_path, MAX_PATH);
        last_slash = strrchr(exe_path, '\\');
        if (last_slash) {
            *last_slash = '\0';
            SetCurrentDirectoryA(exe_path);
        }
    }

    /* v13: probe RAM BEFORE any allocation */
    {MEMORYSTATUS ms;memset(&ms,0,sizeof(ms));ms.dwLength=sizeof(ms);GlobalMemoryStatus(&ms);
     early_si.total_mb=(DWORD)(ms.dwTotalPhys/(1024UL*1024UL));
     early_si.avail_mb=(DWORD)(ms.dwAvailPhys/(1024UL*1024UL));
     if     (early_si.total_mb<256)  early_si.tier=RAM_TIER_TINY;
     else if(early_si.total_mb<512)  early_si.tier=RAM_TIER_SMALL;
     else if(early_si.total_mb<1024) early_si.tier=RAM_TIER_MEDIUM;
     else if(early_si.total_mb<4096) early_si.tier=RAM_TIER_LARGE;
     else                            early_si.tier=RAM_TIER_XLARGE;
     g_sysinfo=early_si;}

    /* v13: open brain.log before anything else */
    blog_init("brain.log");
    BLOG_INFO("WinMain: RAM=%luMB avail=%luMB tier=%d",
              (unsigned long)early_si.total_mb,(unsigned long)early_si.avail_mb,early_si.tier);

    g_hRichEdit=LoadLibraryA("riched20.dll");
    if(!g_hRichEdit)g_hRichEdit=LoadLibraryA("riched32.dll");

    memset(&wc,0,sizeof(wc));
    wc.cbSize=sizeof(wc); wc.style=CS_HREDRAW|CS_VREDRAW;
    wc.lpfnWndProc=WndProc; wc.hInstance=hInst;
    wc.hCursor=LoadCursor(NULL,IDC_ARROW);
    wc.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName="TheBrainClass";
    wc.hIcon=LoadIcon(NULL,IDI_APPLICATION);
    wc.hIconSm=LoadIcon(NULL,IDI_APPLICATION);
    if(!RegisterClassExA(&wc)){
        MessageBoxA(NULL,"RegisterClassEx failed","Fatal",MB_OK|MB_ICONERROR);
        blog_close();return 1;}

    {INITCOMMONCONTROLSEX icc;icc.dwSize=sizeof(icc);icc.dwICC=ICC_PROGRESS_CLASS|ICC_WIN95_CLASSES;InitCommonControlsEx(&icc);}

    scr_w=GetSystemMetrics(SM_CXSCREEN); scr_h=GetSystemMetrics(SM_CYSCREEN);
    win_w=(scr_w*78)/100; win_h=(scr_h*78)/100;

    /* v13: dynamic title with RAM tier */
    t=early_si.tier; if(t<0)t=0; if(t>4)t=4;
    safe_fmt(app_title,sizeof(app_title),
             "TheBrain v13.0  |  %s  |  RAM:%luMB  |  %s",
             BRAIN_BUILD_DATE,(unsigned long)early_si.total_mb,tier_names[t]);

    hwnd=CreateWindowExA(WS_EX_ACCEPTFILES,"TheBrainClass",app_title,
                          WS_OVERLAPPEDWINDOW,
                          (scr_w-win_w)/2,(scr_h-win_h)/2,win_w,win_h,
                          NULL,NULL,hInst,NULL);
    if(!hwnd){MessageBoxA(NULL,"CreateWindow failed","Fatal",MB_OK|MB_ICONERROR);blog_close();return 1;}

    ShowWindow(hwnd,nCmdShow); UpdateWindow(hwnd);

    while(GetMessageA(&msg,NULL,0,0)>0){
        if(!IsDialogMessageA(hwnd,&msg)){TranslateMessage(&msg);DispatchMessageA(&msg);}}

    blog_close();
    return (int)msg.wParam;
}

/*
 * ─────────────────────────────────────────────────────────────
 * END OF PART 8B  — COMPLETE, NO STUBS
 *
 * This file replaces both TheBrain_v13_Part8.c AND the
 * "include v12 verbatim" notes from the earlier session.
 *
 * Every section from v12 Part B is fully implemented here:
 *   §B  PE entropy/header/imports/x86 disasm (full)
 *   §C  K-Means/kselect/clustermap (full)
 *   §D  Weak label/self-train (full)
 *   §E  Feature importance/permutation (full)
 *   §F  Anomaly scan recursive (full)
 *   §G  Similar file cosine search (full)
 *   §H  Chain-of-thought report 4-step (full)
 *   §I  GitHub WinInet training (full)
 *   §J  Forum HTML scraper training (full)
 *   §K  XOR/B64/ROT13/Caesar/RC4/Smart decrypt (full)
 *   §L  config_save/load all v13 fields (full)
 *   §M  Checkpoint/rollback/dead-letter/retry (full)
 *   §N  Worker thread all v12+v13 tasks (full)
 *   §O  Watchdog thread (full)
 *   §P  SEH + BLOG_ERROR (full)
 *   §Q  InputSubclassProc F2-F5+history+drag (full)
 *   §R  WndProc + WM_APP_TOKEN + WM_APP_GUARD + BTN_CONVERSE (full)
 *   §S  WinMain GlobalMemoryStatus FIRST + dynamic title (full)
 * ─────────────────────────────────────────────────────────────
 */

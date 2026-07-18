/* file_utils.c - safe file utilities */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long file_size(const char *p){FILE *f=fopen(p,"rb");long s;if(!f)return -1;fseek(f,0,SEEK_END);s=ftell(f);fclose(f);return s;}
char *file_read_all(const char *p,long *sz){
    FILE *f=fopen(p,"rb");char *buf;long s;
    if(!f)return NULL;fseek(f,0,SEEK_END);s=ftell(f);rewind(f);
    buf=(char*)malloc((size_t)s+1);if(!buf){fclose(f);return NULL;}
    fread(buf,1,(size_t)s,f);buf[s]='\0';fclose(f);if(sz)*sz=s;return buf;
}
int file_write_all(const char *p,const char *d,size_t n){FILE *f=fopen(p,"wb");if(!f)return 0;fwrite(d,1,n,f);fclose(f);return 1;}
int file_copy(const char *s,const char *d){
    char buf[4096];size_t n;FILE *fs=fopen(s,"rb"),*fd=fopen(d,"wb");
    if(!fs||!fd){if(fs)fclose(fs);if(fd)fclose(fd);return 0;}
    while((n=fread(buf,1,sizeof(buf),fs))>0)fwrite(buf,1,n,fd);
    fclose(fs);fclose(fd);return 1;
}
const char *file_ext(const char *p){const char *d=strrchr(p,'.');return d?d+1:"";}
int file_exists(const char *p){FILE *f=fopen(p,"rb");if(f){fclose(f);return 1;}return 0;}
int main(void){printf("file_utils ok\n");return 0;}

/* base64.c - encode/decode */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static const char B64[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
void b64_encode(const unsigned char *in,size_t len,char *out){
    size_t i;int j=0;
    for(i=0;i<len;i+=3){
        unsigned int v=in[i]<<16|(i+1<len?in[i+1]<<8:0)|(i+2<len?in[i+2]:0);
        out[j++]=B64[(v>>18)&63];out[j++]=B64[(v>>12)&63];
        out[j++]=(i+1<len)?B64[(v>>6)&63]:'=';
        out[j++]=(i+2<len)?B64[v&63]:'=';}
    out[j]='\0';}
int main(void){
    char out[256];
    b64_encode((unsigned char*)"Hello World",11,out);
    printf("b64: %s\n",out);return 0;}

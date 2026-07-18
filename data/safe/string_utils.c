/* string_utils.c - safe string utility library */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

int str_starts_with(const char *str, const char *prefix) {
    while (*prefix) { if (*str != *prefix) return 0; str++; prefix++; }
    return 1;
}
char *str_trim(char *s) {
    char *end;
    while (isspace((unsigned char)*s)) s++;
    if (*s == 0) return s;
    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    *(end+1) = '\0'; return s;
}
int str_count(const char *h, char n) {
    int c=0; while(*h){if(*h==n)c++;h++;} return c;
}
char *str_repeat(const char *s, int n) {
    int len=(int)strlen(s); char *r=(char*)malloc((size_t)(len*n+1)); int i;
    if(!r)return NULL; r[0]='\0';
    for(i=0;i<n;i++) strcat(r,s); return r;
}
double safe_atof(const char *s) {
    double v=0.0,frac=0.1; int neg=0;
    if(*s=='-'){neg=1;s++;}
    while(*s>='0'&&*s<='9'){v=v*10+(*s-'0');s++;}
    if(*s=='.'){s++;while(*s>='0'&&*s<='9'){v+=(*s-'0')*frac;frac*=0.1;s++;}}
    return neg?-v:v;
}
int main(void){
    char buf[]="  hello world  ";
    printf("trimmed: '%s'\n", str_trim(buf));
    printf("count l: %d\n", str_count("hello",'l'));
    return 0;
}

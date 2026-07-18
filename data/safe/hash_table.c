/* hash_table.c - open addressing hash table */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HT_SZ 256
typedef struct{char key[64];int val;int used;}HEntry;
typedef struct{HEntry s[HT_SZ];int count;}HTable;
static unsigned int hh(const char *k){unsigned int h=5381;while(*k){h=((h<<5)+h)^(unsigned char)*k;k++;}return h%HT_SZ;}
void ht_set(HTable *t,const char *k,int v){unsigned int s=hh(k);while(t->s[s].used&&strcmp(t->s[s].key,k)!=0)s=(s+1)%HT_SZ;if(!t->s[s].used)t->count++;strncpy(t->s[s].key,k,63);t->s[s].val=v;t->s[s].used=1;}
int ht_get(HTable *t,const char *k,int def){unsigned int s=hh(k);while(t->s[s].used){if(strcmp(t->s[s].key,k)==0)return t->s[s].val;s=(s+1)%HT_SZ;}return def;}
void ht_del(HTable *t,const char *k){unsigned int s=hh(k);while(t->s[s].used){if(strcmp(t->s[s].key,k)==0){t->s[s].used=0;t->count--;return;}s=(s+1)%HT_SZ;}}
int main(void){
    HTable t;memset(&t,0,sizeof(t));
    ht_set(&t,"apples",5);ht_set(&t,"bananas",3);ht_set(&t,"cherries",12);
    printf("apples=%d bananas=%d cherries=%d\n",ht_get(&t,"apples",0),ht_get(&t,"bananas",0),ht_get(&t,"cherries",0));
    return 0;
}

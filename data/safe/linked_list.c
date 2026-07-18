/* linked_list.c - simple linked list */
#include <stdio.h>
#include <stdlib.h>

typedef struct Node{int value;struct Node *next;}Node;
Node *node_new(int v){Node *n=(Node*)malloc(sizeof(Node));if(!n)return NULL;n->value=v;n->next=NULL;return n;}
Node *list_push(Node *h,int v){Node *n=node_new(v);if(!n)return h;n->next=h;return n;}
Node *list_append(Node *h,int v){Node *n=node_new(v),*c=h;if(!n)return h;if(!h)return n;while(c->next)c=c->next;c->next=n;return h;}
int list_len(Node *h){int n=0;while(h){n++;h=h->next;}return n;}
void list_free(Node *h){while(h){Node *t=h->next;free(h);h=t;}}
Node *list_reverse(Node *h){Node *p=NULL,*c=h,*n;while(c){n=c->next;c->next=p;p=c;c=n;}return p;}
void list_print(Node *h){while(h){printf("%d->",h->value);h=h->next;}printf("NULL\n");}
int main(void){
    Node *l=NULL;int i;
    for(i=1;i<=5;i++)l=list_append(l,i);
    list_print(l);l=list_reverse(l);list_print(l);
    printf("len=%d\n",list_len(l));list_free(l);return 0;
}

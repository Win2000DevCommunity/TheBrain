/* sort_utils.c - sorting algorithms */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void bubble_sort(int *a,int n){int i,j,t;
    for(i=0;i<n-1;i++)for(j=0;j<n-i-1;j++)
        if(a[j]>a[j+1]){t=a[j];a[j]=a[j+1];a[j+1]=t;}}
void insertion_sort(int *a,int n){int i,k,t;
    for(i=1;i<n;i++){t=a[i];k=i-1;
        while(k>=0&&a[k]>t){a[k+1]=a[k];k--;}a[k+1]=t;}}
static int cmp_int(const void *a,const void *b){return *(int*)a-*(int*)b;}
void qsort_wrap(int *a,int n){qsort(a,n,sizeof(int),cmp_int);}
void print_arr(int *a,int n){int i;for(i=0;i<n;i++)printf("%d ",a[i]);printf("\n");}
int main(void){
    int a[]={5,3,8,1,9,2,7,4,6};
    bubble_sort(a,9);print_arr(a,9);return 0;}

/* sort_algorithms.c - sorting algorithms */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void bubble_sort(int *a,int n){int i,j,t;for(i=0;i<n-1;i++)for(j=0;j<n-i-1;j++)if(a[j]>a[j+1]){t=a[j];a[j]=a[j+1];a[j+1]=t;}}
void insertion_sort(int *a,int n){int i,j,k;for(i=1;i<n;i++){k=a[i];for(j=i-1;j>=0&&a[j]>k;j--)a[j+1]=a[j];a[j+1]=k;}}
int partition(int *a,int lo,int hi){int p=a[hi],i=lo-1,j,t;for(j=lo;j<hi;j++)if(a[j]<=p){i++;t=a[i];a[i]=a[j];a[j]=t;}{int tt=a[i+1];a[i+1]=a[hi];a[hi]=tt;}return i+1;}
void quicksort(int *a,int lo,int hi){if(lo<hi){int p=partition(a,lo,hi);quicksort(a,lo,p-1);quicksort(a,p+1,hi);}}
int binary_search(int *a,int n,int v){int lo=0,hi=n-1;while(lo<=hi){int m=(lo+hi)/2;if(a[m]==v)return m;if(a[m]<v)lo=m+1;else hi=m-1;}return -1;}
int main(void){
    int a[]={5,2,8,1,9,3,7,4,6};int n=9;
    quicksort(a,0,n-1);
    printf("sorted: ");int i;for(i=0;i<n;i++)printf("%d ",a[i]);printf("\n");
    printf("search 7: idx=%d\n",binary_search(a,n,7));
    return 0;
}

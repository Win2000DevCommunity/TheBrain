/* math_utils.c - safe math utilities */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double clamp(double v,double lo,double hi){return v<lo?lo:v>hi?hi:v;}
double lerp(double a,double b,double t){return a+t*(b-a);}
int is_prime(int n){int i;if(n<2)return 0;if(n==2)return 1;if(n%2==0)return 0;for(i=3;i*i<=n;i+=2)if(n%i==0)return 0;return 1;}
unsigned int gcd(unsigned int a,unsigned int b){while(b){unsigned int t=b;b=a%b;a=t;}return a;}
double std_dev(double *a,int n){double m=0,v=0;int i;for(i=0;i<n;i++)m+=a[i];m/=n;for(i=0;i<n;i++){double d=a[i]-m;v+=d*d;}return sqrt(v/n);}
double moving_avg(double *a,int n,int w){double s=0;int i;if(w>n)w=n;for(i=n-w;i<n;i++)s+=a[i];return s/w;}
int factorial(int n){return n<=1?1:n*factorial(n-1);}
double power(double base,int exp){double r=1.0;int i;for(i=0;i<exp;i++)r*=base;return r;}
int main(void){
    double d[]={1,2,3,4,5};
    printf("stddev=%.4f gcd=%u prime97=%d\n",std_dev(d,5),gcd(48,18),is_prime(97));
    return 0;
}

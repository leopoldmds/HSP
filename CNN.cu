#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void cuda_hello(){
    printf("Hello World \n");
}

void MatrixInit(float *M, int n, int p){
    for (int i=0; i<n*p+1;i++){
            M[i]=(float)rand()/(float)(RAND_MAX)*2-1;
        }
}

void MatrixPrint(float *M, int n, int p){
    printf(" %f ",M[0]);
    for (int i=2;i<n*p+1;i++){
    if(i%p!=0){
    printf(" %f ",M[i]);
    }
    else{
    printf(" %f \n",M[i]);}
    }
    printf("\n"); 
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i=0; i<n*p+1;i++){
        Mout[i]=M1[i]+M2[i];
    }
}



int main(){
    int n = 4;
    int p = 4;
    float *M1 = (float *)malloc(n*p*sizeof(float));
    float *M2 = (float *)malloc(n*p*sizeof(float));
    float *Mout = (float *)malloc(n*p*sizeof(float));
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    MatrixAdd(M1,M2,Mout,n,p);
    MatrixPrint(Mout,n,p);
    cudaDeviceSynchronize();
    return 0;
}

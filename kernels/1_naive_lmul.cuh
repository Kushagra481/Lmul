#include <iostream>
#include <math.h>

__device__ 
float Lmul(float x, float y){
    union { int i; float f; } X;
    union { int i; float f; } Y;
    union { int i; float f; } Z;

    X.f = x;
    Y.f = y;
    Z.i = (((X.i >> 31)) ^ ((Y.i >> 31))) << 31 |  // Sign XOR
    ((X.i & 0x7FFFFFFF) + (Y.i & 0x7FFFFFFF) - 0x3F780000) & 0x7FFFFFFF;
    return Z.f;
}

__global__
void sgemm_naive_lmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    const int x= blockIdx.x * blockDim.x + threadIdx.x;
    const int y= blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0;
        for (int i = 0; i < K; ++i) {            
            tmp += Lmul(A[x * K + i], B[i * N + y]);
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = Lmul(alpha, tmp) + Lmul(beta, C[x * N + y]);
    }
}

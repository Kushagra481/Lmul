#include <iostream>
#include <math.h>

__device__
float Lmul(int x, int y){
    if (x == 0 || y == 0) {
        return 0;
    }
    return (((x >> 31)) ^ ((y >> 31))) << 31 | ((x & 0x7FFFFFFF) + (y & 0x7FFFFFFF) - 0x3F780000) & 0x7FFFFFFF;
}

__global__
void sgemm_naive_lmul(int M, int N, int K, int alpha, const int *A, const int *B, int beta, int *C){
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

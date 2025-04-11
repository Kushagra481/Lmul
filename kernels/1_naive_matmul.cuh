#include <iostream>
#include <math.h>

__global__
void sgemm_naive_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    const int x= blockIdx.x * blockDim.x + threadIdx.x;
    const int y= blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
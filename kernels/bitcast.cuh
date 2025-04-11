#include <iostream>

__device__ float int_to_float(int x) {
    union { int i; float f; } converter;
    converter.i = x;
    return converter.f;  // UB in standard C++ but commonly used
}

__device__ int float_to_int(float x) {
    union { float f; int i; } converter;
    converter.f = x;
    return converter.i;
}

__global__
void int_bitcast(const float *in, int *out, int size){
    const int x= blockIdx.x * blockDim.x + threadIdx.x;
    union { int i; float f; } converter;
    if(x < size) {
        out[x] = float_to_int(in[x]);
    }
}

__global__
void float_bitcast(const int *in, float *out, int size){
    const int x= blockIdx.x * blockDim.x + threadIdx.x;
    if(x < size) {
        out[x] = int_to_float(in[x]);
    }
}

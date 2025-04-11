#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  float maxVal = 0.0;
  float minVal = 0.0;
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
    if (tmp > maxVal) {maxVal = tmp;}
    if (tmp < minVal) {minVal = tmp;}
  }
  std::cout << "Max value: " << maxVal << " Min value: " << minVal << std::endl;
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
  }

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C) {
// cuBLAS uses column-major order. So we change the order of our row-major A &
// B, since (B^T*A^T)^T = (A*B)
// This runs cuBLAS in full fp32 mode
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C) {
// This runs cuBLAS with mixed precision (performing the mul with operands
// downcast to bf16), which is ~4x faster
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C) {
// This runs cuBLAS with mixed precision (performing the mul with operands
// downcast to bf16), which is ~4x faster
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_sgemm_naive_matmul(int M, int N, int K, float alpha, float *A, float *B,
  float beta, float *C) {
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32, 32);
sgemm_naive_matmul<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling_matmul(int M, int N, int K, float alpha, float *A, float *B,
float beta, float *C) {
const uint BM = 64;
const uint BN = 64;
const uint BK = 8;
const uint TM = 8;
dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim((BM * BN) / TM);
sgemm1DBlocktiling_matmul<BM, BN, BK, TM>
<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_naive_lmul(int M, int N, int K, int alpha, int *A, int *B,
      int beta, int *C) {
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32, 32);
sgemm_naive_lmul<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling_lmul(int M, int N, int K, int alpha, int *A, int *B,
  int beta, int *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
sgemm1DBlocktiling_lmul<BM, BN, BK, TM>
<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void int_casting_on_device(const float *in, int *out, int size) {
  const int threads = 256;
  const int blocks = (size * size + threads - 1) / threads;
  int_bitcast<<<blocks, threads>>>(in, out, size);
}

void float_casting_on_device(const int *in, float *out, int size){
  const int threads = 256;
  const int blocks = (size * size + threads - 1) / threads;
  float_bitcast<<<blocks, threads>>>(in, out, size);
}

void run_kernel_matmul(int kernel_num, int M, int N, int K, float alpha, float *A,
  float *B, float beta, float *C, cublasHandle_t handle){
switch (kernel_num) {
  case 0:
  runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
  break;
  case 1:
  run_sgemm_naive_matmul(M, N, K, alpha, A, B, beta, C);
  break;
  case 2:
  runSgemm1DBlocktiling_matmul(M, N, K, alpha, A, B, beta, C);
  break;
  default:
  throw std::invalid_argument("Unknown kernel number");
}
}

void run_kernel_lmul(int kernel_num, int M, int N, int K, int alpha, int *A,
  int *B, int beta, int *C, cublasHandle_t handle){
switch (kernel_num) {
  case 1:
  run_sgemm_naive_lmul(M, N, K, alpha, A, B, beta, C);
  break;
  case 2:
  runSgemm1DBlocktiling_lmul(M, N, K, alpha, A, B, beta, C);
  break;
  default:
  throw std::invalid_argument("Unknown kernel number");
}
}
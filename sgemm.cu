#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 2, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-2)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C
  int alpha_i = static_cast<int>(alpha);
  int beta_i = static_cast<int>(beta);

  float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr; // host matrices

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices
  // int *dA_i = nullptr, *dB_i = nullptr, *dC_i = nullptr; // int device matrices

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);
  
  // allocate memory for the device matrices
  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));
  
  // cudaCheck(cudaMalloc((void **)&dA_i, sizeof(int) * max_size * max_size));
  // cudaCheck(cudaMalloc((void **)&dB_i, sizeof(int) * max_size * max_size));
  // cudaCheck(cudaMalloc((void **)&dC_i, sizeof(int) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  // Cast device float matrix to int matrix
  // We don't reset dC between runs to save time
  // int_casting_on_device(dA, dA_i, max_size);
  // int_casting_on_device(dB, dB_i, max_size);
  // int_casting_on_device(dC, dC_i, max_size);
  // cudaCheck(cudaDeviceSynchronize());
  // cudaCheck(cudaGetLastError());

  int repeat_times = 3;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel_matmul(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle); // Executes the kernel, modifies the result matrix
      run_kernel_lmul(kernel_num, m, n, k, alpha_i, dA, dB, beta_i, dC, handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
      
      // float_casting_on_device(dC_i, dC, max_size);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      verify_matrix(C_ref, C, m * n);
    }

    // Testing matmul performance
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel_lmul(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Matmul Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    
    if (kernel_num == 0) {
      continue;
    }
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel_matmul(kernel_num, m, n, k, alpha_i, dA, dB, beta_i, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    printf(
        "Lmul Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  // cudaFree(dA_i);
  // cudaFree(dB_i);
  // cudaFree(dC_i);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
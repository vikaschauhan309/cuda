#include <cuda_runtime.h>
#include "matrix_add.hpp"

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;

    if (row < width && col < width) {
        C[idx] = A[idx] + B[idx];
    }
}

void cpuMatrixAdd(const float* A, const float* B, float* C, int width) {
    for (int i = 0; i < width * width; ++i) {
        C[i] = A[i] + B[i];
    }
}

void launchMatrixAddGPU(const float* A, const float* B, float* C, int width) {
    size_t bytes = width * width * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (width + 15) / 16);

    matrixAddKernel<<<grid, block>>>(d_A, d_B, d_C, width);
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

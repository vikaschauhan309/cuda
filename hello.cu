#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;

    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}

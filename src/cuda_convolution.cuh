#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Timer class using CUDA events
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// V1: Naive global memory convolution
void conv2d_cuda_v1(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16
);

// V1_const: Global memory convolution with constant memory kernel
void conv2d_cuda_v1_const(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16
);

// V2: Shared memory tiled convolution
void conv2d_cuda_v2(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16
);

// Get device info
void print_device_info();

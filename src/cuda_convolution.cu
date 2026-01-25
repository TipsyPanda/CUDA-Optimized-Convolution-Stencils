#include "cuda_convolution.cuh"
#include <algorithm>
#include <vector>

// Maximum kernel size we support (7x7 = 49 elements)
#define MAX_KERNEL_SIZE 7
#define MAX_KERNEL_ELEMENTS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)

// Constant memory for kernel (accessible by all threads, cached)
__constant__ float c_kernel[MAX_KERNEL_ELEMENTS];

// ============================================================================
// V1: Naive Global Memory Convolution
// ============================================================================

__global__ void conv2d_kernel_v1(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel,
    int kernel_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int half_k = kernel_size / 2;
    float acc = 0.0f;

    for (int ky = -half_k; ky <= half_k; ++ky) {
        const int iy = y + ky;
        if (iy < 0 || iy >= height) continue;

        for (int kx = -half_k; kx <= half_k; ++kx) {
            const int ix = x + kx;
            if (ix < 0 || ix >= width) continue;

            const int input_idx = iy * width + ix;
            const int kernel_idx = (ky + half_k) * kernel_size + (kx + half_k);
            acc += input[input_idx] * kernel[kernel_idx];
        }
    }

    output[y * width + x] = acc;
}

void conv2d_cuda_v1(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x,
    int block_y
) {
    dim3 block(block_x, block_y);
    dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);

    conv2d_kernel_v1<<<grid, block>>>(d_input, d_output, width, height, d_kernel, kernel_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// V2: Shared Memory Tiled Convolution
// ============================================================================

// Template for different kernel sizes to allow compile-time tile size calculation
template <int BLOCK_X, int BLOCK_Y, int KERNEL_SIZE>
__global__ void conv2d_kernel_v2_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    constexpr int HALF_K = KERNEL_SIZE / 2;
    constexpr int TILE_W = BLOCK_X;
    constexpr int TILE_H = BLOCK_Y;
    constexpr int SHARED_W = TILE_W + 2 * HALF_K;
    constexpr int SHARED_H = TILE_H + 2 * HALF_K;

    // Shared memory for tile + halo
    __shared__ float tile[SHARED_H][SHARED_W];

    // Output pixel coordinates
    const int out_x = blockIdx.x * TILE_W + threadIdx.x;
    const int out_y = blockIdx.y * TILE_H + threadIdx.y;

    // Load tile + halo into shared memory
    // Each thread may need to load multiple elements
    const int tile_start_x = blockIdx.x * TILE_W - HALF_K;
    const int tile_start_y = blockIdx.y * TILE_H - HALF_K;

    // Calculate how many elements each thread needs to load
    const int num_loads_x = (SHARED_W + BLOCK_X - 1) / BLOCK_X;
    const int num_loads_y = (SHARED_H + BLOCK_Y - 1) / BLOCK_Y;

    for (int ly = 0; ly < num_loads_y; ++ly) {
        for (int lx = 0; lx < num_loads_x; ++lx) {
            const int shared_x = threadIdx.x + lx * BLOCK_X;
            const int shared_y = threadIdx.y + ly * BLOCK_Y;

            if (shared_x < SHARED_W && shared_y < SHARED_H) {
                const int global_x = tile_start_x + shared_x;
                const int global_y = tile_start_y + shared_y;

                float val = 0.0f;
                if (global_x >= 0 && global_x < width &&
                    global_y >= 0 && global_y < height) {
                    val = input[global_y * width + global_x];
                }
                tile[shared_y][shared_x] = val;
            }
        }
    }

    __syncthreads();

    // Compute convolution for this pixel
    if (out_x < width && out_y < height) {
        float acc = 0.0f;

        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                const int sx = threadIdx.x + kx;
                const int sy = threadIdx.y + ky;
                acc += tile[sy][sx] * c_kernel[ky * KERNEL_SIZE + kx];
            }
        }

        output[out_y * width + out_x] = acc;
    }
}

// Wrapper that dispatches to correct template instantiation
void conv2d_cuda_v2(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x,
    int block_y
) {
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, d_kernel, kernel_size * kernel_size * sizeof(float)));

    dim3 block(block_x, block_y);
    dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);

    // Dispatch based on kernel size and block configuration
    #define LAUNCH_V2(BX, BY, KS) \
        if (block_x == BX && block_y == BY && kernel_size == KS) { \
            conv2d_kernel_v2_shared<BX, BY, KS><<<grid, block>>>(d_input, d_output, width, height); \
        }

    // Common configurations
    LAUNCH_V2(8, 8, 3)
    else LAUNCH_V2(8, 8, 5)
    else LAUNCH_V2(8, 8, 7)
    else LAUNCH_V2(16, 16, 3)
    else LAUNCH_V2(16, 16, 5)
    else LAUNCH_V2(16, 16, 7)
    else LAUNCH_V2(32, 8, 3)
    else LAUNCH_V2(32, 8, 5)
    else LAUNCH_V2(32, 8, 7)
    else LAUNCH_V2(32, 16, 3)
    else LAUNCH_V2(32, 16, 5)
    else LAUNCH_V2(32, 16, 7)
    else {
        // Fallback to V1 for unsupported configurations
        fprintf(stderr, "Warning: V2 configuration (%d,%d,%d) not supported, falling back to V1\n",
                block_x, block_y, kernel_size);
        conv2d_cuda_v1(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
        return;
    }

    #undef LAUNCH_V2

    CUDA_CHECK(cudaGetLastError());
}

void set_constant_kernel(const float* kernel, int kernel_size) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel, kernel_size * kernel_size * sizeof(float)));
}

// ============================================================================
// V3: Fused Gaussian Blur + Sobel Magnitude
// ============================================================================

// Hardcoded kernels for fusion
__constant__ float c_gaussian_3x3[9] = {
    1.0f/16, 2.0f/16, 1.0f/16,
    2.0f/16, 4.0f/16, 2.0f/16,
    1.0f/16, 2.0f/16, 1.0f/16
};

__constant__ float c_sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

__constant__ float c_sobel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// Simple version of V3 kernel without shared memory optimization for correctness verification
__global__ void conv2d_kernel_v3_fused_simple(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // First compute 3x3 Gaussian blur values for the 3x3 neighborhood around (x,y)
    float blurred[3][3];

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int gx_pos = x + dx;
            int gy_pos = y + dy;

            // If position is out of bounds, set blurred to 0 (matches CPU zero-padding for Sobel)
            if (gx_pos < 0 || gx_pos >= width || gy_pos < 0 || gy_pos >= height) {
                blurred[dy + 1][dx + 1] = 0.0f;
                continue;
            }

            float gauss_acc = 0.0f;

            // Apply Gaussian at position (gx_pos, gy_pos)
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = gx_pos + kx;
                    int iy = gy_pos + ky;

                    float val = 0.0f;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        val = input[iy * width + ix];
                    }

                    gauss_acc += val * c_gaussian_3x3[(ky + 1) * 3 + (kx + 1)];
                }
            }

            blurred[dy + 1][dx + 1] = gauss_acc;
        }
    }

    // Now compute Sobel X and Y on the blurred values
    float gx = 0.0f, gy = 0.0f;

    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            float val = blurred[ky][kx];
            gx += val * c_sobel_x[ky * 3 + kx];
            gy += val * c_sobel_y[ky * 3 + kx];
        }
    }

    output[y * width + x] = fabsf(gx) + fabsf(gy);
}

void conv2d_cuda_v3_fused_gaussian_sobel(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int block_x,
    int block_y
) {
    dim3 block(block_x, block_y);
    dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);

    // Use the simple version for now (correct but not optimized)
    conv2d_kernel_v3_fused_simple<<<grid, block>>>(d_input, d_output, width, height);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Multi-GPU Support
// ============================================================================

int get_num_gpus() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

void print_device_info() {
    int num_gpus = get_num_gpus();
    printf("Number of CUDA devices: %d\n", num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max block dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth: %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    }
}

void conv2d_multi_gpu(
    const float* h_input,
    float* h_output,
    int width,
    int height,
    const float* h_kernel,
    int kernel_size,
    int num_gpus
) {
    int available_gpus = get_num_gpus();
    if (available_gpus == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        return;
    }

    if (num_gpus <= 0 || num_gpus > available_gpus) {
        num_gpus = available_gpus;
    }

    const int half_k = kernel_size / 2;

    // Calculate rows per GPU with overlap for halos
    std::vector<int> start_rows(num_gpus);
    std::vector<int> end_rows(num_gpus);
    std::vector<int> chunk_heights(num_gpus);

    const int base_rows = height / num_gpus;
    int remaining = height % num_gpus;

    int current_row = 0;
    for (int g = 0; g < num_gpus; ++g) {
        start_rows[g] = current_row;
        int rows_for_this_gpu = base_rows + (g < remaining ? 1 : 0);
        current_row += rows_for_this_gpu;
        end_rows[g] = current_row;

        // Add halo overlap
        int actual_start = std::max(0, start_rows[g] - half_k);
        int actual_end = std::min(height, end_rows[g] + half_k);
        chunk_heights[g] = actual_end - actual_start;
    }

    // Allocate device memory and process each GPU
    std::vector<float*> d_inputs(num_gpus);
    std::vector<float*> d_outputs(num_gpus);
    std::vector<float*> d_kernels(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);

    // Initialize each GPU
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));

        int actual_start = std::max(0, start_rows[g] - half_k);
        int chunk_h = chunk_heights[g];

        CUDA_CHECK(cudaMalloc(&d_inputs[g], width * chunk_h * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs[g], width * chunk_h * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernels[g], kernel_size * kernel_size * sizeof(float)));

        // Copy input chunk with halo
        CUDA_CHECK(cudaMemcpyAsync(d_inputs[g], h_input + actual_start * width,
                                    width * chunk_h * sizeof(float),
                                    cudaMemcpyHostToDevice, streams[g]));

        // Copy kernel
        CUDA_CHECK(cudaMemcpyAsync(d_kernels[g], h_kernel,
                                    kernel_size * kernel_size * sizeof(float),
                                    cudaMemcpyHostToDevice, streams[g]));
    }

    // Launch kernels on each GPU
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        int chunk_h = chunk_heights[g];

        // Copy kernel to constant memory for this device
        CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, d_kernels[g],
                                       kernel_size * kernel_size * sizeof(float)));

        conv2d_cuda_v2(d_inputs[g], d_outputs[g], width, chunk_h,
                       d_kernels[g], kernel_size, 16, 16);
    }

    // Copy results back and stitch
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        int actual_start = std::max(0, start_rows[g] - half_k);
        int chunk_h = chunk_heights[g];

        // Determine which rows from this chunk go into output
        int output_start = start_rows[g];
        int output_end = end_rows[g];
        int offset_in_chunk = start_rows[g] - actual_start;
        int rows_to_copy = output_end - output_start;

        CUDA_CHECK(cudaMemcpyAsync(h_output + output_start * width,
                                    d_outputs[g] + offset_in_chunk * width,
                                    width * rows_to_copy * sizeof(float),
                                    cudaMemcpyDeviceToHost, streams[g]));
    }

    // Synchronize and cleanup
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));

        CUDA_CHECK(cudaFree(d_inputs[g]));
        CUDA_CHECK(cudaFree(d_outputs[g]));
        CUDA_CHECK(cudaFree(d_kernels[g]));
        CUDA_CHECK(cudaStreamDestroy(streams[g]));
    }

    // Reset to device 0
    CUDA_CHECK(cudaSetDevice(0));
}

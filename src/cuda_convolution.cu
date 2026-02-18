#include "cuda_convolution.cuh"

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
// V1_const: Global Memory + Constant Memory Kernel
// ============================================================================

__global__ void conv2d_kernel_v1_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
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
            acc += input[input_idx] * c_kernel[kernel_idx];
        }
    }

    output[y * width + x] = acc;
}

void conv2d_cuda_v1_const(
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

    conv2d_kernel_v1_const<<<grid, block>>>(d_input, d_output, width, height, kernel_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// V2: Shared Memory Tiled Convolution
// ============================================================================

template <int KERNEL_SIZE>
__global__ void conv2d_kernel_v2_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int tile_w,
    int tile_h,
    int shared_w
) {
    constexpr int HALF_K = KERNEL_SIZE / 2;
    extern __shared__ float tile[];

    const int out_x = blockIdx.x * tile_w + threadIdx.x;
    const int out_y = blockIdx.y * tile_h + threadIdx.y;

    const int tile_start_x = blockIdx.x * tile_w - HALF_K;
    const int tile_start_y = blockIdx.y * tile_h - HALF_K;

    const int shared_h = tile_h + 2 * HALF_K;

    // Each thread may need to load multiple elements
    const int num_loads_x = (shared_w + tile_w - 1) / tile_w;
    const int num_loads_y = (shared_h + tile_h - 1) / tile_h;

    for (int ly = 0; ly < num_loads_y; ++ly) {
        for (int lx = 0; lx < num_loads_x; ++lx) {
            const int sx = threadIdx.x + lx * tile_w;
            const int sy = threadIdx.y + ly * tile_h;

            if (sx < shared_w && sy < shared_h) {
                const int gx = tile_start_x + sx;
                const int gy = tile_start_y + sy;

                float val = 0.0f;
                if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                    val = input[gy * width + gx];
                }
                tile[sy * shared_w + sx] = val;
            }
        }
    }

    __syncthreads();

    if (out_x < width && out_y < height) {
        float acc = 0.0f;

        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                acc += tile[(threadIdx.y + ky) * shared_w + (threadIdx.x + kx)]
                     * c_kernel[ky * KERNEL_SIZE + kx];
            }
        }

        output[out_y * width + out_x] = acc;
    }
}

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
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, d_kernel, kernel_size * kernel_size * sizeof(float)));

    const int half_k = kernel_size / 2;
    const int shared_w = block_x + 2 * half_k;
    const int shared_h = block_y + 2 * half_k;
    size_t smem_size = shared_w * shared_h * sizeof(float);

    dim3 block(block_x, block_y);
    dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);

    if (kernel_size == 3) {
        conv2d_kernel_v2_shared<3><<<grid, block, smem_size>>>(d_input, d_output, width, height, block_x, block_y, shared_w);
    } else if (kernel_size == 5) {
        conv2d_kernel_v2_shared<5><<<grid, block, smem_size>>>(d_input, d_output, width, height, block_x, block_y, shared_w);
    } else if (kernel_size == 7) {
        conv2d_kernel_v2_shared<7><<<grid, block, smem_size>>>(d_input, d_output, width, height, block_x, block_y, shared_w);
    } else {
        fprintf(stderr, "V2: kernel size %d not supported, falling back to V1\n", kernel_size);
        conv2d_cuda_v1(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
        return;
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Device Info
// ============================================================================

void print_device_info() {
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    printf("Number of CUDA devices: %d\n", num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth: %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    }
}

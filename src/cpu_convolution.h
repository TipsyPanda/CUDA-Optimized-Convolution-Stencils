#pragma once
#include <vector>

// Generic 2D convolution with zero-padding.
// input:  HxW in row-major (y*W + x)
// kernel: KHxKW in row-major, with center at (KH/2, KW/2)
// output: HxW in row-major
void conv2d_cpu(
    const float* input,
    float* output,
    int width,
    int height,
    const float* kernel,
    int kernel_width,
    int kernel_height
);

// Convenience overload using std::vector
void conv2d_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const std::vector<float>& kernel,
    int kernel_size  // assumes square kernel
);

// Legacy 3x3 function for backward compatibility
void conv2d_cpu_3x3_zeropad_f32(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const float kernel[9]
);

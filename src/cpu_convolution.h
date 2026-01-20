#pragma once
#include <vector>

// 3x3 convolution with zero-padding.
// input:  HxW in row-major (y*W + x)
// kernel: 3x3 in row-major (ky*3 + kx), with center at (1,1)
// output: HxW in row-major
void conv2d_cpu_3x3_zeropad_f32(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const float kernel[9]
);

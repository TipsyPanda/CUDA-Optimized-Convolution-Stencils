#include "cpu_convolution.h"
#include <stdexcept>
#include <cstring>

void conv2d_cpu(
    const float* input,
    float* output,
    int width,
    int height,
    const float* kernel,
    int kernel_width,
    int kernel_height
) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width/height must be positive");
    }
    if (kernel_width <= 0 || kernel_height <= 0) {
        throw std::invalid_argument("kernel dimensions must be positive");
    }
    if (kernel_width % 2 == 0 || kernel_height % 2 == 0) {
        throw std::invalid_argument("kernel dimensions must be odd");
    }

    const int half_kw = kernel_width / 2;
    const int half_kh = kernel_height / 2;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);

    std::memset(output, 0, n * sizeof(float));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;

            for (int ky = -half_kh; ky <= half_kh; ++ky) {
                const int iy = y + ky;
                if (iy < 0 || iy >= height) continue;

                for (int kx = -half_kw; kx <= half_kw; ++kx) {
                    const int ix = x + kx;
                    if (ix < 0 || ix >= width) continue;

                    const size_t input_idx = static_cast<size_t>(iy) * width + ix;
                    const size_t kernel_idx = static_cast<size_t>(ky + half_kh) * kernel_width + (kx + half_kw);

                    acc += input[input_idx] * kernel[kernel_idx];
                }
            }

            output[static_cast<size_t>(y) * width + x] = acc;
        }
    }
}

void conv2d_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const std::vector<float>& kernel,
    int kernel_size
) {
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (input.size() != n) {
        throw std::invalid_argument("input.size() != width*height");
    }
    if (kernel.size() != static_cast<size_t>(kernel_size * kernel_size)) {
        throw std::invalid_argument("kernel.size() != kernel_size^2");
    }

    output.resize(n);
    conv2d_cpu(input.data(), output.data(), width, height,
               kernel.data(), kernel_size, kernel_size);
}

void conv2d_cpu_3x3_zeropad_f32(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const float kernel[9]
) {
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (input.size() != n) {
        throw std::invalid_argument("input.size() != width*height");
    }

    output.resize(n);
    conv2d_cpu(input.data(), output.data(), width, height, kernel, 3, 3);
}

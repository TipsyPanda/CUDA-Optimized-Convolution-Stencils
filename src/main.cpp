#include <iostream>
#include <vector>
#include <cmath>
#include "cpu_convolution.h"
#include <chrono>

int main()
{
    const int W = 5, H = 4;

    // Simple test input: increasing numbers
    std::vector<float> img(W * H);
    for (int i = 0; i < (int)img.size(); ++i)
        img[i] = (float)i;

    // Example 3x3 kernel: identity (copies center pixel)
    const float K_identity[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0};

    // Box blur (average)
    const float K_box_blur[9] = {
        1.f / 9, 1.f / 9, 1.f / 9,
        1.f / 9, 1.f / 9, 1.f / 9,
        1.f / 9, 1.f / 9, 1.f / 9
    };

    // Sobel X (edge detection)
    const float K_sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    // Laplacian (edge detection)
    const float K_laplacian[9] = {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
    };

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<float> out;
    conv2d_cpu_3x3_zeropad_f32(img, out, W, H, K_laplacian);

    // Print a few values
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            std::cout << out[y * W + x] << " ";
        }
        std::cout << "\n";
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Convolution took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << " μs\n";
    return 0;
}


void max_abs_error(const std::vector<float>& cpu, const std::vector<float>& gpu)
{
    if (cpu.size() != gpu.size())
    {
        std::cerr << "Size mismatch between CPU and GPU results\n";
        return;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < cpu.size(); ++i)
    {
        float err = std::abs(cpu[i] - gpu[i]);
        if (err > max_err)
            max_err = err;
    }

    std::cout << "Max absolute error between CPU and GPU: " << max_err << "\n";
}
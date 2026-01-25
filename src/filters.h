#pragma once
#include <vector>
#include <string>

namespace filters {

// Get a filter kernel by name and size
// Supported names: "gaussian", "sobel_x", "sobel_y", "laplacian", "identity", "box_blur"
// Supported sizes: 3, 5, 7 (not all filters support all sizes)
std::vector<float> get_kernel(const std::string& name, int size = 3);

// Individual filter generators

// Gaussian blur kernels (sigma = size/6 approximately)
std::vector<float> gaussian_3x3();
std::vector<float> gaussian_5x5();
std::vector<float> gaussian_7x7();

// Sobel edge detection kernels (3x3 only)
std::vector<float> sobel_x_3x3();
std::vector<float> sobel_y_3x3();

// Laplacian edge detection (3x3 only)
std::vector<float> laplacian_3x3();

// Identity kernel (any odd size)
std::vector<float> identity(int size);

// Box blur (any odd size)
std::vector<float> box_blur(int size);

// Sharpen kernel (3x3)
std::vector<float> sharpen_3x3();

// Emboss kernel (3x3)
std::vector<float> emboss_3x3();

} // namespace filters

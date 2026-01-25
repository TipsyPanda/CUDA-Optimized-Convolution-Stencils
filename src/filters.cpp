#include "filters.h"
#include <stdexcept>
#include <cmath>

namespace filters {

std::vector<float> gaussian_3x3() {
    // Approximation of Gaussian with sigma ~0.85
    return {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
}

std::vector<float> gaussian_5x5() {
    // Approximation of Gaussian with sigma ~1.0
    return {
        1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256,
        4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
        6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
        4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
        1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256
    };
}

std::vector<float> gaussian_7x7() {
    // Approximation of Gaussian with sigma ~1.5
    std::vector<float> kernel = {
        0, 0,  1,  2,  1, 0, 0,
        0, 3, 13, 22, 13, 3, 0,
        1,13, 59, 97, 59,13, 1,
        2,22, 97,159, 97,22, 2,
        1,13, 59, 97, 59,13, 1,
        0, 3, 13, 22, 13, 3, 0,
        0, 0,  1,  2,  1, 0, 0
    };
    // Normalize
    float sum = 0;
    for (float v : kernel) sum += v;
    for (float& v : kernel) v /= sum;
    return kernel;
}

std::vector<float> sobel_x_3x3() {
    return {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
}

std::vector<float> sobel_y_3x3() {
    return {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
}

std::vector<float> laplacian_3x3() {
    return {
        0,  1, 0,
        1, -4, 1,
        0,  1, 0
    };
}

std::vector<float> identity(int size) {
    if (size <= 0 || size % 2 == 0) {
        throw std::invalid_argument("identity kernel size must be positive and odd");
    }
    std::vector<float> kernel(size * size, 0.0f);
    kernel[(size * size) / 2] = 1.0f;
    return kernel;
}

std::vector<float> box_blur(int size) {
    if (size <= 0 || size % 2 == 0) {
        throw std::invalid_argument("box blur kernel size must be positive and odd");
    }
    float val = 1.0f / (size * size);
    return std::vector<float>(size * size, val);
}

std::vector<float> sharpen_3x3() {
    return {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
}

std::vector<float> emboss_3x3() {
    return {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
}

std::vector<float> get_kernel(const std::string& name, int size) {
    if (name == "identity") {
        return identity(size);
    }
    if (name == "box_blur") {
        return box_blur(size);
    }
    if (name == "gaussian") {
        if (size == 3) return gaussian_3x3();
        if (size == 5) return gaussian_5x5();
        if (size == 7) return gaussian_7x7();
        throw std::invalid_argument("Gaussian only supports sizes 3, 5, 7");
    }
    if (name == "sobel_x") {
        if (size != 3) throw std::invalid_argument("Sobel X only supports size 3");
        return sobel_x_3x3();
    }
    if (name == "sobel_y") {
        if (size != 3) throw std::invalid_argument("Sobel Y only supports size 3");
        return sobel_y_3x3();
    }
    if (name == "laplacian") {
        if (size != 3) throw std::invalid_argument("Laplacian only supports size 3");
        return laplacian_3x3();
    }
    if (name == "sharpen") {
        if (size != 3) throw std::invalid_argument("Sharpen only supports size 3");
        return sharpen_3x3();
    }
    if (name == "emboss") {
        if (size != 3) throw std::invalid_argument("Emboss only supports size 3");
        return emboss_3x3();
    }
    throw std::invalid_argument("Unknown filter: " + name);
}

} // namespace filters

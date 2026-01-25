#pragma once
#include <vector>
#include <random>
#include <cmath>

namespace image_utils {

// Generate a random noise image
inline std::vector<float> random_noise(int width, int height, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> image(width * height);
    for (auto& pixel : image) {
        pixel = dist(rng);
    }
    return image;
}

// Generate a checkerboard pattern
inline std::vector<float> checkerboard(int width, int height, int cell_size = 32) {
    std::vector<float> image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int cx = x / cell_size;
            int cy = y / cell_size;
            image[y * width + x] = ((cx + cy) % 2 == 0) ? 1.0f : 0.0f;
        }
    }
    return image;
}

// Generate a horizontal gradient
inline std::vector<float> gradient_horizontal(int width, int height) {
    std::vector<float> image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image[y * width + x] = static_cast<float>(x) / (width - 1);
        }
    }
    return image;
}

// Generate a vertical gradient
inline std::vector<float> gradient_vertical(int width, int height) {
    std::vector<float> image(width * height);
    for (int y = 0; y < height; ++y) {
        float val = static_cast<float>(y) / (height - 1);
        for (int x = 0; x < width; ++x) {
            image[y * width + x] = val;
        }
    }
    return image;
}

// Generate a radial gradient (center bright, edges dark)
inline std::vector<float> gradient_radial(int width, int height) {
    std::vector<float> image(width * height);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float max_dist = std::sqrt(cx * cx + cy * cy);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            image[y * width + x] = 1.0f - (dist / max_dist);
        }
    }
    return image;
}

// Generate a sine wave pattern
inline std::vector<float> sine_pattern(int width, int height, float frequency = 0.05f) {
    std::vector<float> image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = 0.5f + 0.5f * std::sin(2.0f * 3.14159265f * frequency * (x + y));
            image[y * width + x] = val;
        }
    }
    return image;
}

// Compute max absolute difference between two images
inline float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return -1.0f;
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Compute mean absolute error
inline float mean_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / a.size();
}

} // namespace image_utils

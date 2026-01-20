#include "cpu_convolution.h"
#include <stdexcept>

void conv2d_cpu_3x3_zeropad_f32(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const float kernel[9]
) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width/height must be positive");
    }
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (input.size() != n) {
        throw std::invalid_argument("input.size() != width*height");
    }
    output.assign(n, 0.0f);

    auto idx = [width](int x, int y) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;

            // 3x3 kernel window centered at (x,y)
            for (int ky = -1; ky <= 1; ++ky) {
                const int iy = y + ky;
                if (iy < 0 || iy >= height) continue; // zero padding

                for (int kx = -1; kx <= 1; ++kx) {
                    const int ix = x + kx;
                    if (ix < 0 || ix >= width) continue; // zero padding

                    const float v = input[idx(ix, iy)];
                    const float w = kernel[(ky + 1) * 3 + (kx + 1)];
                    acc += v * w;
                }
            }

            output[idx(x, y)] = acc;
        }
    }
}

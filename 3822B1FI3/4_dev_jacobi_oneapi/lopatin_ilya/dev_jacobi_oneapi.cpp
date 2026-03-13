#include "dev_jacobi_oneapi.h"

#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(n * n, queue);
    float* d_b = sycl::malloc_device<float>(n, queue);
    float* d_x = sycl::malloc_device<float>(n, queue);
    float* d_x_new = sycl::malloc_device<float>(n, queue);

    queue.memcpy(d_a, a.data(), sizeof(float) * n * n).wait();
    queue.memcpy(d_b, b.data(), sizeof(float) * n).wait();
    queue.memset(d_x, 0, sizeof(float) * n).wait();
    queue.memset(d_x_new, 0, sizeof(float) * n).wait();

    std::vector<float> x_host(n);
    std::vector<float> x_new_host(n);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float sigma = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += d_a[i * n + j] * d_x[j];
                }
            }
            d_x_new[i] = (d_b[i] - sigma) / d_a[i * n + i];
            }).wait();

            queue.memcpy(x_host.data(), d_x, sizeof(float) * n).wait();
            queue.memcpy(x_new_host.data(), d_x_new, sizeof(float) * n).wait();

            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                max_diff = std::max(max_diff, std::fabs(x_new_host[i] - x_host[i]));
            }

            if (max_diff < accuracy) {
                break;
            }

            queue.memcpy(d_x, d_x_new, sizeof(float) * n).wait();
    }

    std::vector<float> result(n);
    queue.memcpy(result.data(), d_x_new, sizeof(float) * n).wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x, queue);
    sycl::free(d_x_new, queue);

    return result;
}
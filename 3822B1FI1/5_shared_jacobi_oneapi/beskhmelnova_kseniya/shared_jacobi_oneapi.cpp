#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        [[maybe_unused]] float accuracy,
        sycl::device device) {
    
    const size_t n = b.size();
    sycl::queue q(device, sycl::property::queue::in_order{});

    float* s_a = sycl::malloc_shared<float>(n * n, q);
    float* s_b = sycl::malloc_shared<float>(n, q);
    float* s_inv = sycl::malloc_shared<float>(n, q);
    float* s_x_curr = sycl::malloc_shared<float>(n, q);
    float* s_x_next = sycl::malloc_shared<float>(n, q);

    std::copy_n(a.data(), n * n, s_a);
    std::copy_n(b.data(), n, s_b);
    for (size_t i = 0; i < n; i++) {
        s_inv[i] = 1.0f / a[i * n + i];
    }
    std::fill_n(s_x_curr, n, 0.0f);

    const size_t wg_size = std::min<size_t>(
        device.is_gpu() ? 128 : 256,
        device.get_info<sycl::info::device::max_work_group_size>()
    );
    const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        q.parallel_for(sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> item) {
                size_t i = item.get_global_id(0);
                if (i >= n) return;

                float sum = 0.0f;
                size_t row = i * n;
                for (size_t j = 0; j < n; j++) {
                    if (j != i) {
                        sum += s_a[row + j] * s_x_curr[j];
                    }
                }
                s_x_next[i] = s_inv[i] * (s_b[i] - sum);
            });

        q.wait();
        std::swap(s_x_curr, s_x_next);
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), s_x_curr, n * sizeof(float)).wait();

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_inv, q);
    sycl::free(s_x_curr, q);
    sycl::free(s_x_next, q);

    return result;
}

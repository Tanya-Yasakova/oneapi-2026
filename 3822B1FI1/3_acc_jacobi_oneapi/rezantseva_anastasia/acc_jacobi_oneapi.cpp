#include "acc_jacobi_oneapi.h"

#include <algorithm> 

constexpr size_t JACOBI_LOCAL = 64;

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t n = b.size();
    if (n == 0) return {};
    if (a.size() != n * n) return {};
    if (accuracy < 0.0f) accuracy = 0.0f;

    sycl::queue q(device, sycl::property::queue::in_order{});

    std::vector<float> x_old(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> x_old_buf(x_old.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> x_new_buf(x_new.data(), sycl::range<1>(n));

    float max_diff_host = 0.0f;
    sycl::buffer<float, 1> max_diff_buf(&max_diff_host, sycl::range<1>(1));

    auto round_up = [](size_t x, size_t m) {
        return ((x + m - 1) / m) * m;
    };

    const size_t global = round_up(n, JACOBI_LOCAL);

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.submit([&](sycl::handler& h) {
            auto md = max_diff_buf.get_access<sycl::access::mode::discard_write>(h);
            h.fill(md, 0.0f);
        });

        sycl::event e = q.submit([&](sycl::handler& h) {
            auto A  = a_buf.get_access<sycl::access::mode::read>(h);
            auto B  = b_buf.get_access<sycl::access::mode::read>(h);
            auto XO = x_old_buf.get_access<sycl::access::mode::read>(h);
            auto XN = x_new_buf.get_access<sycl::access::mode::discard_write>(h);

            auto md_red = sycl::reduction(max_diff_buf, h, sycl::maximum<float>());

            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(JACOBI_LOCAL)),
                md_red,
                [=](sycl::nd_item<1> it, auto& md) {
                    const size_t i = it.get_global_id(0);
                    if (i >= n) return;

                    const size_t row = i * n;
                    const float aii = A[row + i];

                    if (aii == 0.0f) {
                        XN[i] = XO[i];
                        md.combine(0.0f);
                        return;
                    }

                    float sum = 0.0f;
                    for (size_t j = 0; j < n; ++j) {
                        if (j == i) continue;
                        sum += A[row + j] * XO[j];
                    }

                    const float xi = (B[i] - sum) / aii;
                    XN[i] = xi;

                    md.combine(sycl::fabs(xi - XO[i]));
                }
            );
        });

        e.wait_and_throw();

        float max_diff = 0.0f;
        {
            sycl::host_accessor md(max_diff_buf, sycl::read_only);
            max_diff = md[0];
        }

        std::swap(x_old, x_new);
        std::swap(x_old_buf, x_new_buf);

        if (max_diff < accuracy) break;
    }

    return x_old;
}
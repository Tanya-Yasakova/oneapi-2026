#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float> a,
        const std::vector<float> b,
        float accuracy,
        sycl::device device)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    try {
        sycl::queue q{device};

        float* A_shared = sycl::malloc_shared<float>(a.size(), q);
        float* b_shared = sycl::malloc_shared<float>(n, q);
        float* x_curr = sycl::malloc_shared<float>(n, q);
        float* x_next = sycl::malloc_shared<float>(n, q);
        float* max_diff_ptr = sycl::malloc_shared<float>(1, q);

        if (!A_shared || !b_shared || !x_curr || !x_next || !max_diff_ptr) {
            sycl::free(A_shared, q);
            sycl::free(b_shared, q);
            sycl::free(x_curr, q);
            sycl::free(x_next, q);
            sycl::free(max_diff_ptr, q);
            return {};
        }

        std::copy(a.begin(), a.end(), A_shared);
        std::copy(b.begin(), b.end(), b_shared);
        std::fill(x_curr, x_curr + n, 0.0f);
        std::copy(x_curr, x_curr + n, x_next);
        *max_diff_ptr = 0.0f;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            *max_diff_ptr = 0.0f;

            q.submit([&](sycl::handler& h)
            {
                auto red = sycl::reduction(max_diff_ptr, h, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>{n},
                    red,
                    [=](sycl::id<1> id, auto& local_max)
                    {
                        const size_t i = id[0];
                        float sigma = 0.0f;

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (j != i)
                            {
                                sigma += A_shared[i * n + j] * x_curr[j];
                            }
                        }

                        float diag = A_shared[i * n + i];
                        float new_val = (sycl::fabs(diag) < 1e-12f)
                                        ? x_curr[i]
                                        : (b_shared[i] - sigma) / diag;

                        x_next[i] = new_val;

                        float diff = sycl::fabs(new_val - x_curr[i]);
                        local_max.combine(diff);
                    });
            }).wait();

            if (*max_diff_ptr < accuracy)
            {
                break;
            }

            std::swap(x_curr, x_next);
        }

        std::vector<float> solution(n);
        std::copy(x_curr, x_curr + n, solution.begin());

        sycl::free(A_shared, q);
        sycl::free(b_shared, q);
        sycl::free(x_curr, q);
        sycl::free(x_next, q);
        sycl::free(max_diff_ptr, q);

        return solution;

    }
    catch (sycl::exception const&)
    {
        return {};
    }
}
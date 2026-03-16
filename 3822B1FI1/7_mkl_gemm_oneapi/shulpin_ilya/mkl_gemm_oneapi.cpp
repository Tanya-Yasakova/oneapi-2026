#include "mkl_gemm_oneapi.h"

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device)
{
    if (size == 0 || a.size() != size * size || b.size() != size * size) {
        return {};
    }

    std::vector<float> c(size * size, 0.0f);

    try {
        sycl::queue q{device};

        oneapi::mkl::transpose transA = oneapi::mkl::transpose::T;
        oneapi::mkl::transpose transB = oneapi::mkl::transpose::T;

        float alpha = 1.0f;
        float beta  = 0.0f;

        int64_t m = static_cast<int64_t>(size);
        int64_t n = static_cast<int64_t>(size);
        int64_t k = static_cast<int64_t>(size);

        int64_t lda = static_cast<int64_t>(size);
        int64_t ldb = static_cast<int64_t>(size);
        int64_t ldc = static_cast<int64_t>(size);

        auto event = oneapi::mkl::blas::gemm(
            q,
            transA,
            transB,
            m, n, k,
            alpha,
            a.data(), lda,
            b.data(), ldb,
            beta,
            c.data(), ldc
        );

        event.wait();

        return c;

    } catch (sycl::exception const&) {
        return {};
    }
}
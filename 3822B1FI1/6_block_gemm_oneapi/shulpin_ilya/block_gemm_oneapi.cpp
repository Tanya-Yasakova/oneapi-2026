#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device)
{
    if (size == 0 || a.size() != size * size || b.size() != size * size) {
        return {};
    }

    constexpr size_t BLOCK_SIZE = 64;

    const size_t num_blocks = size / BLOCK_SIZE;

    std::vector<float> c(size * size, 0.0f);

    try {
        sycl::queue q{device};

        sycl::buffer<float, 1> A_buf{a.data(), sycl::range<1>{a.size()}};
        sycl::buffer<float, 1> B_buf{b.data(), sycl::range<1>{b.size()}};
        sycl::buffer<float, 1> C_buf{c.data(), sycl::range<1>{c.size()}};

        q.submit([&](sycl::handler& h) {
            auto A = A_buf.get_access<sycl::access::mode::read>(h);
            auto B = B_buf.get_access<sycl::access::mode::read>(h);
            auto C = C_buf.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for(
                sycl::range<3>{num_blocks, num_blocks, BLOCK_SIZE},
                [=](sycl::id<3> idx) {
                    const size_t bi = idx[0];
                    const size_t bj = idx[1];
                    const size_t ti = idx[2];

                    float acc = 0.0f;

                    for (size_t bk = 0; bk < num_blocks; ++bk) {
                        for (size_t tj = 0; tj < BLOCK_SIZE; ++tj) {
                            size_t a_idx = (bi * BLOCK_SIZE + ti) * size + (bk * BLOCK_SIZE + tj);
                            size_t b_idx = (bk * BLOCK_SIZE + tj) * size + (bj * BLOCK_SIZE + tj);

                            acc += A[a_idx] * B[b_idx];
                        }
                    }

                    size_t c_idx = (bi * BLOCK_SIZE + ti) * size + (bj * BLOCK_SIZE + tj);
                    C[c_idx] = acc;
                });
        }).wait();

        sycl::host_accessor<float, 1, sycl::access::mode::read> result_acc(C_buf);
        std::vector<float> result(size * size);
        for (size_t i = 0; i < size * size; ++i) {
            result[i] = result_acc[i];
        }

        return result;

    } catch (sycl::exception const&) {
        return {};
    }
}
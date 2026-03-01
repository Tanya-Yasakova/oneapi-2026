#include "block_gemm_oneapi.h"
#include <algorithm>

constexpr size_t BLOCK_SIZE = 16;
constexpr size_t PADDED_BLOCK = 17;

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    
    sycl::queue q(device, sycl::property::queue::in_order{});

    float* d_a = sycl::aligned_alloc_device<float>(64, size * size, q);
    float* d_b = sycl::aligned_alloc_device<float>(64, size * size, q);
    float* d_c = sycl::aligned_alloc_device<float>(64, size * size, q);

    q.memcpy(d_a, a.data(), size * size * sizeof(float));
    q.memcpy(d_b, b.data(), size * size * sizeof(float));
    q.memset(d_c, 0, size * size * sizeof(float));
    q.wait();

    const size_t blocks = size / BLOCK_SIZE;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> a_shared(
            sycl::range<2>(BLOCK_SIZE, PADDED_BLOCK), cgh);
        sycl::local_accessor<float, 2> b_shared(
            sycl::range<2>(PADDED_BLOCK, BLOCK_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<2>(
            sycl::range<2>(blocks * BLOCK_SIZE, blocks * BLOCK_SIZE),
            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
        ), [=](sycl::nd_item<2> item) {
            const size_t block_i = item.get_group(0);
            const size_t block_j = item.get_group(1);
            const size_t local_i = item.get_local_id(0);
            const size_t local_j = item.get_local_id(1);

            float acc = 0.0f;

            for (size_t block_k = 0; block_k < blocks; block_k++) {
                a_shared[local_i][local_j] = 
                    d_a[(block_i * BLOCK_SIZE + local_i) * size + 
                        block_k * BLOCK_SIZE + local_j];
                b_shared[local_i][local_j] = 
                    d_b[(block_k * BLOCK_SIZE + local_i) * size + 
                        block_j * BLOCK_SIZE + local_j];

                item.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (size_t k = 0; k < BLOCK_SIZE; k++) {
                    acc += a_shared[local_i][k] * b_shared[k][local_j];
                }

                item.barrier(sycl::access::fence_space::local_space);
            }

            const size_t global_i = block_i * BLOCK_SIZE + local_i;
            const size_t global_j = block_j * BLOCK_SIZE + local_j;
            d_c[global_i * size + global_j] = acc;
        });
    }).wait();

    std::vector<float> result(size * size);
    q.memcpy(result.data(), d_c, size * size * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return result;
}

#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, size_t size,
                                   sycl::device device) {
  constexpr size_t TILE = 16;
  constexpr size_t PAD = TILE + 1;

  const size_t total = size * size;
  std::vector<float> result(total);

  sycl::queue q(device, sycl::property::queue::in_order{});

  float *d_a = sycl::malloc_device<float>(total, q);
  float *d_b = sycl::malloc_device<float>(total, q);
  float *d_c = sycl::malloc_device<float>(total, q);

  q.memcpy(d_a, a.data(), total * sizeof(float));
  q.memcpy(d_b, b.data(), total * sizeof(float));

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE * PAD), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE * PAD), h);

    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(size, size),
                                     sycl::range<2>(TILE, TILE)),
                   [=](sycl::nd_item<2> item) {
                     const size_t row = item.get_global_id(0);
                     const size_t col = item.get_global_id(1);
                     const size_t local_row = item.get_local_id(0);
                     const size_t local_col = item.get_local_id(1);

                     float sum = 0.0f;

                     for (size_t t = 0; t < size; t += TILE) {
                       tile_a[local_row * PAD + local_col] =
                           d_a[row * size + t + local_col];
                       tile_b[local_row * PAD + local_col] =
                           d_b[(t + local_row) * size + col];

                       item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                       for (size_t k = 0; k < TILE; ++k) {
                         sum += tile_a[local_row * PAD + k] *
                                tile_b[k * PAD + local_col];
                       }

                       item.barrier(sycl::access::fence_space::local_space);
                     }

                     d_c[row * size + col] = sum;
                   });
  });

  q.memcpy(result.data(), d_c, total * sizeof(float)).wait();

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);

  return result;
}
#include "shared_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiSharedONEAPI(
  const std::vector<float>& a,
  const std::vector<float>& b,
  float accuracy,
  sycl::device device) {

  const int n = b.size();

  sycl::queue q(device);

  float* a_shared = sycl::malloc_shared<float>(a.size(), q);
  float* b_shared = sycl::malloc_shared<float>(b.size(), q);
  float* prev_shared = sycl::malloc_shared<float>(n, q);
  float* curr_shared = sycl::malloc_shared<float>(n, q);

  q.memcpy(a_shared, a.data(), sizeof(float) * a.size()).wait();
  q.memcpy(b_shared, b.data(), sizeof(float) * b.size()).wait();
  q.memset(prev_shared, 0, sizeof(float) * n).wait();
  q.memset(curr_shared, 0, sizeof(float) * n).wait();

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
      int i = id[0];
      float value = b_shared[i];

      for (int j = 0; j < n; ++j) {
        if (i != j) {
          value -= a_shared[i * n + j] * prev_shared[j];
        }
      }

      curr_shared[i] = value / a_shared[i * n + i];
      }).wait();

    bool ok = true;
    for (int i = 0; i < n; ++i) {
      if (std::fabs(curr_shared[i] - prev_shared[i]) >= accuracy) {
        ok = false;
      }
      prev_shared[i] = curr_shared[i];
    }

    if (ok) {
      break;
    }
  }

  std::vector<float> result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = prev_shared[i];
  }

  sycl::free(a_shared, q);
  sycl::free(b_shared, q);
  sycl::free(prev_shared, q);
  sycl::free(curr_shared, q);

  return result;
}
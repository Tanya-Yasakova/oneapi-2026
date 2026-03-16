#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float res = 0.0f;
  const float step = (end - start) / count;

  sycl::queue q(device);

  {
    sycl::buffer<float> res_buf(&res, 1);

    q.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(res_buf, cgh, sycl::plus<>());

      cgh.parallel_for(
        sycl::range<2>(count, count),
        sum,
        [=](sycl::id<2> id, auto& total) {
          float x = start + step * (id[0] + 0.5f);
          float y = start + step * (id[1] + 0.5f);

          total += sycl::sin(x) * sycl::cos(y);
        }
      );
      }).wait();
  }

  return res * step * step;
}
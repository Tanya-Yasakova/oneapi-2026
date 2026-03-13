#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    const float step = (end - start) / count;
    const float area = step * step;

    float result = 0.0f;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::SYCL>(0, count * count),
        KOKKOS_LAMBDA(const int idx, float& local_sum) {
        int i = idx / count;
        int j = idx % count;

        float x = start + (i + 0.5f) * step;
        float y = start + (j + 0.5f) * step;

        local_sum += sinf(x) * cosf(y);
    },
        result
    );

    result *= area;

    return result;
}
#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;
    
    const float step = (end - start) / static_cast<float>(count);
    const float area = step * step;

    float sum_sin = 0.0f;
    Kokkos::parallel_reduce(
        "IntegralSin",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int i, float& lsum) {
            const float x = start + step * (static_cast<float>(i) + 0.5f);
            lsum += sinf(x);
        },
        sum_sin
    );

    float sum_cos = 0.0f;
    Kokkos::parallel_reduce(
        "IntegralCos",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int j, float& lsum) {
            const float y = start + step * (static_cast<float>(j) + 0.5f);
            lsum += cosf(y);
        },
        sum_cos
    );

    Kokkos::fence();

    return sum_sin * sum_cos * area;
}

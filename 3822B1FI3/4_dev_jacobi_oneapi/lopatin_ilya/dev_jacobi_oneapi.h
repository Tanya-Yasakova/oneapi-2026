#ifndef __JACOBI_DEV_ONEAPI_H
#define __JACOBI_DEV_ONEAPI_H

#include <vector>
#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device);

#endif  // __JACOBI_DEV_ONEAPI_H
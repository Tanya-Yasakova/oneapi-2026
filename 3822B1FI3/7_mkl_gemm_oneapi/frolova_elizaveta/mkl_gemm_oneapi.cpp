#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    sycl::queue queue(device);
    
    std::vector<float> c(size * size, 0.0f);
    
    {
        sycl::buffer<float> a_buf(a.data(), sycl::range<1>(size * size));
        sycl::buffer<float> b_buf(b.data(), sycl::range<1>(size * size));
        sycl::buffer<float> c_buf(c.data(), sycl::range<1>(size * size));
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        oneapi::mkl::blas::row_major::gemm(queue,oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,size,size,size,alpha,a_buf,size,
            b_buf,size,beta,c_buf,size);
    }
    
    queue.wait();
    
    return c;
}
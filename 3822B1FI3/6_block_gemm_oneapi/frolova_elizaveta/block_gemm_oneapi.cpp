#include "block_gemm_oneapi.h"
#include <cmath>
#include <vector>
#include <iostream>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    
    const size_t BLOCK_SIZE = 16;
    size_t num_blocks = size / BLOCK_SIZE;
    
    try {
        sycl::queue queue(device, sycl::property::queue::in_order{});
        
        float* a_dev = sycl::malloc_device<float>(size * size, queue);
        float* b_dev = sycl::malloc_device<float>(size * size, queue);
        float* c_dev = sycl::malloc_device<float>(size * size, queue);
        
        if (!a_dev || !b_dev || !c_dev) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        
        queue.memcpy(a_dev, a.data(), size * size * sizeof(float));
        queue.memcpy(b_dev, b.data(), size * size * sizeof(float));
        queue.memset(c_dev, 0, size * size * sizeof(float));
        queue.wait();
        
        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 2> a_block(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::local_accessor<float, 2> b_block(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            
            cgh.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(size, size),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<2> item) {
                
                size_t global_row = item.get_global_id(0);
                size_t global_col = item.get_global_id(1);
                
                if (global_row >= size || global_col >= size) return;
                
                size_t local_row = item.get_local_id(0);
                size_t local_col = item.get_local_id(1);
                
                size_t block_row = global_row / BLOCK_SIZE;
                size_t block_col = global_col / BLOCK_SIZE;
                
                float sum = 0.0f;
                
                for (size_t block_k = 0; block_k < num_blocks; ++block_k) {
                    
                    size_t a_global_row = block_row * BLOCK_SIZE + local_row;
                    size_t a_global_col = block_k * BLOCK_SIZE + local_col;
                    a_block[local_row][local_col] = a_dev[a_global_row * size + a_global_col];
                    
                    size_t b_global_row = block_k * BLOCK_SIZE + local_row;
                    size_t b_global_col = block_col * BLOCK_SIZE + local_col;
                    b_block[local_row][local_col] = b_dev[b_global_row * size + b_global_col];
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                        sum += a_block[local_row][k] * b_block[k][local_col];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                c_dev[global_row * size + global_col] = sum;
            });
        });
        
        queue.wait();
        queue.memcpy(c.data(), c_dev, size * size * sizeof(float)).wait();
        
        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        sycl::free(c_dev, queue);
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    } catch (std::exception& e) {
        return std::vector<float>();
    }
    
    return c;
}
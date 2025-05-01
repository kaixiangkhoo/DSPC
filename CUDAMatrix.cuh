#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include "Matrix.h" 

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Structure for managing matrix data on the GPU
struct CUDAMatrix {
    float* data;       // Device pointer
    size_t rows;
    size_t cols;
    size_t depth;
    size_t size;       // Total number of elements

    CUDAMatrix() : data(nullptr), rows(0), cols(0), depth(0), size(0) {}

    // Allocate memory on the GPU
    void allocate(size_t r, size_t c, size_t d) {
        rows = r;
        cols = c;
        depth = d;
        size = r * c * d;

        // Prevent zero-size allocation attempt which causes invalid argument error
        if (size == 0) {
            std::cerr << "Warning: Attempted to allocate zero-size CUDA matrix" << std::endl;
            return;
        }

        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    }

    // Copy from host Matrix to GPU
    void copyFromHost(const Matrix& hostMatrix);

    // Copy from GPU to host Matrix
    void copyToHost(Matrix& hostMatrix) const;

    // Free GPU memory
    void free() {
        if (data) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
        }
        rows = cols = depth = size = 0;
    }

    // Destructor
    ~CUDAMatrix() {
        // Use a safer approach for destruction that won't crash if CUDA contexts are invalid
        if (data) {
            cudaError_t error = cudaFree(data);
            if (error != cudaSuccess) {
                std::cerr << "Warning: Error freeing CUDA memory in destructor: "
                    << cudaGetErrorString(error) << std::endl;
            }
            data = nullptr;
        }
        rows = cols = depth = size = 0;
    }
};

#endif // CUDA_MATRIX_H
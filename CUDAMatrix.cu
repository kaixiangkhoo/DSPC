#include "CUDAMatrix.cuh"

// Implementation of CUDAMatrix methods that depend on Matrix class
void CUDAMatrix::copyFromHost(const Matrix& hostMatrix) {
    if (size != hostMatrix.data.size()) {
        if (data) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
        }
        allocate(hostMatrix.rows, hostMatrix.cols, hostMatrix.depth);
    }

    // Check if allocation succeeded before copying
    if (data) {
        CUDA_CHECK(cudaMemcpy(data, hostMatrix.data.data(),
            size * sizeof(float), cudaMemcpyHostToDevice));
    }
    else {
        std::cerr << "Warning: Cannot copy to null CUDA data pointer" << std::endl;
    }
}

void CUDAMatrix::copyToHost(Matrix& hostMatrix) const {
    hostMatrix.rows = rows;
    hostMatrix.cols = cols;
    hostMatrix.depth = depth;
    hostMatrix.data.resize(size);

    // Check if data is valid before copying
    if (data) {
        CUDA_CHECK(cudaMemcpy(hostMatrix.data.data(), data,
            size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    else {
        std::cerr << "Warning: Cannot copy from null CUDA data pointer" << std::endl;
        // Fill with zeros as a fallback
        std::fill(hostMatrix.data.begin(), hostMatrix.data.end(), 0.0f);
    }
}
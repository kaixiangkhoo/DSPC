#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#include "CUDAMatrix.cuh"
#include "Layer.h" 

// Forward declarations for kernel functions
__global__ void convolutionalForwardKernel(
    const float* input, const float* filters, const float* biases,
    float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding);

__global__ void convolutionalBackwardFilterKernel(
    const float* input, const float* gradOutput,
    float* filterGrads,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding);

__global__ void convolutionalBackwardBiasKernel(
    const float* gradOutput,
    float* biasGrads,
    size_t outputRows, size_t outputCols, size_t outputDepth);

__global__ void convolutionalBackwardInputKernel(
    const float* filters, const float* gradOutput,
    float* inputGrads,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding);

// Host-side wrapper class for CUDA Convolutional Layer
class CUDAConvolutionalLayer : public Layer {
private:
    std::vector<CUDAMatrix> filters;
    CUDAMatrix biases;
    size_t stride, padding;
    size_t inputDepth, outputDepth, filterSize;
    Matrix input; // Store for backward pass

    // CUDA memory for temporary storage
    CUDAMatrix cudaInput;
    CUDAMatrix cudaBiases;
    CUDAMatrix cudaOutput;
    CUDAMatrix cudaFilterGrads;
    CUDAMatrix cudaBiasGrads;
    CUDAMatrix cudaInputGrads;

public:
    CUDAConvolutionalLayer(size_t inputDepth, size_t outputDepth, size_t filterSize,
        size_t stride = 1, size_t padding = 0);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput, float learningRate) override;
    ~CUDAConvolutionalLayer();
};

#endif // CUDA_CONVOLUTION_H
#ifndef CUDA_FULLY_CONNECTED_H
#define CUDA_FULLY_CONNECTED_H

#include "CUDAMatrix.cuh"
#include "Layer.h" 

// Forward declarations for kernel functions
__global__ void matrixMultiplyKernel(
    const float* weights, const float* input, const float* biases,
    float* output,
    size_t outputSize, size_t inputSize);

__global__ void weightGradientKernel(
    const float* gradOutput, const float* input,
    float* weightGrads,
    size_t outputSize, size_t inputSize);

__global__ void biasGradientKernel(
    const float* gradOutput, float* biasGrads,
    size_t outputSize);

__global__ void inputGradientKernel(
    const float* weights, const float* gradOutput,
    float* inputGrads,
    size_t outputSize, size_t inputSize);

// Host-side wrapper class for CUDA Fully Connected Layer
class CUDAFullyConnectedLayer : public Layer {
private:
    CUDAMatrix weights;
    CUDAMatrix biases;
    size_t inputSize, outputSize;
    Matrix input; // Store for backward pass

    // CUDA memory for temporary storage
    CUDAMatrix cudaInput;
    CUDAMatrix cudaOutput;
    CUDAMatrix cudaWeightGrads;
    CUDAMatrix cudaBiasGrads;
    CUDAMatrix cudaInputGrads;

public:
    CUDAFullyConnectedLayer(size_t inputSize, size_t outputSize);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput, float learningRate) override;
    ~CUDAFullyConnectedLayer();
};

#endif // CUDA_FULLY_CONNECTED_H
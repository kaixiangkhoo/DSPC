#include "CUDAMatrix.cuh"
#include "CUDAFullyConnected.cuh"

// CUDA kernel for matrix multiplication (forward pass of fully connected layer)
__global__ void matrixMultiplyKernel(
    const float* weights, const float* input, const float* biases,
    float* output,
    size_t outputSize, size_t inputSize) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < outputSize) {
        float sum = biases[row];

        for (int col = 0; col < inputSize; ++col) {
            sum += weights[row * inputSize + col] * input[col];
        }

        output[row] = sum;
    }
}

// CUDA kernel for weight gradient computation
__global__ void weightGradientKernel(
    const float* gradOutput, const float* input,
    float* weightGrads,
    size_t outputSize, size_t inputSize) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize && col < inputSize) {
        weightGrads[row * inputSize + col] = gradOutput[row] * input[col];
    }
}

// CUDA kernel for bias gradient computation
__global__ void biasGradientKernel(
    const float* gradOutput, float* biasGrads,
    size_t outputSize) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize) {
        biasGrads[row] = gradOutput[row];
    }
}

// CUDA kernel for input gradient computation
__global__ void inputGradientKernel(
    const float* weights, const float* gradOutput,
    float* inputGrads,
    size_t outputSize, size_t inputSize) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < inputSize) {
        float sum = 0.0f;

        for (int row = 0; row < outputSize; ++row) {
            sum += weights[row * inputSize + col] * gradOutput[row];
        }

        inputGrads[col] = sum;
    }
}

CUDAFullyConnectedLayer::CUDAFullyConnectedLayer(size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize) {

    // Initialize weights and biases on host
    Matrix hostWeights(outputSize, inputSize, 1);
    float scale = std::sqrt(6.0f / (inputSize + outputSize));
    hostWeights.randomize(-scale, scale);

    // Copy to device
    weights.allocate(outputSize, inputSize, 1);
    weights.copyFromHost(hostWeights);

    // Initialize biases
    biases.allocate(1, outputSize, 1);
    std::vector<float> hostBiases(outputSize, 0.0f);
    CUDA_CHECK(cudaMemcpy(biases.data, hostBiases.data(),
        outputSize * sizeof(float), cudaMemcpyHostToDevice));
}

Matrix CUDAFullyConnectedLayer::forward(const Matrix& input) {
    // Store input for backward pass
    this->input = input;

    // Ensure input is flattened
    Matrix flatInput;
    if (input.rows * input.cols * input.depth != inputSize) {
        flatInput = input.flatten();
    }
    else {
        flatInput = input;
    }

    // Copy input to device
    cudaInput.copyFromHost(flatInput);

    // Prepare output on device
    cudaOutput.allocate(1, outputSize, 1);

    // Launch kernel
    dim3 blockDim(1, 256, 1);
    dim3 gridDim(1, (outputSize + blockDim.y - 1) / blockDim.y, 1);

    matrixMultiplyKernel << <gridDim, blockDim >> > (
        weights.data, cudaInput.data, biases.data,
        cudaOutput.data,
        outputSize, inputSize
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    Matrix output(1, outputSize, 1);
    cudaOutput.copyToHost(output);

    return output;
}

Matrix CUDAFullyConnectedLayer::backward(const Matrix& gradOutput, float learningRate) {
    // Copy gradient output to device
    CUDAMatrix cudaGradOutput;
    cudaGradOutput.copyFromHost(gradOutput);

    // Prepare gradients on device
    cudaWeightGrads.allocate(outputSize, inputSize, 1);
    cudaBiasGrads.allocate(1, outputSize, 1);
    cudaInputGrads.allocate(1, inputSize, 1);

    // Ensure input is flattened
    Matrix flatInput;
    if (input.rows * input.cols * input.depth != inputSize) {
        flatInput = input.flatten();
    }
    else {
        flatInput = input;
    }

    // Copy input to device if not already
    if (cudaInput.size != flatInput.data.size()) {
        cudaInput.copyFromHost(flatInput);
    }

    // Launch kernel for weight gradients
    dim3 blockDimWeight(16, 16, 1);
    dim3 gridDimWeight(
        (inputSize + blockDimWeight.x - 1) / blockDimWeight.x,
        (outputSize + blockDimWeight.y - 1) / blockDimWeight.y,
        1
    );

    weightGradientKernel << <gridDimWeight, blockDimWeight >> > (
        cudaGradOutput.data, cudaInput.data,
        cudaWeightGrads.data,
        outputSize, inputSize
        );

    // Launch kernel for bias gradients
    dim3 blockDimBias(256, 1, 1);
    dim3 gridDimBias((outputSize + blockDimBias.x - 1) / blockDimBias.x, 1, 1);

    biasGradientKernel << <gridDimBias, blockDimBias >> > (
        cudaGradOutput.data, cudaBiasGrads.data,
        outputSize
        );

    // Launch kernel for input gradients
    dim3 blockDimInput(256, 1, 1);
    dim3 gridDimInput((inputSize + blockDimInput.x - 1) / blockDimInput.x, 1, 1);

    inputGradientKernel << <gridDimInput, blockDimInput >> > (
        weights.data, cudaGradOutput.data,
        cudaInputGrads.data,
        outputSize, inputSize
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy gradients back and update weights and biases
    Matrix hostWeightGrads(outputSize, inputSize, 1);
    cudaWeightGrads.copyToHost(hostWeightGrads);

    std::vector<float> hostBiasGrads(outputSize);
    CUDA_CHECK(cudaMemcpy(hostBiasGrads.data(), cudaBiasGrads.data,
        outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Get current weights and biases
    Matrix hostWeights(outputSize, inputSize, 1);
    weights.copyToHost(hostWeights);

    std::vector<float> hostBiases(outputSize);
    CUDA_CHECK(cudaMemcpy(hostBiases.data(), biases.data,
        outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Update weights and biases
    for (size_t i = 0; i < outputSize; ++i) {
        hostBiases[i] -= learningRate * hostBiasGrads[i];

        for (size_t j = 0; j < inputSize; ++j) {
            hostWeights.at(i, j, 0) -= learningRate * hostWeightGrads.at(i, j, 0);
        }
    }

    // Copy updated weights and biases back to device
    weights.copyFromHost(hostWeights);
    CUDA_CHECK(cudaMemcpy(biases.data, hostBiases.data(),
        outputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy input gradients back
    Matrix flatGradient(1, inputSize, 1);
    cudaInputGrads.copyToHost(flatGradient);

    // Reshape input gradient if necessary
    Matrix inputGradient;
    if (input.rows == 1 && input.cols == inputSize && input.depth == 1) {
        // Input was already flat
        inputGradient = flatGradient;
    }
    else {
        // Input was 3D, reshape gradient
        inputGradient = Matrix::reshape(flatGradient, input.rows, input.cols, input.depth);
    }

    // Clean up
    cudaGradOutput.free();
    cudaWeightGrads.free();
    cudaBiasGrads.free();

    return inputGradient;
}

CUDAFullyConnectedLayer::~CUDAFullyConnectedLayer() {
    try {
        if (weights.data) {
            weights.free();
        }

        if (biases.data) {
            biases.free();
        }

        if (cudaInput.data) {
            cudaInput.free();
        }

        if (cudaOutput.data) {
            cudaOutput.free();
        }

        if (cudaInputGrads.data) {
            cudaInputGrads.free();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Warning: Exception in CUDA Fully Connected Layer destructor: "
            << e.what() << std::endl;
    }
}
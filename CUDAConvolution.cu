#include "CUDAConvolution.cuh"
#include <cuda_runtime.h>

// CUDA kernel for the forward pass of the convolutional layer
__global__ void convolutionalForwardKernel(
    const float* input, const float* filters, const float* biases,
    float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding) {

    // Calculate the output position this thread is responsible for
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int filterIdx = blockIdx.z;

    // Check if the thread is within bounds
    if (outputRow < outputRows && outputCol < outputCols && filterIdx < outputDepth) {
        float sum = biases[filterIdx];

        // Convolve filter with input region
        for (int fh = 0; fh < filterSize; ++fh) {
            for (int fw = 0; fw < filterSize; ++fw) {
                for (int fd = 0; fd < inputDepth; ++fd) {
                    int ih = outputRow * stride + fh - padding;
                    int iw = outputCol * stride + fw - padding;

                    if (ih >= 0 && ih < inputRows && iw >= 0 && iw < inputCols) {
                        int inputIdx = fd * (inputRows * inputCols) + ih * inputCols + iw;
                        int filterIdx3D = filterIdx * (filterSize * filterSize * inputDepth) +
                            fd * (filterSize * filterSize) +
                            fh * filterSize + fw;

                        sum += input[inputIdx] * filters[filterIdx3D];
                    }
                }
            }
        }

        // Store the result
        int outputIdx = filterIdx * (outputRows * outputCols) +
            outputRow * outputCols + outputCol;
        output[outputIdx] = sum;
    }
}

// CUDA kernel for the backward pass (computing filter gradients)
__global__ void convolutionalBackwardFilterKernel(
    const float* input, const float* gradOutput,
    float* filterGrads,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding) {

    // Calculate which filter/element this thread is responsible for
    int fw = blockIdx.x * blockDim.x + threadIdx.x;
    int fh = blockIdx.y * blockDim.y + threadIdx.y;
    int fd = blockIdx.z / outputDepth;
    int filterIdx = blockIdx.z % outputDepth;

    if (fw < filterSize && fh < filterSize && fd < inputDepth && filterIdx < outputDepth) {
        float sum = 0.0f;

        // Loop through all positions in the output gradient
        for (int oh = 0; oh < outputRows; ++oh) {
            for (int ow = 0; ow < outputCols; ++ow) {
                int ih = oh * stride + fh - padding;
                int iw = ow * stride + fw - padding;

                if (ih >= 0 && ih < inputRows && iw >= 0 && iw < inputCols) {
                    int inputIdx = fd * (inputRows * inputCols) + ih * inputCols + iw;
                    int gradOutputIdx = filterIdx * (outputRows * outputCols) +
                        oh * outputCols + ow;

                    sum += input[inputIdx] * gradOutput[gradOutputIdx];
                }
            }
        }

        // Store the result
        int filterGradIdx = filterIdx * (filterSize * filterSize * inputDepth) +
            fd * (filterSize * filterSize) +
            fh * filterSize + fw;
        filterGrads[filterGradIdx] = sum;
    }
}

// CUDA kernel for computing bias gradients
__global__ void convolutionalBackwardBiasKernel(
    const float* gradOutput,
    float* biasGrads,
    size_t outputRows, size_t outputCols, size_t outputDepth) {

    int filterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (filterIdx < outputDepth) {
        float sum = 0.0f;

        // Sum over all positions in the output gradient for this filter
        for (int oh = 0; oh < outputRows; ++oh) {
            for (int ow = 0; ow < outputCols; ++ow) {
                int gradOutputIdx = filterIdx * (outputRows * outputCols) +
                    oh * outputCols + ow;
                sum += gradOutput[gradOutputIdx];
            }
        }

        biasGrads[filterIdx] = sum;
    }
}

// CUDA kernel for the backward pass (computing input gradients)
__global__ void convolutionalBackwardInputKernel(
    const float* filters, const float* gradOutput,
    float* inputGrads,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t filterSize, size_t outputDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding) {

    // Calculate which input position this thread is responsible for
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z;

    if (iw < inputCols && ih < inputRows && id < inputDepth) {
        float sum = 0.0f;

        // Loop through all filters and all applicable output gradient positions
        for (int filterIdx = 0; filterIdx < outputDepth; ++filterIdx) {
            for (int fh = 0; fh < filterSize; ++fh) {
                for (int fw = 0; fw < filterSize; ++fw) {
                    // Calculate corresponding output position
                    int oh = (ih + padding - fh) / stride;
                    int ow = (iw + padding - fw) / stride;

                    // Check if output position is valid and aligned with stride
                    if (oh >= 0 && oh < outputRows && ow >= 0 && ow < outputCols &&
                        (ih + padding - fh) % stride == 0 && (iw + padding - fw) % stride == 0) {

                        int gradOutputIdx = filterIdx * (outputRows * outputCols) +
                            oh * outputCols + ow;
                        int filterIdx3D = filterIdx * (filterSize * filterSize * inputDepth) +
                            id * (filterSize * filterSize) +
                            fh * filterSize + fw;

                        sum += filters[filterIdx3D] * gradOutput[gradOutputIdx];
                    }
                }
            }
        }

        // Store the result
        int inputGradIdx = id * (inputRows * inputCols) + ih * inputCols + iw;
        inputGrads[inputGradIdx] = sum;
    }
}

CUDAConvolutionalLayer::CUDAConvolutionalLayer(size_t inputDepth, size_t outputDepth, size_t filterSize,
    size_t stride, size_t padding)
    : inputDepth(inputDepth), outputDepth(outputDepth), filterSize(filterSize),
    stride(stride), padding(padding) {

    // Initialize filters on host and copy to device
    filters.resize(outputDepth);

    // Allocate memory for biases on device
    biases.allocate(1, outputDepth, 1);

    // Initialize filters and biases
    std::vector<float> hostBiases(outputDepth, 0.0f);
    CUDA_CHECK(cudaMemcpy(biases.data, hostBiases.data(),
        outputDepth * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate proper filter scale using Xavier initialization, debug
    size_t fan_in = inputDepth * filterSize * filterSize;
    size_t fan_out = outputDepth * filterSize * filterSize;
    float scale = std::sqrt(6.0f / (fan_in + fan_out));

    for (size_t i = 0; i < outputDepth; ++i) {
        Matrix hostFilter(filterSize, filterSize, inputDepth);
        hostFilter.randomize(-scale, scale);

        filters[i].allocate(filterSize, filterSize, inputDepth);
        filters[i].copyFromHost(hostFilter);
    }
}

Matrix CUDAConvolutionalLayer::forward(const Matrix& input) {
    this->input = input;

    size_t outputHeight = (input.rows - filterSize + 2 * padding) / stride + 1;
    size_t outputWidth = (input.cols - filterSize + 2 * padding) / stride + 1;
    Matrix output(outputHeight, outputWidth, outputDepth);

    // Copy input to device
    cudaInput.copyFromHost(input);

    // Prepare output on device
    cudaOutput.allocate(outputHeight, outputWidth, outputDepth);

    // Prepare filter data as a single array (for easier kernel access)
    size_t filterElements = filterSize * filterSize * inputDepth;
    std::vector<float> allFiltersHost(outputDepth * filterElements);

    for (size_t i = 0; i < outputDepth; ++i) {
        Matrix hostFilter(filterSize, filterSize, inputDepth);
        filters[i].copyToHost(hostFilter);

        for (size_t j = 0; j < filterElements; ++j) {
            allFiltersHost[i * filterElements + j] = hostFilter.data[j];
        }
    }

    CUDAMatrix allFilters;
    allFilters.allocate(outputDepth, filterSize * filterSize, inputDepth);
    CUDA_CHECK(cudaMemcpy(allFilters.data, allFiltersHost.data(),
        outputDepth * filterElements * sizeof(float),
        cudaMemcpyHostToDevice));

    // Get biases ready
    std::vector<float> hostBiases(outputDepth);
    CUDA_CHECK(cudaMemcpy(hostBiases.data(), biases.data,
        outputDepth * sizeof(float), cudaMemcpyDeviceToHost));

    // Launch kernel
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y,
        outputDepth
    );

    convolutionalForwardKernel << <gridDim, blockDim >> > (
        cudaInput.data, allFilters.data, biases.data,
        cudaOutput.data,
        input.rows, input.cols, input.depth,
        filterSize, outputDepth,
        outputHeight, outputWidth,
        stride, padding
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    cudaOutput.copyToHost(output);

    // Clean up
    allFilters.free();

    return output;
}

Matrix CUDAConvolutionalLayer::backward(const Matrix& gradOutput, float learningRate) {
    size_t outputHeight = gradOutput.rows;
    size_t outputWidth = gradOutput.cols;

    // Copy gradient output to device
    CUDAMatrix cudaGradOutput;
    cudaGradOutput.copyFromHost(gradOutput);

    // Prepare gradients on device
    cudaFilterGrads.allocate(outputDepth, filterSize * filterSize, inputDepth);
    cudaBiasGrads.allocate(1, outputDepth, 1);
    cudaInputGrads.allocate(input.rows, input.cols, input.depth);

    // Prepare filter data as a single array
    size_t filterElements = filterSize * filterSize * inputDepth;
    std::vector<float> allFiltersHost(outputDepth * filterElements);

    for (size_t i = 0; i < outputDepth; ++i) {
        Matrix hostFilter(filterSize, filterSize, inputDepth);
        filters[i].copyToHost(hostFilter);

        for (size_t j = 0; j < filterElements; ++j) {
            allFiltersHost[i * filterElements + j] = hostFilter.data[j];
        }
    }

    CUDAMatrix allFilters;
    allFilters.allocate(outputDepth, filterSize * filterSize, inputDepth);
    CUDA_CHECK(cudaMemcpy(allFilters.data, allFiltersHost.data(),
        outputDepth * filterElements * sizeof(float),
        cudaMemcpyHostToDevice));

    // Launch kernel for filter gradients
    dim3 blockDimFilter(8, 8, 1);
    dim3 gridDimFilter(
        (filterSize + blockDimFilter.x - 1) / blockDimFilter.x,
        (filterSize + blockDimFilter.y - 1) / blockDimFilter.y,
        outputDepth * inputDepth
    );

    convolutionalBackwardFilterKernel << <gridDimFilter, blockDimFilter >> > (
        cudaInput.data, cudaGradOutput.data,
        cudaFilterGrads.data,
        input.rows, input.cols, input.depth,
        filterSize, outputDepth,
        outputHeight, outputWidth,
        stride, padding
        );

    // Launch kernel for bias gradients
    dim3 blockDimBias(256, 1, 1);
    dim3 gridDimBias(
        (outputDepth + blockDimBias.x - 1) / blockDimBias.x,
        1, 1
    );

    convolutionalBackwardBiasKernel << <gridDimBias, blockDimBias >> > (
        cudaGradOutput.data, cudaBiasGrads.data,
        outputHeight, outputWidth, outputDepth
        );

    // Launch kernel for input gradients
    dim3 blockDimInput(16, 16, 1);
    dim3 gridDimInput(
        (input.cols + blockDimInput.x - 1) / blockDimInput.x,
        (input.rows + blockDimInput.y - 1) / blockDimInput.y,
        input.depth
    );

    convolutionalBackwardInputKernel << <gridDimInput, blockDimInput >> > (
        allFilters.data, cudaGradOutput.data,
        cudaInputGrads.data,
        input.rows, input.cols, input.depth,
        filterSize, outputDepth,
        outputHeight, outputWidth,
        stride, padding
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy filter and bias gradients back and update
    Matrix hostFilterGrads(outputDepth, filterSize * filterSize, inputDepth);
    cudaFilterGrads.copyToHost(hostFilterGrads);

    std::vector<float> hostBiasGrads(outputDepth);
    CUDA_CHECK(cudaMemcpy(hostBiasGrads.data(), cudaBiasGrads.data,
        outputDepth * sizeof(float), cudaMemcpyDeviceToHost));

    // Update filters and biases
    for (size_t i = 0; i < outputDepth; ++i) {
        Matrix hostFilter(filterSize, filterSize, inputDepth);
        filters[i].copyToHost(hostFilter);

        for (size_t j = 0; j < filterElements; ++j) {
            hostFilter.data[j] -= learningRate * hostFilterGrads.data[i * filterElements + j];
        }

        filters[i].copyFromHost(hostFilter);

        // Update bias
        std::vector<float> hostBiases(outputDepth);
        CUDA_CHECK(cudaMemcpy(hostBiases.data(), biases.data,
            outputDepth * sizeof(float), cudaMemcpyDeviceToHost));
        hostBiases[i] -= learningRate * hostBiasGrads[i];
        CUDA_CHECK(cudaMemcpy(biases.data, hostBiases.data(),
            outputDepth * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copy input gradients back
    Matrix inputGradient(input.rows, input.cols, input.depth);
    cudaInputGrads.copyToHost(inputGradient);

    // Clean up
    allFilters.free();
    cudaGradOutput.free();
    cudaFilterGrads.free();
    cudaBiasGrads.free();

    return inputGradient;
}

CUDAConvolutionalLayer::~CUDAConvolutionalLayer() {
    try {
        for (auto& filter : filters) {
            filter.free();
        }
        filters.clear();

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
        std::cerr << "Warning: Exception in CUDA Convolutional Layer destructor: "
            << e.what() << std::endl;
    }
}
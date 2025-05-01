#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <memory>
#include <limits>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>  // OpenCV main header
#include <opencv2/core.hpp>    // Core functionality
#include <opencv2/imgproc.hpp> // Image processing
#include <opencv2/highgui.hpp>
#include <omp.h>
#include <atomic>
#include <mutex>

void writePredictionResultsToCSV(const std::string& outputFile,
    const std::vector<std::string>& imageFiles,
    const std::vector<float>& benignProbs,
    const std::vector<float>& malignantProbs,
    const std::vector<bool>& predictions);

// Utility functions for OMPMatrix operations
class OMPMatrix
{
public:
    std::vector<float> data;
    size_t rows, cols, depth;

    OMPMatrix() : rows(0), cols(0), depth(0) {}

    OMPMatrix(size_t rows, size_t cols, size_t depth)
        : rows(rows), cols(cols), depth(depth)
    {
        size_t size = rows * cols * depth;
        if (size == 0)
        {
            throw std::invalid_argument("Cannot create OMPMatrix with zero elements");
        }
        data.resize(size, 0.0f);
    }

    float& at(size_t row, size_t col, size_t d)
    {
        size_t index = d * (rows * cols) + row * cols + col;
        if (index >= data.size())
        {
            throw std::out_of_range("OMPMatrix subscript out of range");
        }
        return data[index];
    }

    float at(size_t row, size_t col, size_t d) const
    {
        size_t index = d * (rows * cols) + row * cols + col;
        if (index >= data.size())
        {
            std::cout << "Out of bounds access: [" << row << ", " << col << ", " << d
                << "] in OMPMatrix of size [" << rows << ", " << cols << ", " << depth << "]" << std::endl;
            throw std::out_of_range("OMPMatrix subscript out of range");
        }
        return data[index];
    }

    // Flatten a 3D OMPMatrix to a 1D vector (for fully connected layers)
    OMPMatrix flatten() const
    {
        size_t totalElements = rows * cols * depth;
        OMPMatrix result(1, totalElements, 1);

#pragma omp parallel for
        for (int i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i];
        }
        return result;
    }

    // Create a OMPMatrix from a flattened vector with specified dimensions
    static OMPMatrix reshape(const OMPMatrix& flat, size_t rows, size_t cols, size_t depth)
    {
        OMPMatrix result(rows, cols, depth);
        size_t totalElements = rows * cols * depth;

        if (flat.data.size() != totalElements)
        {
            std::cout << "ERROR in reshape: Source has " << flat.data.size()
                << " elements but destination requires " << totalElements << std::endl;
            throw std::runtime_error("Reshape dimensions don't match data size");
        }

        // Copy the data directly
        result.data = flat.data;

        return result;
    }

    // Fill OMPMatrix with random values
    void randomize(float min = -0.1f, float max = 0.1f)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        // Using sequential randomization to avoid race conditions
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = dist(gen);
        }
    }

    // Element-wise operations
    OMPMatrix operator+(const OMPMatrix& other) const
    {
        if (rows != other.rows || cols != other.cols || depth != other.depth)
        {
            throw std::runtime_error("Matrix dimensions don't match for addition");
        }

        OMPMatrix result(rows, cols, depth);

#pragma omp parallel for if (data.size() > 1000)
        for (int i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    OMPMatrix operator-(const OMPMatrix& other) const
    {
        if (rows != other.rows || cols != other.cols || depth != other.depth)
        {
            throw std::runtime_error("Matrix dimensions don't match for substraction");
        }

        OMPMatrix result(rows, cols, depth);

#pragma omp parallel for if (data.size() > 1000)
        for (int i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    OMPMatrix hadamard(const OMPMatrix& other) const
    {
        if (rows != other.rows || cols != other.cols || depth != other.depth)
        {
            throw std::runtime_error("Matrix dimensions don't match for Handamard product");
        }

        OMPMatrix result(rows, cols, depth);

#pragma omp parallel for if (data.size() > 1000)
        for (int i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    OMPMatrix operator*(float scalar) const
    {
        OMPMatrix result(rows, cols, depth);

#pragma omp parallel for
        for (int i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // OMPMatrix multiplication (for fully connected layers)
    OMPMatrix matmul(const OMPMatrix& other) const
    {
        if (cols != other.rows)
        {
            throw std::runtime_error("OMPMatrix dimensions don't match for multiplication");
        }

        OMPMatrix result(rows, other.cols, 1);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k)
                {
                    sum += at(i, k, 0) * other.at(k, j, 0);
                }
                result.at(i, j, 0) = sum;
            }
        }
        return result;
    }

    // Transpose operation (for fully connected layer backpropagation)
    OMPMatrix transpose() const
    {
        OMPMatrix result(cols, rows, depth);

#pragma omp parallel for collapse(3)
        for (int d = 0; d < depth; ++d)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    result.at(j, i, d) = at(i, j, d);
                }
            }
        }
        return result;
    }

    // Sum all elements
    float sum() const
    {
        float total = 0.0f;

#pragma omp parallel for reduction(+ : total)
        for (int i = 0; i < data.size(); ++i)
        {
            total += data[i];
        }
        return total;
    }
};

// Dataset class for cancer detection using OpenCV

void printLayerInfo(const std::string& layerName, const OMPMatrix& input)
{
    std::cout << "--- " << layerName << " Layer ---" << std::endl;
    std::cout << "Input dimensions: [" << input.rows << ", " << input.cols << ", " << input.depth << "]" << std::endl;
    std::cout << "Total elements: " << input.data.size() << std::endl;
}

// Abstract base class for layers
class Layer
{
public:
    virtual OMPMatrix forward(const OMPMatrix& input) = 0;
    virtual OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) = 0;
    virtual ~Layer() {}
};

// Convolutional Layer (Parallel implementation)
class ConvolutionalLayer : public Layer
{
private:
    mutable std::mutex mtx;
    std::vector<OMPMatrix> filters;
    std::vector<float> biases;
    size_t stride, padding;
    OMPMatrix input;
    size_t inputDepth, outputDepth, filterSize;

public:
    ConvolutionalLayer(size_t inputDepth, size_t outputDepth, size_t filterSize,
        size_t stride = 1, size_t padding = 0)
        : inputDepth(inputDepth), outputDepth(outputDepth), filterSize(filterSize),
        stride(stride), padding(padding)
    {

        // Initialize filters and biases
        for (size_t i = 0; i < outputDepth; ++i)
        {
            OMPMatrix filter(filterSize, filterSize, inputDepth);
            filter.randomize(-0.1f, 0.1f);
            filters.push_back(filter);
            biases.push_back(0.0f);
        }
    }

    OMPMatrix forward(const OMPMatrix& input) override
    {
        this->input = input;

        size_t outputHeight = (input.rows - filterSize + 2 * padding) / stride + 1;
        size_t outputWidth = (input.cols - filterSize + 2 * padding) / stride + 1;
        OMPMatrix output(outputHeight, outputWidth, outputDepth);

        // Parallelize over filters
#pragma omp parallel for if (outputDepth > 4)
        for (int filterIdx = 0; filterIdx < outputDepth; ++filterIdx)
        {
            // Apply filter to input volume
            for (size_t h = 0; h < outputHeight; ++h)
            {
                for (size_t w = 0; w < outputWidth; ++w)
                {
                    float sum = biases[filterIdx];

                    // Convolve filter with input region
                    for (size_t fh = 0; fh < filterSize; ++fh)
                    {
                        for (size_t fw = 0; fw < filterSize; ++fw)
                        {
                            for (size_t fd = 0; fd < inputDepth; ++fd)
                            {
                                int ih = static_cast<int>(h * stride + fh) - static_cast<int>(padding);
                                int iw = static_cast<int>(w * stride + fw) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(input.rows) &&
                                    iw >= 0 && iw < static_cast<int>(input.cols))
                                {
                                    sum += input.at(ih, iw, fd) *
                                        filters[filterIdx].at(fh, fw, fd);
                                }
                            }
                        }
                    }

                    output.at(h, w, filterIdx) = sum;
                }
            }
        }
        return output;
    }

    OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) override {
        size_t outputHeight = gradOutput.rows;
        size_t outputWidth = gradOutput.cols;

        int numThreads = omp_get_max_threads();
        std::vector<std::vector<OMPMatrix>> threadFilterGrads(numThreads);
        std::vector<std::vector<float>> threadBiasGrads(numThreads);

        for (int t = 0; t < numThreads; ++t) {
            threadFilterGrads[t].resize(outputDepth, OMPMatrix(filterSize, filterSize, inputDepth));
            threadBiasGrads[t].resize(outputDepth, 0.0f);
        }

        OMPMatrix inputGradient(input.rows, input.cols, input.depth);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& localFilterGrad = threadFilterGrads[tid];
            auto& localBiasGrad = threadBiasGrads[tid];

#pragma omp for schedule(dynamic)
            for (int filterIdx = 0; filterIdx < outputDepth; ++filterIdx) {
                for (size_t h = 0; h < outputHeight; ++h) {
                    for (size_t w = 0; w < outputWidth; ++w) {
                        float gradValue = gradOutput.at(h, w, filterIdx);
                        localBiasGrad[filterIdx] += gradValue;

                        for (size_t fh = 0; fh < filterSize; ++fh) {
                            for (size_t fw = 0; fw < filterSize; ++fw) {
                                for (size_t fd = 0; fd < inputDepth; ++fd) {
                                    int ih = static_cast<int>(h * stride + fh) - static_cast<int>(padding);
                                    int iw = static_cast<int>(w * stride + fw) - static_cast<int>(padding);

                                    if (ih >= 0 && ih < static_cast<int>(input.rows) &&
                                        iw >= 0 && iw < static_cast<int>(input.cols)) {
                                        localFilterGrad[filterIdx].at(fh, fw, fd) +=
                                            input.at(ih, iw, fd) * gradValue;
#pragma omp atomic
                                        inputGradient.at(ih, iw, fd) += filters[filterIdx].at(fh, fw, fd) * gradValue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reduce gradients across threads
        for (int i = 0; i < outputDepth; ++i) {
            for (int t = 0; t < numThreads; ++t) {
                for (size_t fh = 0; fh < filterSize; ++fh) {
                    for (size_t fw = 0; fw < filterSize; ++fw) {
                        for (size_t fd = 0; fd < inputDepth; ++fd) {
                            filters[i].at(fh, fw, fd) -= learningRate * threadFilterGrads[t][i].at(fh, fw, fd);
                        }
                    }
                }
                biases[i] -= learningRate * threadBiasGrads[t][i];
            }
        }

        return inputGradient;
    }

};

// Max Pooling Layer (Parallel implementation)
class MaxPoolingLayer : public Layer
{
private:
    size_t poolSize;
    OMPMatrix input;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> maxIndices;

public:
    MaxPoolingLayer(size_t poolSize) : poolSize(poolSize) {}

    OMPMatrix forward(const OMPMatrix& input) override
    {
        this->input = input;

        size_t outputHeight = input.rows / poolSize;
        size_t outputWidth = input.cols / poolSize;
        OMPMatrix output(outputHeight, outputWidth, input.depth);

        // Reset max indices tracking
        maxIndices.resize(input.depth);
        for (size_t d = 0; d < input.depth; ++d)
        {
            maxIndices[d].resize(outputHeight);
            for (size_t h = 0; h < outputHeight; ++h)
            {
                maxIndices[d][h].resize(outputWidth, { 0, 0 });
            }
        }

        // Parallelize over depth
#pragma omp parallel for collapse(3)
        for (int d = 0; d < input.depth; ++d)
        {
            for (size_t h = 0; h < outputHeight; ++h)
            {
                for (size_t w = 0; w < outputWidth; ++w)
                {
                    float maxVal = std::numeric_limits<float>::lowest();
                    size_t maxI = 0, maxJ = 0;

                    // Find maximum in pooling region
                    for (size_t i = 0; i < poolSize; ++i)
                    {
                        for (size_t j = 0; j < poolSize; ++j)
                        {
                            size_t inputI = h * poolSize + i;
                            size_t inputJ = w * poolSize + j;

                            // Make sure inputI and inputJ are within bounds
                            if (inputI < input.rows && inputJ < input.cols)
                            {
                                if (input.at(inputI, inputJ, d) > maxVal)
                                {
                                    maxVal = input.at(inputI, inputJ, d);
                                    maxI = i;
                                    maxJ = j;
                                }
                            }
                        }
                    }

                    output.at(h, w, d) = maxVal;
                    maxIndices[d][h][w] = { maxI, maxJ };
                }
            }
        }
        return output;
    }

    OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) override
    {
        size_t outputHeight = gradOutput.rows;
        size_t outputWidth = gradOutput.cols;
        OMPMatrix inputGradient(input.rows, input.cols, input.depth);

        // Parallelize over depth
#pragma omp parallel for collapse(3)
        for (int d = 0; d < input.depth; ++d)
        {
            for (size_t h = 0; h < outputHeight; ++h)
            {
                for (size_t w = 0; w < outputWidth; ++w)
                {
                    // Ensure we're not accessing out of bounds in maxIndices
                    if (d < maxIndices.size() && h < maxIndices[d].size() && w < maxIndices[d][h].size())
                    {
                        // Get stored indices of max value
                        auto [maxI, maxJ] = maxIndices[d][h][w];

                        // Pass gradient to the max element's position
                        size_t inputI = h * poolSize + maxI;
                        size_t inputJ = w * poolSize + maxJ;

                        // Add bounds checking
                        if (inputI < inputGradient.rows && inputJ < inputGradient.cols)
                        {
                            inputGradient.at(inputI, inputJ, d) = gradOutput.at(h, w, d);
                        }
                    }
                }
            }
        }

        return inputGradient;
    }
};

// ReLU Activation Layer (Parallel implementation)
class ReLULayer : public Layer
{
private:
    OMPMatrix input;

public:
    OMPMatrix forward(const OMPMatrix& input) override
    {
        this->input = input;
        OMPMatrix output(input.rows, input.cols, input.depth);

#pragma omp parallel for
        for (int i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = std::max(0.0f, input.data[i]);
        }

        return output;
    }

    OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) override
    {
        // Ensure output has same dimensions as input
        OMPMatrix inputGradient(input.rows, input.cols, input.depth);

        // Make sure we're computing with the right dimensions
        if (gradOutput.data.size() != input.data.size())
        {
            std::cout << "WARNING: Gradient size mismatch in ReLU backward!" << std::endl;
            return inputGradient; // Return zero gradient to prevent crash
        }

#pragma omp parallel for
        for (int i = 0; i < input.data.size(); ++i)
        {
            // ReLU derivative: 1 if input > 0, 0 otherwise
            inputGradient.data[i] = (input.data[i] > 0) ? gradOutput.data[i] : 0;
        }

        return inputGradient;
    }
};

// Fully Connected Layer (Parallel implementation)
class FullyConnectedLayer : public Layer
{
private:
    OMPMatrix weights;
    std::vector<float> biases;
    OMPMatrix input;
    size_t inputSize, outputSize;

public:
    FullyConnectedLayer(size_t inputSize, size_t outputSize)
        : inputSize(inputSize), outputSize(outputSize)
    {

        // Initialize weights and biases
        weights = OMPMatrix(outputSize, inputSize, 1);
        weights.randomize(-0.1f, 0.1f);
        biases.resize(outputSize, 0.0f);
    }

    OMPMatrix forward(const OMPMatrix& input) override
    {
        // Ensure input is flattened
        OMPMatrix flatInput;
        if (input.rows * input.cols * input.depth != inputSize)
        {
            flatInput = input.flatten();
        }
        else
        {
            flatInput = input;
        }

        this->input = flatInput;
        OMPMatrix output(1, outputSize, 1);

        // Check if our flattened input matches expected size
        if (flatInput.rows * flatInput.cols * flatInput.depth != inputSize)
        {
            std::cout << "ERROR: After flattening, size mismatch: " << std::endl;
            std::cout << "  Expected: " << inputSize << std::endl;
            std::cout << "  Actual: " << flatInput.rows * flatInput.cols * flatInput.depth << std::endl;
        }

        // Compute output = weights * input + biases
        try
        {
#pragma omp parallel for
            for (int i = 0; i < outputSize; ++i)
            {
                float sum = biases[i];
                for (size_t j = 0; j < inputSize && j < flatInput.cols; ++j)
                {
                    // Add bounds checking
                    if (j < flatInput.cols)
                    {
                        sum += weights.at(i, j, 0) * flatInput.at(0, j, 0);
                    }
                    else
                    {
                        std::cout << "WARN: Input index " << j << " out of bounds!" << std::endl;
                        break;
                    }
                }
                output.at(0, i, 0) = sum;
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "Exception in FC forward: " << e.what() << std::endl;
            // Return empty OMPMatrix as fallback
            return OMPMatrix(1, outputSize, 1);
        }

        return output;
    }

    OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) override
    {
        OMPMatrix weightGradients(outputSize, inputSize, 1);
        std::vector<float> biasGradients(outputSize, 0.0f);

        // Compute gradients in parallel
#pragma omp parallel
        {
            // Use thread-local gradients to avoid race conditions
            OMPMatrix local_weightGradients(outputSize, inputSize, 1);
            std::vector<float> local_biasGradients(outputSize, 0.0f);

#pragma omp for
            for (int i = 0; i < outputSize; ++i)
            {
                float gradValue = gradOutput.at(0, i, 0);
                local_biasGradients[i] = gradValue;

                for (size_t j = 0; j < inputSize; ++j)
                {
                    local_weightGradients.at(i, j, 0) = gradValue * input.at(0, j, 0);
                }
            }

            // Merge thread-local gradients
#pragma omp critical
            {
                for (int i = 0; i < outputSize; ++i)
                {
                    biasGradients[i] += local_biasGradients[i];
                    for (size_t j = 0; j < inputSize; ++j)
                    {
                        weightGradients.at(i, j, 0) += local_weightGradients.at(i, j, 0);
                    }
                }
            }
        }

        // Compute input gradients in parallel
        OMPMatrix flatGradient(1, inputSize, 1);
#pragma omp parallel for
        for (int j = 0; j < inputSize; ++j)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < outputSize; ++i)
            {
                sum += weights.at(i, j, 0) * gradOutput.at(0, i, 0);
            }
            flatGradient.at(0, j, 0) = sum;
        }

        // Update weights and biases in parallel
#pragma omp parallel for collapse(2)
        for (int i = 0; i < outputSize; ++i)
        {
            for (size_t j = 0; j < inputSize; ++j)
            {
                weights.at(i, j, 0) -= learningRate * weightGradients.at(i, j, 0);
            }
        }

#pragma omp parallel for
        for (int i = 0; i < outputSize; ++i)
        {
            biases[i] -= learningRate * biasGradients[i];
        }

        // If input was originally 3D (not flat), reshape the gradient back to 3D
        OMPMatrix inputGradient;
        if (input.rows == 1 && input.cols == inputSize && input.depth == 1)
        {
            // Input was already flat, return flat gradient
            inputGradient = flatGradient;
        }
        else
        {
            // Input was 3D, need to reshape gradient to match original dimensions
            inputGradient = OMPMatrix::reshape(flatGradient, input.rows, input.cols, input.depth);
        }

        return inputGradient;
    }
};

// Softmax Layer (Parallel implementation)
class SoftmaxLayer : public Layer
{
private:
    OMPMatrix output;

public:
    OMPMatrix forward(const OMPMatrix& input) override
    {
        OMPMatrix result(input.rows, input.cols, input.depth);

        // Find maximum value for numerical stability
        float maxVal = *std::max_element(input.data.begin(), input.data.end());

        // Compute exponentials
        std::vector<float> expValues(input.data.size());
        float sumExp = 0.0f;

#pragma omp parallel
        {
            float local_sumExp = 0.0f;

#pragma omp for
            for (int i = 0; i < input.data.size(); ++i)
            {
                expValues[i] = std::exp(input.data[i] - maxVal);
                local_sumExp += expValues[i];
            }

#pragma omp atomic
            sumExp += local_sumExp;
        }

        // Normalize
#pragma omp parallel for
        for (int i = 0; i < input.data.size(); ++i)
        {
            result.data[i] = expValues[i] / sumExp;
        }

        this->output = result;
        return result;
    }

    OMPMatrix backward(const OMPMatrix& gradOutput, float learningRate) override
    {
        // For softmax with cross-entropy loss, the gradient simplifies
        OMPMatrix result(gradOutput.rows, gradOutput.cols, gradOutput.depth);

#pragma omp parallel for
        for (int i = 0; i < gradOutput.data.size(); ++i)
        {
            result.data[i] = gradOutput.data[i];
        }

        return result;
    }
};

// Cross-entropy loss function
float crossEntropyLoss(const OMPMatrix& output, const OMPMatrix& target)
{
    float loss = 0.0f;

#pragma omp parallel
    {
        float local_loss = 0.0f;

#pragma omp for
        for (int i = 0; i < output.data.size(); ++i)
        {
            // Clip predictions to avoid log(0)
            float pred = std::max(std::min(output.data[i], 1.0f - 1e-7f), 1e-7f);
            local_loss -= target.data[i] * std::log(pred);
        }

#pragma omp atomic
        loss += local_loss;
    }

    return loss;
}

// Derivative of cross-entropy loss with respect to softmax output
OMPMatrix crossEntropyGradient(const OMPMatrix& output, const OMPMatrix& target)
{
    // For softmax + cross-entropy, gradient is (output - target)
    return output - target;
}

// Parallel CNN class
class ParallelCNN
{
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void addLayer(std::unique_ptr<Layer> layer)
    {
        layers.push_back(std::move(layer));
    }

    OMPMatrix forward(const OMPMatrix& input)
    {
        OMPMatrix current = input;

        for (auto& layer : layers)
        {
            current = layer->forward(current);
        }

        return current;
    }

    float train(const OMPMatrix& input, const OMPMatrix& target, float learningRate)
    {
        // Forward pass
        OMPMatrix output = forward(input);

        // Compute loss
        float loss = crossEntropyLoss(output, target);

        // Compute output gradient
        OMPMatrix gradient = crossEntropyGradient(output, target);

        // Backward pass
        for (int i = layers.size() - 1; i >= 0; --i)
        {
            gradient = layers[i]->backward(gradient, learningRate);
        }

        return loss;
    }

    void trainBatch(const std::vector<OMPMatrix>& batchInputs,
        const std::vector<OMPMatrix>& batchTargets,
        float learningRate) {
        float batchLoss = 0.0f;

        for (size_t i = 0; i < batchInputs.size(); ++i) {
            batchLoss += train(batchInputs[i], batchTargets[i], learningRate);
        }
    }

    float evaluate(const std::vector<OMPMatrix>& inputs, const std::vector<OMPMatrix>& targets)
    {
        size_t correct = 0;

        if (inputs.size() > 8) {
            std::vector<size_t> localCorrect(omp_get_max_threads(), 0);

#pragma omp parallel
            {
                int threadID = omp_get_thread_num();

#pragma omp for schedule(dynamic)
                for (int i = 0; i < inputs.size(); ++i) {
                    OMPMatrix output = forward(inputs[i]);

                    // Find index of highest value in output and target
                    size_t predIndex = std::distance(
                        output.data.begin(),
                        std::max_element(output.data.begin(), output.data.end())
                    );

                    size_t targetIndex = std::distance(
                        targets[i].data.begin(),
                        std::max_element(targets[i].data.begin(), targets[i].data.end())
                    );

                    if (predIndex == targetIndex) {
                        localCorrect[threadID]++;
                    }
                }
            }

            // Sum up thread-local counters
            for (auto count : localCorrect) {
                correct += count;
            }
        }
        else {
            // Sequential for small datasets
            for (size_t i = 0; i < inputs.size(); ++i) {
                OMPMatrix output = forward(inputs[i]);

                size_t predIndex = std::distance(
                    output.data.begin(),
                    std::max_element(output.data.begin(), output.data.end())
                );

                size_t targetIndex = std::distance(
                    targets[i].data.begin(),
                    std::max_element(targets[i].data.begin(), targets[i].data.end())
                );

                if (predIndex == targetIndex) {
                    correct++;
                }
            }
        }

        return static_cast<float>(correct) / inputs.size();
    }
};



class CancerDataset
{
private:
    std::vector<OMPMatrix> images;
    std::vector<OMPMatrix> labels;
    size_t numClasses;
    size_t imageSize;

    // Helper function to load CSV with format: id,label,filename
    std::map<std::string, int> loadCSV(const std::string& csvPath)
    {
        std::map<std::string, int> labelMap;
        std::ifstream file(csvPath);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open CSV file: " << csvPath << std::endl;
            return labelMap;
        }

        std::string line, id, label, filename;

        // Skip header row if it exists
        std::getline(file, line);

        while (std::getline(file, line))
        {
            std::stringstream ss(line);

            // Parse the three columns: id,label,filename
            std::getline(ss, id, ',');
            std::getline(ss, label, ',');
            std::getline(ss, filename, ',');

            // Remove any quotes around the filename if they exist
            if (!filename.empty() && (filename.front() == '"' && filename.back() == '"'))
            {
                filename = filename.substr(1, filename.length() - 2);
            }

            // Convert label to integer
            int labelValue;
            try
            {
                labelValue = std::stoi(label);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error converting label to integer for " << filename << ": " << e.what() << std::endl;
                continue;
            }

            // Use the filename as the key in our map
            labelMap[filename] = labelValue;

            // Also try with just the base filename (no path or extension)
            size_t lastSlash = filename.find_last_of("/\\");
            size_t lastDot = filename.find_last_of(".");
            if (lastSlash != std::string::npos && lastDot != std::string::npos)
            {
                std::string baseFilename = filename.substr(lastSlash + 1, lastDot - lastSlash - 1);
                labelMap[baseFilename] = labelValue;
            }
        }
        return labelMap;
    }

    // Helper function to find image files by trying different extensions
    std::string findImageFile(const std::string& imageFolderPath, const std::string& baseFilename)
    {
        // Try different extensions
        std::vector<std::string> extensions = { ".tif", ".tiff", ".jpg", ".jpeg", ".png" };

        for (const auto& ext : extensions)
        {
            std::string fullPath = imageFolderPath + "/" + baseFilename + ext;
            if (std::ifstream(fullPath).good())
            {
                return fullPath; // File exists
            }
        }

        return ""; // File not found with any extension
    }

public:
    CancerDataset(const std::string& imageFolderPath, const std::string& labelPath,
        size_t numClasses, size_t targetSize = 32)
        : numClasses(numClasses), imageSize(targetSize)
    {

        // Load labels from CSV
        std::map<std::string, int> fileToLabel = loadCSV(labelPath);

        // Try different file extensions (tif, tiff, jpg, png)
        std::vector<cv::String> filenames;
        cv::glob(imageFolderPath + "/*.tif", filenames);

        if (filenames.empty())
        {
            std::cout << "No .tif files found, trying .tiff..." << std::endl;
            cv::glob(imageFolderPath + "/*.tiff", filenames);
        }

        if (filenames.empty())
        {
            std::cout << "No .tiff files found, trying .jpg..." << std::endl;
            cv::glob(imageFolderPath + "/*.jpg", filenames);
        }

        if (filenames.empty())
        {
            std::cout << "No .jpg files found, trying .png..." << std::endl;
            cv::glob(imageFolderPath + "/*.png", filenames);
        }

        // If no files were found using glob patterns, try to find files based on CSV entries
        if (filenames.empty())
        {
            std::cout << "No image files found using glob pattern. Trying to match from CSV entries..." << std::endl;

            // For each label entry, try to find a matching image file
            for (const auto& entry : fileToLabel)
            {
                std::string baseFilename = entry.first;

                // Remove extension if it exists
                size_t lastDot = baseFilename.find_last_of(".");
                if (lastDot != std::string::npos)
                {
                    baseFilename = baseFilename.substr(0, lastDot);
                }

                // Try to find the file with different extensions
                std::string imagePath = findImageFile(imageFolderPath, baseFilename);
                if (!imagePath.empty())
                {
                    filenames.push_back(imagePath);
                    std::cout << "Found image for " << entry.first << ": " << imagePath << std::endl;
                }
            }
        }

        // Process each image - can be parallelized
        images.resize(filenames.size());
        labels.resize(filenames.size());

        // Use std::atomic to track valid images (thread-safe counter)
        std::atomic<size_t> validImages(0);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < filenames.size(); ++i)
        {
            // Extract just the filename from the path
            std::string fullPath = filenames[i];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string filename = fullPath.substr(lastSlash + 1);

            // Extract the base filename without extension for matching
            size_t lastDot = filename.find_last_of(".");
            std::string baseFilename = (lastDot != std::string::npos) ? filename.substr(0, lastDot) : filename;

            // Try several variants to match with the label file
            auto labelIt = fileToLabel.find(filename);
            if (labelIt == fileToLabel.end())
            {
                // Try with just the base filename (no extension)
                labelIt = fileToLabel.find(baseFilename);
            }

            bool found = false;
            if (labelIt == fileToLabel.end())
            {
                // Try just matching the end of the filename (some datasets have prefixes in the actual files)
                for (const auto& entry : fileToLabel)
                {
                    if (filename.find(entry.first) != std::string::npos)
                    {
                        labelIt = fileToLabel.find(entry.first);
                        found = true;
#pragma omp critical
                        {
                            std::cout << "Matched " << filename << " with label for " << entry.first << std::endl;
                        }
                        break;
                    }
                }

                if (!found)
                {
#pragma omp critical
                    {
                        std::cout << "Warning: No label found for file " << filename << std::endl;
                    }
                    continue;
                }
            }

            // Load image using OpenCV
            cv::Mat img = cv::imread(fullPath, cv::IMREAD_COLOR);
            if (img.empty())
            {
#pragma omp critical
                {
                    std::cout << "Warning: Could not load image " << fullPath << std::endl;
                }
                continue;
            }

            // Resize image to target size
            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

            // Convert OpenCV Mat to our OMPMatrix format
            OMPMatrix imageOMPMatrix(imageSize, imageSize, 3);
            for (size_t r = 0; r < imageSize; ++r)
            {
                for (size_t c = 0; c < imageSize; ++c)
                {
                    cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                    // OpenCV uses BGR, normalize to 0-1 range
                    imageOMPMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                    imageOMPMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                    imageOMPMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
                }
            }

            // Create one-hot encoded label
            OMPMatrix labelOMPMatrix(1, numClasses, 1);
            int classId = labelIt->second;
            labelOMPMatrix.at(0, classId, 0) = 1.0f;

            // Get the next valid index
            size_t index = validImages.fetch_add(1);

            // Store in final containers
            if (index < images.size())
            {
                images[index] = imageOMPMatrix;
                labels[index] = labelOMPMatrix;
            }
        }

        // Resize containers to actual number of valid images
        size_t actualSize = validImages.load();
        if (actualSize < images.size())
        {
            images.resize(actualSize);
            labels.resize(actualSize);
        }

        std::cout << "Loaded " << images.size() << " images with labels" << std::endl;
    }

    // Split dataset into training and testing sets
    void splitTrainTest(std::vector<OMPMatrix>& trainImages, std::vector<OMPMatrix>& trainLabels,
        std::vector<OMPMatrix>& testImages, std::vector<OMPMatrix>& testLabels,
        float testRatio = 0.2f)
    {
        size_t numSamples = images.size();
        size_t numTest = static_cast<size_t>(numSamples * testRatio);
        size_t numTrain = numSamples - numTest;

        // Create a random permutation
        std::vector<size_t> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Allocate space
        trainImages.resize(numTrain);
        trainLabels.resize(numTrain);
        testImages.resize(numTest);
        testLabels.resize(numTest);

        // Fill training set in parallel
#pragma omp parallel for
        for (int i = 0; i < numTrain; ++i)
        {
            trainImages[i] = images[indices[i]];
            trainLabels[i] = labels[indices[i]];
        }

        // Fill testing set in parallel
#pragma omp parallel for
        for (int i = 0; i < numTest; ++i)
        {
            testImages[i] = images[indices[numTrain + i]];
            testLabels[i] = labels[indices[numTrain + i]];
        }

        std::cout << "Split dataset into " << numTrain << " training and "
            << numTest << " testing samples" << std::endl;
    }
};

bool predictCancerImageOMP(ParallelCNN& trainedModel, const std::string& imagePath);
void predictMultipleImagesOMP(ParallelCNN& trainedModel);

bool predictCancerImageOMP(ParallelCNN& trainedModel, const std::string& imagePath) {
    // Initialize image size to match the model's expected input
    const size_t imageSize = 32;
    const float MALIGNANT_THRESHOLD = 0.5f;  // Add threshold constant

    try {
        // Load image using OpenCV
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: Could not load image " << imagePath << std::endl;
            return false;
        }

        // Resize image to target size
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

        // Convert OpenCV Mat to our OMPMatrix format
        OMPMatrix imageMatrix(imageSize, imageSize, 3);

        // Parallelize image processing
#pragma omp parallel for collapse(2)
        for (int r = 0; r < imageSize; ++r) {
            for (int c = 0; c < imageSize; ++c) {
                cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                // OpenCV uses BGR, normalize to 0-1 range
                imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
            }
        }

        // Run forward pass through the trained model
        OMPMatrix output = trainedModel.forward(imageMatrix);

        // Get the predicted class
        float benignProb = output.at(0, 0, 0);  // Probability of benign
        float malignantProb = output.at(0, 1, 0); // Probability of malignant

        // Print results
        std::cout << "\n===== OpenMP Model Prediction Results =====" << std::endl;
        std::cout << "Image: " << imagePath << std::endl;
        std::cout << "Benign probability: " << benignProb * 100.0f << "%" << std::endl;
        std::cout << "Malignant probability: " << malignantProb * 100.0f << "%" << std::endl;

        // Modified threshold logic
        if (malignantProb > MALIGNANT_THRESHOLD) {
            std::cout << "Prediction: MALIGNANT (Cancer detected)" << std::endl;
            std::cout << "Confidence: " << malignantProb * 100.0f << "% (threshold: "
                << MALIGNANT_THRESHOLD * 100.0f << "%)" << std::endl;
        }
        else {
            std::cout << "Prediction: BENIGN (No cancer detected)" << std::endl;
        }
        std::cout << "==========================================\n" << std::endl;

        return true;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during OpenMP prediction: " << e.what() << std::endl;
        return false;
    }
}

void predictMultipleImagesOMP(ParallelCNN& trainedModel) {
    std::string imagePath;

    while (true) {
        std::cout << "\nEnter image path for OpenMP model prediction (or 'q' to quit): ";
        std::cin >> imagePath;

        if (imagePath == "q" || imagePath == "Q") {
            break;
        }

        predictCancerImageOMP(trainedModel, imagePath);
    }
}

std::pair<std::pair<std::chrono::duration<double>, float>, ParallelCNN*> ompWithModel(std::string ImagePath, std::string LabelPath)
{
    std::cout << "Cancer Detection using CNN with OpenMP Parallelization" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Get number of available OpenMP threads
    int maxThreads = omp_get_max_threads();
    std::cout << "Using OpenMP with " << maxThreads << " threads" << std::endl;

    // Create dataset using OpenCV
    std::cout << "Loading cancer detection dataset from " << ImagePath << "..." << std::endl;
    CancerDataset dataset(ImagePath, LabelPath, 2); // 2 classes: benign and malignant

    // Split dataset
    std::vector<OMPMatrix> trainImages, trainLabels, testImages, testLabels;
    dataset.splitTrainTest(trainImages, trainLabels, testImages, testLabels);

    // Hyperparameters
    float learningRate = 0.001f;
    size_t batchSize = 16;
    size_t numEpochs = 5;
    size_t inputSize = 32;  // 32x32 pixel images (from resizing)
    size_t numChannels = 3; // RGB images
    size_t numClasses = 2;  // Benign or malignant

    // Create and configure Parallel CNN
    std::cout << "\nInitializing OpenMP CNN..." << std::endl;
    ParallelCNN* parallelCNN = new ParallelCNN();

    // Add layers to parallel CNN
    parallelCNN->addLayer(std::make_unique<ConvolutionalLayer>(numChannels, 16, 3, 1, 1));
    parallelCNN->addLayer(std::make_unique<ReLULayer>());
    parallelCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    parallelCNN->addLayer(std::make_unique<ConvolutionalLayer>(16, 32, 3, 1, 1));
    parallelCNN->addLayer(std::make_unique<ReLULayer>());
    parallelCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    parallelCNN->addLayer(std::make_unique<ConvolutionalLayer>(32, 64, 3, 1, 1));
    parallelCNN->addLayer(std::make_unique<ReLULayer>());
    parallelCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    // After pooling, we have 64 feature maps of size 4x4
    parallelCNN->addLayer(std::make_unique<FullyConnectedLayer>(64 * 4 * 4, 128));
    parallelCNN->addLayer(std::make_unique<ReLULayer>());
    parallelCNN->addLayer(std::make_unique<FullyConnectedLayer>(128, numClasses));
    parallelCNN->addLayer(std::make_unique<SoftmaxLayer>());

    // Train and evaluate Parallel CNN
    std::cout << "\n===== Training OpenMP CNN =====" << std::endl;
    auto parallelStart = std::chrono::high_resolution_clock::now();

    // Training loop with epoch progress
    for (size_t epoch = 0; epoch < numEpochs; ++epoch)
    {
        float epochLoss = 0.0f;
        size_t batchCount = 0;

        // Process mini-batches
        for (size_t batchStart = 0; batchStart < trainImages.size(); batchStart += batchSize)
        {
            size_t currentBatchSize = std::min(batchSize, trainImages.size() - batchStart);
            std::vector<OMPMatrix> batchImages(currentBatchSize);
            std::vector<OMPMatrix> batchLabels(currentBatchSize);

            for (size_t i = 0; i < currentBatchSize; ++i)
            {
                batchImages[i] = trainImages[batchStart + i];
                batchLabels[i] = trainLabels[batchStart + i];
            }

            // Train on batch using parallel processing
            parallelCNN->trainBatch(batchImages, batchLabels, learningRate);

            batchCount++;
        }

        // Evaluate accuracy on training set
        float trainAccuracy = parallelCNN->evaluate(trainImages, trainLabels) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs
            << ", Training Accuracy: " << trainAccuracy << "%" << std::endl;
    }

    auto parallelEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallelDuration = parallelEnd - parallelStart;

    // Print performance
    std::cout << "\n===== Performance =====" << std::endl;
    std::cout << "OpenMP CNN execution time: " << parallelDuration.count() << " seconds" << std::endl;

    // Final evaluation
    std::cout << "\n===== Final Model Evaluation =====" << std::endl;
    float parallelAccuracy = parallelCNN->evaluate(testImages, testLabels) * 100.0f;
    std::cout << "OpenMP CNN accuracy: " << parallelAccuracy << "%" << std::endl;

    return { { parallelDuration, parallelAccuracy }, parallelCNN };
}

void batchPredictOMP(ParallelCNN& model, const std::string& imageFolder, const std::string& outputFile) {
    const size_t imageSize = 32;
    const float MALIGNANT_THRESHOLD = 0.5f;

    // Get all image files in the folder
    std::vector<std::string> imageFiles;
    for (const auto& entry : std::filesystem::directory_iterator(imageFolder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    std::cout << "Found " << imageFiles.size() << " images in " << imageFolder << std::endl;

    // Prepare result vectors
    std::vector<float> benignProbs;
    std::vector<float> malignantProbs;
    std::vector<bool> predictions;
    std::vector<std::string> validImageFiles;

    // Process images in parallel
    size_t processedCount = 0;
    size_t totalImages = imageFiles.size();

    // Thread-safe progress reporting
    std::mutex progressMutex;

    // Process images in batches to manage memory
    const size_t batchSize = 100;

    for (size_t batchStart = 0; batchStart < imageFiles.size(); batchStart += batchSize) {
        size_t currentBatchSize = std::min(batchSize, imageFiles.size() - batchStart);

        // Process a batch of images in parallel
#pragma omp parallel for
        for (int i = 0; i < currentBatchSize; ++i) {
            size_t imageIndex = batchStart + i;
            const std::string& imagePath = imageFiles[imageIndex];

            try {
                // Load and preprocess image
                cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (img.empty()) {
#pragma omp critical
                    {
                        std::cerr << "Warning: Could not load image " << imagePath << std::endl;
                    }
                    continue;
                }

                // Resize image
                cv::Mat resizedImg;
                cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

                // Convert to OMPMatrix format
                OMPMatrix imageMatrix(imageSize, imageSize, 3);
                for (size_t r = 0; r < imageSize; ++r) {
                    for (size_t c = 0; c < imageSize; ++c) {
                        cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                        imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                        imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                        imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
                    }
                }

                // Run prediction
                OMPMatrix output = model.forward(imageMatrix);

                float benignProb = output.at(0, 0, 0);
                float malignantProb = output.at(0, 1, 0);
                bool isMalignant = (malignantProb > MALIGNANT_THRESHOLD);

                // Store results (thread-safe)
#pragma omp critical
                {
                    validImageFiles.push_back(imagePath);
                    benignProbs.push_back(benignProb);
                    malignantProbs.push_back(malignantProb);
                    predictions.push_back(isMalignant);
                }
            }
            catch (const std::exception& e) {
#pragma omp critical
                {
                    std::cerr << "Error processing image " << imagePath << ": " << e.what() << std::endl;
                }
            }

            // Update progress (thread-safe)
#pragma omp critical
            {
                processedCount++;
                if (processedCount % 100 == 0 || processedCount == totalImages) {
                    std::cout << "Progress: " << processedCount << "/" << totalImages
                        << " (" << (processedCount * 100 / totalImages) << "%)" << std::endl;
                }
            }
        }
    }

    // Use the writePredictionResultsToCSV function from main.cpp
    writePredictionResultsToCSV(outputFile, validImageFiles, benignProbs, malignantProbs, predictions);

    // Print summary
    size_t malignantCount = std::count(predictions.begin(), predictions.end(), true);
    size_t benignCount = predictions.size() - malignantCount;

    std::cout << "\n===== Prediction Summary =====" << std::endl;
    std::cout << "Total images processed: " << predictions.size() << std::endl;
    std::cout << "Benign predictions: " << benignCount
        << " (" << (benignCount * 100.0f / predictions.size()) << "%)" << std::endl;
    std::cout << "Malignant predictions: " << malignantCount
        << " (" << (malignantCount * 100.0f / predictions.size()) << "%)" << std::endl;
}
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
#include <opencv2/opencv.hpp>  // OpenCV main header
#include <opencv2/core.hpp>    // Core functionality
#include <opencv2/imgproc.hpp> // Image processing
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include "CUDACNN.h"
#include "common.h"
#include "Matrix.h"
#include "Layer.h"
#include "CUDAMatrix.cuh"
#include "CUDAConvolution.cuh"
#include "CUDAFullyConnected.cuh"

// Forward declarations for classes defined elsewhere
class CancerDataset;

// Non-CUDA layer implementations needed for the CUDA CNN
// ReLU Layer
class ReLULayer : public Layer {
private:
    Matrix input;

public:
    Matrix forward(const Matrix& input) override {
        this->input = input;
        Matrix output(input.rows, input.cols, input.depth);
        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = std::max(0.0f, input.data[i]);
        }
        return output;
    }

    Matrix backward(const Matrix& gradOutput, float learningRate) override {
        Matrix inputGradient(input.rows, input.cols, input.depth);
        for (size_t i = 0; i < input.data.size(); ++i) {
            inputGradient.data[i] = (input.data[i] > 0) ? gradOutput.data[i] : 0;
        }
        return inputGradient;
    }
};

// Max Pooling Layer 
class MaxPoolingLayer : public Layer {
private:
    size_t poolSize;
    Matrix input;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> maxIndices;

public:
    MaxPoolingLayer(size_t poolSize) : poolSize(poolSize) {}

    Matrix forward(const Matrix& input) override {
        this->input = input;
        size_t outputHeight = input.rows / poolSize;
        size_t outputWidth = input.cols / poolSize;
        Matrix output(outputHeight, outputWidth, input.depth);

        // Reset max indices tracking
        maxIndices.resize(input.depth);
        for (size_t d = 0; d < input.depth; ++d) {
            maxIndices[d].resize(outputHeight);
            for (size_t h = 0; h < outputHeight; ++h) {
                maxIndices[d][h].resize(outputWidth, { 0, 0 });
            }
        }

        // Find max values in each pooling region
        for (size_t d = 0; d < input.depth; ++d) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float maxVal = std::numeric_limits<float>::lowest();
                    size_t maxI = 0, maxJ = 0;

                    for (size_t i = 0; i < poolSize; ++i) {
                        for (size_t j = 0; j < poolSize; ++j) {
                            size_t inputI = h * poolSize + i;
                            size_t inputJ = w * poolSize + j;

                            if (inputI < input.rows && inputJ < input.cols) {
                                if (input.at(inputI, inputJ, d) > maxVal) {
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

    Matrix backward(const Matrix& gradOutput, float learningRate) override {
        size_t outputHeight = gradOutput.rows;
        size_t outputWidth = gradOutput.cols;
        Matrix inputGradient(input.rows, input.cols, input.depth);

        for (size_t d = 0; d < input.depth; ++d) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    if (d < maxIndices.size() && h < maxIndices[d].size() && w < maxIndices[d][h].size()) {
                        auto [maxI, maxJ] = maxIndices[d][h][w];
                        size_t inputI = h * poolSize + maxI;
                        size_t inputJ = w * poolSize + maxJ;

                        if (inputI < inputGradient.rows && inputJ < inputGradient.cols) {
                            inputGradient.at(inputI, inputJ, d) = gradOutput.at(h, w, d);
                        }
                    }
                }
            }
        }
        return inputGradient;
    }
};

// Softmax Layer
class SoftmaxLayer : public Layer {
private:
    Matrix output;

public:
    Matrix forward(const Matrix& input) override {
        Matrix result(input.rows, input.cols, input.depth);

        // Find maximum value for numerical stability
        float maxVal = *std::max_element(input.data.begin(), input.data.end());

        // Compute exponentials
        std::vector<float> expValues(input.data.size());
        float sumExp = 0.0f;

        for (size_t i = 0; i < input.data.size(); ++i) {
            expValues[i] = std::exp(input.data[i] - maxVal);
            sumExp += expValues[i];
        }

        // Normalize
        for (size_t i = 0; i < input.data.size(); ++i) {
            result.data[i] = expValues[i] / sumExp;
        }

        this->output = result;
        return result;
    }

    Matrix backward(const Matrix& gradOutput, float learningRate) override {
        // For softmax with cross-entropy loss, the gradient simplifies
        Matrix result = gradOutput;
        return result;
    }
};

// Function to train and evaluate the CUDA CNN model
std::pair<std::chrono::duration<double>, float> cuda(CancerDataset& dataset,
    std::vector<Matrix>& trainImages,
    std::vector<Matrix>& trainLabels,
    std::vector<Matrix>& testImages,
    std::vector<Matrix>& testLabels,
    CUDACNN*& trainedCUDACNN) {
    std::cout << "Cancer Detection using CNN with CUDA" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Initialize CUDA device
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable device detected!" << std::endl;
        return { std::chrono::duration<double>(0), 0.0f };
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    // Hyperparameters
    float learningRate = 0.001f;
    size_t batchSize = 16;
    size_t numEpochs = 5;
    size_t inputSize = 32;  // 32x32 pixel images (from resizing)
    size_t numChannels = 3; // RGB images
    size_t numClasses = 2;  // Benign or malignant

    // Create and configure CUDA CNN
    std::cout << "\nInitializing CUDA CNN..." << std::endl;
    CUDACNN* cudaCNN = new CUDACNN();  // Create on heap

    // Add layers to CUDA CNN
    cudaCNN->addLayer(std::make_unique<CUDAConvolutionalLayer>(numChannels, 16, 3, 1, 1));
    cudaCNN->addLayer(std::make_unique<ReLULayer>());
    cudaCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    cudaCNN->addLayer(std::make_unique<CUDAConvolutionalLayer>(16, 32, 3, 1, 1));
    cudaCNN->addLayer(std::make_unique<ReLULayer>());
    cudaCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    cudaCNN->addLayer(std::make_unique<CUDAConvolutionalLayer>(32, 64, 3, 1, 1));
    cudaCNN->addLayer(std::make_unique<ReLULayer>());
    cudaCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

    // After pooling, we have 64 feature maps of size 4x4
    cudaCNN->addLayer(std::make_unique<CUDAFullyConnectedLayer>(64 * 4 * 4, 128));
    cudaCNN->addLayer(std::make_unique<ReLULayer>());
    cudaCNN->addLayer(std::make_unique<CUDAFullyConnectedLayer>(128, numClasses));
    cudaCNN->addLayer(std::make_unique<SoftmaxLayer>());

    // Train and evaluate CUDA CNN
    std::cout << "\n===== Training CUDA CNN =====" << std::endl;
    auto cudaStart = std::chrono::high_resolution_clock::now();

    // Training loop with epoch progress
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {
        float epochLoss = 0.0f;
        size_t batchCount = 0;

        // Process mini-batches
        for (size_t batchStart = 0; batchStart < trainImages.size(); batchStart += batchSize) {
            size_t currentBatchSize = std::min(batchSize, trainImages.size() - batchStart);
            std::vector<Matrix> batchImages(currentBatchSize);
            std::vector<Matrix> batchLabels(currentBatchSize);

            for (size_t i = 0; i < currentBatchSize; ++i) {
                batchImages[i] = trainImages[batchStart + i];
                batchLabels[i] = trainLabels[batchStart + i];
            }

            // Train on batch using CUDA acceleration
            cudaCNN->trainBatch(batchImages, batchLabels, learningRate);
            batchCount++;
        }

        // Evaluate accuracy on training set
        float trainAccuracy = cudaCNN->evaluate(trainImages, trainLabels) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs
            << ", Training Accuracy: " << trainAccuracy << "%" << std::endl;
    }

    auto cudaEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cudaDuration = cudaEnd - cudaStart;

    // Print performance information
    std::cout << "\n===== Performance =====" << std::endl;
    std::cout << "CUDA CNN execution time: " << cudaDuration.count() << " seconds" << std::endl;

    // Final evaluation
    std::cout << "\n===== Final Model Evaluation =====" << std::endl;
    float cudaAccuracy = cudaCNN->evaluate(testImages, testLabels) * 100.0f;
    std::cout << "CUDA CNN accuracy: " << cudaAccuracy << "%" << std::endl;

    // Clean up CUDA resources
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        std::cerr << "Warning: CUDA synchronization error: "
            << cudaGetErrorString(syncError) << std::endl;
    }

    // Store the trained model for later use
    trainedCUDACNN = cudaCNN;  // Pass the trained model back

    return { cudaDuration, cudaAccuracy };
}
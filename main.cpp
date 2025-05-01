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
#include <mutex> 
#include <filesystem>
#include <opencv2/opencv.hpp>  // OpenCV main header
#include <opencv2/core.hpp>    // Core functionality
#include <opencv2/imgproc.hpp>  // Image processing
#include <opencv2/highgui.hpp>
#include "omp.h"
#include "common.h"
#include "CUDACNN.h"
#include <fstream>

// Forward declaration for CUDA CNN class
class CUDACNN;

// Convolutional Layer (Serial implementation)
class ConvolutionalLayer : public Layer {
private:
    std::vector<Matrix> filters;
    std::vector<float> biases;
    size_t stride, padding;
    Matrix input;
    size_t inputDepth, outputDepth, filterSize;

public:
    ConvolutionalLayer(size_t inputDepth, size_t outputDepth, size_t filterSize,
        size_t stride = 1, size_t padding = 0)
        : inputDepth(inputDepth), outputDepth(outputDepth), filterSize(filterSize),
        stride(stride), padding(padding) {

        // Initialize filters and biases
        for (size_t i = 0; i < outputDepth; ++i) {
            Matrix filter(filterSize, filterSize, inputDepth);
            filter.randomize(-0.1f, 0.1f);
            filters.push_back(filter);
            biases.push_back(0.0f);
        }
    }

    Matrix forward(const Matrix& input) override {
        this->input = input;

        size_t outputHeight = (input.rows - filterSize + 2 * padding) / stride + 1;
        size_t outputWidth = (input.cols - filterSize + 2 * padding) / stride + 1;
        Matrix output(outputHeight, outputWidth, outputDepth);

        // For each filter
        for (size_t filterIdx = 0; filterIdx < outputDepth; ++filterIdx) {
            // Apply filter to input volume
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float sum = biases[filterIdx];

                    // Convolve filter with input region
                    for (size_t fh = 0; fh < filterSize; ++fh) {
                        for (size_t fw = 0; fw < filterSize; ++fw) {
                            for (size_t fd = 0; fd < inputDepth; ++fd) {
                                int ih = static_cast<int>(h * stride + fh) - static_cast<int>(padding);
                                int iw = static_cast<int>(w * stride + fw) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(input.rows) &&
                                    iw >= 0 && iw < static_cast<int>(input.cols)) {
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

    Matrix backward(const Matrix& gradOutput, float learningRate) override {
        size_t outputHeight = gradOutput.rows;
        size_t outputWidth = gradOutput.cols;

        // Initialize gradients for filters and biases
        std::vector<Matrix> filterGradients(outputDepth, Matrix(filterSize, filterSize, inputDepth));
        std::vector<float> biasGradients(outputDepth, 0.0f);

        // Initialize gradient for input
        Matrix inputGradient(input.rows, input.cols, input.depth);

        // For each filter
        for (size_t filterIdx = 0; filterIdx < outputDepth; ++filterIdx) {
            // Accumulate bias gradients (sum of gradOutput for this filter)
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    biasGradients[filterIdx] += gradOutput.at(h, w, filterIdx);
                }
            }

            // Accumulate filter gradients
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float gradValue = gradOutput.at(h, w, filterIdx);

                    for (size_t fh = 0; fh < filterSize; ++fh) {
                        for (size_t fw = 0; fw < filterSize; ++fw) {
                            for (size_t fd = 0; fd < inputDepth; ++fd) {
                                int ih = static_cast<int>(h * stride + fh) - static_cast<int>(padding);
                                int iw = static_cast<int>(w * stride + fw) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(input.rows) &&
                                    iw >= 0 && iw < static_cast<int>(input.cols)) {
                                    filterGradients[filterIdx].at(fh, fw, fd) += input.at(ih, iw, fd) * gradValue;
                                }
                            }
                        }
                    }
                }
            }

            // Compute input gradients
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float gradValue = gradOutput.at(h, w, filterIdx);

                    for (size_t fh = 0; fh < filterSize; ++fh) {
                        for (size_t fw = 0; fw < filterSize; ++fw) {
                            for (size_t fd = 0; fd < inputDepth; ++fd) {
                                int ih = static_cast<int>(h * stride + fh) - static_cast<int>(padding);
                                int iw = static_cast<int>(w * stride + fw) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(input.rows) &&
                                    iw >= 0 && iw < static_cast<int>(input.cols)) {
                                    inputGradient.at(ih, iw, fd) += filters[filterIdx].at(fh, fw, fd) * gradValue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update filters and biases
        for (size_t i = 0; i < outputDepth; ++i) {
            for (size_t fh = 0; fh < filterSize; ++fh) {
                for (size_t fw = 0; fw < filterSize; ++fw) {
                    for (size_t fd = 0; fd < inputDepth; ++fd) {
                        filters[i].at(fh, fw, fd) -= learningRate * filterGradients[i].at(fh, fw, fd);
                    }
                }
            }
            biases[i] -= learningRate * biasGradients[i];
        }
        return inputGradient;
    }
};

// Max Pooling Layer (Serial implementation)
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

        // For each depth slice
        for (size_t d = 0; d < input.depth; ++d) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float maxVal = std::numeric_limits<float>::lowest();
                    size_t maxI = 0, maxJ = 0;

                    // Find maximum in pooling region
                    for (size_t i = 0; i < poolSize; ++i) {
                        for (size_t j = 0; j < poolSize; ++j) {
                            size_t inputI = h * poolSize + i;
                            size_t inputJ = w * poolSize + j;

                            // Make sure inputI and inputJ are within bounds
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

        // For each depth slice
        for (size_t d = 0; d < input.depth; ++d) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    // Ensure we're not accessing out of bounds in maxIndices
                    if (d < maxIndices.size() && h < maxIndices[d].size() && w < maxIndices[d][h].size()) {
                        // Get stored indices of max value
                        auto [maxI, maxJ] = maxIndices[d][h][w];

                        // Pass gradient to the max element's position
                        size_t inputI = h * poolSize + maxI;
                        size_t inputJ = w * poolSize + maxJ;

                        // Add bounds checking
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

// ReLU Activation Layer (Serial implementation)
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
        // Ensure output has same dimensions as input
        Matrix inputGradient(input.rows, input.cols, input.depth);

        // Make sure we're computing with the right dimensions
        if (gradOutput.data.size() != input.data.size()) {
            std::cout << "WARNING: Gradient size mismatch in ReLU backward!" << std::endl;
            return inputGradient; // Return zero gradient to prevent crash
        }

        for (size_t i = 0; i < input.data.size(); ++i) {
            // ReLU derivative: 1 if input > 0, 0 otherwise
            inputGradient.data[i] = (input.data[i] > 0) ? gradOutput.data[i] : 0;
        }

        return inputGradient;
    }
};

// Fully Connected Layer (Serial implementation)
class FullyConnectedLayer : public Layer {
private:
    Matrix weights;
    std::vector<float> biases;
    Matrix input;
    size_t inputSize, outputSize;

public:
    FullyConnectedLayer(size_t inputSize, size_t outputSize)
        : inputSize(inputSize), outputSize(outputSize) {

        // Initialize weights and biases
        weights = Matrix(outputSize, inputSize, 1);
        weights.randomize(-0.1f, 0.1f);
        biases.resize(outputSize, 0.0f);
    }

    Matrix forward(const Matrix& input) override {
        // Ensure input is flattened
        Matrix flatInput;
        if (input.rows * input.cols * input.depth != inputSize) {
            flatInput = input.flatten();
        }
        else {
            flatInput = input;
        }

        this->input = flatInput;
        Matrix output(1, outputSize, 1);

        // Check if our flattened input matches expected size
        if (flatInput.rows * flatInput.cols * flatInput.depth != inputSize) {
            std::cout << "ERROR: After flattening, size mismatch: " << std::endl;
            std::cout << "  Expected: " << inputSize << std::endl;
            std::cout << "  Actual: " << flatInput.rows * flatInput.cols * flatInput.depth << std::endl;
        }

        // Compute output = weights * input + biases
        try {
            for (size_t i = 0; i < outputSize; ++i) {
                float sum = biases[i];
                for (size_t j = 0; j < inputSize && j < flatInput.cols; ++j) {
                    // Add bounds checking
                    if (j < flatInput.cols) {
                        sum += weights.at(i, j, 0) * flatInput.at(0, j, 0);
                    }
                    else {
                        std::cout << "WARN: Input index " << j << " out of bounds!" << std::endl;
                        break;
                    }
                }
                output.at(0, i, 0) = sum;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Exception in FC forward: " << e.what() << std::endl;
            // Return empty matrix as fallback
            return Matrix(1, outputSize, 1);
        }

        return output;
    }

    Matrix backward(const Matrix& gradOutput, float learningRate) override {
        Matrix weightGradients(outputSize, inputSize, 1);
        std::vector<float> biasGradients(outputSize, 0.0f);

        // Compute gradients
        for (size_t i = 0; i < outputSize; ++i) {
            float gradValue = gradOutput.at(0, i, 0);
            biasGradients[i] = gradValue;

            for (size_t j = 0; j < inputSize; ++j) {
                weightGradients.at(i, j, 0) = gradValue * input.at(0, j, 0);
            }
        }

        // Compute input gradients
        Matrix flatGradient(1, inputSize, 1);
        for (size_t j = 0; j < inputSize; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < outputSize; ++i) {
                sum += weights.at(i, j, 0) * gradOutput.at(0, i, 0);
            }
            flatGradient.at(0, j, 0) = sum;
        }

        // Update weights and biases
        for (size_t i = 0; i < outputSize; ++i) {
            biases[i] -= learningRate * biasGradients[i];
            for (size_t j = 0; j < inputSize; ++j) {
                weights.at(i, j, 0) -= learningRate * weightGradients.at(i, j, 0);
            }
        }

        // If input was originally 3D (not flat), reshape the gradient back to 3D
        Matrix inputGradient;
        if (input.rows == 1 && input.cols == inputSize && input.depth == 1) {
            // Input was already flat, return flat gradient
            inputGradient = flatGradient;
        }
        else {
            // Input was 3D, need to reshape gradient to match original dimensions
            inputGradient = Matrix::reshape(flatGradient, input.rows, input.cols, input.depth);
        }

        return inputGradient;
    }
};

// Softmax Layer (Serial implementation)
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
        Matrix result(gradOutput.rows, gradOutput.cols, gradOutput.depth);
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            result.data[i] = gradOutput.data[i];
        }

        return result;
    }
};

// Serial CNN class
class SerialCNN {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Matrix forward(const Matrix& input) {
        Matrix current = input;

        for (auto& layer : layers) {
            current = layer->forward(current);
        }

        return current;
    }

    float train(const Matrix& input, const Matrix& target, float learningRate) {
        // Forward pass
        Matrix output = forward(input);

        // Compute loss
        float loss = crossEntropyLoss(output, target);

        // Compute output gradient
        Matrix gradient = crossEntropyGradient(output, target);

        // Backward pass
        for (int i = layers.size() - 1; i >= 0; --i) {
            gradient = layers[i]->backward(gradient, learningRate);
        }

        return loss;
    }

    void trainBatch(const std::vector<Matrix>& batchInputs,
        const std::vector<Matrix>& batchTargets,
        float learningRate) {
        float batchLoss = 0.0f;

        for (size_t i = 0; i < batchInputs.size(); ++i) {
            batchLoss += train(batchInputs[i], batchTargets[i], learningRate);
        }
    }

    float evaluate(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets) {
        size_t correct = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Matrix output = forward(inputs[i]);

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
                correct++;
            }
        }

        return static_cast<float>(correct) / inputs.size();
    }
};

// Dataset class for cancer detection using OpenCV
class CancerDataset {
private:
    std::vector<Matrix> images;
    std::vector<Matrix> labels;
    size_t numClasses;
    size_t imageSize;

    // Helper function to load CSV with format: id,label,filename
    std::map<std::string, int> loadCSV(const std::string& csvPath) {
        std::map<std::string, int> labelMap;
        std::ifstream file(csvPath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open CSV file: " << csvPath << std::endl;
            return labelMap;
        }

        std::string line, id, label, filename;

        // Skip header row if it exists
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);

            // Parse the three columns: id,label,filename
            std::getline(ss, id, ',');
            std::getline(ss, label, ',');
            std::getline(ss, filename, ',');

            // Remove any quotes around the filename if they exist
            if (!filename.empty() && (filename.front() == '"' && filename.back() == '"')) {
                filename = filename.substr(1, filename.length() - 2);
            }

            // Convert label to integer
            int labelValue;
            try {
                labelValue = std::stoi(label);
            }
            catch (const std::exception& e) {
                std::cerr << "Error converting label to integer for " << filename << ": " << e.what() << std::endl;
                continue;
            }

            // Use the filename as the key in our map
            labelMap[filename] = labelValue;

            // Also try with just the base filename (no path or extension)
            size_t lastSlash = filename.find_last_of("/\\");
            size_t lastDot = filename.find_last_of(".");
            if (lastSlash != std::string::npos && lastDot != std::string::npos) {
                std::string baseFilename = filename.substr(lastSlash + 1, lastDot - lastSlash - 1);
                labelMap[baseFilename] = labelValue;
            }
        }
        return labelMap;
    }

    // Helper function to find image files by trying different extensions
    std::string findImageFile(const std::string& imageFolderPath, const std::string& baseFilename) {
        // Try different extensions
        std::vector<std::string> extensions = { ".tif", ".tiff", ".jpg", ".jpeg", ".png" };

        for (const auto& ext : extensions) {
            std::string fullPath = imageFolderPath + "/" + baseFilename + ext;
            if (std::ifstream(fullPath).good()) {
                return fullPath; // File exists
            }
        }

        return ""; // File not found with any extension
    }

public:

    CancerDataset(const std::string& imageFolderPath, const std::string& labelPath,
        size_t numClasses, size_t targetSize = 32)
        : numClasses(numClasses), imageSize(targetSize) {

        // Load labels from CSV
        std::map<std::string, int> fileToLabel = loadCSV(labelPath);

        // Try different file extensions (tif, tiff, jpg, png)
        std::vector<cv::String> filenames;
        cv::glob(imageFolderPath + "/*.tif", filenames);

        if (filenames.empty()) {
            std::cout << "No .tif files found, trying .tiff..." << std::endl;
            cv::glob(imageFolderPath + "/*.tiff", filenames);
        }

        if (filenames.empty()) {
            std::cout << "No .tiff files found, trying .jpg..." << std::endl;
            cv::glob(imageFolderPath + "/*.jpg", filenames);
        }

        if (filenames.empty()) {
            std::cout << "No .jpg files found, trying .png..." << std::endl;
            cv::glob(imageFolderPath + "/*.png", filenames);
        }

        // If no files were found using glob patterns, try to find files based on CSV entries
        if (filenames.empty()) {
            std::cout << "No image files found using glob pattern. Trying to match from CSV entries..." << std::endl;

            // For each label entry, try to find a matching image file
            for (const auto& entry : fileToLabel) {
                std::string baseFilename = entry.first;

                // Remove extension if it exists
                size_t lastDot = baseFilename.find_last_of(".");
                if (lastDot != std::string::npos) {
                    baseFilename = baseFilename.substr(0, lastDot);
                }

                // Try to find the file with different extensions
                std::string imagePath = findImageFile(imageFolderPath, baseFilename);
                if (!imagePath.empty()) {
                    filenames.push_back(imagePath);
                    std::cout << "Found image for " << entry.first << ": " << imagePath << std::endl;
                }
            }
        }

        // Process each image
        for (const auto& file : filenames) {
            // Extract just the filename from the path
            std::string fullPath = file;
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string filename = fullPath.substr(lastSlash + 1);

            // Extract the base filename without extension for matching
            size_t lastDot = filename.find_last_of(".");
            std::string baseFilename = (lastDot != std::string::npos) ?
                filename.substr(0, lastDot) : filename;

            // Try several variants to match with the label file
            auto labelIt = fileToLabel.find(filename);
            if (labelIt == fileToLabel.end()) {
                // Try with just the base filename (no extension)
                labelIt = fileToLabel.find(baseFilename);
            }

            if (labelIt == fileToLabel.end()) {
                // Try just matching the end of the filename (some datasets have prefixes in the actual files)
                bool found = false;
                for (const auto& entry : fileToLabel) {
                    if (filename.find(entry.first) != std::string::npos) {
                        labelIt = fileToLabel.find(entry.first);
                        found = true;
                        std::cout << "Matched " << filename << " with label for " << entry.first << std::endl;
                        break;
                    }
                }

                if (!found) {
                    std::cout << "Warning: No label found for file " << filename << std::endl;
                    continue;
                }
            }

            // Load image using OpenCV
            cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cout << "Warning: Could not load image " << file << std::endl;
                continue;
            }

            // Resize image to target size
            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

            // Convert OpenCV Mat to our Matrix format
            Matrix imageMatrix(imageSize, imageSize, 3);
            for (size_t r = 0; r < imageSize; ++r) {
                for (size_t c = 0; c < imageSize; ++c) {
                    cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                    // OpenCV uses BGR, normalize to 0-1 range
                    imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                    imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                    imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
                }
            }

            // Create one-hot encoded label
            Matrix labelMatrix(1, numClasses, 1);
            int classId = labelIt->second;
            labelMatrix.at(0, classId, 0) = 1.0f;

            // Add to dataset
            images.push_back(imageMatrix);
            labels.push_back(labelMatrix);
        }

        std::cout << "Loaded " << images.size() << " images with labels" << std::endl;
    }


    // Split dataset into training and testing sets
    size_t splitTrainTest(std::vector<Matrix>& trainImages, std::vector<Matrix>& trainLabels,
        std::vector<Matrix>& testImages, std::vector<Matrix>& testLabels,
        float testRatio = 0.2f) {
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

        // Fill training set
        for (size_t i = 0; i < numTrain; ++i) {
            trainImages[i] = images[indices[i]];
            trainLabels[i] = labels[indices[i]];
        }

        // Fill testing set
        for (size_t i = 0; i < numTest; ++i) {
            testImages[i] = images[indices[numTrain + i]];
            testLabels[i] = labels[indices[numTrain + i]];
        }

        std::cout << "Split dataset into " << numTrain << " training and "
            << numTest << " testing samples" << std::endl;

        return numSamples;
    }
};

// Function declarations (add these at the beginning of your serialOMP.cpp file)

// Serial implementation functions
std::pair<std::chrono::duration<double>, float> serial(CancerDataset*& pDataset,
    std::vector<Matrix>& trainImages,
    std::vector<Matrix>& trainLabels,
    std::vector<Matrix>& testImages,
    std::vector<Matrix>& testLabels);

bool predictCancerImageSerial(SerialCNN& trainedModel, const std::string& imagePath);
void predictMultipleImagesSerial(SerialCNN& trainedModel);

// OpenMP implementation functions
std::pair<std::chrono::duration<double>, float> omp();
std::pair<std::pair<std::chrono::duration<double>, float>, ParallelCNN*> ompWithModel();
bool predictCancerImageOMP(ParallelCNN& trainedModel, const std::string& imagePath);
void predictMultipleImagesOMP(ParallelCNN& trainedModel);

// CUDA implementation functions
std::pair<std::chrono::duration<double>, float> cuda(CancerDataset& dataset,
    std::vector<Matrix>& trainImages,
    std::vector<Matrix>& trainLabels,
    std::vector<Matrix>& testImages,
    std::vector<Matrix>& testLabels,
    CUDACNN*& trainedCUDACNN);

bool predictCancerImageCUDA(CUDACNN& trainedModel, const std::string& imagePath);
void predictMultipleImagesCUDA(CUDACNN& trainedModel);

// Function implementations
std::pair<std::chrono::duration<double>, float> serial(CancerDataset*& pDataset,
    std::vector<Matrix>& trainImages,
    std::vector<Matrix>& trainLabels,
    std::vector<Matrix>& testImages,
    std::vector<Matrix>& testLabels) {
    std::cout << "Cancer Detection using CNN" << std::endl;
    std::cout << "===============================" << std::endl;

    // Hyperparameters
    float learningRate = 0.001f;
    size_t batchSize = 16;
    size_t numEpochs = 5;
    size_t inputSize = 32; // 32x32 pixel images (from resizing)
    size_t numChannels = 3; // RGB images
    size_t numClasses = 2; // Benign or malignant

    // Create and configure Serial CNN
    std::cout << "\nInitializing Serial CNN..." << std::endl;
    SerialCNN serialCNN;

    // Add layers to serial CNN
    serialCNN.addLayer(std::make_unique<ConvolutionalLayer>(numChannels, 16, 3, 1, 1));
    serialCNN.addLayer(std::make_unique<ReLULayer>());
    serialCNN.addLayer(std::make_unique<MaxPoolingLayer>(2));

    serialCNN.addLayer(std::make_unique<ConvolutionalLayer>(16, 32, 3, 1, 1));
    serialCNN.addLayer(std::make_unique<ReLULayer>());
    serialCNN.addLayer(std::make_unique<MaxPoolingLayer>(2));

    serialCNN.addLayer(std::make_unique<ConvolutionalLayer>(32, 64, 3, 1, 1));
    serialCNN.addLayer(std::make_unique<ReLULayer>());
    serialCNN.addLayer(std::make_unique<MaxPoolingLayer>(2));

    // After pooling, we have 64 feature maps of size 4x4
    serialCNN.addLayer(std::make_unique<FullyConnectedLayer>(64 * 4 * 4, 128));
    serialCNN.addLayer(std::make_unique<ReLULayer>());
    serialCNN.addLayer(std::make_unique<FullyConnectedLayer>(128, numClasses));
    serialCNN.addLayer(std::make_unique<SoftmaxLayer>());

    // Train and evaluate Serial CNN
    std::cout << "\n===== Training Serial CNN =====" << std::endl;
    auto serialStart = std::chrono::high_resolution_clock::now();

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

            // Train on batch and accumulate loss
            serialCNN.trainBatch(batchImages, batchLabels, learningRate);

            batchCount++;
        }

        // Evaluate accuracy on training set
        float trainAccuracy = serialCNN.evaluate(trainImages, trainLabels) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs
            << ", Training Accuracy: " << trainAccuracy << "%" << std::endl;
    }

    auto serialEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serialDuration = serialEnd - serialStart;

    // Print performance
    std::cout << "\n===== Performance =====" << std::endl;
    std::cout << "Serial CNN execution time: " << serialDuration.count() << " seconds" << std::endl;

    // Final evaluation
    std::cout << "\n===== Final Model Evaluation =====" << std::endl;
    float  serialAccuracy = serialCNN.evaluate(testImages, testLabels) * 100.0f;
    std::cout << "Serial CNN accuracy: " << serialAccuracy << "%" << std::endl;

    return { serialDuration, serialAccuracy };
}

// Serial prediction function
bool predictCancerImageSerial(SerialCNN& trainedModel, const std::string& imagePath) {
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

        // Convert OpenCV Mat to our Matrix format
        Matrix imageMatrix(imageSize, imageSize, 3);
        for (size_t r = 0; r < imageSize; ++r) {
            for (size_t c = 0; c < imageSize; ++c) {
                cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                // OpenCV uses BGR, normalize to 0-1 range
                imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
            }
        }

        // Run forward pass through the trained model
        Matrix output = trainedModel.forward(imageMatrix);

        // Get the predicted class
        float benignProb = output.at(0, 0, 0);  // Probability of benign
        float malignantProb = output.at(0, 1, 0); // Probability of malignant

        // Print results
        std::cout << "\n===== Serial Model Prediction Results =====" << std::endl;
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
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        return false;
    }
}


void predictMultipleImagesSerial(SerialCNN& trainedModel) {
    std::string imagePath;

    while (true) {
        std::cout << "\nEnter image path for Serial model prediction (or 'q' to quit): ";
        std::cin >> imagePath;

        if (imagePath == "q" || imagePath == "Q") {
            break;
        }

        predictCancerImageSerial(trainedModel, imagePath);
    }
}

// CUDA prediction function
bool predictCancerImageCUDA(CUDACNN& trainedModel, const std::string& imagePath) {
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

        // Convert OpenCV Mat to our Matrix format
        Matrix imageMatrix(imageSize, imageSize, 3);
        for (size_t r = 0; r < imageSize; ++r) {
            for (size_t c = 0; c < imageSize; ++c) {
                cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                // OpenCV uses BGR, normalize to 0-1 range
                imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
            }
        }

        // Run forward pass through the trained CUDA model
        Matrix output = trainedModel.forward(imageMatrix);

        // Get the predicted class
        float benignProb = output.at(0, 0, 0);  // Probability of benign
        float malignantProb = output.at(0, 1, 0); // Probability of malignant

        // Print results
        std::cout << "\n===== CUDA Model Prediction Results =====" << std::endl;
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
        std::cout << "========================================\n" << std::endl;

        return true;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during CUDA prediction: " << e.what() << std::endl;
        return false;
    }
}

void predictMultipleImagesCUDA(CUDACNN& trainedModel) {
    std::string imagePath;

    while (true) {
        std::cout << "\nEnter image path for CUDA model prediction (or 'q' to quit): ";
        std::cin >> imagePath;

        if (imagePath == "q" || imagePath == "Q") {
            break;
        }

        predictCancerImageCUDA(trainedModel, imagePath);
    }
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Function to detect available datasets
std::vector<std::pair<int, std::string>> detectAvailableDatasets(const std::string& baseFolder) {
    std::vector<std::pair<int, std::string>> datasets;

    try {
        // Check if base folder exists
        if (!std::filesystem::exists(baseFolder)) {
            std::cerr << "Base folder doesn't exist: " << baseFolder << std::endl;
            return datasets;
        }

        // Iterate through directories in the base folder
        for (const auto& entry : std::filesystem::directory_iterator(baseFolder)) {
            if (entry.is_directory()) {
                std::string dirName = entry.path().filename().string();

                // Check for data folders with numeric size in the name
                if (dirName.find("data") == 0) {
                    std::string sizeStr = dirName.substr(4); // Skip "data" prefix
                    try {
                        int size = std::stoi(sizeStr);

                        // Verify that this folder has the expected structure
                        std::string imagesPath = entry.path().string() + "/sampled_images";
                        std::string labelsPath = entry.path().string() + "/sampled_labels.csv";

                        if (std::filesystem::exists(imagesPath) &&
                            std::filesystem::exists(labelsPath)) {
                            datasets.push_back({ size, entry.path().string() });
                        }
                    }
                    catch (...) {
                        // Not a valid size number, ignore this folder
                    }
                }
            }
        }

        // Sort datasets by size
        std::sort(datasets.begin(), datasets.end());

    }
    catch (const std::exception& e) {
        std::cerr << "Error detecting datasets: " << e.what() << std::endl;
    }

    return datasets;
}

// Common function to write prediction results to CSV
void writePredictionResultsToCSV(const std::string& outputFile,
    const std::vector<std::string>& imageFiles,
    const std::vector<float>& benignProbs,
    const std::vector<float>& malignantProbs,
    const std::vector<bool>& predictions) {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create prediction results file: " << outputFile << std::endl;
        return;
    }

    // Write header
    file << "Image,Benign_Probability,Malignant_Probability,Prediction\n";

    // Write data for each image
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        // Extract just the filename without the path
        std::string filename = imageFiles[i];
        size_t lastSlash = filename.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            filename = filename.substr(lastSlash + 1);
        }

        file << filename << ","
            << benignProbs[i] << ","
            << malignantProbs[i] << ","
            << (predictions[i] ? "Malignant" : "Benign") << "\n";
    }

    file.close();
    std::cout << "Prediction results saved to " << outputFile << std::endl;
}

// Serial model batch prediction
void batchPredictSerial(SerialCNN& model, const std::string& imageFolder, const std::string& outputFile) {
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
    std::vector<bool> predictions;  // true for malignant, false for benign

    // Process each image
    size_t processedCount = 0;
    size_t totalImages = imageFiles.size();

    for (const auto& imagePath : imageFiles) {
        try {
            // Load and preprocess image
            cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not load image " << imagePath << std::endl;
                continue;
            }

            // Resize image
            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

            // Convert to Matrix format
            Matrix imageMatrix(imageSize, imageSize, 3);
            for (size_t r = 0; r < imageSize; ++r) {
                for (size_t c = 0; c < imageSize; ++c) {
                    cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                    imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                    imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                    imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
                }
            }

            // Run prediction
            Matrix output = model.forward(imageMatrix);
            float benignProb = output.at(0, 0, 0);
            float malignantProb = output.at(0, 1, 0);
            bool isMalignant = (malignantProb > MALIGNANT_THRESHOLD);

            // Store results
            benignProbs.push_back(benignProb);
            malignantProbs.push_back(malignantProb);
            predictions.push_back(isMalignant);

            // Update progress
            processedCount++;
            if (processedCount % 100 == 0 || processedCount == totalImages) {
                std::cout << "Progress: " << processedCount << "/" << totalImages
                    << " (" << (processedCount * 100 / totalImages) << "%)" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing image " << imagePath << ": " << e.what() << std::endl;
        }
    }

    // Write results to CSV
    writePredictionResultsToCSV(outputFile, imageFiles, benignProbs, malignantProbs, predictions);

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

// CUDA model batch prediction
void batchPredictCUDA(CUDACNN& model, const std::string& imageFolder, const std::string& outputFile) {
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

    // Process each image
    size_t processedCount = 0;
    size_t totalImages = imageFiles.size();

    // Batch processing for CUDA is more efficient
    const size_t batchSize = 16;  // Can be adjusted based on GPU memory

    for (size_t batchStart = 0; batchStart < imageFiles.size(); batchStart += batchSize) {
        size_t currentBatchSize = std::min(batchSize, imageFiles.size() - batchStart);
        std::vector<Matrix> batchImages(currentBatchSize);

        // Load and preprocess batch
        for (size_t i = 0; i < currentBatchSize; ++i) {
            try {
                const std::string& imagePath = imageFiles[batchStart + i];

                // Load image
                cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Warning: Could not load image " << imagePath << std::endl;
                    // Use a blank image instead to maintain batch size
                    img = cv::Mat::zeros(imageSize, imageSize, CV_8UC3);
                }

                // Resize image
                cv::Mat resizedImg;
                cv::resize(img, resizedImg, cv::Size(imageSize, imageSize));

                // Convert to Matrix format
                Matrix imageMatrix(imageSize, imageSize, 3);
                for (size_t r = 0; r < imageSize; ++r) {
                    for (size_t c = 0; c < imageSize; ++c) {
                        cv::Vec3b pixel = resizedImg.at<cv::Vec3b>(r, c);
                        imageMatrix.at(r, c, 0) = pixel[2] / 255.0f; // R
                        imageMatrix.at(r, c, 1) = pixel[1] / 255.0f; // G
                        imageMatrix.at(r, c, 2) = pixel[0] / 255.0f; // B
                    }
                }

                batchImages[i] = imageMatrix;
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing image " << imageFiles[batchStart + i] << ": " << e.what() << std::endl;
                // Use a blank image instead to maintain batch size
                Matrix blankMatrix(imageSize, imageSize, 3); // All zeros by default
                batchImages[i] = blankMatrix;
            }
        }

        // Run batch prediction
        for (size_t i = 0; i < currentBatchSize; ++i) {
            Matrix output = model.forward(batchImages[i]);
            float benignProb = output.at(0, 0, 0);
            float malignantProb = output.at(0, 1, 0);

            benignProbs.push_back(benignProb);
            malignantProbs.push_back(malignantProb);
            predictions.push_back(malignantProb > MALIGNANT_THRESHOLD);
        }

        // Update progress
        processedCount += currentBatchSize;
        std::cout << "Progress: " << processedCount << "/" << totalImages
            << " (" << (processedCount * 100 / totalImages) << "%)" << std::endl;
    }

    // Write results to CSV
    writePredictionResultsToCSV(outputFile, imageFiles, benignProbs, malignantProbs, predictions);

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

int main() {
    std::pair<std::chrono::duration<double>, float> Serialresult;
    std::pair<std::chrono::duration<double>, float> OMPresult;
    std::pair<std::chrono::duration<double>, float> CUDAresult;
    double speedupOMP, speedupCUDA;

    // We'll create a shared dataset for all implementations
    CancerDataset* pDataset = nullptr;
    std::vector<Matrix> trainImages, trainLabels, testImages, testLabels;

    // Track trained models
    SerialCNN* trainedSerialCNN = nullptr;
    ParallelCNN* trainedOMPCNN = nullptr; // Add OpenMP model
    CUDACNN* trainedCUDACNN = nullptr;
    bool isSerialModelTrained = false;
    bool isOMPModelTrained = false; // Add OpenMP flag
    bool isCUDAModelTrained = false;

    std::string baseDatasetPath = "D:/TARUMT/DSPC/imageDataset";
    std::string imageFolderPath = "";
    std::string labelPath = "";
    int selectedDatasetSize = 0;

    // Detect available datasets
    std::cout << "Scanning for available datasets in " << baseDatasetPath << "..." << std::endl;
    auto availableDatasets = detectAvailableDatasets(baseDatasetPath);

    if (availableDatasets.empty()) {
        std::cerr << "Error: No valid datasets found in " << baseDatasetPath << std::endl;
        std::cerr << "Please ensure datasets are organized as data<SIZE>/sampled_images and data<SIZE>/sampled_labels.csv" << std::endl;
        return 1;
    }

    std::cout << "Found " << availableDatasets.size() << " datasets:" << std::endl;
    for (const auto& dataset : availableDatasets) {
        std::cout << "- Dataset with " << dataset.first << " images" << std::endl;
    }

    while (true) {
        int choice;
        std::cout << "\n=== Cancer Detection CNN Implementation ===" << std::endl;

        // Show current dataset if selected
        if (selectedDatasetSize > 0) {
            std::cout << "Current dataset: " << selectedDatasetSize << " images" << std::endl;
        }
        else {
            std::cout << "No dataset selected" << std::endl;
        }

        std::cout << "0. Select Dataset" << std::endl;
        std::cout << "1. Run Serial Implementation" << std::endl;
        std::cout << "2. Run OpenMP Implementation" << std::endl;
        std::cout << "3. Run CUDA Implementation" << std::endl;
        std::cout << "4. Show Comparison" << std::endl;
        std::cout << "5. Predict Cancer from Uploaded Image" << std::endl;
        std::cout << "6. Exit" << std::endl;
        std::cout << "Your choice: ";
        std::cin >> choice;

        switch (choice) {
        case 0: {
            // Dataset selection
            std::cout << "\nAvailable datasets:" << std::endl;
            for (size_t i = 0; i < availableDatasets.size(); ++i) {
                std::cout << i + 1 << ". Dataset with " << availableDatasets[i].first
                    << " images" << std::endl;
            }

            int datasetChoice;
            std::cout << "Select dataset (1-" << availableDatasets.size() << "): ";
            std::cin >> datasetChoice;

            if (datasetChoice < 1 || datasetChoice > availableDatasets.size()) {
                std::cout << "Invalid selection." << std::endl;
            }
            else {
                // Update selected dataset
                selectedDatasetSize = availableDatasets[datasetChoice - 1].first;
                std::string selectedDatasetPath = availableDatasets[datasetChoice - 1].second;

                imageFolderPath = selectedDatasetPath + "/sampled_images";
                labelPath = selectedDatasetPath + "/sampled_labels.csv";

                std::cout << "Selected dataset with " << selectedDatasetSize << " images" << std::endl;
                std::cout << "Images: " << imageFolderPath << std::endl;
                std::cout << "Labels: " << labelPath << std::endl;

                // If models were already trained, they need to be retrained with new dataset
                if (isSerialModelTrained || isOMPModelTrained || isCUDAModelTrained) {
                    std::cout << "Warning: Changing dataset requires retraining models." << std::endl;

                    // Clean up existing models and dataset
                    if (pDataset) {
                        delete pDataset;
                        pDataset = nullptr;
                    }
                    if (trainedSerialCNN) {
                        delete trainedSerialCNN;
                        trainedSerialCNN = nullptr;
                        isSerialModelTrained = false;
                    }
                    if (trainedOMPCNN) {
                        delete trainedOMPCNN;
                        trainedOMPCNN = nullptr;
                        isOMPModelTrained = false;
                    }
                    if (trainedCUDACNN) {
                        delete trainedCUDACNN;
                        trainedCUDACNN = nullptr;
                        isCUDAModelTrained = false;
                    }

                    // Clear vectors
                    trainImages.clear();
                    trainLabels.clear();
                    testImages.clear();
                    testLabels.clear();
                }

                // Load the selected dataset
                std::cout << "Loading cancer detection dataset from " << imageFolderPath << "..." << std::endl;
                pDataset = new CancerDataset(imageFolderPath, labelPath, 2); // 2 classes: benign and malignant

                size_t imageCount = pDataset->splitTrainTest(trainImages, trainLabels, testImages, testLabels);
                std::cout << "Dataset loaded and split successfully!\n" << std::endl;
            }
            break;
        }

        case 1: { // Serial Implementation
            if (selectedDatasetSize == 0) {
                std::cout << "Please select a dataset first (option 0)." << std::endl;
                break;
            }

            std::cout << "\nRunning Serial Implementation...\n" << std::endl;

            // Configure the CNN
            trainedSerialCNN = new SerialCNN();
            size_t numChannels = 3;

            // Add layers
            trainedSerialCNN->addLayer(std::make_unique<ConvolutionalLayer>(numChannels, 16, 3, 1, 1));
            trainedSerialCNN->addLayer(std::make_unique<ReLULayer>());
            trainedSerialCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

            trainedSerialCNN->addLayer(std::make_unique<ConvolutionalLayer>(16, 32, 3, 1, 1));
            trainedSerialCNN->addLayer(std::make_unique<ReLULayer>());
            trainedSerialCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

            trainedSerialCNN->addLayer(std::make_unique<ConvolutionalLayer>(32, 64, 3, 1, 1));
            trainedSerialCNN->addLayer(std::make_unique<ReLULayer>());
            trainedSerialCNN->addLayer(std::make_unique<MaxPoolingLayer>(2));

            trainedSerialCNN->addLayer(std::make_unique<FullyConnectedLayer>(64 * 4 * 4, 128));
            trainedSerialCNN->addLayer(std::make_unique<ReLULayer>());
            trainedSerialCNN->addLayer(std::make_unique<FullyConnectedLayer>(128, 2));
            trainedSerialCNN->addLayer(std::make_unique<SoftmaxLayer>());

            // Train the model
            float learningRate = 0.001f;
            size_t batchSize = 16;
            size_t numEpochs = 5;

            auto serialStart = std::chrono::high_resolution_clock::now();

            for (size_t epoch = 0; epoch < numEpochs; ++epoch) {
                for (size_t batchStart = 0; batchStart < trainImages.size(); batchStart += batchSize) {
                    size_t currentBatchSize = std::min(batchSize, trainImages.size() - batchStart);
                    std::vector<Matrix> batchImages(currentBatchSize);
                    std::vector<Matrix> batchLabels(currentBatchSize);

                    for (size_t i = 0; i < currentBatchSize; ++i) {
                        batchImages[i] = trainImages[batchStart + i];
                        batchLabels[i] = trainLabels[batchStart + i];
                    }

                    trainedSerialCNN->trainBatch(batchImages, batchLabels, learningRate);
                }

                float trainAccuracy = trainedSerialCNN->evaluate(trainImages, trainLabels) * 100.0f;
                std::cout << "Epoch " << epoch + 1 << "/" << numEpochs
                    << ", Training Accuracy: " << trainAccuracy << "%" << std::endl;
            }

            auto serialEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> serialDuration = serialEnd - serialStart;
            float serialAccuracy = trainedSerialCNN->evaluate(testImages, testLabels) * 100.0f;

            Serialresult = { serialDuration, serialAccuracy };
            isSerialModelTrained = true;

            std::cout << "Serial CNN execution time: " << serialDuration.count() << " seconds" << std::endl;
            std::cout << "Serial CNN accuracy: " << serialAccuracy << "%" << std::endl;
            break;
        }

        case 2: { // OpenMP Implementation
            if (selectedDatasetSize == 0) {
                std::cout << "Please select a dataset first (option 0)." << std::endl;
                break;
            }

            std::cout << "\nRunning OpenMP Parallel Implementation...\n" << std::endl;

            // Use the existing function signature
            auto [result, modelPtr] = ompWithModel(imageFolderPath, labelPath);
            OMPresult = result;
            trainedOMPCNN = modelPtr;
            isOMPModelTrained = true;
            break;
        }

        case 3: { // CUDA Implementation
            if (selectedDatasetSize == 0) {
                std::cout << "Please select a dataset first (option 0)." << std::endl;
                break;
            }

            std::cout << "\nRunning CUDA Implementation...\n" << std::endl;

            // Use the existing function signature
            CUDAresult = cuda(*pDataset, trainImages, trainLabels, testImages, testLabels, trainedCUDACNN);
            isCUDAModelTrained = true;
            break;
        }

        case 4: { // Show Comparison
            if (!isSerialModelTrained && !isOMPModelTrained && !isCUDAModelTrained) {
                std::cout << "\nError: No models have been trained. Please run at least one implementation first." << std::endl;
                break;
            }

            std::cout << "\n===== Performance Comparison =====" << std::endl;

            if (isSerialModelTrained) {
                std::cout << "Serial Implementation:" << std::endl;
                std::cout << "  - Execution time: " << Serialresult.first.count() << " seconds" << std::endl;
                std::cout << "  - Accuracy: " << Serialresult.second << "%" << std::endl;
            }

            if (isOMPModelTrained) {
                std::cout << "OpenMP Implementation:" << std::endl;
                std::cout << "  - Execution time: " << OMPresult.first.count() << " seconds" << std::endl;
                std::cout << "  - Accuracy: " << OMPresult.second << "%" << std::endl;
            }

            if (isCUDAModelTrained) {
                std::cout << "CUDA Implementation:" << std::endl;
                std::cout << "  - Execution time: " << CUDAresult.first.count() << " seconds" << std::endl;
                std::cout << "  - Accuracy: " << CUDAresult.second << "%" << std::endl;
            }

            if (isSerialModelTrained) {
                // Calculate speedups only if serial model is available
                if (isOMPModelTrained) {
                    speedupOMP = Serialresult.first.count() / OMPresult.first.count();
                    std::cout << "\nOpenMP Performance Gain: " << speedupOMP << "x" << std::endl;
                    std::cout << "OpenMP Accuracy difference: " << (OMPresult.second - Serialresult.second) << "%" << std::endl;
                }

                if (isCUDAModelTrained) {
                    speedupCUDA = Serialresult.first.count() / CUDAresult.first.count();
                    std::cout << "CUDA Performance Gain: " << speedupCUDA << "x" << std::endl;
                    std::cout << "CUDA Accuracy difference: " << (CUDAresult.second - Serialresult.second) << "%" << std::endl;
                }
            }

            // Generate benchmark CSV file for the Python script to use
            std::ofstream file("benchmark.csv");

            if (file.is_open()) {
                // Write the header
                file << "Implementation,Execution Time (s),Accuracy (%),Speedup (vs Serial),Accuracy Difference (%),Dataset Size\n";

                // Write the data for each trained model
                if (isSerialModelTrained) {
                    file << "Serial,"
                        << Serialresult.first.count() << ","
                        << Serialresult.second << ","
                        << "1,"
                        << "0,"
                        << selectedDatasetSize << "\n";
                }

                if (isOMPModelTrained) {
                    float speedup = isSerialModelTrained ? Serialresult.first.count() / OMPresult.first.count() : 0;
                    float accDiff = isSerialModelTrained ? (OMPresult.second - Serialresult.second) : 0;

                    file << "OpenMP,"
                        << OMPresult.first.count() << ","
                        << OMPresult.second << ","
                        << speedup << ","
                        << accDiff << ","
                        << selectedDatasetSize << "\n";
                }

                if (isCUDAModelTrained) {
                    float speedup = isSerialModelTrained ? Serialresult.first.count() / CUDAresult.first.count() : 0;
                    float accDiff = isSerialModelTrained ? (CUDAresult.second - Serialresult.second) : 0;

                    file << "CUDA,"
                        << CUDAresult.first.count() << ","
                        << CUDAresult.second << ","
                        << speedup << ","
                        << accDiff << ","
                        << selectedDatasetSize << "\n";
                }

                file.close();
                std::cout << "\nBenchmark results have been saved to benchmark.csv" << std::endl;
                std::cout << "Run the Python script to generate plots and visualizations." << std::endl;
            }
            else {
                std::cerr << "Error: Could not open benchmark.csv for writing!" << std::endl;
            }
            break;
        }

              // Modify the case 5 in the switch statement in main() function to include batch prediction

        case 5: { // Cancer Image Prediction
            if (!isSerialModelTrained && !isOMPModelTrained && !isCUDAModelTrained) {
                std::cout << "\nError: No model has been trained. Please run option 1, 2, or 3 first." << std::endl;
                break;
            }

            std::cout << "\n===== Cancer Image Prediction =====" << std::endl;
            std::cout << "1. Predict Single Image" << std::endl;
            std::cout << "2. Predict Batch of Images (Folder)" << std::endl;
            std::cout << "3. Back to Main Menu" << std::endl;
            std::cout << "Your choice: ";

            int predictionChoice;
            std::cin >> predictionChoice;

            if (predictionChoice == 1) {
                // Single image prediction - existing code
                std::cout << "Which model would you like to use?" << std::endl;

                if (isSerialModelTrained) {
                    std::cout << "1. Serial Model" << std::endl;
                }
                if (isOMPModelTrained) {
                    std::cout << "2. OpenMP Model" << std::endl;
                }
                if (isCUDAModelTrained) {
                    std::cout << "3. CUDA Model" << std::endl;
                }

                int modelChoice;
                std::cin >> modelChoice;

                std::string imagePath;
                std::cout << "Enter path to image: ";
                std::cin >> imagePath;

                if (modelChoice == 1 && isSerialModelTrained) {
                    predictCancerImageSerial(*trainedSerialCNN, imagePath);
                }
                else if (modelChoice == 2 && isOMPModelTrained) {
                    predictCancerImageOMP(*trainedOMPCNN, imagePath);
                }
                else if (modelChoice == 3 && isCUDAModelTrained) {
                    predictCancerImageCUDA(*trainedCUDACNN, imagePath);
                }
                else {
                    std::cout << "Invalid model selection." << std::endl;
                }
            }
            else if (predictionChoice == 2) {
                // Batch prediction (folder of images)
                std::cout << "Which model would you like to use?" << std::endl;

                if (isSerialModelTrained) {
                    std::cout << "1. Serial Model" << std::endl;
                }
                if (isOMPModelTrained) {
                    std::cout << "2. OpenMP Model" << std::endl;
                }
                if (isCUDAModelTrained) {
                    std::cout << "3. CUDA Model" << std::endl;
                }

                int modelChoice;
                std::cin >> modelChoice;

                std::string imageFolder;
                std::cout << "Enter path to folder containing images: ";
                std::cin >> imageFolder;

                // Check if folder exists
                if (!std::filesystem::exists(imageFolder) || !std::filesystem::is_directory(imageFolder)) {
                    std::cout << "Error: The specified folder does not exist or is not a directory." << std::endl;
                    break;
                }

                std::string outputFile;
                std::cout << "Enter output CSV filename: ";
                std::cin >> outputFile;

                std::cout << "\nStarting batch prediction process. This may take a while..." << std::endl;

                auto start = std::chrono::high_resolution_clock::now();

                // Call the appropriate batch prediction function based on model choice
                if (modelChoice == 1 && isSerialModelTrained) {
                    // Call the Serial batch prediction function
                    batchPredictSerial(*trainedSerialCNN, imageFolder, outputFile);
                }
                else if (modelChoice == 2 && isOMPModelTrained) {
                    // Call the OMP batch prediction function
                    batchPredictOMP(*trainedOMPCNN, imageFolder, outputFile);
                }
                else if (modelChoice == 3 && isCUDAModelTrained) {
                    // Call the CUDA batch prediction function
                    batchPredictCUDA(*trainedCUDACNN, imageFolder, outputFile);
                }
                else {
                    std::cout << "Invalid model selection." << std::endl;
                    break;
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;

                std::cout << "Batch prediction completed in " << duration.count() << " seconds." << std::endl;
                std::cout << "To visualize the results, run the Python script with the prediction output file." << std::endl;
            }
            break;
        }
        case 6: { // Exit
            std::cout << "Exiting program..." << std::endl;
            // Clean up
            if (pDataset) {
                delete pDataset;
            }
            if (trainedSerialCNN) {
                delete trainedSerialCNN;
            }
            if (trainedOMPCNN) {
                delete trainedOMPCNN;
            }
            if (trainedCUDACNN) {
                delete trainedCUDACNN;
            }
            return 0;
        }

        default:
            std::cout << "Invalid choice." << std::endl;
        }
    }

    return 0;
}

 


#pragma once
#include <vector>
#include <memory>
#include "common.h"
#include "Matrix.h"
#include "Layer.h"

// CUDA CNN class
class CUDACNN {
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
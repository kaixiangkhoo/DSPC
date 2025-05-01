#ifndef COMMON_H
#define COMMON_H

#include "Matrix.h"
#include "Layer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <memory>

// Cross-entropy loss function
inline float crossEntropyLoss(const Matrix& output, const Matrix& target) {
    float loss = 0.0f;

    for (size_t i = 0; i < output.data.size(); ++i) {
        // Clip predictions to avoid log(0)
        float pred = std::max(std::min(output.data[i], 1.0f - 1e-7f), 1e-7f);
        loss -= target.data[i] * std::log(pred);
    }

    return loss;
}

// Derivative of cross-entropy loss with respect to softmax output
inline Matrix crossEntropyGradient(const Matrix& output, const Matrix& target) {
    // For softmax + cross-entropy, gradient is (output - target)
    return output - target;
}

#endif // COMMON_H
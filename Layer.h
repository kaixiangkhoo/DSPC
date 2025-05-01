#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"

class Layer {
public:
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradOutput, float learningRate) = 0;
    virtual ~Layer() {}
};

#endif // LAYER_H
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>

class Matrix {
public:
    std::vector<float> data;
    size_t rows, cols, depth;

    Matrix() : rows(0), cols(0), depth(0) {}

    Matrix(size_t rows, size_t cols, size_t depth)
        : rows(rows), cols(cols), depth(depth) {
        size_t size = rows * cols * depth;
        if (size == 0) {
            throw std::invalid_argument("Cannot create matrix with zero elements");
        }
        data.resize(size, 0.0f);
    }

    float& at(size_t row, size_t col, size_t d) {
        size_t index = d * (rows * cols) + row * cols + col;
        if (index >= data.size()) {
            throw std::out_of_range("Matrix subscript out of range");
        }
        return data[index];
    }

    float at(size_t row, size_t col, size_t d) const {
        size_t index = d * (rows * cols) + row * cols + col;
        if (index >= data.size()) {
            std::cout << "Out of bounds access: [" << row << ", " << col << ", " << d
                << "] in matrix of size [" << rows << ", " << cols << ", " << depth << "]" << std::endl;
            throw std::out_of_range("Matrix subscript out of range");
        }
        return data[index];
    }

    // Flatten a 3D matrix to a 1D vector (for fully connected layers)
    Matrix flatten() const {
        size_t totalElements = rows * cols * depth;
        Matrix result(1, totalElements, 1);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i];
        }
        return result;
    }

    // Create a matrix from a flattened vector with specified dimensions
    static Matrix reshape(const Matrix& flat, size_t rows, size_t cols, size_t depth) {
        Matrix result(rows, cols, depth);
        size_t totalElements = rows * cols * depth;

        if (flat.data.size() != totalElements) {
            std::cout << "ERROR in reshape: Source has " << flat.data.size()
                << " elements but destination requires " << totalElements << std::endl;
            throw std::runtime_error("Reshape dimensions don't match data size");
        }

        // Copy the data directly
        result.data = flat.data;

        return result;
    }

    // Fill matrix with random values
    void randomize(float min = -0.1f, float max = 0.1f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
        }
    }

    // Element-wise operations
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols || depth != other.depth) {
            throw std::runtime_error("Matrix dimensions don't match for addition");
        }

        Matrix result(rows, cols, depth);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols || depth != other.depth) {
            throw std::runtime_error("Matrix dimensions don't match for subtraction");
        }

        Matrix result(rows, cols, depth);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols || depth != other.depth) {
            throw std::runtime_error("Matrix dimensions don't match for Hadamard product");
        }

        Matrix result(rows, cols, depth);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(float scalar) const {
        Matrix result(rows, cols, depth);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Matrix multiplication (for fully connected layers)
    Matrix matmul(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }

        Matrix result(rows, other.cols, 1);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    sum += at(i, k, 0) * other.at(k, j, 0);
                }
                result.at(i, j, 0) = sum;
            }
        }
        return result;
    }

    // Transpose operation (for fully connected layer backpropagation)
    Matrix transpose() const {
        Matrix result(cols, rows, depth);
        for (size_t d = 0; d < depth; ++d) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result.at(j, i, d) = at(i, j, d);
                }
            }
        }
        return result;
    }

    // Sum all elements
    float sum() const {
        float total = 0.0f;
        for (const auto& val : data) {
            total += val;
        }
        return total;
    }
};

#endif // MATRIX_H
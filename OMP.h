
#ifndef OMP_CNN_H
#define OMP_CNN_H

// Forward declarations of classes
class OMPMatrix;
class Layer;
class ConvolutionalLayer;
class MaxPoolingLayer;
class ReLULayer;
class FullyConnectedLayer;
class SoftmaxLayer;
class ParallelCNN;
class CancerDataset;

// Function declaration for the main test function
std::pair<std::pair<std::chrono::duration<double>, float>, ParallelCNN*> ompWithModel(std::string ImagePath, std::string LabelPath);
void batchPredictOMP(ParallelCNN& model, const std::string& imageFolder, const std::string& outputFile);

#endif // OMP_CNN_H

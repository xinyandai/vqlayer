#include <iostream>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "include/network.h"
using namespace std;


Network::Network( int *sizesOfLayers, const vector<Activation >& layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,
                  int* K, int* L, int* RangePow, float* Sparsity) {
  layer_size_ = sizesOfLayers;
  num_layers_ = noOfLayers;
  batch_size_ = batchSize;
  input_dim_ = inputdim;
  optimizer_.lr = lr;
  layer_.emplace_back(input_dim_, layer_size_[0], layersTypes[0]);
  for (int i = 1; i < num_layers_-1; ++i) {
    layer_.emplace_back(layer_size_[i-1], layer_size_[i], layersTypes[i]);
  }
}

Network::~Network() = default;

int Network::predictClass(int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;
    return correctPred;
}


float Network::ProcessInput( int **inputIndices, float **inputValues, int *lengths,
                             int **labels, int *labelsize, int iter) {
  // forward pass for one sample not batched
  vector<SparseVector > activations ((size_t)num_layers_ + 1);
  vector<int > labels_;
  // construct from input
  // TODO  activations[0] = ...
  for (int i = 0; i < num_layers_; ++i) {
    activations[i+1] = layer_[i].forward(activations[i]) ;
  }
  float loss;
  // compute loss
  SparseVector grad = SoftMaxCrossEntropy::compute(activations[num_layers_], labels_, loss);
  // backward
  for (int i = num_layers_-1; i > 0; --i) {
    grad = layer_[i].backward(grad, activations[i+1], activations[i], optimizer_, i!=0);
  }
  return loss;
}


void Network::saveWeights(string file) {
}



#include <iostream>
#include <math.h>
#include <algorithm>
#include <limits>
#include <omp.h>
#include "../include/network.h"
using namespace std;


Network::Network( int *sizesOfLayers, vector<Activation >& layersTypes,
                  int noOfLayers, int batchSize, float lr,
                  int inputdim, int* K, int* L, int*
                  RangePow, float* Sparsity) {
  layer_size_ = sizesOfLayers;
  num_layers_ = noOfLayers;
  batch_size_ = batchSize;
  input_dim_ = inputdim;
  optimizer_.lr = lr;
  layer_.reserve(num_layers_);
  layer_.emplace_back(input_dim_, layer_size_[0], layersTypes[0]);
  for (int i = 1; i < num_layers_; ++i) {
    layer_.emplace_back(layer_size_[i-1], layer_size_[i], layersTypes[i]);
  }
}

Network::~Network() = default;

int Network::predictClass(int **inputIndices, float **inputValues,
                          int *lengths, int **labels, int *labelsize) {
  std::cout << "predict \t";
  int correctPred = 0;

#pragma omp parallel for reduction(+:correctPred)
  for (int b = 0; b < batch_size_; ++b) {
    // construct from input
    SparseVector activation = SparseVector(inputIndices[b], inputValues[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b]+labelsize[b]);
    // forward pass for one sample
    for (int i = 0; i < num_layers_; ++i) {
      activation = layer_[i].forward(activation) ;
    }
    T max_act = std::numeric_limits<float >::min();
    int predict_class = activation.index_[0];
    for (int k = 0; k < activation.size(); k++) {
      T cur_act = activation.value_[k];
      if (max_act < cur_act) {
        max_act = cur_act;
        predict_class = activation.index_[k];
      }
    }

    if (std::find (labels[b], labels[b]+labelsize[b], predict_class)!= labels[b]+labelsize[b]) {
      correctPred++;
    }
  }
  return correctPred;
}


float Network::ProcessInput(int **inputIndices, float **inputValues,
                            int *lengths, int **labels,
                            int *labelsize, int iter) {
// std::cout << "training\t";
  float loss = 0;
#pragma omp parallel for reduction(+:loss)
  for (int b = 0; b < batch_size_; ++b) {
    vector<SparseVector > activations ((size_t)num_layers_ + 1);
    // construct from input
    activations[0] = SparseVector(inputIndices[b], inputValues[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b]+labelsize[b]);
    // forward pass for one sample
    for (int i = 0; i < num_layers_; ++i) {
      activations[i+1] = layer_[i].forward(activations[i]) ;
    }
    // compute loss
    float loss_b = 0;
    // gradient with respect to last layer output(pre SoftMax)
    SparseVector grad = SoftMaxCrossEntropy::compute(
        activations[num_layers_], labels_, &loss_b);
    loss += loss_b;
    // backward and update
    for (int i = num_layers_-1; i >= 0; --i) {
      grad = layer_[i].backward(grad, activations[i], optimizer_, i!=0);
      activations[i].clear();
    }
  }
//  std::cout << loss << std::endl;
  return loss;
}


void Network::saveWeights(string file) {
}



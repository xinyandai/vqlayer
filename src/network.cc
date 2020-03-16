#include <iostream>
#include <math.h>
#include <algorithm>
#include <limits>
#include <omp.h>
#include "../include/network.h"
using namespace std;


Network::Network(int *layer_size, int num_layers, int batch_size,
                 const Optimizer& optimizer, int input_dim)
                 : optimizer_(optimizer){
  layer_size_ = layer_size;
  num_layers_ = num_layers;
  batch_size_ = batch_size;
  input_dim_ = input_dim;
  layer_.reserve(num_layers_);
  std::cout << "building layer " << input_dim
            << " x " << layer_size_[0] << std::endl;
  layer_.emplace_back(new Layer<ReLu, false>(
            input_dim_, layer_size_[0]));
  int i = 1;
  for (; i < num_layers_-1; ++i) {
    std::cout << "building layer " << layer_size_[i-1]
              << " x " << layer_size_[i] << std::endl;
    layer_.emplace_back(new Layer<ReLu, false>(
              layer_size_[i-1], layer_size_[i]));
  }
  std::cout << "building layer " << layer_size_[i-1]
            << " x " << layer_size_[i] << std::endl;
  layer_.emplace_back(new PQLayer<SoftMax, true, true>(
            layer_size_[i-1], layer_size_[i]));
  std::cout << "building network, done" << std::endl;
}

Network::~Network() = default;

int Network::predict(int **input_indices, float **input_values,
                     int *lengths, int **labels, int *labelsize) {
//  std::cout << "predict \t";
  int correctPred = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+:correctPred)
#endif
  for (int b = 0; b < batch_size_; ++b) {
    // construct from input
    SparseVector activation = SparseVector(
                                input_indices[b], input_values[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b]+labelsize[b]);
    // forward pass for one sample
    for (int i = 0; i < num_layers_; ++i) {
      activation = layer_[i]->forward(activation) ;
    }
    if (activation.size() == 0)
      throw std::runtime_error("predict 0 classed");
    T max_act = activation.value_[0];
    int predict_class = activation.index_[0];
    for (int k = 1; k < activation.size(); k++) {
      T cur_act = activation.value_[k];
      if (max_act < cur_act) {
        max_act = cur_act;
        predict_class = activation.index_[k];
      }
    }

    if (std::find (labels[b], labels[b]+labelsize[b], predict_class)
        != labels[b]+labelsize[b]) {
      correctPred++;
    }
  }
  return correctPred;
}


float Network::train(int **input_indices, float **input_values,
                     int *lengths, int **labels,
                     int *label_size) {
// std::cout << "training\t";
  float loss = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+:loss)
#endif
  for (int b = 0; b < batch_size_; ++b) {
    vector<SparseVector > activations ((size_t)num_layers_ + 1);
    // construct from input
    activations[0] = SparseVector(input_indices[b], input_values[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b]+label_size[b]);
    // forward pass for one sample
    for (int i = 0; i < num_layers_; ++i) {
      activations[i+1] = layer_[i]->forward(activations[i]);
    }
    // compute loss
    float loss_b = 0;
    // gradient with respect to last layer output(pre SoftMax)
    SparseVector grad = SoftMaxCrossEntropy::compute(
        activations[num_layers_], labels_, &loss_b);
    loss += loss_b;
    // backward and update
    for (int i = num_layers_-1; i >= 0; --i) {
      if (grad.size() == 0) {
        // std::cerr << "No gradient Since Layer " << 1 + i << std::endl;
        break;
      }
      grad = layer_[i]->backward(grad, activations[i], optimizer_, i!=0);
      activations[i].clear();
    }
  }
//  std::cout << loss << std::endl;
  return loss;
}


void Network::save_weight(string file) {
}



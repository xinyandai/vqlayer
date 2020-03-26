#include <omp.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include "../include/network.h"


Interface* create_layer(size_type I, size_type O,
                        size_type layer, size_type num_layers) {
  const size_type THRESHOLD = 1 << 12;
  if (layer == num_layers - 1) {
    if (O >= THRESHOLD) {
      std::cout << "building PQLayer<SoftMax> "
                << I << " x " << O << std::endl;
      return new PQLayer<SoftMax, true, false>(I, O);
    }

    std::cout << "building Layer<SoftMax> "
              << I << " x " << O << std::endl;
    return new Layer<SoftMax, false>(I, O);

  } else {
    if (O >= THRESHOLD) {
      std::cout << "building PQLayer<ReLu> "
                << I << " x " << O << std::endl;
      return new PQLayer<ReLu, true, false>(I, O);

    } else if (I >= THRESHOLD) {
      std::cout << "building CPQLayer<ReLu> "
                << I << " x " << O << std::endl;
      return new CPQLayer<ReLu, false, false>(I, O);
    }

    std::cout << "building Layer<ReLu> "
              << I << " x " << O << std::endl;
    return new Layer<ReLu, false>(I, O);
  }
}

Network::Network(int *layer_size,
                 const int num_layers,
                 const int batch_size,
                 const Optimizer& optimizer,
                 const int input_dim) : optimizer_(optimizer) {
  layer_size_ = layer_size;
  num_layers_ = num_layers;
  batch_size_ = batch_size;
  input_dim_ = input_dim;
  layer_.reserve(static_cast<size_t >(num_layers_));

  layer_.emplace_back(
    create_layer(input_dim_, layer_size_[0], 0, num_layers_));
  for (int i = 1; i < num_layers_; ++i) {
    layer_.emplace_back(
      create_layer(layer_size_[i-1], layer_size_[i], i, num_layers_));
  }
  std::cout << "building network, done" << std::endl;
}

Network::~Network() {
  for (auto l : layer_) {
    delete l;
  }
  layer_.clear();
}

int Network::predict(int **input_indices, float **input_values,
                     int *lengths, int **labels, int *label_size) {
  int correct = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+:correct)
#endif
  for (int b = 0; b < batch_size_; ++b) {
    // construct from input
    SparseVector activation = SparseVector(input_indices[b],
                                           input_values[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b]+label_size[b]);

    // forward pass for one sample
    for (int i = 0; i < num_layers_; ++i) {
      activation = layer_[i]->forward(activation);
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

    if (labels[b]+label_size[b] !=
        std::find(labels[b], labels[b] + label_size[b], predict_class)) {
      correct++;
    }
  }
  return correct;
}


float Network::train(int **input_indices, float **input_values,
                     int *lengths, int **labels, int *label_size) {
  float loss = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+:loss)
#endif
  for (int b = 0; b < batch_size_; ++b) {
    vector<SparseVector > activations((size_t)num_layers_ + 1);

    // construct from input
    activations[0] = SparseVector(input_indices[b],
                                  input_values[b], lengths[b]);
    vector<int > labels_(labels[b], labels[b] + label_size[b]);

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
        break;
      }
      grad = layer_[i]->backward(grad, activations[i], optimizer_, i != 0);
      activations[i].clear();
    }
  }
  return loss;
}


void Network::save_weight(string file) {
}



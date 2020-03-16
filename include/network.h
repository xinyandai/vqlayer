#pragma once
#include <chrono>
#include <vector>
#include "iostream"
#include "string"
#include "layer.h"

using namespace std;

class Network {
 public:
  Network(int* layer_size, int num_layers, int batch_size,
          const Optimizer& optimizer, int input_dim);
  int predict(int **input_indices, float **input_values,
              int *length, int **labels, int *label_size);
  float train(int **input_indices, float **input_values,
              int *lengths, int **labels, int *label_size);
  void save_weight(string file);
  ~Network();
 private:
  size_type              batch_size_;
  size_type              num_layers_;
  size_type              input_dim_;
  size_type*             layer_size_;
  vector<Interface*>     layer_;
  const Optimizer&       optimizer_;
};


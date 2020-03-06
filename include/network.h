#pragma once
#include <chrono>
#include <vector>
#include "iostream"
#include "layer.h"

using namespace std;


class Network {
 public:
  Network(int* sizesOfLayers, vector<Activation >& layersTypes, int noOfLayers, int batchsize,
      const Optimizer& optimizer, int inputdim, int* K, int* L, int* RangePow, float* Sparsity);
  int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
  float ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize);
  void saveWeights(string file);
  ~Network();
 private:
  size_type              batch_size_;
  size_type              num_layers_;
  size_type              input_dim_;
  size_type*             layer_size_;
  vector<AbstractLayer*> layer_;
  const Optimizer&       optimizer_;
};


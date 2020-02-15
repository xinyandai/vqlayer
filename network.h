#pragma once
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>
#include "layer.h"

using namespace std;

class Network
{
 private:
  Layer** _hiddenlayers;
  float _learningRate;
  int _numberOfLayers;
  int* _sizesOfLayers;
  NodeType* _layersTypes;
  float * _Sparsity;
  int  _currentBatchSize;


 public:
  Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity);
  Layer* getLayer(int LayerID);
  int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
  float ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter);
  void saveWeights(string file);
  ~Network();
};


#pragma once
#include <chrono>

using namespace std;

class Network
{
 private:
  float _learningRate;
  int _numberOfLayers;
  int* _sizesOfLayers;
  float * _Sparsity;
  int  _currentBatchSize;


 public:
  Network(int* sizesOfLayers, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity);
  int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
  float ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter);
  void saveWeights(string file);
  ~Network();
};


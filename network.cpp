#include <iostream>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "network.h"
using namespace std;


Network::Network( int *sizesOfLayers, int noOfLayers, int batchSize, float lr, int inputdim,
                  int* K, int* L, int* RangePow, float* Sparsity) {
}
Network::~Network() {
}

int Network::predictClass(int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;
    return correctPred;
}


float Network::ProcessInput( int **inputIndices, float **inputValues, int *lengths,
                             int **labels, int *labelsize, int iter) {
  float log_loss;
  return log_loss;
}


void Network::saveWeights(string file) {
}



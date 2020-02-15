#pragma once
#include <string>
#include <sys/mman.h>

using namespace std;

enum NodeType
{ ReLU, SoftMax};

class Layer
{
 private:
  NodeType _type;
  int * _randNode;
  float* _normalizationConstants;
  int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;



 public:
  int _layerID, _noOfActive;
  int _noOfNodes;
  float* _weights;
  float* _adamAvgMom;
  float* _adamAvgVel;
  float* _bias;

  int * _binids;
  Layer(int _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
  int getNodeCount();
  void addtoHashTable(float* weights, int length, float bias, int id);
  float getNomalizationConstant(int inputID);
  int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
  int queryActiveNodes(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
  int computeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
  int computeSoftmax(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
  void saveWeights(string file);
  void updateTable();
  void updateRandomNodes();

  ~Layer();


};


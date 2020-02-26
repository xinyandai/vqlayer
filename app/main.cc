
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <cfloat>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "../include/network.h"

int *RangePow;
int *K;
int *L;
float *Sparsity;


int Batchsize = 1000;
int Rehash = 1000;
int Rebuild = 1000;
int InputDim = 784;
int totRecords = 60000;
int totRecordsTest = 10000;
float Lr = 0.0001;
int Epoch = 5;
int Stepsize = 20;
int *sizesOfLayers;
int numLayer = 3;
string trainData = "";
string testData = "";
string Weights = "";
string savedWeights = "";
string logFile = "";
using namespace std;
int globalTime = 0;

#define ALL(c) c.begin(), c.end()
#define FOR(i,c) for(typeof(c.begin())i=c.begin();i!=c.end();++i)
#define REP(i,n) for(int i=0;i<n;++i)
#define fst first
#define snd second


void endianSwap(unsigned int &x) {
  x = (x>>24)|((x<<8)&0x00FF0000)|((x>>8)&0x0000FF00)|(x<<24);
}
typedef vector<unsigned int> Image;


string trim(string& str)
{
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}


void parseconfig(string filename)
{
  std::ifstream file(filename);
  if(!file)
  {
    cout<<"Error Config file not found: Given Filename "<< filename << endl;
  }
  std::string str;
  while (getline(file, str))
  {
    if (str == "")
      continue;

    std::size_t found = str.find("#");
    if (found != std::string::npos)
      continue;

    if (trim(str).length() < 3)
      continue;

    int index = str.find_first_of("=");
    string first = str.substr(0, index);
    string second = str.substr(index + 1, str.length());

    if (trim(first) == "RangePow")
    {
      string str = trim(second).c_str();
      RangePow = new int[numLayer];
      char *mystring = &str[0];
      char *pch;
      pch = strtok(mystring, ",");
      int i=0;
      while (pch != NULL) {
        RangePow[i] = atoi(pch);
        pch = strtok(NULL, ",");
        i++;
      }
    }
    else if (trim(first) == "K")
    {
      string str = trim(second).c_str();
      K = new int[numLayer];
      char *mystring = &str[0];
      char *pch;
      pch = strtok(mystring, ",");
      int i=0;
      while (pch != NULL) {
        K[i] = atoi(pch);
        pch = strtok(NULL, ",");
        i++;
      }
    }
    else if (trim(first) == "L")
    {
      string str = trim(second).c_str();
      L = new int[numLayer];
      char *mystring = &str[0];
      char *pch;
      pch = strtok(mystring, ",");
      int i=0;
      while (pch != NULL) {
        L[i] = atoi(pch);
        pch = strtok(NULL, ",");
        i++;
      }
    }
    else if (trim(first) == "Sparsity")
    {
      string str = trim(second).c_str();
      Sparsity = new float[numLayer*2];
      char *mystring = &str[0];
      char *pch;
      pch = strtok(mystring, ",");
      int i=0;
      while (pch != NULL) {
        Sparsity[i] = atof(pch);
        pch = strtok(NULL, ",");
        i++;
      }
    }
    else if (trim(first) == "Batchsize")
    {
      Batchsize = atoi(trim(second).c_str());
    }
    else if (trim(first) == "Rehash")
    {
      Rehash = atoi(trim(second).c_str());
    }
    else if (trim(first) == "Rebuild")
    {
      Rebuild = atoi(trim(second).c_str());
    }
    else if (trim(first) == "InputDim")
    {
      InputDim = atoi(trim(second).c_str());
    }
    else if (trim(first) == "totRecords")
    {
      totRecords = atoi(trim(second).c_str());
    }
    else if (trim(first) == "totRecordsTest")
    {
      totRecordsTest = atoi(trim(second).c_str());
    }
    else if (trim(first) == "Epoch")
    {
      Epoch = atoi(trim(second).c_str());
    }
    else if (trim(first) == "Lr")
    {
      Lr = atof(trim(second).c_str());
    }
    else if (trim(first) == "Stepsize")
    {
      Stepsize = atoi(trim(second).c_str());
    }
    else if (trim(first) == "numLayer")
    {
      numLayer = atoi(trim(second).c_str());
    }
    else if (trim(first) == "logFile")
    {
      logFile = trim(second).c_str();
    }
    else if (trim(first) == "sizesOfLayers")
    {
      string str = trim(second).c_str();
      sizesOfLayers = new int[numLayer];
      char *mystring = &str[0];
      char *pch;
      pch = strtok(mystring, ",");
      int i=0;
      while (pch != NULL) {
        sizesOfLayers[i] = atoi(pch);
        pch = strtok(NULL, ",");
        i++;
      }
    }
    else if (trim(first) == "trainData")
    {
      trainData = trim(second).c_str();
    }
    else if (trim(first) == "testData")
    {
      testData = trim(second).c_str();
    }
    else if (trim(first) == "weight")
    {
      Weights = trim(second).c_str();
    }
    else if (trim(first) == "savedweight")
    {
      savedWeights = trim(second).c_str();
    }
    else
    {
      cout << "Error Parsing conf File at Line" << endl;
      cout << str << endl;
    }
  }
}

void EvalDataSVM(int numBatchesTest,  Network* _mynet, int iter){
  int totCorrect = 0;
  int debugnumber = 0;
  std::ifstream testfile(testData);
  string str;
  //Skipe header
  std::getline( testfile, str );

  ofstream outputFile(logFile,  std::ios_base::app);
  for (size_t i = 0; i < numBatchesTest; i++) {
    int **records = new int *[Batchsize];
    float **values = new float *[Batchsize];
    int *sizes = new int[Batchsize];
    int **labels = new int *[Batchsize];
    int *labelsize = new int[Batchsize];
    int nonzeros = 0;
    int count = 0;
    vector<string> list;
    vector<string> value;
    vector<string> label;
    while (std::getline(testfile, str)) {

      char *mystring = &str[0];
      char *pch, *pchlabel;
      int track = 0;
      list.clear();
      value.clear();
      label.clear();
      pch = strtok(mystring, " ");
      pch = strtok(NULL, " :");
      while (pch != NULL) {
        if (track % 2 == 0)
          list.push_back(pch);
        else if (track%2==1)
          value.push_back(pch);
        track++;
        pch = strtok(NULL, " :");
      }

      pchlabel = strtok(mystring, ",");
      while (pchlabel != NULL) {
        label.push_back(pchlabel);
        pchlabel = strtok(NULL, ",");
      }

      nonzeros += list.size();
      records[count] = new int[list.size()];
      values[count] = new float[list.size()];
      labels[count] = new int[label.size()];
      sizes[count] = list.size();
      labelsize[count] = label.size();

      int currcount = 0;
      debugnumber++;
      vector<string>::iterator it;
      for (it = list.begin(); it < list.end(); it++) {
        records[count][currcount] = stoi(*it);
        currcount++;
      }
      currcount = 0;
      for (it = value.begin(); it < value.end(); it++) {
        values[count][currcount] = stof(*it);
        currcount++;
      }
      currcount = 0;
      for (it = label.begin(); it < label.end(); it++) {
        labels[count][currcount] = stoi(*it);
        currcount++;
      }

      count++;
      if (count >= Batchsize)
        break;
    }

    int num_features = 0, num_labels = 0;
    for (int i = 0; i < Batchsize; i++)
    {
      num_features += sizes[i];
      num_labels += labelsize[i];
    }

//    std::cout << Batchsize << " records, with "<< num_features << " features and " << num_labels << " labels" << std::endl;
    auto correctPredict = _mynet->predictClass(records, values, sizes, labels, labelsize);
    totCorrect += correctPredict;
//    std::cout <<" iter "<< i << ": " << totCorrect*1.0/(Batchsize*(i+1)) << " correct" << std::endl;

    delete[] sizes;
    delete[] labels;
    for (size_t d = 0; d < Batchsize; d++) {
      delete[] records[d];
      delete[] values[d];
    }
    delete[] records;
    delete[] values;

  }
  testfile.close();
  cout << "over all " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
  outputFile << iter << " " << globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;

}

void ReadDataSVM(int numBatches,  Network* _mynet, int epoch){
  std::ifstream file(trainData);
  float accumlogss = 0;
  std::string str;
  //skipe header
  std::getline( file, str );
  int totalTime = 0;
  for (size_t i = 0; i < numBatches; i++) {

    int **records = new int *[Batchsize];
    float **values = new float *[Batchsize];
    int *sizes = new int[Batchsize];
    int **labels = new int *[Batchsize];
    int *labelsize = new int[Batchsize];
    int nonzeros = 0;
    int count = 0;
    vector<string> list;
    vector<string> value;
    vector<string> label;
    while (std::getline(file, str)) {
      char *mystring = &str[0];
      char *pch, *pchlabel;
      int track = 0;
      list.clear();
      value.clear();
      label.clear();
      pch = strtok(mystring, " ");
      pch = strtok(NULL, " :");
      while (pch != NULL) {
        if (track % 2 == 0)
          list.push_back(pch);
        else if (track%2==1)
          value.push_back(pch);
        track++;
        pch = strtok(NULL, " :");
      }

      pchlabel = strtok(mystring, ",");
      while (pchlabel != NULL) {
        label.push_back(pchlabel);
        pchlabel = strtok(NULL, ",");
      }

      nonzeros += list.size();
      records[count] = new int[list.size()];
      values[count] = new float[list.size()];
      labels[count] = new int[label.size()];
      sizes[count] = list.size();
      labelsize[count] = label.size();
      int currcount = 0;
      vector<string>::iterator it;
      for (it = list.begin(); it < list.end(); it++) {
        records[count][currcount] = stoi(*it);
        currcount++;
      }
      currcount = 0;
      for (it = value.begin(); it < value.end(); it++) {
        values[count][currcount] = stof(*it);
        currcount++;
      }

      currcount = 0;
      for (it = label.begin(); it < label.end(); it++) {
        labels[count][currcount] = stoi(*it);
        currcount++;
      }

      count++;
      if (count >= Batchsize)
        break;
    }



    auto t1 = std::chrono::high_resolution_clock::now();


    auto loss = _mynet->ProcessInput(records, values, sizes, labels, labelsize);

    auto t2 = std::chrono::high_resolution_clock::now();

    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    globalTime+= timeDiffInMiliseconds;

    delete[] sizes;

    for (size_t d = 0; d < Batchsize; d++) {
      delete[] records[d];
      delete[] values[d];
      delete[] labels[d];
    }
    delete[] records;
    delete[] values;
    delete[] labels;

  }
  file.close();

}


int main(int argc, char* argv[])
{
  //***********************************
  // Parse Config File
  //***********************************
  parseconfig(argc > 1 ? argv[1]  : "amazon.cfg");

  //***********************************
  // Initialize Network
  //***********************************
  int numBatches = totRecords/Batchsize;
  int numBatchesTest = totRecordsTest/Batchsize;

  vector<Activation > layersTypes(numLayer, Activation::ReLu);
  layersTypes[numLayer-1] = Activation::SoftMax;


  auto t1 = std::chrono::high_resolution_clock::now();
  Optimizer optimizer = {0.1}; // TODO modify config file later
  Network *_mynet = new Network(sizesOfLayers, layersTypes, numLayer, Batchsize, optimizer, InputDim, K, L, RangePow, Sparsity);
  auto t2 = std::chrono::high_resolution_clock::now();
  float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Network Initialization takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

  //***********************************
  // Start Training
  //***********************************

  const int schedule_epoch = Epoch / 5;
  for (int e=0; e< Epoch; e++) {
    ofstream outputFile(logFile,  std::ios_base::app);
    outputFile<<"Epoch "<<e<<endl;
    if (e > 0 && e % schedule_epoch == 0) {
      optimizer.lr /= 3.0;
    }
    // train
    ReadDataSVM(numBatches, _mynet, e);
    // test
    EvalDataSVM(numBatchesTest, _mynet, (e+1)*numBatches);
    _mynet->saveWeights(savedWeights);

  }

  delete [] RangePow;
  delete [] K;
  delete [] L;
  delete [] Sparsity;

  return 0;

}

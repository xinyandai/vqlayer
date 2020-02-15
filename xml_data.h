//
// Created by xinyan on 2020/2/14.
//

#ifndef VQ_LAYER__XML_DATA_H_
#define VQ_LAYER__XML_DATA_H_

#include <thread>
#include <deque>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>

using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::istringstream;

class SparseData {
 public:
  SparseData(const string& file_name, const int batch_size)
  : data_reader_(file_name), batch_size_(batch_size) {
    string line;
    std::getline(data_reader_,  line);
    istringstream iss(line);
    iss >> num_items_ >> num_features_ >> num_labels_;

    std::thread([this](){
        loadData();
    });
  }

  void nextBatch() {

  }
  int getBatchSize() const {
    return batch_size_;
  }
  int getNumItems() const {
    return num_items_;
  }
  int getNumFeatures() const {
    return num_features_;
  }
  int getNumLabels() const {
    return num_labels_;
  }
 private:
  void loadData() {

  }

 private:
  ifstream data_reader_;
  int batch_size_;
  int num_items_;
  int num_features_;
  int num_labels_;
};


void EvalDataSVM(const string& file_name){

  std::ifstream data_reader(file_name);
  string str;
  std::getline( data_reader, str );

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
    while (std::getline(data_reader, str)) {

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

    int num_features = 0, num_labels = 0;
    for (int i = 0; i < Batchsize; i++)
    {
      num_features += sizes[i];
      num_labels += labelsize[i];
    }

    delete[] sizes;
    delete[] labels;
    for (size_t d = 0; d < Batchsize; d++) {
      delete[] records[d];
      delete[] values[d];
    }
    delete[] records;
    delete[] values;

  }
  data_reader.close();

}

#endif //VQ_LAYER__XML_DATA_H_
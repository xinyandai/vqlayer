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
#include <string>
#include <string.h>

using std::stoi;
using std::stof;
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

  ~SparseData() {
    data_reader_.close();
  }

  void nextBatch(const int batch_size) {
    // TODO
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

  void loadLine() {
    string str;
    if (std::getline(data_reader_, str)) {
      char *mystring = &str[0];
      char *pch, *pchlabel;
      int track = 0;
      index_.clear();
      value_.clear();
      label_.clear();
      pch = strtok(mystring, " ");
      pch = strtok(NULL, " :");
      while (pch != NULL) {
        if (track % 2 == 0)
          index_.push_back(stoi(pch));
        else if (track%2==1)
          value_.push_back(stof(pch));
        track++;
        pch = strtok(NULL, " :");
      }

      pchlabel = strtok(mystring, ",");
      while (pchlabel != NULL) {
        label_.push_back(stoi(pchlabel));
        pchlabel = strtok(NULL, ",");
      }
    }
  }

 private:
  ifstream data_reader_;
  vector<int > label_;
  vector<int > index_;
  vector<float > value_;
  int batch_size_;
  int num_items_;
  int num_features_;
  int num_labels_;
};



#endif //VQ_LAYER__XML_DATA_H_
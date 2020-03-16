//
// Created by xinyan on 16/3/2020.
//
#include "test.h"
#include "../include/util.h"

int main() {

  TopSelector selector(3);
  int id = 100;
  selector.insert(0.1, id++);
  selector.insert(0.7, id++);
  selector.insert(200, 1);
  selector.insert(100, 0);
  selector.insert(0.0, id++);
  selector.insert(0.5, id++);
  selector.insert(200, 2);

  SparseVector top = selector.select();
  vector<float > top_ = {100, 200, 200};
  compare("top", top, top_.data(), top.size());
}

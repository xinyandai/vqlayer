//
// Created by xinyan on 16/3/2020.
//
#include "test.h"
#include "../include/util.h"

int main() {

  TopSelector selector(3);
  int id = 100;
  selector.insert(id++, 0.1);
  selector.insert(id++, 0.7);
  selector.insert(1, 200);
  selector.insert(0, 100);
  selector.insert(id++, 0.0);
  selector.insert(id++, 0.5);
  selector.insert(2, 200);

  SparseVector top = selector.select();
  vector<float > top_ = {100, 200, 200};
  compare("top", top, top_.data(), top.size());
}

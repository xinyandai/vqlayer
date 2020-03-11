//
// Created by xinyan on 10/3/2020.
//
#pragma once
#include "../include/layer.h"

void dump(const SparseVector& s) {
  for (int i = 0; i < s.size(); ++i) {
    std::cout << s.index_[i] << " : " << s.value_[i] <<"\t";
  }
  std::cout << std::endl;
}

void dump(const T* p, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << i << " : " << *(p++) <<"\t";
  }
  std::cout << std::endl;
}

void compare(std::string variable, T a, T b) {
  bool success = (std::abs(a - b) < 0.001);
  std::cout << (success?"[PASS]":"[FAIL]") ;
  std::cout << " " << variable << std::endl;
  std::cout << "\t" << a << "\n";
  std::cout << "\t" << b << "\n";
}

void compare(std::string variable,
             SparseVector s, const T* p, int size) {
  bool success = true;
  for (int i = 0; i < s.size(); ++i) {
    if (std::abs(s.value_[i] - p[i]) > 0.001) {
      success = false;
      break;
    }
  }
  std::cout << (success?"[PASS]":"[FAIL]") ;
  std::cout << "\t" << variable << std::endl;
  std::cout << "\t\t";
  dump(s);
  std::cout << "\t\t";
  dump(p, size);
}

void compare(std::string variable,
             SparseVector s, SparseVector p) {
  bool success = true;
  if (s.size() != p.size()) {
    success = false;
  }
  else for (int i = 0; i < s.size(); ++i) {
    if (s.index_[i] != p.index_[i] || std::abs(s.value_[i] - p.value_[i]) > 0.001) {
      success = false;
      break;
    }
  }

  std::cout << (success?"[PASS]":"[FAIL]") ;
  std::cout << "\t" << variable << std::endl;
  std::cout << "\t\t";
  dump(s);
  std::cout << "\t\t";
  dump(p);
}
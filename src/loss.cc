//
// Created by xinyan on 2020/3/3.
//
//
// Created by xinyan on 2020/2/18.
//

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include "../include/layer.h"


SparseVector SoftMaxCrossEntropy:: compute(
    const SparseVector& p,
    const vector<size_type >& y, T* loss) {


  SparseVector grad;
  size_t reserve_size = std::max(p.size(), y.size());
  grad.reserve(reserve_size);

  T y_prob = (T)1.0 / y.size();

  if (loss) { // compute loss = Sum(yi log pi)
    T loss_ = 0;
    size_type i_p = 0;
    size_type i_y = 0;
    while (i_p < p.size() && i_y < y.size()) {
      if (p.index_[i_p] == y[i_y]) {
        loss_ += y_prob * std::log(p.value_[i_p]);
        i_p++, i_y++;
      } else if (p.index_[i_p] < y[i_y]){
        i_p++;
      } else {
        i_y++;
      }
    }
    *loss = -loss_;
  }

  size_type i_p = 0;
  size_type i_y = 0;
  // compute gradient : g_i = p_i - y_i
  while (i_p < p.size() && i_y < y.size()) {
    if (p.index_[i_p] == y[i_y]) {
      grad.push_back(y[i_y], p.value_[i_p] - y_prob);
      i_p++, i_y++;
    } else if (p.index_[i_p] < y[i_y]) {
      grad.push_back(p.index_[i_p], p.value_[i_p]);
      i_p++;
    } else {
      grad.push_back(y[i_y], - y_prob);
      i_y++;
    }
  }
  while (i_p < p.size()) {
    grad.push_back(p.index_[i_p], p.value_[i_p]);
    i_p++;
  }
  while (i_y < y.size()) {
    grad.push_back(y[i_y], - y_prob);
    i_y++;
  }

  return grad;
}


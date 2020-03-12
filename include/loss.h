//
// Created by xinyan on 12/3/2020.
//
#pragma once
#include <vector>
#include "tensor.h"

using std::vector;

class SoftMaxCrossEntropy {
 public:
  /**
   * \param p SoftMax estimate
   * \param y true labels for classification task,
   * \return gradient with respect to the pre-SoftMax output
   *         according to formula: g_i = p_i - y_i
   */
  static SparseVector compute(const SparseVector& p,
                              const vector<size_type >& y, T* loss);
};

//
// Created by xinyan on 2020/2/16.
//

#pragma once
#include <memory>
#include <vector>
#include "operand.h"

using std::vector;

class Operator {
};

//            | tensor            | gradient
// X input    | NxD sparse-tensor | GW' NxD
// W weight   | Dxl dense-tensor  | X'G DxL
// b bias     | l   dense-tensor  | sum(G) L
// Y output   | NxL sparse-tensor | G NxL sparse-tensor for last layer

// forward:   Y = XW + b
// backward:  GW'; X'G
// sgd: W -= X'G
template <typename value_type>
class SparseDenseMul : Operator {
 public:
  static void forward( std::array<Operand*, 3>& input,
                       SparseRows<value_type >& output) {
    auto x = static_cast<SparseRows<value_type > *>(input[0]);
    auto w = static_cast<Tensor<value_type, 2> *>(input[1]);
    auto b = static_cast<Tensor<value_type, 1> *>(input[2]);
  }
  static void backward(SparseRows<value_type >& grad_output, std::array<Operand*, 3>& grads) {

  }
};

template <typename value_type>
class Relu  : Operator{
 public:
  void forward();
  void backward();
};

template <typename value_type>
class Dropout : Operator {
 public:
  void forward();
  void backward();
};


class SoftMaxLoss : Operator {
 public:
  void go(double& loss);
};
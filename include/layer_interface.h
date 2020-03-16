//
// Created by xinyan on 16/3/2020.
//

#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <memory>
#include <numeric>
#include <utility>
#include <mutex>
#include <thread>

#include "vq.h"
#include "loss.h"
#include "tensor.h"


using std::mutex;
using std::vector;
using std::shared_ptr;


enum Activation {
  ReLu, SoftMax
};

typedef struct {
  T lr;
} Optimizer;

class Interface {
 public:
  /**
 * \brief y = \sigma(xW + b), where sigma is the activation function
 * \param x Sparse Vector
 * \return y Sparse Vector
 */
  virtual SparseVector forward(const SparseVector& x) = 0;
  /**
 * \brief calculated gradient with respect to weight and input
 *        according to formula: g_W = gx; g_b = g; g_I = gW';
 *        update the parameters with Optimization Algorithm:
 *        P -=  lr * Gradient
 * \param a current layer output activation
 * \param x sparse vector, memorize input for calculate gradient
 *        with respect to weight_
 *
 * \param g gradient with respect to the output of forward
 * \param compute_gx should compute gradient with respect to x
 * \return gradient with respect to x
 */
  virtual SparseVector backward(const SparseVector& g,
                                const SparseVector& x,
                                const Optimizer& optimizer,
                                bool compute_gx) = 0;
};

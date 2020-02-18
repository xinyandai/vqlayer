//
// Created by xinyan on 17/2/2020.
//

#pragma once
#include "../include/operator.h"
using T = float;

enum Activation {

};

/***
 * \author Xinyan DAI (xinyan.dai@outlook.com)
 * \brief Sparse Matrix Multiplication Layer
 */
class Layer {
 public:
  Layer() {}
  virtual ~Layer() {}
  /**
   * \brief y = xW + b
   * \param x Sparse Vector
   * \return y Sparse Vector
   */
  virtual void* forward(void* x) {
  }
  /**
   * \brief calculated gradient with respect to weight and input
   *        according to formula: g_W = gx; g_b = g; g_I = gW';
   *        update the parameters with Optimization Algorithm:
   *        P -=  lr * Gradient
   * \param x sparse vector, memorize input for calculate gradient
   *        with respect to weight_
   * \param g gradient with respect to the output of forward
   * \input
   */
  virtual void* backward(void* g. void* x) {

  }
 private:
  size_type I_;
  size_type O_;
  T* weight_;
  T* bias_;
};

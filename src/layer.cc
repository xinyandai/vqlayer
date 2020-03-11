//
// Created by xinyan on 2020/2/18.
//

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include "../include/layer.h"

Layer::Layer(size_type I, size_type O, Activation type)
: AbstractLayer(I, O, type)
#ifdef ThreadSafe
  , weight_lock_(I), bias_lock_(O)
#endif
{
  weight_ = new T[I * O];
  if (weight_==nullptr)
    throw std::runtime_error("Failed to allocate memory for weight");
  bias_ = new T[O];
  if (bias_==nullptr)
    throw std::runtime_error("Failed to allocate memory for bias");
  initialize();
}

Layer::Layer(const Layer& c) : Layer(c.I_, c.O_, c.type_) {
  std::memcpy(weight_, c.weight_, I_ * O_ * sizeof(T));
  std::memcpy(bias_, c.bias_,  O_ * sizeof(T));
}

Layer::Layer(Layer&& c) noexcept : AbstractLayer(c.I_, c.O_, c.type_),
                                   weight_(c.weight_), bias_(c.bias_)
#ifdef ThreadSafe
                                   ,weight_lock_(c.I_), bias_lock_(c.O_)
#endif
{
  c.weight_ = NULL;
  c.bias_ = NULL;
}

Layer::~Layer() {
  delete [] weight_;
  delete [] bias_;
}

void Layer::initialize(const vector<T >& w, const vector<T >& b) {
  if (w.size() != I_ * O_) {
    throw std::runtime_error("Weight size not matched! ");
  }
  if (b.size() != O_) {
    throw std::runtime_error("Bias size not matched! ");
  }
  std::memcpy(weight_, w.data(), w.size() * sizeof(T));
  std::memcpy(bias_, b.data(), b.size() * sizeof(T));
}

void Layer::initialize() {
  std::default_random_engine generator(1016);
  std::uniform_real_distribution<T > distribution(0.0, 1.0 / std::sqrt(I_ / 2.0));
  T* w = weight_;
  for (int i = 0; i < I_; i++) {
    for (int j = 0; j < O_; j++) {
      *(w++) = distribution(generator);
    }
  }
  T* b = bias_;
  for (int o = 0; o < O_; ++o) {
    *(b++) = distribution(generator);
  }
}

T Layer::get_w(size_type i, size_type o)  {
  return weight_[i * O_ + o];
}

T Layer::get_b(size_type o)  {
  return bias_[o];
}


SparseVector Layer::forward(const SparseVector& x) {
  return AbstractLayer::forward(x);
}


SparseVector Layer::backward_x(const SparseVector& g,
                        const SparseVector& x) {
  return AbstractLayer::backward_x(g, x);
}


void Layer::backward_w(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer) {
  T lr = optimizer.lr;
  SparseVector gx;
  volatile T* weight = weight_;
  volatile T* bias = bias_;

  // compute gradient and update with respect to the weight
  // gw[I_, O_] = x[1, I_]' g[1, O_]
  for (int i = 0; i < x.size(); ++i) {
    volatile T* w = weight + O_ * x.index_[i];
#ifdef ThreadSafe
    std::lock_guard<std::mutex> lock(weight_lock_[x.index_[i]]);
#endif
    for (int o = 0; o < g.size(); ++o) {
      T grad = x.value_[i] * g.value_[o];
      w[g.index_[o]] -= lr * grad;
    }
  }
  // update bias (gradient of bias is equivalent to g)
  for (int i = 0; i < g.size(); ++i) {
    T grad = g.value_[i];
    size_type index = g.index_[i];
#ifdef ThreadSafe
    std::lock_guard<std::mutex> lock(bias_lock_[index]);
#endif
    bias[index] -= lr * grad;
  }
}


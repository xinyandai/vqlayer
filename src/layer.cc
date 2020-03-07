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
: I_(I), O_(O), type_(type)
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

Layer::Layer(Layer&& c) : I_(c.I_), O_(c.O_), type_(c.type_),
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

SparseVector Layer::forward(const SparseVector& x) {
  SparseVector y;
  volatile T* weight = weight_;
  volatile T* bias = bias_;
  // Relu activation remove the negative output
  if (type_ == Activation::ReLu) {
    for (int o = 0; o < O_; ++o) {
      T mm = bias[o];
      for (int s = 0; s < x.size(); ++s) {
        size_type i = x.index_[s];
        mm += x.value_[s] * weight[O_ * i + o];
      }
      if (mm > 0) {
        y.push_back(o, mm);
      }
    }
  }

  // Compute SoftMax, see @link{https://deepnotes.io/softmax-crossentropy}
  // TODO replace expensive std::exp operation with lookup table
  else if (type_ == Activation::SoftMax) {
    T max_v = std::numeric_limits<T>::min();
    for (int o = 0; o < O_; ++o) {
      T mm = bias[o];
      for (int s = 0; s < x.size(); ++s) {
        size_type i = x.index_[s];
        mm += x.value_[s] * weight[O_ * i + o];
      }
      y.push_back(o, mm);
      if (mm > max_v)
        max_v = mm;
    }
    // C is a constant for computation stability
    const T C = -max_v;
    T sum = 0;
    for (auto& v : y.value_) {
      v = std::exp(v+C);
      sum += v;
    }
    if (sum > 0) {
      for (auto& v : y.value_) {
        v = v / sum;
      }
    }
  }
  return y;
}


SparseVector Layer::backward( const SparseVector& g,
                              const SparseVector& x,
                              const Optimizer& optimizer,
                              bool compute_gx ) {
  T lr = optimizer.lr;
  SparseVector gx;
  volatile T* weight = weight_;
  volatile T* bias = bias_;
  if (compute_gx) {
    // Compute gradient  with respect to the input:
    // gx[I_] = w[I_, O_], g[O_].
    // Previous layer's activation function must be ReLu,
    // since SoftMax only exist in last layer.
    gx = x;
    for (int i = 0; i < x.size(); ++i) {
      volatile T* w = weight + O_ * x.index_[i];
      T grad = 0;
      for (int o = 0; o < g.size(); ++o) {
        grad += g.value_[o] * w[g.index_[o]];
      }
      gx.value_[i] = grad;
    }
  }

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

  return gx;
}
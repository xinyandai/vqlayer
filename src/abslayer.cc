//
// Created by xinyan on 11/3/2020.
//

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include "../include/layer.h"

AbstractLayer::AbstractLayer(size_type I, size_type O, Activation type)
                           : I_(I), O_(O), type_(type) {

  bias_ = new T[O];
  initialize();
}

void AbstractLayer::initialize() {
  std::default_random_engine generator(1016);
  std::uniform_real_distribution<T > dist(0.0f, 1.0f / std::sqrt(I_ / 2.0f));
  T* b = bias_;
  for (int o = 0; o < O_; ++o) {
    *(b++) = dist(generator);
  }
}

AbstractLayer::AbstractLayer(const AbstractLayer& c)
                           : AbstractLayer(c.I_, c.O_, c.type_) {
  std::memcpy(bias_, c.bias_,  O_ * sizeof(T));
}

AbstractLayer::AbstractLayer(AbstractLayer&& c) noexcept
    : AbstractLayer(c.I_, c.O_, c.type_) {
  bias_ = c.bias_;
  c.bias_ = nullptr;
}

AbstractLayer::~AbstractLayer() {
  delete [] bias_;
}



T AbstractLayer::get_w(size_type i, size_type o)  {
  throw std::runtime_error("get_w not implemented.");
}

T AbstractLayer::get_b(size_type o)  {
  return bias_[0];
}

SparseVector AbstractLayer::forward(const SparseVector &x) {
  return default_forward(x);
}

SparseVector AbstractLayer::backward(const SparseVector& g,
                                     const SparseVector& x,
                                     const Optimizer& optimizer,
                                     bool compute_gx ) {
  SparseVector gx;
  if (compute_gx) {
    gx = backward_x(g, x);
  }
  backward_w(g, x, optimizer);
  backward_b(g, x, optimizer);
  return gx;
}

SparseVector AbstractLayer::default_forward(const SparseVector &x) {
  SparseVector y;

  // Relu activation remove the negative output
  if (type_ == Activation::ReLu) {
    for (int o = 0; o < O_; ++o) {
      T mm = get_b(o);
      for (int s = 0; s < x.size(); ++s) {
        size_type i = x.index_[s];
        mm += x.value_[s] * get_w(i, o);
      }
      if (mm > 0) {
        y.push_back(o, mm);
      }
    }
  }

  // Compute SoftMax, see @link{https://deepnotes.io/softmax-crossentropy}
  else if (type_ == Activation::SoftMax) {
    T max_v = std::numeric_limits<T>::min();
    for (int o = 0; o < O_; ++o) {
      T mm = get_b(o);
      for (int s = 0; s < x.size(); ++s) {
        size_type i = x.index_[s];
        mm += x.value_[s] * get_w(i, o);
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

SparseVector AbstractLayer::default_backward_x(const SparseVector &g,
                                               const SparseVector &x) {
  // Compute gradient  with respect to the input:
  // gx[I_] = w[I_, O_], g[O_].
  // Previous layer's activation function must be ReLu,
  // since SoftMax only exist in last layer.
  SparseVector gx = x;
  for (int i = 0; i < x.size(); ++i) {
    T grad = 0;
    for (int o = 0; o < g.size(); ++o) {
      grad += g.value_[o] * get_w(x.index_[i], g.index_[o]);
    }
    gx.value_[i] = grad;
  }
  return gx;
}
void AbstractLayer::backward_w(const SparseVector& g,
                               const SparseVector& x,
                               const Optimizer& optimizer) {
  throw std::runtime_error("backward for weights is not implemented");
}

void AbstractLayer::backward_b(const SparseVector& g,
                               const SparseVector& x,
                               const Optimizer& optimizer) {
  T lr = optimizer.lr;
  SparseVector gx;
  volatile T* bias = bias_;
  // update bias (gradient of bias is equivalent to g)
  for (int i = 0; i < g.size(); ++i) {
    T grad = g.value_[i];
    size_type index = g.index_[i];
    bias[index] -= lr * grad;
  }
}
//
// Created by xinyan on 16/3/2020.
//
#pragma once
#include <limits>
#include <utility>
#include "util.h"
#include "layer_interface.h"


template <Activation Act, bool Select>
class AbstractLayer : public Interface {
 public:
  AbstractLayer(size_type I, size_type O) : I_(I), O_(O) {
    bias_ = new T[O];
    initialize();
  }

  virtual ~AbstractLayer() {
    delete [] bias_;
  }

  virtual T get_w(size_type i, size_type o) const = 0;

  T get_b(size_type o) const {
    return bias_[o];
  }

  void initialize();

  SparseVector forward(const SparseVector& x) override;

  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx) override {
    SparseVector gx;
    if (compute_gx) {
      gx = backward_x(g, x);
    }
    backward_w(g, x, optimizer);
    backward_b(g, x, optimizer);
    return gx;
  }

  virtual SparseVector backward_x(const SparseVector& g,
                                  const SparseVector& x) {
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
  virtual void backward_w(const SparseVector& g,
                          const SparseVector& x,
                          const Optimizer& optimizer) = 0;

  virtual void backward_b(const SparseVector& g,
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

 public:
  const size_type  I_;
  const size_type  O_;

 protected:
  T*               bias_;
};

template <Activation Act, bool Select>
void AbstractLayer<Act, Select>::initialize() {
  std::default_random_engine generator(1016);
  std::uniform_real_distribution<T > dist(0.f, 1.f / std::sqrt(I_ / 2.f));
  T* b = bias_;
  for (int o = 0; o < O_; ++o) {
    *(b++) = dist(generator);
  }
}

template <Activation Act, bool Select>
inline void insert(size_type o, T mm, T& max_v,
                   TopSelector<size_type, T>& selector, SparseVector& y) {
  if constexpr (Act == ReLu) {
    if (mm > 0) {
      y.push_back(o, mm);
    }
  }
  if constexpr (Act == SoftMax)  {
    if constexpr (Select) {
      selector.insert(o, mm);
    } else {
      y.push_back(o, mm);
    }
    if (mm > max_v)
      max_v = mm;
  }
}

template <Activation Act, bool Select>
inline SparseVector softmax(TopSelector<size_type, T>& selector,
                            SparseVector& y, T max_v) {
  if constexpr (Act == SoftMax) {
    SparseVector selected;
    if constexpr (Select) {
      selected = selector.select();
      y = selected;
    }
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

template <Activation Act, bool Select>
SparseVector AbstractLayer<Act, Select>::forward(const SparseVector& x) {
  SparseVector y;
  TopSelector selector(10 + O_/10);
  T max_v = std::numeric_limits<T>::min();
  for (int o = 0; o < this->O_; ++o) {
    T mm = this->get_b(o);
    for (int s = 0; s < x.size(); ++s) {
      size_type i = x.index_[s];
      mm += x.value_[s] * this->get_w(i, o);
    }
    insert<Act, Select>(o, mm, max_v, selector, y);
  }
  return softmax<Act, Select>(selector, y, max_v);
}

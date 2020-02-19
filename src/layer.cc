//
// Created by xinyan on 2020/2/18.
//

#include <cmath>
#include <iostream>
#include <limits>
#include <cstring>
#include "../include/layer.h"

Layer::Layer(size_type I, size_type O, Activation type)
: I_(I), O_(O), type_(type) {
  weight_ = new T[I * O];
  if (weight_==nullptr)
    throw std::runtime_error("Failed to allocate memory for weight");
  bias_ = new T[O];
  if (bias_==nullptr)
    throw std::runtime_error("Failed to allocate memory for bias");
}

Layer::Layer(const Layer& c) : Layer(c.I_, c.O_, c.type_) {
  std::memcpy(weight_, c.weight_, I_ * O_);
  std::memcpy(bias_, c.bias_,  O_);
}
Layer::Layer(Layer&& c) : I_(c.I_), O_(c.O_), type_(c.type_),
                          weight_(c.weight_), bias_(c.bias_) {
  c.weight_ = NULL;
  c.bias_ = NULL;
}

Layer::~Layer() {
  if (weight_)
    delete [] weight_;
  if (bias_)
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

/**
 * \brief y = \sigma(xW + b)
 * \param x Sparse Vector
 * \return y Sparse Vector
 */
SparseVector Layer::forward(const SparseVector& x) {
  SparseVector y;
  // Relu activation remove the negative output
  // SoftMax activation remove the extremely small output
  T threshold = (type_ == Activation::ReLu) ? 0.f : -1e10f;
  T max_v = threshold;
  for (int o = 0; o < O_; ++o) {
    T mm = bias_[o];
    for (int s = 0; s < x.size(); ++s) {
      size_type i = x.index_[s];
      mm += x.value_[s] * weight_[O_ * i + o];
    }
    if (mm > threshold) {
      y.push_back(o, mm);
      if (mm > max_v)
        max_v = mm;
    }
  }
  // Compute SoftMax, see @link{https://deepnotes.io/softmax-crossentropy}
  // TODO replace expensive log operation with lookup table
  if (type_ == Activation::SoftMax) {
    // C is a constant for computation stability
    const T C = -max_v;
    T sum = 0;
    for (auto& v : y.value_) {
      v = std::log(v+C);
      sum += v;
    }
    for (auto& v : y.value_) {
      v = v / sum;
    }
  }
  return y;
}

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
SparseVector Layer::backward( const SparseVector& g,
                              const SparseVector& x,
                              const Optimizer& optimizer,
                              bool compute_gx ) {
  T lr = optimizer.lr;
  SparseVector gx;
  if (compute_gx) {
    // Compute gradient  with respect to the input:
    // gx[I_] = w[I_, O_], g[O_].
    // Previous layer's activation function must be ReLu,
    // since SoftMax only exist in last layer.
    gx = x;
    for (int i = 0; i < x.size(); ++i) {
      T* w = weight_ + O_ * x.index_[i];

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
    T* w = weight_ + O_ * x.index_[i];
    for (int o = 0; o < g.size(); ++o) {
      T grad = x.value_[i] * g.value_[o];
      w[g.index_[o]] -= lr * grad;
    }
  }
  // update bias (gradient of bias is equivalent to g)
  for (int i = 0; i < g.size(); ++i) {
    T grad = g.value_[i];
    bias_[g.index_[i]] -= lr * grad;
  }

  return gx;
}


/**
 * \param p SoftMax estimate
 * \param y true labels for classification task,
 * \return gradient with respect to the pre-SoftMax output
 *         according to formula: g_i = p_i - y_i
 */
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
    *loss = loss_;
  }

  size_type i_p = 0;
  size_type i_y = 0;
  // compute gradient : g_i = p_i - y_i
  while (i_p < p.size() && i_y < y.size()) {
    if (p.index_[i_p] == y[i_y]) {
      grad.push_back(y[i_y], p.value_[i_p] - y_prob);
      i_p++, i_y++;
    } else if (p.index_[i_p] < y[i_y]){
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

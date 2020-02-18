//
// \author Xinyan DAI (xinyan.dai@outlook.com)
// Created by xinyan DAI on 17/2/2020.
//

#pragma once
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

using T = float;
using size_type = int;

enum Activation {
  ReLu, SoftMax
};

typedef struct {
  T lr;
} Optimizer;

class SparseVector {
 public:
  SparseVector()  = default;
  SparseVector(vector<size_type >&& idx, vector<T >&& val)
    : index_(idx), value_(val) {};
  SparseVector(size_type* idx, T* val, int len)
    : index_(idx, idx+len), value_(val, val+len) {};

  SparseVector(const SparseVector& s) = default;
  SparseVector(SparseVector&& s) = default;

  SparseVector&operator=(const SparseVector& s) = default;
  SparseVector&operator=(SparseVector&& s) = default;

  size_t size() const {
    return index_.size();
  }
  void reserve(size_t size) {
    index_.reserve(size);
    value_.reserve(size);
  }
  void push_back(size_type idx, T val) {
    index_.push_back(idx);
    value_.push_back(val);
  }
  vector<size_type > index_;
  vector<T >         value_;
};


/***
 *
 * \brief Sparse Matrix Multiplication Layer
 */
class Layer {
 public:
  Layer(size_type I, size_type O, Activation type)
    : I_(I), O_(O), type_(type) {
    weight_ = new T[I * O];
    bias_ = new T[O];
  }

  virtual ~Layer() {
    delete [] weight_;
    delete [] bias_;
  }

  /**
   * \brief y = \sigma(xW + b)
   * \param x Sparse Vector
   * \return y Sparse Vector
   */
  SparseVector forward(const SparseVector& x) {
    SparseVector y;
    T threshold = (type_ == Activation::ReLu) ? 0.f : -1e10f;
    for (int o = 0; o < O_; ++o) {
      T mm = 0;
      for (int s = 0; s < x.size(); ++s) {
        size_type i = x.index_[s];
        mm += * (weight_ + O_ * x.index_[i] + o);
      }
      if (mm > threshold)
        y.push_back(o, mm);
    }

    if (type_ == Activation::SoftMax) {
      // TODO
    }
    return y;
  }

  /**
   * \brief calculated gradient with respect to weight and input
   *        according to formula: g_W = gx; g_b = g; g_I = gW';
   *        update the parameters with Optimization Algorithm:
   *        P -=  lr * Gradient
   * \param x sparse vector, memorize input for calculate gradient
   *        with respect to weight_
   * \param g gradient with respect to the output of forward
   * \param compute_gx should compute gradient with respect to x
   * \return gradient with respect to x
   */
  SparseVector backward(const SparseVector& g,
                        const SparseVector& a,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx) {
    T lr = optimizer.lr;
    // compute gradient and update with respect to the weight
    // gw[I_, O_] = g[O_] @ x[I_]
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
    return gx;
  }

 private:
  size_type  I_;
  size_type  O_;
  Activation type_;
  T*         weight_;
  T*         bias_;
};


class SoftMaxCrossEntropy {
 public:
  /**
   * \param p SoftMax estimate
   * \param y true labels for classification task,
   * \return gradient with respect to the pre-SoftMax output
   *         according to formula: g_i = p_i - y_i
   */
  static SparseVector compute(const SparseVector& p,
                         const vector<size_type >& y, T& loss) {
    loss = 0;

    SparseVector grad;
    size_t reserve_size = std::max(p.size(), y.size());
    grad.reserve(reserve_size);

    T y_prob = (T)1.0 / y.size();

    size_type i_p = 0;
    size_type i_y = 0;
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
};
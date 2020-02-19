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

  SparseVector& operator=(const SparseVector& s) = default;
  SparseVector& operator=(SparseVector&& s) = default;

  void clear() {
    index_.clear();
    value_.clear();
  }
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
  Layer(size_type I, size_type O, Activation type);
  ~Layer() ;

  /**
   * \brief y = \sigma(xW + b)
   * \param x Sparse Vector
   * \return y Sparse Vector
   */
  SparseVector forward(const SparseVector& x);

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
  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx);

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
                         const vector<size_type >& y, T* loss) ;
};
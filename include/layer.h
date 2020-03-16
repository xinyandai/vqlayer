//
// \author Xinyan DAI (xinyan.dai@outlook.com)
// Created by xinyan DAI on 17/2/2020.
//

#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
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

class AbstractLayer : public Interface {
 public:
  AbstractLayer(size_type I, size_type O, Activation type);
  virtual ~AbstractLayer() ;

  AbstractLayer(const AbstractLayer& layer);
  AbstractLayer(AbstractLayer&& layer) noexcept;

  virtual T get_w(size_type i, size_type o) const;
  T get_b(size_type o) const {
    return bias_[o];
  }
  void initialize();

  SparseVector forward(const SparseVector& x) override;
  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx) override;

  virtual SparseVector backward_x(const SparseVector& g,
                                  const SparseVector& x) = 0;
  virtual void backward_w(const SparseVector& g,
                          const SparseVector& x,
                          const Optimizer& optimizer);

  virtual void backward_b(const SparseVector& g,
                          const SparseVector& x,
                          const Optimizer& optimizer);
  /**
   * \brief default implementation for forward(x)
   * \param x
   * \return
   */
  SparseVector default_forward(const SparseVector &x);
  /**
   * \brief default implementation for backward_x(g, x)
   * \param g
   * \param x
   * \return
   */
  SparseVector default_backward_x(const SparseVector &g,
                                  const SparseVector &x);



 public:
  const size_type  I_;
  const size_type  O_;
  const Activation type_;
 protected:
  T*               bias_;
};

/**
 * \brief Sparse Matrix Multiplication Layer
 */
class Layer : public AbstractLayer {
 public:
  Layer(size_type I, size_type O, Activation type);
  ~Layer() override;

  Layer(const Layer& l);
  Layer(Layer&& l) noexcept;

  const T* weight() { return weight_; }
  const T* bias() { return bias_; }

  void initialize(const vector<T >& w, const vector<T >& b);
  void initialize();

  T get_w(size_type i, size_type o) const override;

  SparseVector forward(const SparseVector& x) override;

  SparseVector backward_x(const SparseVector& g,
                          const SparseVector& x) override;

  void backward_w(const SparseVector& g,
                  const SparseVector& x,
                  const Optimizer& optimizer) override;

 protected:
  T*               weight_;
};




/**
* \brief Product Vector Quantized Sparse Matrix Multiplication Layer
*/
class PQLayer : public AbstractLayer {
 public:
  PQLayer(size_type I, size_type O, Activation type);
  ~PQLayer() override;

  PQLayer(const PQLayer& l);
  PQLayer(PQLayer&& l) noexcept;

  void initialize();
  T get_w(size_type i, size_type o) const override;
  SparseVector forward(const SparseVector& x) override;

  SparseVector backward_x(const SparseVector& g,
                          const SparseVector& x) override;

  void backward_w(const SparseVector& g,
                  const SparseVector& x,
                  const Optimizer& optimizer) override;

 private:
  const size_type  D_;     // sub dimension D_ = O_ / M_
  T*               dict_;  // shape of [M_, Ks, D_]
  CodeType *       code_;  // shape of [O_, M_]
#ifdef NEQ
  T*               norm_;  // shape of [O_, M_]
#endif
};

/**
* \brief Residual Vector Quantized Sparse Matrix Multiplication Layer
*/
class RQLayer : public AbstractLayer {
 public:
  RQLayer(size_type I, size_type O, Activation type);
  ~RQLayer() override;

  RQLayer(const RQLayer& l);
  RQLayer(RQLayer&& l) noexcept;

  void initialize();
  T get_w(size_type i, size_type o) const override;
  SparseVector forward(const SparseVector& x) override;

  SparseVector backward_x(const SparseVector& g,
                          const SparseVector& x) override;

  void backward_w(const SparseVector& g,
                  const SparseVector& x,
                  const Optimizer& optimizer) override;

 protected:
  T*               norm_;  //
  T*               dict_;  // shape of [R_, Ks, I_]
  CodeType *       code_;  // shape of [O_, R_]
};

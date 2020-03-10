//
// \author Xinyan DAI (xinyan.dai@outlook.com)
// Created by xinyan DAI on 17/2/2020.
//

#pragma once
#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <mutex>
#include <thread>

//#define ThreadSafe

using std::mutex;
using std::vector;
using std::shared_ptr;

using T = float;
using size_type = int;

#define CodeType uint8_t
#define Ks 256
#define M_ 8 // M of Product Quantization

size_type vq(const T* w, const T* dict, size_type ks, size_type d);
void rq(const T* w, const T* dict, CodeType* code, T* norm,
        size_type ks, size_type m, size_type d);

T normalize(T* w, size_type d);
T l2dist_sqr(const T *a, const T *b, size_type d);
void normalize_codebook(T* dict, size_type m, size_type ks, size_type d);
void rq_codebook(T* centroid, size_type M, size_type n,
                 size_type ks, size_type d, size_type iter);
void vq_codebook(T* centroid, const size_type n,
                 size_type ks, size_type d, size_type iter);
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
  SparseVector(const vector<T>& s): index_(s.size()), value_(s) {
    std::iota(index_.begin(), index_.end(), 0);
  };

  SparseVector& operator=(const SparseVector& s) = default;
  SparseVector& operator=(SparseVector&& s) = default;
  SparseVector& operator=(const vector<T>& s) {
    index_.resize(s.size());
    std::iota(index_.begin(), index_.end(), 0);
    value_ = s;
  };
  SparseVector& operator=(vector<T >&& s) {
    index_.resize(s.size());
    std::iota(index_.begin(), index_.end(), 0);
    value_ = s;
  };

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

class AbstractLayer {
 public:
  AbstractLayer() = default;
  virtual ~AbstractLayer() = default;

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
                                bool compute_gx)
                                = 0;
};

/**
 * \brief Sparse Matrix Multiplication Layer
 */
class Layer : public AbstractLayer {
 public:
  Layer(size_type I, size_type O, Activation type);
  ~Layer() override ;

  Layer(const Layer& l);
  Layer(Layer&& l) noexcept;

  const T* weight() { return weight_; }
  const T* bias() { return bias_; }

  void initialize(const vector<T >& w, const vector<T >& b);
  void initialize();
  SparseVector forward(const SparseVector& x) override ;
  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx)
  override;

 private:
  const size_type  I_;
  const size_type  O_;
  const Activation type_;
  T*               weight_;
  T*               bias_;
#ifdef ThreadSafe
  vector<mutex >   weight_lock_;
  vector<mutex >   bias_lock_;
#endif
};




/**
* \brief Vectorized Sparse Matrix Multiplication Layer
*/
class VQLayer : public AbstractLayer {
 public:
  VQLayer(size_type I, size_type O, Activation type);
  ~VQLayer() override;

  VQLayer(const VQLayer& l);
  VQLayer(VQLayer&& l) noexcept;

  void initialize();
  SparseVector forward(const SparseVector& x) override ;
  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx) override;

 private:
  const size_type  I_;
  const size_type  O_;
  const size_type  D_; //sub dimension D_ = O_ / M_
  const Activation type_;
  T*               dict_; // shape of [M_, Ks, D_]
  CodeType *       code_; // shape of [O_, M_]
};

/**
* \brief Vectorized Sparse Matrix Multiplication Layer
*/
class RQLayer : public AbstractLayer {
 public:
  RQLayer(size_type I, size_type O, Activation type);
  ~RQLayer() override;

  RQLayer(const RQLayer& l);
  RQLayer(RQLayer&& l) noexcept;

  void initialize();
  SparseVector forward(const SparseVector& x) override ;
  SparseVector backward(const SparseVector& g,
                        const SparseVector& x,
                        const Optimizer& optimizer,
                        bool compute_gx) override;

 private:
  const size_type  I_;
  const size_type  O_;
  const Activation type_;
  T*               norm_; //
  T*               dict_; // shape of [R_, Ks, I_]
  CodeType *       code_; // shape of [O_, R_]
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
                              const vector<size_type >& y, T* loss);
};
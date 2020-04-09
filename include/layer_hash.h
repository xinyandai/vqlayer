//
// Created by xinyan on 16/3/2020.
//

/**
 * \brief Sparse Matrix Multiplication Layer
 */
#include "layer_abstract.h"

template <Activation Act, bool Select>
class HashLayer : public AbstractLayer<Act, Select> {
 public:
  HashLayer(size_type I, size_type O, size_type S)
  : AbstractLayer<Act, Select>(I, O), S_(S) {
    bucket_ = new T[S];
    initialize();
  }
  ~HashLayer() override {
    delete [] bucket_;
  }


  void initialize() {
    AbstractLayer<Act, Select>::initialize();
    std::default_random_engine generator(1016);
    std::uniform_real_distribution<T > dist(
      0.f, 1.f / std::sqrt(this->I_ / 2.0f));
    T* w = bucket_;
    for (int i = 0; i < this->S_; i++) {
      *(w++) = dist(generator);
    }
  }

  size_type hash(size_type i, size_type o) const {
    return std::hash<size_type >{}(i * this->O_ + o) % this->S_;
  }

  T get_w(size_type i, size_type o) const override {
    return bucket_[this->hash(i, o)];
  }


  void backward_w(const SparseVector& g,
                  const SparseVector& x,
                  const Optimizer& optimizer) override {
    T lr = optimizer.lr;

    // compute gradient and update with respect to the weight
    // gw[this->I_, this->O_] = x[1, this->I_]' g[1, this->O_]
    for (int i = 0; i < x.size(); ++i) {
      for (int o = 0; o < g.size(); ++o) {
        T grad = x.value_[i] * g.value_[o];
        bucket_[this->hash(x.index_[i], g.index_[o]) ]-= lr * grad;
      }
    }
  }
 public:
  const size_type  S_;  // size of memory usage
 protected:
  T*               bucket_;
};

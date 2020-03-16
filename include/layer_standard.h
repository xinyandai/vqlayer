//
// Created by xinyan on 16/3/2020.
//

/**
 * \brief Sparse Matrix Multiplication Layer
 */
#include "layer_abstract.h"

template <Activation Act, bool Select>
class Layer : public AbstractLayer<Act, Select> {
 public:
  Layer(size_type I, size_type O)
  : AbstractLayer<Act, Select>(I, O) {
    weight_ = new T[I * O];
    initialize();
  }
  ~Layer() override {
    delete [] weight_;
  }

  const T* weight() { return this->weight_; }
  const T* bias() { return this->bias_; }

  void initialize(const vector<T >& w, const vector<T >& b) {
    std::memcpy(this->weight_, w.data(), w.size() * sizeof(T));
    std::memcpy(this->bias_, b.data(), b.size() * sizeof(T));
  }

  void initialize() {
    AbstractLayer<Act, Select>::initialize();
    std::default_random_engine generator(1016);
    std::uniform_real_distribution<T > dist(
      0.f, 1.f / std::sqrt(this->I_ / 2.0f));
    T* w = weight_;
    for (int i = 0; i < this->I_; i++) {
      for (int j = 0; j < this->O_; j++) {
        *(w++) = dist(generator);
      }
    }
  }

  T get_w(size_type i, size_type o) const override {
    return weight_[i * this->O_ + o];
  }

  void backward_w(const SparseVector& g,
                  const SparseVector& x,
                  const Optimizer& optimizer) override {
    T lr = optimizer.lr;
    volatile T* weight = weight_;

    // compute gradient and update with respect to the weight
    // gw[this->I_, this->O_] = x[1, this->I_]' g[1, this->O_]
    for (int i = 0; i < x.size(); ++i) {
      volatile T* w = weight + this->O_ * x.index_[i];
      for (int o = 0; o < g.size(); ++o) {
        T grad = x.value_[i] * g.value_[o];
        w[g.index_[o]] -= lr * grad;
      }
    }
  }

 protected:
  T*               weight_;
};

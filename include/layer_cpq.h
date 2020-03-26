//
// Created by xinyan on 18/3/2020.
//
#include <limits>
#include <utility>
#include "layer_abstract.h"
/**
* \brief Column-wise Product Vector Quantized Sparse Matrix Multiplication Layer
*/
template <
  Activation Act, bool Select, bool NQ,
  size_type M_ = 2, size_type Ks = 256, typename CodeType = uint8_t
>
class CPQLayer : public AbstractLayer<Act, Select> {
 public:
  CPQLayer(size_type I, size_type O)
    : AbstractLayer<Act, Select>(I, O), D_(this->O_/M_) {
    if (this->O_ % M_ > 0)
      throw std::runtime_error("O_ is not dividable by M_");

    code_ = new CodeType[this->I_ * M_];
    dict_ = new T[M_ * Ks * D_];
    if constexpr (NQ)
      norm_ = new T[this->I_ * M_];
    else
      norm_ = nullptr;
    initialize();
  }
  ~CPQLayer() {
    delete [] code_;
    delete [] dict_;
    delete [] norm_;
  }

  void initialize();

  T get_w(size_type i, size_type o) const override;
  SparseVector forward(const SparseVector& x) override;

  SparseVector backward_x(const SparseVector& g,
                          const SparseVector& x) override;

  void backward_w(const SparseVector& x,
                  const SparseVector& g,
                  const Optimizer& optimizer) override;

 private:
  const size_type  D_;     // sub dimension D_ = O_ / M_
  T*               dict_;  // shape of [M_, Ks, D_]
  CodeType *       code_;  // shape of [I_, M_]
  T*               norm_;  // shape of [I_, M_]
};

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
>
void CPQLayer<Act, Select, NQ, M_, Ks, CodeType>::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = this->code_;
  for (int i = 0; i < M_ * this->I_; ++i) {
    *(code++) = static_cast<CodeType>(codes_dist(generator));
  }
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  if constexpr (NQ) {
    T* norm = norm_;
    for (int i = 0; i < this->I_ * M_; ++i) {
      *(norm++) = distribution(generator);
    }
  }
//  #define LEARNED_CODE_BOOK
#ifndef LEARNED_CODE_BOOK
  T* w = dict_;
  for (int i = 0; i < M_ * Ks * D_; i++) {
    *(w++) = distribution(generator);
  }
#else
  vq_codebook(dict_, /*n*/65536, Ks, D_, /*iter*/20);
  for (int i = 1; i < M_; ++i) {
    std::memcpy(&dict_[i * Ks * D_], dict_, Ks * D_ * sizeof(T));
  }
#endif
  if constexpr (NQ) {
    normalize_codebook(dict_, M_, Ks, D_);
  }
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
>
T CPQLayer<Act, Select, NQ, M_, Ks, CodeType>
::get_w(size_type i, size_type o) const {
  static size_type count = 0;
  if (count++ == 0)
    std::cerr << "Not efficient, for test only" << std::endl;
  int m = 0;
  while (o >= D_) {
    m++;
    o -= D_;
  }
  CodeType c = code_[i * M_ + m];
  if constexpr (NQ) {
    return dict_[m * Ks * D_ + c * D_ + o] * norm_[i * M_ + m];
  } else {
    return dict_[m * Ks * D_ + c * D_ + o];
  }
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType>
SparseVector CPQLayer<Act, Select, NQ, M_, Ks, CodeType>
::forward(const SparseVector& x) {
  SparseVector y;

  volatile T* dict = dict_;         // shape of [M_, Ks, D_]
  volatile CodeType* code = code_;  // shape of [I_, M_]
  TopSelector selector(10 + this->O_/10);
  T max_v = std::numeric_limits<T>::min();

  T* result = new T[this->D_];
  for (int m = 0, start_idx = 0; m < M_; ++m, start_idx += this->D_) {
    for (int dim = 0; dim < this->D_; ++dim) {
      result[dim] = this->get_b(dim + start_idx);
    }
    for (int i = 0; i < x.size(); ++i) {
      T xv = x.value_[i];
      CodeType c = code[x.index_[i] * M_ + m];
      auto w = &dict[m * Ks * this->D_  + c * this->D_];
      for (int dim = 0; dim < this->D_; ++dim) {
        if constexpr (NQ)
          result[dim] += xv * w[dim] * norm_[x.index_[i] * M_ + m];
        else
          result[dim] += xv * w[dim];
      }
    }
    for (int dim = 0; dim < this->D_; ++dim) {
      insert<Act, Select>(dim + start_idx, result[dim], max_v, selector, y);
    }
  }
  delete[] result;
  return softmax<Act, Select>(selector, y, max_v);
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType>
SparseVector CPQLayer<Act, Select, NQ, M_, Ks, CodeType>
::backward_x(const SparseVector& g, const SparseVector& x) {
  volatile T* dict = dict_;         // shape of [M_, Ks, D_]
  volatile CodeType* code = code_;  // shape of [I_, M_]

  // calculate look up table:  [M_, Ks]
  T tables[M_][Ks];

  for (int k = 0; k < Ks; ++k) {
    size_type idx = 0;
    for (int m = 0; m < M_; ++m) {
      // TODO(Xinyan) to be optimized
      volatile T* d = dict + m * Ks * D_ + k * D_;
      T mm = 0;
      size_type begin_idx = m * D_;
      size_type end_idx = begin_idx + D_;
      while (g.size() > idx && g.index_[idx] < end_idx) {
        mm += g.value_[idx] * d[g.index_[idx] - begin_idx];
        idx++;
      }
      tables[m][k] = mm;
    }
  }

  SparseVector y = x;
  for (int i = 0; i < x.size(); ++i) {
    T mm = 0;
    volatile CodeType* c = &code[x.index_[i] * M_];
    T* norm = &norm_[x.index_[i] * M_];
#pragma unroll
    for (int m = 0; m < M_; ++m) {
      if constexpr (NQ) {
        // *c = code[i * M_ + m] * norm_[i * M_ + m]
        mm += tables[m][*(c++)] *  (*(norm++));
      } else {
        // *c = code[i * M_ + m]
        mm += tables[m][*(c++)];
      }
    }
    y.value_[i] = mm;
  }

  return y;
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
>
void CPQLayer<Act, Select, NQ, M_, Ks, CodeType>
::backward_w(const SparseVector& g,
             const SparseVector& x,
             const Optimizer& optimizer) {
  // compute gradient and update with respect to the weight
  // gw[i_, o_] = x[1, i_]' g[1, o_]
  T* const dict = dict_;         // shape of [M_, Ks, D_]
  CodeType* const code = code_;  // shape of [I_, M_]
  T * w = new T[D_];
  T lr = optimizer.lr;

  for (int i = 0; i < x.size(); ++i) {
    size_type o = 0;
    for (int m = 0, begin_idx = 0; m < M_; ++m, begin_idx+=D_) {
      size_type end_idx = begin_idx + D_;
      auto& c = code[x.index_[i] * M_ + m];
      std::memcpy(w, &dict[m * Ks * D_ + c * D_], D_ * sizeof(T));

      T* norm = nullptr;
      if constexpr (NQ) {
        norm = &norm_[x.index_[i] * M_ + m];
        for (int dim = 0; dim < D_; ++dim) {
          w[dim] *= *norm;
        }
      }
      for (; g.size() > o && g.index_[o] < end_idx; o++) {
        T grad = g.value_[o] * x.value_[i];
        w[g.index_[o] - begin_idx] -= lr * grad;
      }

      if constexpr (NQ) {
        c = static_cast<CodeType>(nvq(norm, w, &dict[m * Ks * D_], Ks, D_));
      } else {
        c = static_cast<CodeType>(vq(w, &dict[m * Ks * D_], Ks, D_));
      }
    }
  }

  delete [] w;
}

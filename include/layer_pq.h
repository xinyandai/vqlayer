//
// Created by xinyan on 16/3/2020.
//
#pragma once
#include "vq.h"
#include "layer_abstract.h"
/**
* \brief Product Vector Quantized Sparse Matrix Multiplication Layer
*/
template <
  Activation Act, bool Select, bool NQ,
  size_type M_ = 2, size_type Ks = 256, typename CodeType = uint8_t
    >
class PQLayer : public AbstractLayer<Act, Select> {
 public:
  PQLayer(size_type I, size_type O)
        : AbstractLayer<Act, Select>(I, O), D_(this->I_/M_) {
    if (this->I_ % M_ > 0)
      throw std::runtime_error("I_ is not dividable by M_");

    code_ = new CodeType[this->O_ * M_];
    dict_ = new T[M_ * Ks * D_];
    if (NQ)
      norm_ = new T[this->O_ * M_];
    initialize();
  }
  ~PQLayer() {
    delete [] code_;
    delete [] dict_;
    delete [] norm_;
  }

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
  T*               norm_;  // shape of [O_, M_]
};

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
  >
void PQLayer<Act, Select, NQ, M_, Ks, CodeType>::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = this->code_;
  for (int i = 0; i < M_ * this->O_; ++i) {
    *(code++) = static_cast<CodeType>(codes_dist(generator));
  }
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  if constexpr (NQ) {
    T* norm = norm_;
    for (int i = 0; i < this->O_ * M_; ++i) {
      *(norm++) = distribution(generator);
    }
  }
// #define LEARNED_CODE_BOOK
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
T PQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::get_w(size_type i, size_type o) const {
  static size_type count = 0;
  if (count++ == 0)
    std::cerr << "Not efficient, for test only" << std::endl;
  int m = 0;
  while (i >= D_) {
    m++;
    i -= D_;
  }
  CodeType c = code_[o * M_ + m];
  if constexpr (NQ) {
    return dict_[m * Ks * D_ + c * D_ + i] * norm_[o * M_ + m];
  } else {
    return dict_[m * Ks * D_ + c * D_ + i];
  }
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType>
SparseVector PQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::forward(const SparseVector& x) {
  SparseVector y;

  volatile T* dict = dict_;         // shape of [M_, Ks, D_]
  volatile CodeType* code = code_;  // shape of [O_, M_]

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
      while (x.size() > idx && x.index_[idx] < end_idx) {
        mm += x.value_[idx] * d[x.index_[idx] - begin_idx];
        idx++;
      }
      tables[m][k] = mm;
    }
  }

  TopSelector selector(10 + this->O_/10);
  T max_v = std::numeric_limits<T>::min();
  volatile CodeType* c = code;
  T* norm = norm_;

  for (int o = 0; o < this->O_; ++o) {
    T mm = this->get_b(o);
#pragma unroll
    for (int m = 0; m < M_; ++m) {
      if constexpr (NQ) {
        // *c = code[o * M_ + m] * norm_[o * M_ + m]
        mm += tables[m][*(c++)] * (*(norm++));
      }
      else {
        mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
      }
    }

    insert<Act, Select>(o, mm, max_v, selector, y);
  }

  return softmax<Act, Select>(selector, y, max_v);
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType>
SparseVector PQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::backward_x(const SparseVector& g, const SparseVector& x) {
  // Compute gradient  with respect to the input:
  // gx[I_] = w[I_, O_], g[O_].
  // Previous layer's activation function must be ReLu,
  // since SoftMax only exist in last layer.
  T* const dict = dict_;         // shape of [M_, Ks, D_]
  CodeType* const code = code_;  // shape of [O_, M_]
  SparseVector gx = x;
  gx = x;
  size_type idx = 0;
  for (int m = 0; m < M_; ++m) {
    size_type begin_idx = m * D_;
    size_type end_idx = begin_idx + D_;

    for (; x.size() > idx && x.index_[idx] < end_idx; idx++) {
      T grad = 0;
      for (int o = 0; o < g.size(); ++o) {
        CodeType c = code[g.index_[o] * M_ + m];
        // TODO(Xinyan) to be optimized
        if constexpr (NQ) {
          grad += g.value_[o] * dict[m * Ks * D_ + c * D_ +
            x.index_[idx] - begin_idx] * norm_[g.index_[o] * M_ + m];
        }
        else {
          grad += g.value_[o] * dict[m * Ks * D_ + c * D_ +
            x.index_[idx] - begin_idx];
        }
      }
      gx.value_[idx] = grad;
    }
  }
  return gx;
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
  >
void PQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::backward_w(const SparseVector& g,
               const SparseVector& x,
               const Optimizer& optimizer) {
  // compute gradient and update with respect to the weight
  // gw[i_, o_] = x[1, i_]' g[1, o_]
  T* const dict = dict_;         // shape of [M_, Ks, D_]
  CodeType* const code = code_;  // shape of [O_, M_]
  T * w = new T[D_];
  T lr = optimizer.lr;

  for (int o = 0; o < g.size(); ++o) {
    size_type idx = 0;
    for (int m = 0; m < M_; ++m) {
      size_type begin_idx = m * D_;
      size_type end_idx = begin_idx + D_;
      auto& c = code[g.index_[o] * M_ + m];
      std::memcpy(w, &dict[m * Ks * D_ + c * D_], D_ * sizeof(T));

      T* norm;
      if constexpr (NQ) {
        norm = &norm_[g.index_[o] * M_ + m];
        for (int dim = 0; dim < D_; ++dim) {
          w[dim] *= *norm;
        }
      }
      for (; x.size() > idx && x.index_[idx] < end_idx; idx++) {
        T grad = x.value_[idx] * g.value_[o];
        w[x.index_[idx] - begin_idx] -= lr * grad;
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
//
// Created by xinyan on 2020/3/3.
//


#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>

#include "../include/layer.h"


PQLayer::PQLayer(size_type I, size_type O,
                 Activation type)
                 : AbstractLayer(I, O, type), D_(I_/M_) {
  if (I_ % M_ > 0)
    throw std::runtime_error("I_ is not dividable by M_");

  code_ = new CodeType[O_ * M_];
  dict_ = new T[M_ * Ks * D_];
#ifdef NEQ
  norm_ = new T[O_ * M_];
#endif
  initialize();
}

PQLayer::PQLayer(const PQLayer& c) : PQLayer(c.I_, c.O_, c.type_) {
  std::memcpy(code_, c.code_, O_ * M_ * sizeof(CodeType));
  std::memcpy(dict_, c.dict_,  M_ * Ks * D_ * sizeof(T));
#ifdef NEQ
  std::memcpy(norm_, c.norm_,  M_ * D_ * sizeof(T));
#endif
}

PQLayer::PQLayer(PQLayer&& c) noexcept:
                 AbstractLayer(c.I_, c.O_, c.type_),
#ifdef NEQ
                 norm_(c.norm_),
#endif
                 D_(c.D_), code_(c.code_), dict_(c.dict_) {
  c.dict_ = nullptr;
  c.code_ = nullptr;
#ifdef NEQ
  c.norm_ = nullptr;
#endif
}

PQLayer::~PQLayer() {
  delete [] code_;
  delete [] dict_;
#ifdef NEQ
  delete [] norm_;
#endif
}

void PQLayer::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = code_;
  for (int i = 0; i < M_ * O_; ++i) {
    *(code++) = static_cast<CodeType>(codes_dist(generator));
  }
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
#ifdef NEQ
  T* norm = norm_;
  for (int i = 0; i < O_ * M_; ++i) {
    *(norm++) = distribution(generator);
  }
#endif
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
#ifdef NEQ
  normalize_codebook(dict_, M_, Ks, D_);
#endif
}

T PQLayer::get_w(size_type i, size_type o)  {
  static size_type count = 0;
  if (count++ == 0)
    std::cerr << "Not efficient, for test only" << std::endl;
  int m = 0;
  while (i >= D_) {
    m++;
    i -= D_;
  }
  CodeType c = code_[o * M_ + m];
#ifdef NEQ
  return dict_[m * Ks * D_ + c * D_ + i] * norm_[o * M_ + m];
#else
  return dict_[m * Ks * D_ + c * D_ + i];
#endif
}

SparseVector PQLayer::forward(const SparseVector& x) {
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

  // Relu activation remove the negative output
  if (type_ == Activation::ReLu) {
    volatile CodeType* c = code;
#ifdef NEQ
    T* norm = norm_;
#endif
    for (int o = 0; o < O_; ++o) {
      T mm = get_b(o);
#pragma unroll
      for (int m = 0; m < M_; ++m) {
#ifdef NEQ
        // *c = code[o * M_ + m] * norm_[o * M_ + m]
        mm += tables[m][*(c++)] * (*(norm++));
#else
        mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
#endif
      }
      if (mm > 0) {
        y.push_back(o, mm);
      }
    }
  } else if (type_ == Activation::SoftMax) {
    T max_v = std::numeric_limits<T>::min();
    volatile CodeType* c = code;
#ifdef NEQ
    T* norm = norm_;
#endif
    for (int o = 0; o < O_; ++o) {
      T mm = get_b(o);
#pragma unroll
      for (int m = 0; m < M_; ++m) {
#ifdef NEQ
        // *c = code[o * M_ + m] * norm_[o * M_ + m]
        mm += tables[m][*(c++)] * (*(norm++));
#else
        mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
#endif
      }

      y.push_back(o, mm);
      if (mm > max_v)
        max_v = mm;
    }
    // C is a constant for computation stability
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


SparseVector PQLayer::backward_x(const SparseVector& g,
                                 const SparseVector& x) {
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
#ifdef NEQ
        grad += g.value_[o] * dict[m * Ks * D_ + c * D_ +
          x.index_[idx] - begin_idx] * norm_[g.index_[o] * M_ + m];
#else
        grad += g.value_[o] * dict[m * Ks * D_ + c * D_ +
          x.index_[idx] - begin_idx];
#endif
      }
      gx.value_[idx] = grad;
    }
  }
  return gx;
}

void PQLayer::backward_w(const SparseVector& g,
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
#ifdef NEQ
      T* norm = &norm_[g.index_[o] * M_ + m];
      for (int dim = 0; dim < D_; ++dim) {
        w[dim] *= *norm;
      }
#endif
      for (; x.size() > idx && x.index_[idx] < end_idx; idx++) {
        T grad = x.value_[idx] * g.value_[o];
        w[x.index_[idx] - begin_idx] -= lr * grad;
      }
#ifdef NEQ
      c = static_cast<CodeType>(nvq(norm, w, &dict[m * Ks * D_], Ks, D_));
#else
      c = static_cast<CodeType>(vq(w, &dict[m * Ks * D_], Ks, D_));
#endif
    }
  }
  delete [] w;
}



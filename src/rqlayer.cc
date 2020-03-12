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



RQLayer::RQLayer(size_type I, size_type O,
                 Activation type) : AbstractLayer(I, O, type) {

  code_ = new CodeType[O_ * M_];
  dict_ = new T[M_ * Ks * I_];
  norm_ = new T[O_];
  initialize();
}

RQLayer::RQLayer(const RQLayer& c) : RQLayer(c.I_, c.O_, c.type_) {
  std::memcpy(code_, c.code_, O_ * M_ * sizeof(CodeType));
  std::memcpy(dict_, c.dict_,  M_ * Ks * I_ * sizeof(T));
  std::memcpy(norm_, c.norm_,  O_ * sizeof(T));
}

RQLayer::RQLayer(RQLayer&& c) noexcept:
                 AbstractLayer(c.I_, c.O_, c.type_),
                 code_(c.code_), dict_(c.dict_), norm_(c.norm_) {
  c.dict_ = nullptr;
  c.code_ = nullptr;
  c.norm_ = nullptr;
}

RQLayer::~RQLayer() {
  delete [] code_;
  delete [] dict_;
  delete [] norm_;
}

void RQLayer::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = code_;
  for (int i = 0; i < M_ * O_; ++i) {
    *(code++) = static_cast<CodeType>(codes_dist(generator));
  }
  
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  T* norm = norm_;
  for (int i = 0; i < O_; ++i) {
    *(norm++) = distribution(generator);
  }
// #define LEARNED_CODE_BOOK
#ifndef LEARNED_CODE_BOOK
  T* w = dict_;
  for (int i = 0; i < M_ * Ks * I_; ++i) {
    *(w++) = distribution(generator);
  }
  normalize_codebook(dict_, M_, Ks, I_);
#else
  rq_codebook(/*centroid*/dict_, M_, /*n*/ 65536,
              /*ks*/Ks, /*d*/I_, /*iter*/20);
#endif
}

T RQLayer::get_w(size_type i, size_type o)  {
  static size_type count = 0;
  if (count++ == 0)
    std::cerr << "Not efficient, for test only" << std::endl;
  T w = 0;
  for (int m = 0; m < M_; ++m) {
    CodeType c = code_[o * M_ + m];
    w += dict_[m * Ks * I_ + c * I_ + i];
  }
  return w * norm_[o];
}


SparseVector RQLayer::forward(const SparseVector& x) {
//  return AbstractLayer::forward(x);
  SparseVector y;

  volatile T* dict = dict_;         // shape of [M_, Ks, I_]
  volatile CodeType* code = code_;  // shape of [O_, M_]

  // calculate look up table:  [M_, Ks]
  T tables[M_][Ks];
  volatile T* d = dict; 
  for (int m = 0; m < M_; ++m) {
    for (int k = 0; k < Ks; ++k, d+=I_) {
      T mm = 0;
      for (size_type idx = 0; idx < x.size(); ++idx) {
        mm += x.value_[idx] * d[x.index_[idx]];
      }
      tables[m][k] = mm;
    }
  }

  // Relu activation remove the negative output
  if (type_ == Activation::ReLu) {
    volatile T* norm = norm_;
    volatile CodeType* c = code;
    for (int o = 0; o < O_; ++o) {
      T mm = 0;
#pragma unroll
      for (int m = 0; m < M_; ++m) {
        mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
      }
      mm *= *(norm++);
      if (mm > 0) {
        y.push_back(o, mm);
      }
    }
  }

  else if (type_ == Activation::SoftMax) {
    T max_v = std::numeric_limits<T>::min();
    volatile T* norm = norm_;
    volatile CodeType* c = code;
    for (int o = 0; o < O_; ++o) {
      T mm = 0;
#pragma unroll
      for (int m = 0; m < M_; ++m) {
        mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
      }
      mm *= *(norm++);
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


SparseVector RQLayer::backward_x(const SparseVector& g,
                                 const SparseVector& x) {
//  return AbstractLayer::backward_x(g, x);
  T* const norm = norm_;         // shape of [O_]
  CodeType* const code = code_;  // shape of [O_, M_]
  // Compute gradient  with respect to the input:
  // gx[I_] = w[I_, O_], g[O_].
  // Previous layer's activation function must be ReLu,
  // since SoftMax only exist in last layer.
  SparseVector gx = x;
  // gx[I_] = w[I_, O_], g[O_].
  std::memset(gx.value_.data(), 0, gx.size() * sizeof(T));
  for (int o = 0; o < g.size(); ++o) {
    T grad_o = g.value_[o] * norm[g.index_[o]];
    for (int m = 0; m < M_; ++m) {
      CodeType c = code[g.index_[o] * M_ + m];
      T* d = &dict_[m * Ks * I_ + c * I_];
      for (int idx = 0; idx < x.size(); ++idx) {
        gx.value_[idx] += d[x.index_[idx]] * grad_o;
      }
    }
  }
  return gx;
}

void RQLayer::backward_w(const SparseVector& g,
                         const SparseVector& x,
                         const Optimizer& optimizer) {
  T lr = optimizer.lr;
  SparseVector gx;
  T* const dict = dict_;        // shape of [M_, Ks, D_]
  T* const norm = norm_;        // shape of [O_]
  CodeType* const code = code_; // shape of [O_, M_]
  // compute gradient and update with respect to the weight
  // gw[i_, o_] = x[1, i_]' g[1, o_]
  T * w = new T[I_];
  for (int o = 0; o < g.size(); ++o) {
    std::memset(w, 0, I_ * sizeof(T));
    for (int m = 0; m < M_; ++m) {
      auto& c = code[g.index_[o] * M_ + m];
      T* d = &dict[m * Ks * I_ + c * I_];
      for (int i = 0; i < I_; ++i) {
        w[i] += d[i];
      }
    }
    T norm_o = norm[g.index_[o]];
    for (int i = 0; i < I_; ++i) {
      w[i] *= norm_o;
    }

    T grad_o = g.value_[o];
    for (size_type idx = 0;  idx < x.size(); idx++) {
      T grad = x.value_[idx] * grad_o;
      w[x.index_[idx]] -= lr * grad;
    }
    rq(w, dict, &code[g.index_[o] * M_], &norm[g.index_[o]], Ks, M_, I_);
  }
  delete [] w;
}

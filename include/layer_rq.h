//
// Created by xinyan on 16/3/2020.
//
#pragma once
#include <limits>

/**
* \brief Residual Vector Quantized Sparse Matrix Multiplication Layer
*/
template <
  Activation Act, bool Select, bool NQ,
  size_type M_=2, size_type Ks = 256, typename CodeType = uint8_t
    >
class RQLayer : public AbstractLayer<Act, Select> {
 public:
  RQLayer(size_type I, size_type O)
        : AbstractLayer<Act, Select>(I, O) {
    code_ = new CodeType[O * M_];
    dict_ = new T[M_ * Ks * I];
    norm_ = new T[O];
    initialize();
  }

  ~RQLayer() override {
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

 protected:
  T*               norm_;  //
  T*               dict_;  // shape of [R_, Ks, I_]
  CodeType *       code_;  // shape of [O_, R_]
};

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
  >
void RQLayer<Act, Select, NQ, M_, Ks, CodeType>::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = code_;
  for (int i = 0; i < M_ * this->O_; ++i) {
    *(code++) = static_cast<CodeType>(codes_dist(generator));
  }

  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  T* norm = norm_;
  for (int i = 0; i < this->O_; ++i) {
    *(norm++) = distribution(generator);
  }
// #define LEARNED_CODE_BOOK
#ifndef LEARNED_CODE_BOOK
  T* w = dict_;
  for (int i = 0; i < M_ * Ks * this->I_; ++i) {
    *(w++) = distribution(generator);
  }
  normalize_codebook(dict_, M_, Ks, this->I_);
#else
  rq_codebook(/*centroid*/dict_, M_, /*n*/65536,
              /*ks*/Ks, /*d*/I_, /*iter*/20);
#endif
}

template <
  Activation Act, bool Select,bool NQ,
  size_type M_, size_type Ks, typename CodeType
  >
T RQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::get_w(size_type i, size_type o) const {
  static size_type count = 0;
  if (count++ == 0)
    std::cerr << "Not efficient, for test only" << std::endl;
  T w = 0;
  for (int m = 0; m < M_; ++m) {
    CodeType c = code_[o * M_ + m];
    w += dict_[m * Ks * this->I_ + c * this->I_ + i];
  }
  return w * norm_[o];
}

template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType
  >
SparseVector RQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::forward(const SparseVector& x) {
  SparseVector y;

  volatile T* dict = dict_;         // shape of [M_, Ks, I_]
  volatile CodeType* code = code_;  // shape of [O_, M_]

  // calculate look up table:  [M_, Ks]
  T tables[M_][Ks];
  volatile T* d = dict;
  for (int m = 0; m < M_; ++m) {
    for (int k = 0; k < Ks; ++k, d+=this->I_) {
      T mm = 0;
      for (size_type idx = 0; idx < x.size(); ++idx) {
        mm += x.value_[idx] * d[x.index_[idx]];
      }
      tables[m][k] = mm;
    }
  }

  volatile T* norm = norm_;
  volatile CodeType* c = code;

  TopSelector selector(10 + this->O_/10);
  T max_v = std::numeric_limits<T>::min();

  for (int o = 0; o < this->O_; ++o) {
    T mm = this->get_b(o);
#pragma unroll
    for (int m = 0; m < M_; ++m) {
      mm += tables[m][*(c++)];  // *c = code[o * M_ + m]
    }
    mm *= *(norm++);
    if (mm > 0) {
      y.push_back(o, mm);
    }

    insert<Act, Select>(o, mm, max_v, selector, y);
  }

  return softmax<Act, Select>(selector, y, max_v);
}

template <Activation Act, bool Select, bool NQ, size_type M_, size_type Ks, typename CodeType>
SparseVector RQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::backward_x(const SparseVector& g, const SparseVector& x) {
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
      T* d = &dict_[m * Ks * this->I_ + c * this->I_];
      for (int idx = 0; idx < x.size(); ++idx) {
        gx.value_[idx] += d[x.index_[idx]] * grad_o;
      }
    }
  }
  return gx;
}
template <
  Activation Act, bool Select, bool NQ,
  size_type M_, size_type Ks, typename CodeType>
void RQLayer<Act, Select, NQ, M_, Ks, CodeType>
  ::backward_w(const SparseVector& g,
               const SparseVector& x,
               const Optimizer& optimizer) {
  T lr = optimizer.lr;
  SparseVector gx;
  T* const dict = dict_;         // shape of [M_, Ks, D_]
  T* const norm = norm_;         // shape of [O_]
  CodeType* const code = code_;  // shape of [O_, M_]
  // compute gradient and update with respect to the weight
  // gw[i_, o_] = x[1, i_]' g[1, o_]
  T * w = new T[this->I_];
  for (int o = 0; o < g.size(); ++o) {
    std::memset(w, 0, this->I_ * sizeof(T));
    for (int m = 0; m < M_; ++m) {
      auto& c = code[g.index_[o] * M_ + m];
      T* d = &dict[m * Ks * this->I_ + c * this->I_];
      for (int i = 0; i < this->I_; ++i) {
        w[i] += d[i];
      }
    }
    T norm_o = norm[g.index_[o]];
    for (int i = 0; i < this->I_; ++i) {
      w[i] *= norm_o;
    }

    T grad_o = g.value_[o];
    for (size_type idx = 0;  idx < x.size(); idx++) {
      T grad = x.value_[idx] * grad_o;
      w[x.index_[idx]] -= lr * grad;
    }
    rq(w, dict, &code[g.index_[o] * M_], &norm[g.index_[o]], Ks, M_, this->I_);
  }
  delete [] w;
}
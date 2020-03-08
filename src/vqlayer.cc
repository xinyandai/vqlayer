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

T l2dist(const T* a, const T* b, size_type d) {
  T dist = 0;
  for (int i = 0; i < d; ++i) {
    T diff = (*(a++)) - (*(b++));
    dist += diff * diff;
  }
  return dist;
}

size_type vq(const T* w, const T* dict, size_type ks, size_type d) {
  size_type re = 0;
  T min_dist = l2dist(w, dict, d);
  for (int i = 1; i < ks; ++i) {
    dict += d;
    T dist = l2dist(w, dict, d);
    if (dist < min_dist) {
      re = i;
    }
  }
  return re;
}

void nomalize_codebook(T* dict, size_type m, size_type ks, size_type d) {
  for (int i = 0; i < m * ks; ++i, dict+=d) {
    T norm_sqr = 0;
    for (int j = 0; j < d; ++j) {
      norm_sqr += dict[j] * dict[j];
    }
    if (norm_sqr <= 0) {
      throw std::runtime_error("zero norm");
    }
    T norm = sqrt(norm_sqr);
    for (int j = 0; j < d; ++j) {
      dict[j] /= norm;
    }
  }
}

template <typename DataType>
void load_data(DataType* data, size_t D, size_t N, const char* inputPath) {
  std::ifstream fin(inputPath, std::ios::binary | std::ios::ate);
  if (!fin) {
    throw std::runtime_error("cannot open file ");
  }

  size_t fileSize = fin.tellg();
  fin.seekg(0, fin.beg);
  if (fileSize == 0) {
    throw std::runtime_error("file size is 0 ");
  }

  int dim;
  fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
  if (D != dim)
    throw std::runtime_error("Dimension not matched when reading file ");
  size_t bytesPerRecord = dim * sizeof(DataType) + 4;
  if (fileSize % bytesPerRecord != 0) {
    throw std::runtime_error("File not aligned");
  }
  size_t cardinality = fileSize / bytesPerRecord;
  if (N != cardinality) {
    throw std::runtime_error("cardinality not matched");
  }
  fin.read((char*)data, sizeof(DataType) * dim);

  for (int i = 1; i < cardinality; ++i) {
    fin.read((char*)&dim, 4);
    fin.read((char*)(data + i * dim), sizeof(DataType) * dim);
  }
  fin.close();
}

VQLayer::VQLayer(size_type I, size_type O,
                 Activation type)
                 : I_(I), O_(O), type_(type), D_(I_/M_) {
  if (I_ % M_ > 0)
    throw std::runtime_error("I_ is not dividable by M_");

  code_ = new CodeType[O_ * M_];
  dict_ = new T[M_ * Ks * D_];
  initialize();
}

VQLayer::VQLayer(const VQLayer& c) : VQLayer(c.I_, c.O_, c.type_) {
  std::memcpy(code_, c.code_, O_ * M_ * sizeof(CodeType));
  std::memcpy(dict_, c.dict_,  M_ * Ks * D_ * sizeof(T));
}

VQLayer::VQLayer(VQLayer&& c) noexcept: I_(c.I_), O_(c.O_),
                                        D_(c.D_), type_(c.type_),
                                        code_(c.code_), dict_(c.dict_) {
  c.dict_ = nullptr;
  c.code_ = nullptr;
}

VQLayer::~VQLayer() {
  delete [] code_;
  delete [] dict_;
}

void VQLayer::initialize() {
  std::default_random_engine generator(1016);

  std::uniform_int_distribution<> codes_dist(0, Ks-1);
  CodeType* code = code_;
  for (int i = 0; i < M_ * O_; ++i) {
    *(code++) = codes_dist(generator);
  }
//#define LEARNED_CODE_BOOK
#ifndef LEARNED_CODE_BOOK
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  T* w = dict_;
  for (int i = 0; i < M_ * Ks * D_; i++) {
      *(w++) = distribution(generator);
  }
#else
  std::string file_name = "../codebooks/learned_codebook/angular_dim_"
                          + std::to_string(D_) + "_Ks_"
                          + std::to_string(Ks) + ".fvecs";
  load_data<float >(dict_, D_, Ks, file_name.c_str());
  for (int i = 1; i < M_; ++i) {
    std::memcpy(&dict_[i * Ks * D_], dict_, Ks * D_ * sizeof(T));
  }
#endif
//  nomalize_codebook(dict_, M_, Ks, D_);
}

SparseVector VQLayer::forward(const SparseVector& x) {
  SparseVector y;

  volatile T* dict = dict_;        // shape of [M_, Ks, D_]
  volatile CodeType* code = code_; // shape of [O_, M_]

  // calculate look up table:  [M_, Ks]
  T tables[M_][Ks];

  for (int k = 0; k < Ks; ++k) {
    size_type idx = 0;
    for (int m = 0; m < M_; ++m) {
      volatile T* d = dict + m * Ks * D_ + k * D_; // TODO to be optimized
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
    for (int o = 0; o < O_; ++o) {
      T mm = 0;
#pragma unroll
      for (int m = 0; m < M_; ++m) {
        mm += tables[m][*(c++)]; // *c = code[o * M_ + m]
      }
      if (mm > 0) {
        y.push_back(o, mm);
      }
    }
  }

  else if (type_ == Activation::SoftMax) {
    T max_v = std::numeric_limits<T>::min();
    volatile CodeType* c = code;
    for (int o = 0; o < O_; ++o) {
      T mm = 0;
#pragma unroll
      for (int m = 0; m < M_; ++m) {
        mm += tables[m][*(c++)]; // *c = code[o * M_ + m]
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


SparseVector VQLayer::backward(const SparseVector& g,
                               const SparseVector& x,
                               const Optimizer& optimizer,
                               bool compute_gx ) {
  T lr = optimizer.lr;
  SparseVector gx;
  T* const dict = dict_;        // shape of [M_, Ks, D_]
  CodeType* const code = code_; // shape of [O_, M_]

  if (compute_gx) {
    // gx[I_] = w[I_, O_], g[O_].
    gx = x;
    size_type idx = 0;
    for (int m = 0; m < M_; ++m) {
      size_type begin_idx = m * D_;
      size_type end_idx = begin_idx + D_;

      for (; x.size() > idx && x.index_[idx] < end_idx; idx++) {
        T grad = 0;
        for (int o = 0; o < g.size(); ++o) {
          CodeType c = code[g.index_[o] * M_ + m];
          // TODO to be optimized
          grad += g.value_[o] * dict[m * Ks * D_ + c * D_ +
              x.index_[idx] - begin_idx];
        }
        gx.value_[idx] = grad;
      }
    }
  }

  // compute gradient and update with respect to the weight
  // gw[i_, o_] = x[1, i_]' g[1, o_]
  T * w = new T[D_];
  for (int o = 0; o < g.size(); ++o) {
    size_type idx = 0;
    for (int m = 0; m < M_; ++m) {
      size_type begin_idx = m * D_;
      size_type end_idx = begin_idx + D_;
      auto& c = code[g.index_[o] * M_ + m];
      std::memcpy(w, &dict[m * Ks * D_ + c * D_], D_ * sizeof(T));
      for (; x.size() > idx && x.index_[idx] < end_idx; idx++) {
        T grad = x.value_[idx] * g.value_[o];
        w[x.index_[idx] - begin_idx] -= lr * grad;
      }
      c = vq(w, &dict[m * Ks * D_], Ks, D_);
    }
  }
  delete [] w;


  return gx;
}

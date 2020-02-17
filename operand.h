//
// Created by xinyan on 2020/2/16.
//

#pragma once
#include <memory>
#include <vector>



using std::vector;


using index_type=int;
using size_type=index_type ;

class Operand {};

template <typename value_type>
class SparseVector : Operand {
 public:
  SparseVector() = default;
  index_type size() const {
    return (index_type)index_.size();
  }
  const vector<index_type >& getIndex() const {
    return index_;
  }
  const vector<value_type>& getValue() const {
    return value_;
  }
 private:
  vector<index_type> index_;
  vector<value_type> value_;
};

template <typename value_type>
class SparseRows : Operand {
 public:
  SparseRows() = default;
  ~SparseRows() = default;
 private:
  vector<SparseVector<value_type > > rows_;
};

template<typename T, size_type D, bool variable>
class Tensor : Operand {
 public:
  explicit Tensor(const std::array<size_type, D> & shapes) : (shape_(shapes)), size_(0) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      size_ *= a[i];
    }
    data_ = new T[size_];
  }
  ~Tensor() {
    delete[] data_;
  }

  T& operator[] (std::array<size_type , D> slices) {
    size_type offset = 0;
#pragma unroll
    for (int i = 0; i < D-1; ++i) {
      offset += slices[i];
      offset *= shape_[i];
    }
    return *(offset + slices[D-1] + data_);
  }

  T* data() {return data_;}
  const T* data() const {return data_;}
  size_type size() const {return size_;}
  static constexpr size_type dim() {return D;}
  static constexpr bool differentiable() {return variable;}
  const std::array<size_type, D > & shape() const {return shape_;}

 protected:
  size_type size_;
  std::array<size_type, D> shape_;
  T *data_;
};


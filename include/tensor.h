//
// Created by xinyan on 12/3/2020.
//

#pragma once
#include <vector>

using std::vector;

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

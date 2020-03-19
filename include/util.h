//
// Created by xinyan on 16/3/2020.
//

#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "tensor.h"

using std::vector;
using std::pair;


template <typename ID = size_type, typename V = T>
class TopSelector {
 public:
  explicit TopSelector(int k): k_(k) {
    heap_.reserve(k);
  }
  
  bool insert(ID id, V value) {
    if (heap_.size() < k_) {
      heap_.emplace_back(value, id);
      std::push_heap(heap_.begin(), heap_.end(), std::greater<>());
      return true;
    } else {
      if (value > heap_[0].first) {
        std::pop_heap(heap_.begin(), heap_.end(), std::greater<>());
        heap_[k_-1] = {value, id};  // pop the max one and swap it
        std::push_heap(heap_.begin(), heap_.end(), std::greater<>());
        return true;
      }
      return false;
    }
  }
  
  SparseVector select() {

    std::sort(
      heap_.begin(),
      heap_.end(),
      [](const pair<V, ID >& a, const pair<V, ID>& b) {
        return a.second < b.second;
      });

    SparseVector selected;
    selected.reserve(heap_.size());
    for (auto& value_id : heap_) {
      selected.push_back(value_id.second, value_id.first);
    }
    return selected;
  }
  
  ID k() {
    return k_;
  }

 private:
  ID                   k_;
  vector<pair<V, ID> > heap_;
};

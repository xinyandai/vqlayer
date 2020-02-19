//
// Created by xinyan on 2020/2/19.
//
#include <iostream>
#include "../include/layer.h"

void dump(const SparseVector& s) {

  for (int i = 0; i < s.size(); ++i) {
    std::cout << s.index_[i] << " : " << s.value_[i] <<"\t";
  }
  std::cout << std::endl;
}

void dump(const T* p, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << i << " : " << *(p++) <<"\t";
  }
  std::cout << std::endl;
}

void compare(const std::string& variable, const SparseVector& s, const T* p, int size) {
  bool success = true;
  for (int i = 0; i < s.size(); ++i) {
    if (std::abs(s.value_[i] - p[i]) > 0.001) {
      success = false;
      break;
    }
  }
  std::cout << (success?"[PASS]":"[FAIL]") ;
  std::cout << " " << variable << std::endl;
  std::cout << "\t\t";
  dump(s);
  std::cout << "\t\t";
  dump(p, size);
}

void test_smm_relu() {

  vector<T>  x_  = { 0.1357047994833403, -0.9500842332443221 };
  vector<T>  w_  = { 0.12282730752559588, 1.901416969226425, -1.573997496240008, 0.20564681374665741 };
  vector<T>  b_  = { -1.2207054473768937, 1.5676540439571955 };
  vector<T>  g_  = { 1.7100876807416627, -0.0916384140982638 };
  vector<T>  o_  = { 0.29139301210561674, 1.6303036571426572 };
  vector<T>  gx_  = { 0.035802629858752405, -2.710518875714364 };
  vector<T>  update_w_  = { 0.0996205969441981, 1.9026605464874424, -1.4115247619462077, 0.19694039250722994 };
  vector<T>  update_b_  = { -1.39171421545106, 1.5768178853670218 };

  vector<T>  gw_  = { 0.2320671058139778, -0.012435772610176194, -1.6247273429380038, 0.08706421239427464 };
  vector<T>  gb_  = { 1.7100876807416627, -0.0916384140982638 };

  size_type I=2, O=2;
  Optimizer optimizer = {0.1};
  Layer layer(I, O, Activation::ReLu);
  layer.initialize(w_, b_);
  SparseVector x = x_, g=g_;

  SparseVector c_o = layer.forward(x);

  compare("forward", c_o, o_.data(), o_.size());

  SparseVector c_gx = layer.backward(g, x, optimizer, true);

  compare("update_w_", update_w_, layer.weight(), I*O);
  compare("update_b_", update_b_, layer.bias(), O);
  compare("gradient_x",c_gx, gx_.data(), gx_.size());



}


int main() {
  std::cout << "Start Testing Sparse Matrix Multiplication Layer" << std::endl;
  test_smm_relu();
}
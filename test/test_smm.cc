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

void test_smm_softmax() {
  size_type I=8, O=4;
  vector<T>  x_  = { 1.5516870102023173, -0.9251752728867606, -0.014352775709434534,
                     0.22134742695532325, 0.15594105122622795, 0.5030637979205353,
                     -0.7834341624394263, 0.7741397728809003 };
  vector<T>  w_  = { 0.18785237059264645, -0.8551286574783469, -0.3720122543464393,
                     -1.1245439053024775, 1.1008960707603663, 1.2242894316303707,
                     0.6083893460380334, -0.1844802709171102, 0.4925100769615539,
                     0.07239803849750229, 0.13475540977024994, -1.5512873822380544,
                     -1.7792852223291857, 0.7621670322758677, 1.7524477864075518,
                     -0.528663313911014, 0.10796811613632247, -1.684173979989373,
                     0.9861907424803545, 0.5204656840717133, -1.409598693782954,
                     -1.1382040754746086, -0.4954915626388351, -0.297129980483365,
                     -0.29775108503821784, 0.6222751276617549, 1.5488036759136954,
                     2.35938171417603, 0.6042851340815848, -0.06642175572357562,
                     1.3076468425197068, 0.36328624713534113 };
  vector<T>  b_  = { 0.886633630580316, -0.8271605666946733,
                     -0.182012857677164, -0.17508423942467125 };
  vector<T>  o_  = { 0.7038258993214305, 0.009932670338585496,
                     0.2588713071665085, 0.027370123173475553 };


  vector<size_type >  y_  = { 3 };
  vector<T>  g_  = { 0.7038258993214305, 0.009932670338585496, 0.2588713071665085,
                     -0.9726298768265245 };
  T loss_ = 3.59830325563001;

  Layer layer(I, O, Activation::SoftMax);
  layer.initialize(w_, b_);
  SparseVector x = x_;
  T loss;

  SparseVector c_o = layer.forward(x);
  SparseVector g = SoftMaxCrossEntropy::compute(c_o, y_, &loss);

  compare("softmax forward", c_o, o_.data(), o_.size());

  compare("cross entropy gradient", g, g_.data(), g_.size());
  std::cout <<  (( std::abs(loss - loss_) < 0.0001)? "[PASS]":"[FAIL]")
            << " Cross Entropy Loss" << std::endl;


}

int main() {
  std::cout << "Start Testing Sparse Matrix Multiplication Layer" << std::endl;
  test_smm_relu();
  test_smm_softmax();
}
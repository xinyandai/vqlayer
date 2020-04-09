//
// Created by xinyan on 11/3/2020.
//

#include "test.h"

template <Activation Act, bool Select>
void test_hash(int seed) {
  const size_type I = 16, O = 16;
  HashLayer<Act, Select> layer(I, O, I * O / 8);
  FakeLayer<Act, Select> fakeRQLayer(layer);

  vector<T > x(I, 0);
  vector<T > g(O, 0);

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  for (int i = 0; i < I; ++i) {
    x[i] = distribution(generator);
  }
  for (int o = 0; o < O; ++o) {
    g[o] = distribution(generator);
  }

  SparseVector sx = x;
  SparseVector sg = g;
  SparseVector y = layer.forward(sx);
  SparseVector y_ = fakeRQLayer.forward(sx);

  compare("hash forward", y, y_);

  SparseVector gx = layer.backward_x(g, sx);
  SparseVector gx_ = fakeRQLayer.backward_x(g, sx);
  compare("hash backward gx", gx, gx_);
}

int main() {
  int i = 808;
  test_hash<Activation::ReLu, true>(i++);
  test_hash<Activation::ReLu, false>(i++);

  test_hash<Activation::SoftMax, true>(i++);
  test_hash<Activation::SoftMax, false>(i++);
}

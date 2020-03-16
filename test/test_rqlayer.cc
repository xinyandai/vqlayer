//
// Created by xinyan on 11/3/2020.
//

#include "test.h"


void test_rq(Activation activation, int seed) {
  const size_type I = 16, O = 16;
  RQLayer rq(I, O, activation);
  FakeLayer fakeRQLayer(rq);

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
  SparseVector y = rq.forward(sx);
  SparseVector y_ = fakeRQLayer.forward(sx);

  compare("RQ forward", y, y_);

  SparseVector gx = rq.backward_x(g, sx);
  SparseVector gx_ = fakeRQLayer.backward_x(g, sx);
  compare("RQ backward gx", gx, gx_);
}

int main() {
  test_rq(Activation::ReLu, 1016);
  test_rq(Activation::SoftMax, 808);
}

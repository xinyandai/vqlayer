//
// Created by xinyan on 11/3/2020.
//

/**
* \brief Vectorized Sparse Matrix Multiplication Layer
*/
#include "test.h"

template <Activation Act, bool Select, bool NQ>
void test_pq(int seed) {
  const size_type I = 16, O = 16;
  PQLayer<Act, Select, NQ> rq(I, O);
  FakeLayer<Act, Select> fakeVQLayer(rq);

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
  SparseVector y_ = fakeVQLayer.forward(sx);

  compare("RQ forward", y, y_);

  SparseVector gx = rq.backward_x(g, sx);
  SparseVector gx_ = fakeVQLayer.backward_x(g, sx);
  compare("RQ backward gx", gx, gx_);
}

int main() {
  int i = 1016;
  test_pq<Activation::ReLu, true, true>(i++);
  test_pq<Activation::ReLu, true, false>(i++);
  test_pq<Activation::ReLu, false, false>(i++);
  test_pq<Activation::ReLu, false, false>(i++);

  test_pq<Activation::SoftMax, true, true>(i++);
  test_pq<Activation::SoftMax, true, false>(i++);
  test_pq<Activation::SoftMax, false, false>(i++);
  test_pq<Activation::SoftMax, true, true>(i++);
}

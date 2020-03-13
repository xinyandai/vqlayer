//
// Created by xinyan on 9/3/2020.
//
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <iterator>
#include <algorithm>
#include <vector>
#include "../include/vq.h"
#include "../include/progress_bar.h"

using std::vector;

T l2dist_sqr(const T *a, const T *b, size_type d) {
  T dist = 0;
  for (int i = 0; i < d; ++i) {
    T diff = (*(a++)) - (*(b++));
    dist += diff * diff;
  }
  return dist;
}

size_type vq(const T* w, const T* dict, size_type ks, size_type d) {
  size_type re = 0;
  T min_dist = l2dist_sqr(w, dict, d);
  for (int i = 1; i < ks; ++i) {
    dict += d;
    T dist = l2dist_sqr(w, dict, d);
    if (dist < min_dist) {
      re = i;
      min_dist = dist;
    }
  }
  return re;
}

size_type nvq(T* norm, T* w, const T* dict, size_type ks, size_type d) {
  *norm = normalize(w, d);
  return vq(w, dict, ks, d);
}

T norm_sqr(T* w, size_type d) {
  T norm_sqr = 0;
  for (int i = 0; i < d; ++i) {
    norm_sqr += w[i] * w[i];
  }
  return norm_sqr;
}

T normalize(T* w, size_type d) {
  T n_sqr = norm_sqr(w, d);
  if (n_sqr <= 0) {
    throw std::runtime_error("zero norm");
  }
  T norm = std::sqrt(n_sqr);
  for (int i = 0; i < d; ++i) {
    w[i] /= norm;
  }
  return norm;
}

void normalize_codebook(T* dict, size_type m, size_type ks, size_type d) {
  for (int i = 0; i < m * ks; ++i) {
    normalize(&dict[i * d], d);
  }
}

/**
 * \param w    shape of [d]
 * \param dict shape of [m, ks, d]
 * \param code shape of [m]
 * \param norm shape of [1]
 * \param ks   size of code book
 * \param m    number of code book
 * \param d    dimension
 */
void rq(const T* w, const T* dict, CodeType* code, T* norm,
        const size_type ks, const size_type m, const size_type d) {
  T * residue = new T[d];
  std::memcpy(residue, w, d * sizeof(T));
  *norm = normalize(residue, d);

  for (int i = 0; i < m; ++i) {
    const T* codebook = &dict[i * ks * d];
    auto c = (CodeType)vq(residue, codebook, ks, d);
    code[i] = c;

    for (int dim = 0; dim < d; ++dim) {
      residue[dim] -= codebook[c * d + dim];
    }
  }

  // use relative norm here
  T quantized_norm_sqr = 0;
  for (int dim = 0; dim < d; ++dim) {
    T recover = w[dim] - residue[dim];
    quantized_norm_sqr += recover * recover;
  }
  T quantized_norm = std::sqrt(quantized_norm_sqr);
  *norm = *norm / quantized_norm;

  delete[] residue;
}


/**
 * \param centroids [ks, d]
 * \param code      [n]
 * \param data      [n, d]
 * \param n 
 * \param k 
 * \param d 
 * \param iter 
 */
void kmeans(T* centroids, CodeType* code, const T* data,
            const size_type n, const size_type ks,
            const size_type d, const size_type iter) {

  if (ks > n) {
    throw std::runtime_error("too many centroids");
  }
  std::default_random_engine generator(1016);
  std::uniform_int_distribution<> distribution(0, n-1);
  // random initialization
  std::memcpy(centroids, data, ks * d * sizeof(T));

  ProgressBar bar(iter, std::string("k-means"));
  for (int i = 0; i < iter; ++i, ++bar) {
    // assign
#pragma omp parallel for
    for (int t = 0; t < n; ++t) {
      code[t] = static_cast<CodeType>(vq(&data[t * d], centroids, ks, d));
    }

    // recenter
    vector<size_type > count(ks, 0);
    std::memset(centroids, 0, ks * d * sizeof(T));
    for (int t = 0; t < n; ++t) {
      CodeType c = code[t];
      count[c]++;
      for (int dim = 0; dim < d; ++dim) {
        centroids[c * d + dim] += data[t * d + dim];
      }
    }

    for (int c = 0; c < ks; ++c) {
      if (count[c] == 0) {
        size_type t = distribution(generator);
        std::memcpy(&centroids[c * d], &data[t * d], d * sizeof(T));
      } else {
        for (int dim = 0; dim < d; ++dim) {
          centroids[c * d + dim] /= count[c];
        }
      }
    }
  }

}

/**
 * \param centroids [M, ks, d]
 * \param code      [M, n]
 * \param data      [n, d]
 * \param M
 * \param n
 * \param k
 * \param d
 * \param iter
 */
void kmeans_residual(T* centroids, CodeType* code,
                     const T* data, const size_type M,
                     const size_type n, const size_type ks,
                     const size_type d, const size_type iter) {
  T* residue = new T[n * d];
  std::memcpy(residue, data, n * d * sizeof(T));

  for (int m = 0; m < M; ++m) {
    T* codebook = &centroids[m * ks * d];
    CodeType* assign_code = &code[m * n];
    kmeans(codebook, assign_code, residue, n, ks, d, iter);
    for (int i = 0; i < n; ++i) {
      CodeType c = assign_code[i];
      for (int dim = 0; dim < d; ++dim) {
        residue[i * d + dim] -= codebook[c * d + dim];
      }
    }
  }

  delete [] residue;
}

void rq_codebook(T* centroid, const size_type M, const size_type n,
                 const size_type ks, const size_type d, const size_type iter) {
  std::default_random_engine generator(1016);
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  T* x = new T[n * d];
  auto* code = new CodeType[n * M];
  for (int i = 0; i < n * d; ++i) {
    x[i] = distribution(generator);
  }

  for (int i = 0; i < n; ++i) {
    normalize(&x[i * d], d);
  }

  kmeans_residual(centroid, code, x, M, n, ks, d, iter);

  delete [] code;
  delete [] x;
}

void vq_codebook(T* centroid, const size_type n,
                 const size_type ks, const size_type d, const size_type iter) {
  std::default_random_engine generator(1016);
  std::uniform_real_distribution<T > distribution(0.0, 1.0);
  T* x = new T[n * d];
  auto* code = new CodeType[n];

  for (int i = 0; i < n * d; ++i) {
    x[i] = distribution(generator);
  }

  for (int i = 0; i < n; ++i) {
    normalize(&x[i * d], d);
  }

  kmeans(centroid, code, x, n, ks, d, iter);

  delete [] code;
  delete [] x;
}

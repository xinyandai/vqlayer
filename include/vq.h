//
// Created by xinyan on 12/3/2020.
//

#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <memory>
#include <numeric>
#include <mutex>
#include <thread>

#include "tensor.h"

#define CodeType uint8_t


size_type vq(const T* w, const T* dict, size_type ks, size_type d);
size_type nvq(T* norm, T* w, const T* dict, size_type ks, size_type d);
void rq(const T* w, const T* dict, CodeType* code, T* norm,
        size_type ks, size_type m, size_type d);

T norm_sqr(T* w, size_type d);
T normalize(T* w, size_type d);

T l2dist_sqr(const T *a, const T *b, size_type d);

void normalize_codebook(T* dict, size_type m, size_type ks, size_type d);

void kmeans(T* centroids, CodeType* code, const T* data,
            size_type n, size_type ks, size_type d, size_type iter);
void kmeans_residual(T* centroids, CodeType* code,
                     const T* data, const size_type M,
                     const size_type n, const size_type ks,
                     const size_type d, const size_type iter);

void rq_codebook(T* centroid, size_type M, size_type n,
                 size_type ks, size_type d, size_type iter);
void vq_codebook(T* centroid, size_type n,
                 size_type ks, size_type d, size_type iter);

//
// Created by xinyan on 12/3/2020.
//

//
// \author Xinyan DAI (xinyan.dai@outlook.com)
// Created by xinyan DAI on 17/2/2020.
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


using T = float;
using size_type = int;

#define CodeType uint8_t
#define Ks 256
#define M_ 8 // M of Product Quantization

size_type vq(const T* w, const T* dict, size_type ks, size_type d);
void rq(const T* w, const T* dict, CodeType* code, T* norm,
        size_type ks, size_type m, size_type d);

T normalize(T* w, size_type d);
T l2dist_sqr(const T *a, const T *b, size_type d);
void normalize_codebook(T* dict, size_type m, size_type ks, size_type d);
void rq_codebook(T* centroid, size_type M, size_type n,
                 size_type ks, size_type d, size_type iter);
void vq_codebook(T* centroid, const size_type n,
                 size_type ks, size_type d, size_type iter);





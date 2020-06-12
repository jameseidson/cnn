#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#define FLAT2D(i, j, max_j) ((i * max_j) + j)
#define FLAT3D(i, j, k, max_j, max_k) ((i * max_j * max_k) + (j * max_k) + k)
#define FLAT4D(i, j, k, l, max_j, max_k, max_l) ((i * max_j * max_k * max_l) + (j * max_k * max_l) + (k * max_l) + l)

typedef struct Matrix1D {
  size_t rows;
  double *elems;
} Matrix1D_T;

typedef struct Matrix2D {
  size_t rows;
  size_t cols;
  double *elems;
} Matrix2D_T;

typedef struct Matrix3D {
  size_t rows;
  size_t cols;
  size_t chnls;
  double *elems;
} Matrix3D_T;


__global__ void MAT_init(size_t rows, double val, Matrix1D_T *out);
__global__ void MAT_init(size_t rows, size_t cols, double val, Matrix2D_T *out);
__global__ void MAT_init(size_t rows, size_t cols, size_t chnls, double val, Matrix3D_T *out);

__global__ void MAT_mul(Matrix2D_T *a, Matrix2D_T *b, Matrix2D_T *c);
__global__ void MAT_mul(Matrix2D_T *a, Matrix1D_T *b, Matrix1D_T *c);

__global__ void MAT_print(Matrix2D_T *a);

__global__ void MAT_free(Matrix1D_T *out);
__global__ void MAT_free(Matrix2D_T *out);
__global__ void MAT_free(Matrix3D_T *out);

#endif

#include "mat.h"

/* launch with (rows) threads */
__global__ void MAT_init(size_t rows, double val, Matrix1D_T *out) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x == 0) {
    out->rows = rows;
    cudaMalloc((void **)&out->elems, rows * sizeof(double));
  }
  __syncthreads();

  if (x < rows) {
    out->elems[x] = val;
  }
}

/* launch with (rows * cols) threads */
__global__ void MAT_init(size_t rows, size_t cols, double val, Matrix2D_T *out) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t numElems = rows * cols;

  if (x == 0) {
    out->rows = rows;
    out->cols = cols;
    cudaMalloc((void **)&out->elems, numElems * sizeof(double));
  }
  __syncthreads();

  if (x < numElems) {
    out->elems[x] = val;
  }
}

/* launch with (rows * cols * chnls) threads */
__global__ void MAT_init(size_t rows, size_t cols, size_t chnls, double val, Matrix3D_T *out) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t numElems = rows * cols * chnls;

  if (x == 0) {
    out->rows = rows;
    out->cols = cols;
    out->chnls = chnls;
    cudaMalloc((void **)&out->elems, numElems * sizeof(double));
  }
  __syncthreads();

  if (x < numElems) {
    out->elems[x] = val;
  }
}

/* c must be of size b.rows */
/* required that a.cols == b.rows */
/* launch with (c->rows) threads */
__global__ void MAT_mul(Matrix2D_T *a, Matrix1D_T *b, Matrix1D_T *c) {
  size_t row = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (row < c->rows) {
    double dotProd = 0.0f;
    for (size_t i = 0; i < b->rows; i++) {
      dotProd += a->elems[FLAT2D(row, i, a->cols)] * b->elems[i];
    }
    c->elems[row] = dotProd;
  }
}

/* c must be of size a.rows * b.cols */
/* required that a.cols == b.rows */
/* launch with (a->rows) * (b->cols) threads */
__global__ void MAT_mul(Matrix2D_T *a, Matrix2D_T *b, Matrix2D_T *c) {
  size_t row = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t col = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (row < c->rows && col < c->cols) {
    double dotProd = 0.0f;
    for (size_t i = 0; i < a->cols; i++) {
      dotProd += a->elems[FLAT2D(row, i, a->cols)] * b->elems[FLAT2D(i, col, c->cols)];
    }
    c->elems[FLAT2D(row, col, c->cols)] = dotProd;
  }
}

/* launch with 1 thread */
__global__ void MAT_print(Matrix2D_T *a) {
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < a->cols; j++) {
      printf("%0.3f ", a->elems[FLAT2D(i, j, a->cols)]);
    }
    printf("\n");
  }
}

/* launch free functions with 1 thread */
__global__ void MAT_free(Matrix1D_T *out) {
  cudaFree(out->elems);
}

__global__ void MAT_free(Matrix2D_T *out) {
  cudaFree(out->elems);
}

__global__ void MAT_free(Matrix3D_T *out) {
  cudaFree(out->elems);
}


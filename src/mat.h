#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

#define FLAT2D(i, j, max_j) ((i * max_j) + j)
#define FLAT3D(i, j, k, max_j, max_k) ((i * max_j * max_k) + (j * max_k) + k)
#define FLAT4D(i, j, k, l, max_j, max_k, max_l) ((i * max_j * max_k * max_l) + (j * max_k * max_l) + (k * max_l) + l)

#define NUMBLK(dim, blkSize) (dim / blkSize) + 1

static const uint16_t BLKS_1D = 256;
static const uint16_t BLKS_2D = 16;
static const uint8_t BLKS_3D = 8;

static const uint8_t NUM_CHNL = 3;

typedef enum NonLin {
  RELU,
  SIG
} NonLin_T;

typedef struct MatList {
  size_t num;
  size_t rows;
  size_t cols;
  double *mats;
} MatList_T;

__global__ void MAT_randomize(double *m, size_t numElm);

/* launch with (numElm) threads */
__global__ void MAT_setVal(double *m, size_t numElm, double val);

/* launch with (numElm) threads */
__global__ void MAT_randomize(double *m, size_t numElm);

/* deep copies src into dest */
/* launch with (numElm) threads */
__global__ void MAT_assign(double *src, double *dst, size_t numElm);

/* mC is output */
/* launch with (numElm) threads */
__global__ void MAT_ewMul(double *mA, double *mB, double *mC, size_t numElm);

/* vC is output */
/* required that cRows == aRows and bRows == aCols */
/* launch with threads equal to output rows, aka (aRows) */
__global__ void MAT_mvMul(double *mA, double *vB, double *vC, size_t aRows, size_t aCols);

/* launch with (numElm) threads */
/* applies nonlinearity to (mA) and puts result in (mB) */
__global__ void MAT_ReLU(double *mA, double *mB, size_t numElm);
__global__ void MAT_sigmoid(double *mA, double *mB, size_t numElm);

/* launch with (numElm) threads */
__global__ void MAT_loss(double *m, size_t numElm, size_t lbl, double *loss);

/* launch with wRows threads */
__global__ void MAT_fwdProp(double *mWgt, double *vAct, double *vNxtAct, size_t wRows, size_t wCols, NonLin_T fType);

/* applies delta equation */
/* launch with (numElm) threads */
__global__ void MAT_deltas_out(double *vOut, double *vDelt, size_t numElm, size_t lbl, NonLin_T fType);
/* launch with (wCols) threads */
__global__ void MAT_deltas_hidden(double *vAct, double *vDelt, double *mWgt, double *vNxtDelt, size_t wRows, size_t wCols, NonLin_T fType);

/* launch with (wRows) threads */
__global__ void MAT_applyGradient(double *vAct, double *vNxtDelt, double *mWgt, size_t wRows, size_t wCols, double scaleFac);

/* mB is output */
/* launch with (bRows), (bCols) threads */
__global__ void MAT_convolve(double *mA, double *mB, double *mKern, size_t aRows, size_t aCols, size_t kRows, size_t kCols);

/* mB is output */
/* launch with (bRows), (bCols) threads */
__global__ void MAT_pool(double *mA, double *mB, size_t aRows, size_t aCols, size_t wDim, size_t stride);

/* launch with 1 thread */
__global__ void MAT_print(double *m, size_t rows, size_t cols, bool is3D);

#endif

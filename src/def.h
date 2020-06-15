#ifndef DEF_H
#define DEF_H

#include "../include/data.h"
#include "mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>

#define RED 0
#define GRN 1
#define BLU 2

typedef struct MatList Features_T;
typedef struct MatList Forward_T;

typedef struct Pool {
  size_t dim;
  size_t stride;
} Pool_T;

typedef struct Softmax {
  size_t numLyr;
  double lrnRate;
  NonLin_T fType;

  size_t *aIdx;
  size_t *wIdx;
  size_t *aTopo;
  size_t *wTopo;

  double *deltas;
  double *activs;
  double *wgts;
} Softmax_T;

MatList_T *CNN_initFwd(size_t maxNum, size_t hgt, size_t wid);
Features_T *CNN_initFtrs(size_t num, size_t hgt, size_t wid);
Pool_T *CNN_initPool(size_t winDim, size_t stride);
Softmax_T *CNN_initSoftmax(size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType);

void CNN_freeFwd(MatList_T *fwd);
void CNN_freeFtrs(Features_T *kern);
void CNN_freePool(Pool_T *pool);
void CNN_freeSoftmax(Softmax_T *sm);

#endif

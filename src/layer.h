#ifndef LAYER_H
#define LAYER_H

#include "data.h"
#include "mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

static const unsigned RED = 0;
static const unsigned GRN = 1;
static const unsigned BLU = 2;

typedef struct MatList Net_T;

typedef struct Conv {
  double lrnRate;
  MatList_T fltrs;
  MatList_T inp;
} Conv_T;

typedef struct Pool {
  size_t dim;
  size_t stride;
  size_t rows;
  size_t cols;
  size_t *idxs;
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

Net_T *LYR_net_init(size_t maxNum, size_t maxHgt, size_t maxWid);
__global__ void LYR_net_update(Net_T *net, double *imgs, size_t num, size_t rows, size_t cols);
void LYR_net_free(Net_T *net);

Conv_T *LYR_conv_init(size_t fNum, size_t fHgt, size_t fWid, size_t iNum, size_t iHgt, size_t iWid);
__global__ void LYR_conv_fwd(Net_T *net, Conv_T *kern);
__global__ void LYR_conv_back(Conv_T *kern, Net_T *net, double *buf);
void LYR_conv_free(Conv_T *kern);

Pool_T *LYR_pool_init(size_t dim, size_t stride, size_t iNum, size_t iRows, size_t iCols);
__global__ void LYR_pool_fwd(Pool_T *pool, Net_T *net, double *buf);
__global__ void LYR_pool_back(Pool_T *pool, Net_T *net, double *buf);
void LYR_pool_free(Pool_T *pool);

__global__ void LYR_norm_fwd(Net_T *net, NonLin_T func);

Softmax_T *LYR_softmax_init(size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType);
__global__ void LYR_softmax_fwd(Softmax_T *sm, Net_T *net);
__global__ void LYR_softmax_back(Softmax_T *sm, Net_T *net, size_t lbl);
__global__ void LYR_softmax_loss(Softmax_T *sm, size_t lbl, double *loss);
__global__ void LYR_softmax_cpyOut(Softmax_T *sm, double *output);
void LYR_softmax_free(Softmax_T *sm);

#endif

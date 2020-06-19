#ifndef LAYER_H
#define LAYER_H

#include "../include/data.h"
#include "mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

static const unsigned RED = 0;
static const unsigned GRN = 1;
static const unsigned BLU = 2;

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

Forward_T *LYR_fwd_init(size_t maxNum, size_t hgt, size_t wid);
__global__ void LYR_fwd_prep(Forward_T *fwd, double *imgs, size_t num, size_t hgt, size_t wid);
void LYR_fwd_free(Forward_T *fwd);

Features_T *LYR_conv_init(size_t num, size_t hgt, size_t wid);
__global__ void LYR_conv_fwd(Forward_T *fwd, Features_T *kern, double *buf);
void LYR_conv_free(Features_T *kern);

Pool_T *LYR_pool_init(size_t dim, size_t stride);
__global__ void LYR_pool_fwd(Pool_T *pool, Forward_T *fwd, double *buf);
void LYR_pool_free(Pool_T *pool);

__global__ void LYR_norm_fwd(Forward_T *fwd, NonLin_T func);

Softmax_T *LYR_softmax_init(size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType);
__global__ void LYR_softmax_fwd(Softmax_T *sm, Forward_T *fwd);
__global__ void LYR_softmax_back(Softmax_T *sm, size_t lbl);
__global__ void LYR_softmax_loss(Softmax_T *sm, size_t lbl, double *loss);
__global__ void LYR_softmax_cpyOut(Softmax_T *sm, double *output);
void LYR_softmax_free(Softmax_T *sm);

#endif

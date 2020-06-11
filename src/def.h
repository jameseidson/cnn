#ifndef DEF_H
#define DEF_H

#include "../include/data.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>

#define RED 0
#define GRN 1
#define BLU 2

#define FLAT2D(i, j, max_j) ((i * max_j) + j)
#define FLAT3D(i, j, k, max_j, max_k) ((i * max_j * max_k) + (j * max_k) + k)
#define FLAT4D(i, j, k, l, max_j, max_k, max_l) ((i * max_j * max_k * max_l) + (j * max_k * max_l) + (k * max_l) + l)

#define I_IDX(idx, j, k, l, max_i, max_j, max_k, max_l) (((idx - j * max_k * max_l - k * max_l - l) / (max_l * max_k * max_j)) % max_i)
#define J_IDX(idx, k, l, max_j, max_k, max_l) (((idx - k * max_l - l) / (max_l * max_k)) % max_j)
#define K_IDX(idx, l, max_k, max_l) (((idx - l) / max_l) % max_k)
#define L_IDX(idx, max_l) (idx % max_l)

#define NUMBLK(dim, blkSize) (dim / blkSize) + 1

static const uint8_t NUM_CHNL = 3;
static const uint16_t BLKS_1D = 256;
static const uint8_t BLKS_3D = 8;

struct ImgList {
  size_t num;
  size_t hgt;
  size_t wid;
  double *imgs; /* 4d arr: num * hgt * wid * colors (r, g, b) */
};

typedef struct ImgList Net_T;
typedef struct ImgList Features_T;

typedef struct Pool {
  size_t winDim;
  size_t stride;
} Pool_T;

typedef struct Classify {
  size_t numLyr;
  double lrnRate;
  size_t totalWgt;

  size_t *topo;
  size_t *wgtTopo;

  double *activs; /* 2d arr: netSize * nrnsPerLyr */
  double *wgts; /* 3d arr: netSize * nrnsPerLyr * wgtsPerNrn */
  double *errs; /* 1d arr: maxNrn */

  double *errBuf; /* 1d arr: maxNrn */
  double *wgtBuf; /* 3d arr: netSize * nrnsPerLyr * wgtsPerNrn */
} Classify_T;


Net_T *CNN_initNet(size_t maxNum, size_t hgt, size_t wid);
Features_T *CNN_initFtrs(size_t num, size_t hgt, size_t wid);
Pool_T *CNN_initPool(size_t winDim, size_t stride);
Classify_T *CNN_initClsfier(size_t *topology, size_t numLyr, double lrnRate);

void CNN_freeNet(Net_T *net);
void CNN_freeFtrs(Features_T *kern);
void CNN_freePool(Pool_T *pool);
void CNN_freeClsfier(Classify_T *cls);

#endif

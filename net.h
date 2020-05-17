#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>

#define FLAT2D(i, j, max_j) (i * max_j) + j 
#define FLAT3D(i, j, k, max_j, max_k) (i * max_j * max_k) + (j * max_k) + k
#define FLAT4D(i, j, k, l, max_j, max_k, max_l) (i * max_j * max_k * max_l) + (j * max_k * max_l) + (k * max_l) + l
#define NUMBLK(dim, blkSize) (dim / blkSize) + 1

static const uint8_t NUM_CHNL = 3;
static const uint16_t BLKS_1D = 256;
static const uint8_t BLKS_3D = 8;

struct Features { /* whole thing is stored in device mem */
  size_t num;
  size_t hgt;
  size_t wid;
  double *imgs;  /* 4d arr: num * hgt * wid * colors (r, g, b) */
};

typedef struct Convlvd { /* whole thing is in device mem */
  size_t stride;
  size_t winDim;
  size_t num;
  size_t hgt;
  size_t wid;
  double *imgs;  /* 4d arr: num * hgt * wid * colors (r, g, b) */
} Convlvd_T;

typedef struct Data {
  size_t num;
  size_t hgt;
  size_t wid;
  size_t *lbls;
  double *imgs; /* 4d arr: num * hgt * wid * colors (r, g, b) */
} Data_T;

typedef struct Classify Classify_T;
typedef struct Features Features_T;

__global__ void CNN_pool(Convlvd_T *conv, double *buffer);

__global__ void CNN_convolve(Convlvd_T *conv, Features_T *kern, double *img);
Convlvd_T *CNN_initConvlvd(Features_T *kern, Data_T *data, size_t winDim, size_t stride);
__global__ void CNN_testConvolve(Convlvd_T *conv);
void CNN_freeConvlvd(Convlvd_T *conv);

void CNN_testData(Data_T *data, size_t idx);
void CNN_freeData(Data_T *data);

Features_T *CNN_initFtrs(size_t num, size_t hgt, size_t wid);
void CNN_freeFtrs(Features_T *kern);

Classify_T *CNN_initClsfier(size_t *topology, size_t netSize);
void CNN_testClsfier(Classify_T *net);
void CNN_freeClsfier(Classify_T *net);

#endif

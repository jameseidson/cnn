#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

#define flat4d(i, j, k, l, max_j, max_k, max_l) (i * max_j * max_k * max_l) + (j * max_k * max_l) + (k * max_l) + l

static const uint8_t NUM_CHNL = 3;

typedef struct Classify Classify_T;
typedef struct Features Features_T;

typedef struct Data {
  size_t num;
  size_t wid;
  size_t hgt;
  size_t *lbls;
  double *imgs; /* 4d arr: num * wid * hgt * colors (r, g, b) */
} Data_T;

void CNN_freeData(Data_T *data);

Features_T *CNN_initFtrs(size_t numFeat, size_t featWid, size_t featHgt);
void CNN_freeFtrs(Features_T *kernel);

Classify_T *CNN_initClsfier(size_t *topology, size_t netSize);
void CNN_testClsfier(Classify_T *net);
void CNN_freeClsfier(Classify_T *net);

#endif

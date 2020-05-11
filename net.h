#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

const uint8_t NUM_CHNL = 3;

typedef struct Img {
  uint8_t lbl;
  uint8_t r;
  uint8_t g;
  uint8_t b;
} Img_T;

typedef struct NetCfg {
  size_t numFltr;
  size_t fltrWid;
  size_t fltrHgt;
  size_t stride;
} NetCfg_T;

typedef struct NetDat {
  size_t num;
  size_t hgt;
  size_t wid;
  Img_T *dat;
} NetDat_T;

typedef struct Classify Classify_T;
typedef struct FeatLrn FeatLrn_T;

/* note: assumes image is in device mem */
void CNN_convolve(FeatLrn_T *feats, NetCfg_T *img);

FeatLrn_T *CNN_initFL(NetCfg_T spec);
void CNN_freeFC(FeatLrn_T *net);

Classify_T *CNN_initC(size_t *topology, size_t netSize);
void CNN_testC(Classify_T *net);
void CNN_freeC(Classify_T *net);

#endif

#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

static const uint8_t NUM_CHNL = 3;

typedef struct ImgList {
  size_t numImg;
  size_t wid;
  size_t hgt;

  double *r; 
  double *g;
  double *b;
} ImgList_T;

typedef struct Data {
  size_t *lbls;
  ImgList_T imgDat;
} Data_T;

typedef struct Classify Classify_T;
typedef struct FLearn FLearn_T;

ImgList_T *CNN_convolve(FLearn_T *net, Data_T *input);

FLearn_T *CNN_initFL(size_t numFltr, size_t fltrWid, size_t fltrHgt, size_t stride);
void CNN_freeFL(FLearn_T *net);

Classify_T *CNN_initC(size_t *topology, size_t netSize);
void CNN_testC(Classify_T *net);
void CNN_freeC(Classify_T *net);

void CNN_freeData(Data_T* input);

#endif

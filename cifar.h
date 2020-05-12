#ifndef CIFAR_H
#define CIFAR_H

#include "net.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define TEST_SET 5;

static const size_t NUM_IMG = 60000;
static const size_t BATCH_SIZE = 10000;
static const size_t NUM_BATCH = 6;
static const size_t DIM = 32;
static const size_t CHNL_SIZE = 1024;

typedef struct Img Img_T;

Img_T *readImg(FILE **batchBins);
Data_T *prepData(Img_T *data);
void exportPPM(Img_T *data, size_t imgIdx, FILE *ppmOut);
void freeCifarImg(Img_T *data);

#endif

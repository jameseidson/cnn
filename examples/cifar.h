#ifndef CIFAR_H
#define CIFAR_H

#include "net.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static const size_t BATCH_SIZE = 10000;
static const size_t NUM_BATCH = 6;
static const size_t NUM_IMG = BATCH_SIZE * NUM_BATCH;
static const size_t DIM = 32;
static const size_t CHNL_SIZE = 1024;
static const uint8_t TEST_SET = 5;
static const uint8_t NUM_OUT = 10; 

typedef struct Cifar_Img Cifar_Img_T;

Cifar_Img_T *Cifar_readAll(FILE **batchBins);
Data_T *Cifar_prepData(Cifar_Img_T *cifar, size_t idx, size_t num);
double *Cifar_getImg(Cifar_Img_T *cifar, size_t idx);
uint8_t Cifar_getLbl(Cifar_Img_T *cifar, size_t idx);
void Cifar_exportPPM(Cifar_Img_T *cifar, size_t idx, FILE *ppmOut);
void Cifar_freeImg(double *img);
void Cifar_freeAll(Cifar_Img_T *cifar);

#endif

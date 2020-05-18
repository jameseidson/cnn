#ifndef NET_H
#define NET_H

#include "def.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

__global__ void CNN_feedForward(Classify_T *cls);
__global__ void CNN_normalize(Convlvd_T *conv);
__global__ void CNN_pool(Convlvd_T *conv, double *buffer);
__global__ void CNN_convolve(Convlvd_T *conv, Features_T *kern, double *img);

/* TODO: delete these */
void CNN_testClsfier(Classify_T *cls);
__global__ void CNN_testConvolve(Convlvd_T *conv);
void CNN_testData(Data_T *data, size_t idx);

#endif

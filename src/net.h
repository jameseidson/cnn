#ifndef NET_H
#define NET_H

#include "def.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

__global__ void CNN_prepNet(Net_T *net, double *imgs, size_t num, size_t hgt, size_t wid);
__global__ void CNN_convolve(Net_T *net, Features_T *kern, double *buf);
__global__ void CNN_pool(Net_T *net, Pool_T *pool, double *buf);
__global__ void CNN_normalize(Net_T *net);
__global__ void CNN_feedForward(Classify_T *cls);

/* TODO: delete these */
void CNN_testClsfier(Classify_T *cls);
__global__ void CNN_testNet(Net_T *net);
void CNN_testData(Data_T *data, size_t idx);

#endif

#ifndef NET_H
#define NET_H

#include "def.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>

__global__ void CNN_prepFwd(Forward_T *fwd, double *imgs, size_t num, size_t hgt, size_t wid);

__global__ void CNN_convolve(Forward_T *fwd, Features_T *kern, double *buf);
__global__ void CNN_pool(Pool_T *pool, Forward_T *fwd, double *buf);
__global__ void CNN_normalize(Forward_T *fwd, NonLin_T func);

__global__ void CNN_softmax_fwd(Softmax_T *sm, Forward_T *fwd);
__global__ void CNN_softmax_back(Softmax_T *sm, size_t lbl);
__global__ void CNN_softmax_loss(Softmax_T *sm, size_t lbl, double *loss);
__global__ void CNN_softmax_cpyOut(Softmax_T *sm, double *output);

/* TODO: delete this */
void CNN_testData(Data_T *data, size_t idx);

#endif

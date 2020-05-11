#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

typedef struct Net Net_T;
typedef struct Filter Filter_T;

Filter_T *CNN_initFltrs(size_t wid, size_t hgt, size_t numFltr);
void CNN_freeFltrs(Filter_T *fltrs);

Net_T *CNN_initFC(size_t *topology, size_t netSize);
void CNN_testFC(Net_T *net);
void CNN_freeFC(Net_T *net);

#endif

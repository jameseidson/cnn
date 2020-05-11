#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

typedef struct Net Net_T;

Net_T *CNN_initFC(size_t *topology, size_t netSize);
void CNN_testFC(Net_T *net);
void CNN_freeFC(Net_T *net);

#endif

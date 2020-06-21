#ifndef DATA_H
#define DATA_H

#include "../src/mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct Data {
  size_t numEpoch;
  size_t num;
  size_t hgt;
  size_t wid;
  size_t *lbls; /* must be stored in host memory */
  double *imgs; /* must be stored in device memory */ /* 4d array: num * hgt * wid * colors (r, g, b) */
} Data_T;

Data_T *CNN_data_init(size_t numEpoch, size_t num, size_t hgt, size_t wid, size_t *lbls, double *imgs);
void CNN_data_free(Data_T *data);

#endif

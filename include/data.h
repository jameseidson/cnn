#ifndef DATA_H
#define DATA_H

typedef struct Data {
  size_t numEpoch;
  size_t num;
  size_t hgt;
  size_t wid;
  size_t *lbls; /* must be stored in device memory */
  double *imgs; /* must be stored in device memory */ /* 4d array: num * hgt * wid * colors (r, g, b) */
} Data_T;

Data_T *CNN_initData(size_t numEpoch, size_t num, size_t hgt, size_t wid, size_t *lbls, double *imgs);
void CNN_freeData(Data_T *data);

#endif

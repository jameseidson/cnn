#include "data.h"

Data_T *CNN_data_init(size_t numEpoch, size_t num, size_t hgt, size_t wid, size_t *lbls, double *imgs) {
  Data_T *data = (Data_T *)malloc(sizeof(Data_T));
  data->numEpoch = numEpoch;
  data->num = num;
  data->hgt = hgt;
  data->wid = wid;

  size_t lblBytes = num * sizeof(size_t);
  data->lbls = (size_t *)malloc(lblBytes);
  memcpy(data->lbls, lbls, lblBytes);

  size_t imgBytes = num * hgt * wid * NUM_CHNL * sizeof(double);
  cudaMalloc((void **)&data->imgs, imgBytes);
  cudaMemcpy(data->imgs, imgs, imgBytes, cudaMemcpyHostToDevice);

  return data;
}

void CNN_data_free(Data_T *data) {
  free(data->lbls);
  cudaFree(data->imgs);
  free(data);
}

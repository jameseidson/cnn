#include "../include/cnn.h"
#include "../include/data.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

const size_t ALLOWED_BYTES = 5000000000; /* gtx 1070 has 8GB of VRAM */

void printOut(double *out, uint8_t lbl) {
  printf("Output layer:\n");
  for (uint8_t i = 0; i < NUM_OUT; i++) {
    printf("[%u]: %f", i, out[i]);
    if (i == lbl) {
      printf(" <--- truth\n");
    } else {
      printf("\n");
    }
  }
}

int main() {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, ALLOWED_BYTES);

  FILE *batchBins[NUM_BATCH];
  batchBins[0] = fopen("./cifar-10-batches-bin/data_batch_1.bin", "rb");
  batchBins[1] = fopen("./cifar-10-batches-bin/data_batch_2.bin", "rb");
  batchBins[2] = fopen("./cifar-10-batches-bin/data_batch_3.bin", "rb");
  batchBins[3] = fopen("./cifar-10-batches-bin/data_batch_4.bin", "rb");
  batchBins[4] = fopen("./cifar-10-batches-bin/data_batch_5.bin", "rb");
  batchBins[5] = fopen("./cifar-10-batches-bin/test_batch.bin", "rb");
  for (size_t i = 0; i < NUM_BATCH; i++) {
    assert(batchBins[i]);
  }
  Cifar_Img_T *cifar = Cifar_readAll(batchBins);
  for (int i = 0; i < NUM_BATCH; i++) {
    fclose(batchBins[i]);
  }
  Data_T *data = Cifar_prepData(cifar, 0, 2, 3);

  FILE *cfgFile = fopen("./cifar.cfg", "r");
  assert(cfgFile);

  CNN_T *cnn = CNN_init(cfgFile, data);

  size_t testIdx = 0;
  double *out = (double *)malloc(NUM_OUT * sizeof(double));
  double *img = Cifar_getImg(cifar, testIdx);
  uint8_t lbl = Cifar_getLbl(cifar, testIdx);

  CNN_train(cnn, data);

  CNN_predict(cnn, img, out);
  printOut(out, lbl);

  Cifar_freeAll(cifar);
  CNN_data_free(data);
  fclose(cfgFile);
  CNN_free(cnn);
  free(out);
  Cifar_freeImg(img);
  cudaDeviceReset();
  return 0;
}

#include "../include/cnn.h"
#include "../include/data.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const size_t ALLOWED_BYTES = 5000000000; /* gtx 1070 has 8GB of VRAM */

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
  Data_T *data = Cifar_prepData(cifar, 0, 60000);
  FILE *cfgFile = fopen("./cifar.cfg", "r");
  assert(cfgFile);
  CNN_T *cnn = CNN_init(cfgFile, data);
  fclose(cfgFile);

  double *img = Cifar_getImg(cifar, TEST_SET * BATCH_SIZE);
  CNN_feed(cnn, img);

  CNN_free(cnn);
  CNN_freeData(data);
  Cifar_freeAll(cifar);
  Cifar_freeImg(img);

  cudaDeviceReset();
  return 0;
}

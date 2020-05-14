#include "net.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>


int main() {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, ULLONG_MAX);
  FILE *batchBins[NUM_BATCH];
  
  batchBins[0] = fopen("./cifar-10-batches-bin/data_batch_1.bin", "rb");
  batchBins[1] = fopen("./cifar-10-batches-bin/data_batch_2.bin", "rb");
  batchBins[2] = fopen("./cifar-10-batches-bin/data_batch_3.bin", "rb");
  batchBins[3] = fopen("./cifar-10-batches-bin/data_batch_4.bin", "rb");
  batchBins[4] = fopen("./cifar-10-batches-bin/data_batch_5.bin", "rb");
  batchBins[5] = fopen("./cifar-10-batches-bin/test_batch.bin", "rb");

  Cifar_Img_T *cifar = Cifar_readImg(batchBins);

  for (int i = 0; i < NUM_BATCH; i++) {
    fclose(batchBins[i]);
  }

  Data_T *netData = Cifar_prepData(cifar, 0, 60000);

  //Cifar_exportPPM(cifar, 0, stdout);

  Cifar_freeImg(cifar);
  CNN_freeData(netData);


  /*
  Features_T *kernel = CNN_initFtrs(5, 2, 2);
  CNN_freeFtrs(kernel);
  */

  cudaDeviceReset();
  return 0;
}

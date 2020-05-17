#include "net.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void convolveIdx(Convlvd_T *conv, Features_T *kern, double *imgs, size_t idx) {
  dim3 blkSize(BLKS_3D, BLKS_3D, BLKS_3D);
  dim3 grdSize(NUMBLK(kern->num, BLKS_3D), NUMBLK(conv->hgt, BLKS_3D), NUMBLK(conv->wid, BLKS_3D));

  CNN_convolve<<<grdSize, blkSize>>>(conv, kern, &imgs[idx * CHNL_SIZE * NUM_CHNL]);
  cudaDeviceSynchronize();
}

__global__ void callPool(Convlvd_T *conv) {
  double *buffer;
  cudaMalloc((void **)&buffer, conv->num * conv->hgt * conv->wid * NUM_CHNL * sizeof(double));

  dim3 blkSize(BLKS_3D, BLKS_3D, BLKS_3D);
  dim3 grdSize(NUMBLK(conv->num, BLKS_3D), NUMBLK(conv->hgt, BLKS_3D), NUMBLK(conv->wid, BLKS_3D));

  CNN_pool<<<grdSize, blkSize>>>(conv, buffer);
  cudaDeviceSynchronize();

  cudaFree(buffer);
}

int main() {
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

  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, ULLONG_MAX);
  Data_T *netData = Cifar_prepData(cifar, 0, 60000);
  //CNN_testData(netData, 0);
  //Cifar_exportPPM(cifar, 0, stdout);

  Features_T *kern = CNN_initFtrs(1, 3, 3);
  Convlvd_T *conv = CNN_initConvlvd(kern, netData, 2, 2);

  convolveIdx<<<1, 1>>>(conv, kern, netData->imgs, 0);
  CNN_testConvolve<<<1, 1>>>(conv);
  callPool<<<1, 1>>>(conv);
  CNN_testConvolve<<<1, 1>>>(conv);
  cudaDeviceSynchronize();

  Cifar_freeImg(cifar);
  CNN_freeData(netData);
  CNN_freeFtrs(kern);
  CNN_freeConvlvd(conv);

  cudaDeviceReset();
  return 0;
}

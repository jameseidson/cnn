#include "net.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>

static const uint8_t NUM_OUT = 10;

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

__global__ void callNormalize(Convlvd_T *conv) {
  size_t numPxl = conv->num * conv->hgt * conv->wid * NUM_CHNL;

  CNN_normalize<<<NUMBLK(numPxl, BLKS_1D), BLKS_1D>>>(conv);
  cudaDeviceSynchronize();
}

__global__ void callFeedForward(Classify_T *cls, Convlvd_T *conv) {
  double *out;
  cudaMalloc((void **)&out, NUM_OUT * sizeof(double));

  CNN_feedForward<<<1, 1>>>(cls);
  cudaDeviceSynchronize();

  printf("Activations of Output Layer:\n");
  for (uint8_t i = 0; i < NUM_OUT; i++) {
    printf("[%u]: %f\n", i, cls->activs[FLAT2D((cls->numLyr - 1), i, NUM_OUT)]);
  }
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

  Features_T *kern = CNN_initFtrs(3, 3, 3);
  Convlvd_T *conv = CNN_initConvlvd(kern, netData, 2, 2);

  convolveIdx<<<1, 1>>>(conv, kern, netData->imgs, 0);
  callPool<<<1, 1>>>(conv);
  callNormalize<<<1, 1>>>(conv);
  cudaDeviceSynchronize();

  /* numFeat * convHgt * convWid * numChnl */
  /* should probably have a way to access these values directly but we'll cross that bridge later */
  size_t numIn = 3 * 15 * 15 * NUM_CHNL;
  size_t topology[] = {numIn, 200, NUM_OUT};

  Classify_T *cls = CNN_initClsfier(topology, 3);
  callFeedForward<<<1, 1>>>(cls, conv);
  cudaDeviceSynchronize();

  Cifar_freeImg(cifar);
  CNN_freeData(netData);
  CNN_freeFtrs(kern);
  CNN_freeConvlvd(conv);
  CNN_freeClsfier(cls);

  cudaDeviceReset();
  return 0;
}

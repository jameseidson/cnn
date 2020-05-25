#include "cnn.h"
#include "net.h"
#include "cifar.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void convolveIdx(Net_T *net, Features_T *kern, double *imgs, size_t idx) {
  dim3 blkSize(BLKS_3D, BLKS_3D, BLKS_3D);
  dim3 grdSize(NUMBLK(kern->num, BLKS_3D), NUMBLK(net->hgt, BLKS_3D), NUMBLK(net->wid, BLKS_3D));

  CNN_convolve<<<grdSize, blkSize>>>(net, kern, &imgs[idx * CHNL_SIZE * NUM_CHNL]);
  cudaDeviceSynchronize();
}

__global__ void callPool(Net_T *net, Pool_T *pool) {
  double *buffer;
  cudaMalloc((void **)&buffer, net->num * net->hgt * net->wid * NUM_CHNL * sizeof(double));

  dim3 blkSize(BLKS_3D, BLKS_3D, BLKS_3D);
  dim3 grdSize(NUMBLK(net->num, BLKS_3D), NUMBLK(net->hgt, BLKS_3D), NUMBLK(net->wid, BLKS_3D));
  CNN_pool<<<grdSize, blkSize>>>(net, pool, buffer);
  cudaDeviceSynchronize();

  cudaFree(buffer);
}

__global__ void callNormalize(Net_T *net) {
  size_t numPxl = net->num * net->hgt * net->wid * NUM_CHNL;

  CNN_normalize<<<NUMBLK(numPxl, BLKS_1D), BLKS_1D>>>(net);
  cudaDeviceSynchronize();
}

__global__ void callFeedForward(Classify_T *cls, Net_T *net) {
  double *out;
  cudaMalloc((void **)&out, NUM_OUT * sizeof(double));

  CNN_feedForward<<<1, 1>>>(cls);
  cudaDeviceSynchronize();

  printf("Activations of Output Layer:\n");
  for (uint8_t i = 0; i < NUM_OUT; i++) {
    printf("[%u]: %f\n", i, cls->activs[FLAT2D((cls->numLyr - 1), i, NUM_OUT)]);
  }
}

int oldmain() {
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

  Features_T *kern = CNN_initFtrs(3, 3, 3);
  Net_T *net = CNN_initNet(kern, netData);
  Pool_T *pool = CNN_initPool(2, 2);

  convolveIdx<<<1, 1>>>(net, kern, netData->imgs, 0);
  callPool<<<1, 1>>>(net, pool);
  callNormalize<<<1, 1>>>(net);
  cudaDeviceSynchronize();

  /* numFeat * netHgt * netWid * numChnl */
  size_t numIn = 3 * 15 * 15 * NUM_CHNL;
  size_t topology[] = {numIn, 200, NUM_OUT};

  Classify_T *cls = CNN_initClsfier(topology, 3, 0.1);
  callFeedForward<<<1, 1>>>(cls, net);
  cudaDeviceSynchronize();

  Cifar_freeImg(cifar);
  CNN_freeData(netData);
  CNN_freeFtrs(kern);
  CNN_freeNet(net);
  CNN_freeClsfier(cls);
  CNN_freePool(pool);

  cudaDeviceReset();
  return 0;
}

int main() {
  /* cudaDeviceSetLimit(cudaLimitMallocHeapSize, ULLONG_MAX); */

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
  Data_T *data = Cifar_prepData(cifar, 0, 60000);

  FILE *cfgFile = fopen("./cifar.cfg", "r");
  CNN_T *cnn = CNN_init(cfgFile, data);


  CNN_free(cnn);
  CNN_freeData(data);
  Cifar_freeImg(cifar);

  return 0;
}

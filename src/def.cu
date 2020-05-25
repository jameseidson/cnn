#include "def.h"

size_t findMax(size_t *, size_t);

Data_T *CNN_initData(size_t numEpoch, size_t num, size_t hgt, size_t wid, size_t *lbls, double *imgs) {
  Data_T *data = (Data_T *)malloc(sizeof(Data_T));
  data->numEpoch = numEpoch;
  data->num = num;
  data->hgt = hgt;
  data->wid = wid;

  size_t lblBytes = num * sizeof(size_t);
  cudaMalloc((void **)&data->lbls, lblBytes);
  cudaMemcpy(data->lbls, lbls, lblBytes, cudaMemcpyHostToDevice);

  size_t imgBytes = num * hgt * wid * NUM_CHNL * sizeof(double);
  cudaMalloc((void **)&data->imgs, imgBytes);
  cudaMemcpy(data->imgs, imgs, imgBytes, cudaMemcpyHostToDevice);

  return data;
}

__global__ void cuda_initNet(Net_T *net, Features_T *kern, size_t imgHgt, size_t imgWid) {
  net->num = kern->num;
  net->hgt = imgHgt - kern->hgt + 1;
  net->wid = imgWid - kern->wid + 1;
  cudaMalloc((void **)&net->imgs, net->num * net->hgt * net->wid * NUM_CHNL * sizeof(double));

  size_t totalPxls = net->num * net->hgt * net->wid * NUM_CHNL;
  for (size_t i = 0; i < totalPxls; i++) {
    net->imgs[i] = 0.1f;
  }
}

Net_T *CNN_initNet(Features_T *kern, Data_T *data) {
  Net_T *net;
  cudaMalloc((void **)&net, sizeof(Net_T));
  cuda_initNet<<<1, 1>>>(net, kern, data->hgt, data->wid);
  cudaDeviceSynchronize();

  return net;
}

__global__ void cuda_initFtrs(Features_T *kern, size_t num, size_t hgt, size_t wid) {
  kern->num = num;
  kern->hgt = hgt;
  kern->wid = wid;

  size_t numImg = num * hgt * wid * NUM_CHNL;
  cudaMalloc((void **)&kern->imgs, numImg * sizeof(double));
  for (size_t i = 0; i < numImg; i++) {
    kern->imgs[i] = 0.1f;
  }
}

Features_T *CNN_initFtrs(size_t numFeat, size_t hgt, size_t wid) {
  Features_T *kern;
  cudaMalloc((void **)&kern, sizeof(Features_T));
  cuda_initFtrs<<<1, 1>>>(kern, numFeat, hgt, wid);
  cudaDeviceSynchronize();

  return kern;
}

Pool_T *CNN_initPool(size_t winDim, size_t stride) {
  Pool_T *pool;
  cudaMalloc((void **)&pool, sizeof(Pool_T));
  cudaMemcpy(&pool->winDim, &winDim, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(&pool->stride, &stride, sizeof(size_t), cudaMemcpyHostToDevice);

  return pool;
}

__global__ void cuda_initClsfier(Classify_T *cls, size_t *topo, size_t numLyr, size_t maxNrn, double lrnRate) {
  cls->numLyr = numLyr;
  cls->maxNrn = maxNrn;
  cls->lrnRate = lrnRate;

  /* deep copy topo otherwise we must free from host, which is not possible because it's a member of a 
  device-stored struct. yeah, it's pretty gross */
  cudaMalloc((void **)&cls->topo, numLyr * sizeof(size_t));
  for (size_t i = 0; i < numLyr; i++) {
    cls->topo[i] = (i == 0 || i == numLyr - 1) ? topo[i] : topo[i] + 1;
  }

  size_t totalNrn = 0;
  for (size_t i = 0; i < numLyr; i++) {
    totalNrn += cls->topo[i];
  }
  cudaMalloc((void **)&cls->activs, totalNrn * sizeof(double));
  for (size_t i = 0; i < totalNrn; i++) {
    cls->activs[i] = 0.0f;
  }


  size_t totalWgt = 0;
  for (size_t i = 0; i < numLyr- 1; i++) {
    totalWgt += cls->topo[i] * topo[i + 1];
  }
  cudaMalloc((void **)&cls->wgts, totalWgt * sizeof(double));
  for (size_t i = 0; i < totalWgt; i++) {
    cls->wgts[i] = 0.01f;
  }
}

Classify_T *CNN_initClsfier(size_t *topology, size_t numLyr, double lrnRate) {
  Classify_T *cls;
  cudaMalloc((void **)&cls, sizeof(Classify_T));

  size_t *topo;
  cudaMalloc((void **)&topo, numLyr * sizeof(size_t));
  cudaMemcpy(topo, topology, numLyr * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initClsfier<<<1, 1>>>(cls, topo, numLyr, findMax(topology, numLyr), lrnRate);
  cudaDeviceSynchronize();

  cudaFree(topo);
  return cls;
}

void CNN_freeData(Data_T *data) {
  cudaFree(data->lbls);
  cudaFree(data->imgs);
  free(data);
}

__global__ void cuda_freeNet(Net_T *net) {
  cudaFree(net->imgs);
}

void CNN_freeNet(Net_T *net) {
  cuda_freeNet<<<1, 1>>>(net);
  cudaDeviceSynchronize();

  cudaFree(net);
}

__global__ void cuda_freeFtrs(Features_T *kern) {
  cudaFree(kern->imgs);
}

void CNN_freeFtrs(Features_T *kern) {
  cuda_freeFtrs<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

void CNN_freePool(Pool_T *pool) {
  cudaFree(pool);
}

__global__ void cuda_freeClsfier(Classify_T *fc) {
  cudaFree(fc->topo);
  cudaFree(fc->activs);
  cudaFree(fc->wgts);
}

void CNN_freeClsfier(Classify_T *cls) {
  cuda_freeClsfier<<<1,1>>>(cls);
  cudaDeviceSynchronize();
  cudaFree(cls);
}

size_t findMax(size_t *arr, size_t len) {
  size_t max = 0;
  for (size_t i = 0; i < len; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }

  return max;
}

#include "def.h"

size_t findMax(size_t *, size_t);

Data_T *CNN_initData(size_t numEpoch, size_t num, size_t hgt, size_t wid, size_t *lbls, double *imgs) {
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

__global__ void cuda_initFwd(Forward_T *fwd, size_t maxNum, size_t hgt, size_t wid) {
  fwd->num = 1;
  fwd->rows = hgt;
  fwd->cols = wid;

  cudaMalloc((void **)&fwd->mats, maxNum * hgt * wid * NUM_CHNL * sizeof(double));
}

Forward_T *CNN_initFwd(size_t maxNum, size_t hgt, size_t wid) {
  Forward_T *fwd;
  cudaMalloc((void **)&fwd, sizeof(Forward_T));
  cuda_initFwd<<<1, 1>>>(fwd, maxNum, hgt, wid);
  cudaDeviceSynchronize();

  return fwd;
}

__global__ void cuda_initFtrs(Features_T *kern, size_t num, size_t hgt, size_t wid) {
  kern->num = num;
  kern->rows = hgt;
  kern->cols = wid;

  size_t numElm = num * hgt * wid * NUM_CHNL;
  cudaMalloc((void **)&kern->mats, numElm * sizeof(double));
  MAT_randomize<<<NUMBLK(numElm, BLKS_1D), BLKS_1D>>>(kern->mats, numElm);
}

Features_T *CNN_initFtrs(size_t num, size_t hgt, size_t wid) {
  Features_T *kern;
  cudaMalloc((void **)&kern, sizeof(Features_T));
  cuda_initFtrs<<<1, 1>>>(kern, num, hgt, wid);

  return kern;
}

Pool_T *CNN_initPool(size_t dim, size_t stride) {
  Pool_T *pool;
  cudaMalloc((void **)&pool, sizeof(Pool_T));
  cudaMemcpy(&pool->dim, &dim, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(&pool->stride, &stride, sizeof(size_t), cudaMemcpyHostToDevice);

  return pool;
}

__global__ void cuda_initSoftmax(Softmax_T *sm, size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType) {
  sm->numLyr = numLyr;
  sm->lrnRate = lrnRate;
  sm->fType = fType;
  size_t lastLyr = numLyr - 1;
  cudaMalloc((void **)&sm->aTopo, numLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->aIdx, numLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->wTopo, lastLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->wIdx, lastLyr * sizeof(size_t));

  size_t totalNrn = 0;
  size_t totalWgt = 0;
  for (size_t i = 0; i < lastLyr; i++) {
    sm->aTopo[i] = topo[i] + 1;
    sm->wTopo[i] = topo[i + 1];
    sm->aIdx[i] = totalNrn;
    sm->wIdx[i] = totalWgt;
    totalNrn += sm->aTopo[i];
    totalWgt += sm->aTopo[i] * sm->wTopo[i];
  }
  sm->aTopo[lastLyr] = topo[lastLyr];
  sm->aIdx[lastLyr] = totalNrn;
  totalNrn += topo[lastLyr];

  cudaMalloc((void **)&sm->activs, totalNrn * sizeof(double));
  cudaMalloc((void **)&sm->deltas, totalNrn * sizeof(double));
  cudaMalloc((void **)&sm->wgts, totalWgt * sizeof(double));
  MAT_setVal<<<NUMBLK(totalNrn, BLKS_1D), BLKS_1D>>>(sm->activs, totalNrn, 1);
  MAT_randomize<<<NUMBLK(totalWgt, BLKS_1D), BLKS_1D>>>(sm->wgts, totalWgt);
}

Softmax_T *CNN_initSoftmax(size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType) {
  Softmax_T *sm;
  cudaMalloc((void **)&sm, sizeof(Softmax_T));

  size_t *topo_d;
  cudaMalloc((void **)&topo_d, numLyr * sizeof(size_t));
  cudaMemcpy(topo_d, topo, numLyr * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initSoftmax<<<1, 1>>>(sm, topo_d, numLyr, lrnRate, fType);
  cudaDeviceSynchronize();

  cudaFree(topo_d);
  return sm;
}

void CNN_freeData(Data_T *data) {
  free(data->lbls);
  cudaFree(data->imgs);
  free(data);
}

__global__ void cuda_freeFwd(Forward_T *fwd) {
  cudaFree(fwd->mats);
}

void CNN_freeFwd(Forward_T *fwd) {
  cuda_freeFwd<<<1, 1>>>(fwd);
  cudaDeviceSynchronize();

  cudaFree(fwd);
}

__global__ void cuda_freeFtrs(Features_T *kern) {
  cudaFree(kern->mats);
}

void CNN_freeFtrs(Features_T *kern) {
  cuda_freeFtrs<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

void CNN_freePool(Pool_T *pool) {
  cudaFree(pool);
}

__global__ void cuda_freeSoftmax(Softmax_T *sm) {
  cudaFree(sm->aTopo);
  cudaFree(sm->wTopo);
  cudaFree(sm->aIdx);
  cudaFree(sm->wIdx);
  cudaFree(sm->activs);
  cudaFree(sm->wgts);
}

void CNN_freeSoftmax(Softmax_T *sm) {
  cuda_freeSoftmax<<<1, 1>>>(sm);
  cudaDeviceSynchronize();

  cudaFree(sm);
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

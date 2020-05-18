#include "def.h"

size_t findMax(size_t *, size_t);

__global__ void cuda_initClsfier(Classify_T *cls, size_t *topo, size_t numLyr, size_t maxNrn) {
  cls->numLyr = numLyr;
  cls->maxNrn = maxNrn;

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

Classify_T *CNN_initClsfier(size_t *topology, size_t numLyr) {
  Classify_T *cls;
  cudaMalloc((void **)&cls, sizeof(Classify_T));

  size_t *topo;
  cudaMalloc((void **)&topo, numLyr * sizeof(size_t));
  cudaMemcpy(topo, topology, numLyr * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initClsfier<<<1, 1>>>(cls, topo, numLyr, findMax(topology, numLyr));
  cudaDeviceSynchronize();

  cudaFree(topo);
  return cls;
} __global__ void cuda_initConvlvd(Convlvd_T *conv, Features_T *kern, size_t imgHgt, size_t imgWid,
                                                                    size_t winDim, size_t stride) {
  conv->num = kern->num;

  conv->hgt = imgHgt - kern->hgt + 1;
  conv->wid = imgWid - kern->wid + 1;

  assert(winDim > 0 && winDim < conv->hgt && winDim < conv->wid);
  assert((conv->hgt - winDim) % stride == 0);
  assert((conv->wid - winDim) % stride == 0);

  cudaMalloc((void **)&conv->imgs, conv->num * conv->hgt * conv->wid * NUM_CHNL * sizeof(double));

  size_t totalPxls = conv->num * conv->hgt * conv->wid * NUM_CHNL;
  for (size_t i = 0; i < totalPxls; i++) {
    conv->imgs[i] = 0.1f;
  }

  conv->winDim = winDim;
  conv->stride = stride;
}

Convlvd_T *CNN_initConvlvd(Features_T *kern, Data_T *data, size_t winDim, size_t stride) {
  Convlvd_T *conv;
  cudaMalloc((void **)&conv, sizeof(Convlvd_T));
  cuda_initConvlvd<<<1, 1>>>(conv, kern, data->hgt, data->wid, winDim, stride);
  cudaDeviceSynchronize();

  return conv;
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

__global__ void cuda_freeClsfier(Classify_T *net) {
  cudaFree(net->topo);
  cudaFree(net->activs);
  cudaFree(net->wgts);
}

void CNN_freeClsfier(Classify_T *cls) {
  cuda_freeClsfier<<<1,1>>>(cls);
  cudaDeviceSynchronize();
  cudaFree(cls);
}

__global__ void cuda_freeConvlvd(Convlvd_T *conv) {
  cudaFree(conv->imgs);
}

void CNN_freeConvlvd(Convlvd_T *conv) {
  cuda_freeConvlvd<<<1, 1>>>(conv);
  cudaDeviceSynchronize();

  cudaFree(conv);
}

__global__ void cuda_freeFtrs(Features_T *kern) {
  cudaFree(kern->imgs);
}

void CNN_freeFtrs(Features_T *kern) {
  cuda_freeFtrs<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

void CNN_freeData(Data_T *data) {
  cudaFree(data->lbls);
  cudaFree(data->imgs);
  free(data);
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

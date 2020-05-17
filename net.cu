#include "net.h"

#define RED 0
#define GRN 1
#define BLU 2

typedef struct GPUClassify {
  size_t *topo;
  double *activs;
  double *wgts; 
} GPUClassify_T;

struct Classify {
  size_t size;
  size_t maxNrn;
  GPUClassify_T *dev;
};

size_t findMax(size_t *, size_t);

__global__ void CNN_pool(Convlvd_T *conv, double *buffer) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t stride = conv->stride;
  size_t hgt = conv->hgt;
  size_t wid = conv->wid;
  size_t poolHgt = ((hgt - conv->winDim) / stride) + 1;
  size_t poolWid = ((wid - conv->winDim) / stride) + 1;

  if (x < conv->num && y < hgt && y % stride == 0 && z < wid && z % stride == 0) {
    double chnlMax[NUM_CHNL] = { DBL_MIN, DBL_MIN, DBL_MIN };
    for (size_t i = y; i - y < conv->winDim; i++) {
      for (size_t j = z; j - z < conv->winDim; j++) {
        for (uint8_t k = 0; k < NUM_CHNL; k++) {
          double curPxl = conv->imgs[FLAT4D(x, i, j, k, hgt, wid, NUM_CHNL)];
          if(curPxl > chnlMax[k]) {
            chnlMax[k] = curPxl;
          }
        }
      }
    }

    for (uint8_t i = 0; i < NUM_CHNL; i++) {
      buffer[FLAT4D(x, (y / stride), (z / stride), i, poolHgt, poolWid, NUM_CHNL)] = chnlMax[i];
    }
  }

  conv->hgt = poolHgt;
  conv->wid = poolWid;

  __syncthreads();
  if (x < conv->num && y < poolHgt && z < poolWid) {
    for (int i = 0; i < NUM_CHNL; i++) {
      size_t idx = FLAT4D(x, y, z, i, poolHgt, poolWid, NUM_CHNL);
      conv->imgs[idx] = buffer[idx];
    }
  }
}

__global__ void CNN_testConvolve(Convlvd_T *conv) {
  for (size_t i = 0; i < conv->num; i++) {
    printf("Printing convolved feature #%lu:\n", i);
    printf("Red Channel:\n");
    for (size_t j = 0; j < conv->hgt; j++) {
      for (size_t k = 0; k < conv->wid; k++) {
        printf("%0.2f ", conv->imgs[FLAT4D(i, j, k, 0, conv->hgt, conv->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Green Channel:\n");
    for (size_t j = 0; j < conv->hgt; j++) {
      for (size_t k = 0; k < conv->wid; k++) {
        printf("%0.2f ", conv->imgs[FLAT4D(i, j, k, 1, conv->hgt, conv->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Blue Channel:\n");
    for (size_t j = 0; j < conv->hgt; j++) {
      for (size_t k = 0; k < conv->wid; k++) {
        printf("%0.2f ", conv->imgs[FLAT4D(i, j, k, 2, conv->hgt, conv->wid, NUM_CHNL)]);
      }
      printf("\n");
    }
  }
}

__global__ void CNN_convolve(Convlvd_T *conv, Features_T *kern, double *img) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t imgWid = conv->wid + kern->wid - 1;
  size_t kernPxls = kern->hgt * kern->wid;
  
  if (x < kern->num && y < conv->hgt && z < conv->wid) {
    double chnlTotal[NUM_CHNL] = { 0.0f, 0.0f, 0.0f };
    for (size_t i = y; i - y < kern->hgt; i++) {
      for (size_t j = z; j - z < kern->wid; j++) {
        for (size_t k = 0; k < NUM_CHNL; k++) {
          chnlTotal[k] += img[FLAT3D(i, j, k, imgWid, NUM_CHNL)] * 
                    kern->imgs[FLAT4D(x, (i - y), (j - z), k, kern->hgt, kern->wid, NUM_CHNL)];
        }
      }
    }

    for (uint8_t i = 0; i < NUM_CHNL; i++) {
      conv->imgs[FLAT4D(x, y, z, i, conv->hgt, conv->wid, NUM_CHNL)] = chnlTotal[i] / kernPxls;
    }
  }
}

__global__ void cuda_initConvlvd(Convlvd_T *conv, Features_T *kern, size_t imgHgt, size_t imgWid,
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
    conv->imgs[i] = 0.5f;
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

__global__ void cuda_freeConvlvd(Convlvd_T *conv) {
  cudaFree(conv->imgs);
}

void CNN_freeConvlvd(Convlvd_T *conv) {
  cuda_freeConvlvd<<<1, 1>>>(conv);
  cudaDeviceSynchronize();

  cudaFree(conv);
}

__global__ void cuda_testData(double *imgs, size_t *lbls, size_t idx, size_t hgt, size_t wid) {
  double sum = 0;
  printf("Red Channel:\n");
  for (size_t i = 0; i < hgt; i++) {
    for (size_t j = 0; j < wid; j++) {
      printf("%0.2f ", imgs[FLAT4D(idx, i, j, 0, hgt, wid, NUM_CHNL)]);
      sum += imgs[FLAT4D(idx, i, j, 0, hgt, wid, NUM_CHNL)];
    }
    printf("\n");
  }

  sum = 0;
  printf("Green Channel:\n");
  for (size_t i = 0; i < hgt; i++) {
    for (size_t j = 0; j < wid; j++) {
      printf("%0.2f ", imgs[FLAT4D(idx, i, j, 1, hgt, wid, NUM_CHNL)]);
      sum += imgs[FLAT4D(idx, i, j, 1, hgt, wid, NUM_CHNL)];
    }
    printf("\n");
  }

  sum = 0;
  printf("Blue Channel:\n");
  for (size_t i = 0; i < hgt; i++) {
    for (size_t j = 0; j < wid; j++) {
      printf("%0.2f ", imgs[FLAT4D(idx, i, j, 2, hgt, wid, NUM_CHNL)]);
      sum += imgs[FLAT4D(idx, i, j, 2, hgt, wid, NUM_CHNL)];
    }
    printf("\n");
  }
}

void CNN_testData(Data_T *data, size_t idx) {
  cuda_testData<<<1, 1>>>(data->imgs, data->lbls, idx, data->hgt, data->wid);
  cudaDeviceSynchronize();
}

void CNN_freeData(Data_T *data) {
  cudaFree(data->lbls);
  cudaFree(data->imgs);
  free(data);
}

__global__ void cuda_initFtrs(Features_T *kern, size_t num, size_t hgt, size_t wid) {
  kern->num = num;
  kern->hgt = hgt;
  kern->wid = wid;

  size_t numImg = num * hgt * wid * NUM_CHNL;
  cudaMalloc((void **)&kern->imgs, numImg * sizeof(double));
  for (size_t i = 0; i < numImg; i++) {
    kern->imgs[i] = 0.5f;
  }
}

Features_T *CNN_initFtrs(size_t numFeat, size_t hgt, size_t wid) {
  Features_T *kern;
  cudaMalloc((void **)&kern, sizeof(Features_T));
  cuda_initFtrs<<<1, 1>>>(kern, numFeat, hgt, wid);
  cudaDeviceSynchronize();

  return kern;
}

__global__ void cuda_freeFtrs(Features_T *kern) {
  cudaFree(kern->imgs);
}

void CNN_freeFtrs(Features_T *kern) {
  cuda_freeFtrs<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

__global__ void cuda_initClsfier(GPUClassify_T *net, size_t *topo, size_t netSize) {
  cudaMalloc((void **)&net->topo, netSize * sizeof(size_t));
  /* deep copy- host memory can only be freed in host */
  for (size_t i = 0; i < netSize; i++) {
    net->topo[i] = topo[i];
  }

  size_t totalNrn = 0;
  for (size_t i = 0; i < netSize; i++) {
    totalNrn += topo[i];
  }
  cudaMalloc((void **)&net->activs, totalNrn * sizeof(double));
  for (size_t i = 0; i < totalNrn; i++) {
    net->activs[i] = 0.5f;
  }

  size_t totalWgt = 0;
  for (size_t i = 0; i < netSize - 1; i++) {
    totalWgt += topo[i] * topo[i + 1];
  }
  cudaMalloc((void **)&net->wgts, totalWgt * sizeof(double));
  for (size_t i = 0; i < totalWgt; i++) {
    net->wgts[i] = 0.5f;
  }
}

Classify_T *CNN_initClsfier(size_t *topology, size_t netSize) {
  Classify_T *net = (Classify_T *)malloc(sizeof(Classify_T));
  net->maxNrn = findMax(topology, netSize);
  net->size = netSize;

  cudaMalloc((void **)&net->dev, sizeof(GPUClassify_T));

  /* init topology */
  size_t *topo_d;
  cudaMalloc((void **)&topo_d, netSize * sizeof(size_t));
  cudaMemcpy(topo_d, topology, netSize * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initClsfier<<<1,1>>>(net->dev, topo_d, netSize);
  cudaDeviceSynchronize();
  cudaFree(topo_d);

  return net;
}

__global__ void cuda_testClsfier(GPUClassify_T *net, size_t size) {
	for (size_t i = 0; i < size; i++) {
		printf("  layer %lu:\n", i);
		printf("    numNrn %lu:\n", net->topo[i]);
		for (size_t j = 0; j < net->topo[i]; j++) {
			printf("    neuron %lu (activ: %.2f) weights:\n      ", j, net->activs[FLAT2D(i, j, size)]);
      if (i != size - 1) {
				for (size_t k = 0; k < net->topo[i + 1]; k++) {
					printf("[%.2f] ", net->wgts[FLAT3D(i, j, k, size, net->topo[i])]);
				}
      }
			printf("\n");
		}
	}
}

void CNN_testClsfier(Classify_T *net) {
  cuda_testClsfier<<<1,1>>>(net->dev, net->size);
  cudaDeviceSynchronize();
}

__global__ void cuda_freeClsfier(GPUClassify_T *net) {
  cudaFree(net->topo);
  cudaFree(net->activs);
  cudaFree(net->wgts);
}

void CNN_freeClsfier(Classify_T *net) {
  cuda_freeClsfier<<<1,1>>>(net->dev);
  cudaDeviceSynchronize();
  cudaFree(net->dev);
  free(net);
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

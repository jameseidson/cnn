#include "net.h"

#define flat2d(i, j, wid) (i * wid) + j 
#define flat3d(i, j, k, wid, hgt) (hgt * wid * i) + (wid * j) + k

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

struct FLearn {
  size_t stride;
  ImgList_T featrs;
};

size_t CNN_findMax(size_t *, size_t);

FLearn_T *CNN_initFL(size_t numFeat, size_t featWid, size_t featHgt, size_t stride) {
  FLearn_T *net = (FLearn_T *)malloc(sizeof(FLearn_T));
  net->stride = stride;
  net->featrs.numImg = numFeat;
  net->featrs.wid = featWid;
  net->featrs.hgt = featHgt;

  size_t chnlSize = numFeat * featWid * featHgt * sizeof(double);
  cudaMalloc((void **)&net->featrs.r, chnlSize);
  cudaMalloc((void **)&net->featrs.g, chnlSize);
  cudaMalloc((void **)&net->featrs.b, chnlSize);

  return net;
}

__global__ void cuda_initC(GPUClassify_T *net, size_t *topo, size_t netSize) {
  cudaMalloc((void **)&net->topo, netSize * sizeof(size_t));
  /* deep copy- host memory can only be freed in host */
  for (int i = 0; i < netSize; i++) {
    net->topo[i] = topo[i];
  }

  size_t totalNrn = 0;
  for (int i = 0; i < netSize; i++) {
    totalNrn += topo[i];
  }
  cudaMalloc((void **)&net->activs, totalNrn * sizeof(double));

  size_t totalWgt = 0;
  for (int i = 0; i < netSize - 1; i++) {
    totalWgt += topo[i] * topo[i + 1];
  }
  cudaMalloc((void **)&net->wgts, totalWgt * sizeof(double));
}

Classify_T *CNN_initC(size_t *topology, size_t netSize) {
  assert(netSize > 0 && topology != NULL);
  for (int i = 0; i < netSize; i++) {
    assert(topology[i] != 0);
  }

  Classify_T *net = (Classify_T *)malloc(sizeof(Classify_T));
  net->maxNrn = CNN_findMax(topology, netSize);
  net->size = netSize;

  cudaMalloc((void **)&net->dev, sizeof(GPUClassify_T));

  /* init topology */
  size_t *topo_d;
  cudaMalloc((void **)&topo_d, netSize * sizeof(size_t));
  cudaMemcpy(topo_d, topology, netSize * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initC<<<1,1>>>(net->dev, topo_d, netSize);
  cudaDeviceSynchronize();
  cudaFree(topo_d);

  return net;
}

__global__ void cuda_freeC(GPUClassify_T *net) {
  cudaFree(net->topo);
  cudaFree(net->activs);
  cudaFree(net->wgts);
}

void CNN_freeC(Classify_T *net) {
  cuda_freeC<<<1,1>>>(net->dev);
  cudaDeviceSynchronize();
  cudaFree(net->dev);
  free(net);
  cudaDeviceReset();
}

__global__ void cuda_testC(GPUClassify_T *net, size_t size) {
	for (size_t i = 0; i < size; i++) {
		printf("  layer %lu:\n", i);
		printf("    numNrn %lu:\n", net->topo[i]);
		for (size_t j = 0; j < net->topo[i]; j++) {
			printf("    neuron %lu (activ: %.2f) weights:\n      ", j, net->activs[flat2d(i, j, size)]);
      if (i != size - 1) {
				for (size_t k = 0; k < net->topo[i + 1]; k++) {
					printf("[%.2f] ", net->wgts[flat3d(i, j, k, size, net->topo[i])]);
				}
      }
			printf("\n");
		}
	}
}

void CNN_testC(Classify_T *net) {
  cuda_testC<<<1,1>>>(net->dev, net->size);
  cudaDeviceSynchronize();
}

void CNN_freeFL(FLearn_T *net) {
  cudaFree(net->featrs.r);
  cudaFree(net->featrs.b);
  cudaFree(net->featrs.g);

  free(net);
}

void CNN_freeData(Data_T* input) {
  cudaFree(input->lbls);
  cudaFree(input->imgDat.r);
  cudaFree(input->imgDat.g);
  cudaFree(input->imgDat.b);
  cudaDeviceReset();
  free(input);
}

size_t CNN_findMax(size_t *arr, size_t len) {
  size_t max = 0;
  for (int i = 0; i < len; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }

  return max;
}

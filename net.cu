#include "net.h"

#define BLK_SIZE 256
#define flat2d(i, j, max_j) (i * max_j) + j 
#define flat3d(i, j, k, max_j, max_k) (i * max_j * max_k) + (j * max_k) + k
#define numBlk(dim) (dim / BLK_SIZE) + 1

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

struct Features {
  size_t numFeat;
  size_t featWid;
  size_t featHgt;
  double *ftrImgs; /* 4d arr: num * wid * hgt * colors (r, g, b) */
};

size_t findMax(size_t *, size_t);

void CNN_freeData(Data_T *data) {
  cudaFree(data->lbls);
  cudaFree(data->imgs);
  free(data);
}

Features_T *CNN_initFtrs(size_t numFeat, size_t featWid, size_t featHgt) {
  Features_T *kernel = (Features_T *)malloc(sizeof(Features_T));
  kernel->numFeat = numFeat;
  kernel->featWid = featWid;
  kernel->featHgt = featHgt;

  cudaMalloc((void **)&kernel->ftrImgs, numFeat * featWid * featHgt * NUM_CHNL * sizeof(double));

  return kernel;
}

void CNN_freeFtrs(Features_T *kernel) {
  cudaFree(kernel->ftrImgs);
  free(kernel);
}

__global__ void cuda_initClsfier(GPUClassify_T *net, size_t *topo, size_t netSize) {
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
  for (int i = 0; i < totalNrn; i++) {
    net->activs[i] = 0.5f;
  }

  size_t totalWgt = 0;
  for (int i = 0; i < netSize - 1; i++) {
    totalWgt += topo[i] * topo[i + 1];
  }
  cudaMalloc((void **)&net->wgts, totalWgt * sizeof(double));
  for (int i = 0; i < totalWgt; i++) {
    net->wgts[i] = 0.5f;
  }
}

Classify_T *CNN_initClsfier(size_t *topology, size_t netSize) {
  assert(netSize > 0 && topology != NULL);
  for (int i = 0; i < netSize; i++) {
    assert(topology[i] != 0);
  }

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
  for (int i = 0; i < len; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }

  return max;
}

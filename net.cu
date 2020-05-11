#include "net.h"

#define flat2d(i, j, wid) (i * wid) + j
#define flat3d(i, j, k, wid, hgt) (hgt * wid * i) + (wid * j) + k

typedef struct GPU_Net {
  size_t *topo;
  double *activs;
  double *wgts;
} GPU_Net_T;

struct Net {
  size_t size;
  size_t maxNrn;
  GPU_Net_T *dev;
};

size_t CNN_findMax(size_t *, size_t);

__global__ void cuda_initFC(GPU_Net_T *net, size_t *topo, size_t netSize) {
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

Net_T *CNN_initFC(size_t *topology, size_t netSize) {
  assert(netSize > 0 && topology != NULL);
  for (int i = 0; i < netSize; i++) {
    assert(topology[i] != 0);
  }

  Net_T *net = (Net_T *)malloc(sizeof(Net_T));
  net->maxNrn = CNN_findMax(topology, netSize);
  net->size = netSize;

  cudaMalloc((void **)&net->dev, sizeof(GPU_Net_T));

  /* init topology */
  size_t *topo_d;
  cudaMalloc((void **)&topo_d, netSize * sizeof(size_t));
  cudaMemcpy(topo_d, topology, netSize * sizeof(size_t), cudaMemcpyHostToDevice);

  cuda_initFC<<<1,1>>>(net->dev, topo_d, netSize);
  cudaDeviceSynchronize();
  cudaFree(topo_d);

  return net;
}

__global__ void cuda_freeFC(GPU_Net_T *net) {
  cudaFree(net->topo);
  cudaFree(net->activs);
  cudaFree(net->wgts);
}

void CNN_freeFC(Net_T *net) {
  cuda_freeFC<<<1,1>>>(net->dev);
  cudaDeviceSynchronize();
  cudaFree(net->dev);
  free(net);
  cudaDeviceReset();
}

__global__ void cuda_testFC(GPU_Net_T *net, size_t size) {
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

void CNN_testFC(Net_T *net) {
  cuda_testFC<<<1,1>>>(net->dev, net->size);
  cudaDeviceSynchronize();
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

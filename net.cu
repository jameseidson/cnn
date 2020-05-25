#include "net.h"

#define SIG(x) 1.0f/(1.0f + exp(-x))
#define DSIG(x) SIG(x) * (1 - SIG(X))

__global__ void CNN_convolve(Net_T *net, Features_T *kern, double *img) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t imgWid = net->wid + kern->wid - 1;
  size_t kernPxls = kern->hgt * kern->wid;
  
  if (x < kern->num && y < net->hgt && z < net->wid) {
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
      net->imgs[FLAT4D(x, y, z, i, net->hgt, net->wid, NUM_CHNL)] = chnlTotal[i] / kernPxls;
    }
  }
}

__global__ void CNN_normalize(Net_T *net) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < net->num * net->hgt * net->wid * NUM_CHNL && net->imgs[x] < 0) {
    net->imgs[x] = 0;
  }
}

__global__ void CNN_pool(Net_T *net, Pool_T *pool, double *buffer) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t hgt = net->hgt;
  size_t wid = net->wid;
  size_t poolHgt = ((hgt - pool->winDim) / pool->stride) + 1;
  size_t poolWid = ((wid - pool->winDim) / pool->stride) + 1;

  if (x < net->num && y < hgt && y % pool->stride == 0 && z < wid && z % pool->stride == 0) {
    double chnlMax[NUM_CHNL] = { DBL_MIN, DBL_MIN, DBL_MIN };
    for (size_t i = y; i - y < pool->winDim; i++) {
      for (size_t j = z; j - z < pool->winDim; j++) {
        for (uint8_t k = 0; k < NUM_CHNL; k++) {
          double curPxl = net->imgs[FLAT4D(x, i, j, k, hgt, wid, NUM_CHNL)];
          if(curPxl > chnlMax[k]) {
            chnlMax[k] = curPxl;
          }
        }
      }
    }

    for (uint8_t i = 0; i < NUM_CHNL; i++) {
      buffer[FLAT4D(x, (y / pool->stride), (z / pool->stride), i, poolHgt, poolWid, NUM_CHNL)] = chnlMax[i];
    }
  }

  net->hgt = poolHgt;
  net->wid = poolWid;

  __syncthreads();
  if (x < net->num && y < poolHgt && z < poolWid) {
    for (int i = 0; i < NUM_CHNL; i++) {
      size_t idx = FLAT4D(x, y, z, i, poolHgt, poolWid, NUM_CHNL);
      net->imgs[idx] = buffer[idx];
    }
  }
}


__global__ void CNN_feedForward(Classify_T *cls) {
  for (size_t i = 1; i < cls->numLyr; i++) {
    size_t numNrn = (i == cls->numLyr - 1) ? cls->topo[i] : cls->topo[i] - 1;
    size_t prevLyr = i - 1;
    for (size_t j = 0; j < numNrn; j++) {
      double sum = 0;
      for (size_t k = 0; k < cls->topo[prevLyr]; k++) {
        sum += cls->wgts[FLAT3D(prevLyr, k, j, cls->topo[prevLyr], numNrn)]
             * cls->activs[FLAT2D(prevLyr, k, cls->topo[prevLyr])];
      }
      cls->activs[FLAT2D(i, j, numNrn)] = SIG(sum);
    }
  }
}

__global__ void CNN_testNet(Net_T *net) {
  printf("Num features: %lu\n", net->num);
  printf("Feature wid: %lu\n", net->hgt);
  printf("Feature Hgt: %lu\n", net->wid);

  for (size_t i = 0; i < net->num; i++) {
    printf("Printing netolved feature #%lu:\n", i);
    printf("Red Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.2f ", net->imgs[FLAT4D(i, j, k, 0, net->hgt, net->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Green Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.2f ", net->imgs[FLAT4D(i, j, k, 1, net->hgt, net->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Blue Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.2f ", net->imgs[FLAT4D(i, j, k, 2, net->hgt, net->wid, NUM_CHNL)]);
      }
      printf("\n");
    }
  }
}

/* TODO: this is bugged- don't call it until you fix */
__global__ void cuda_testClsfier(Classify_T *cls) {
  size_t size = cls->numLyr;
	for (size_t i = 0; i < size; i++) {
		printf("  layer %lu:\n", i);
		printf("    numNrn %lu:\n", cls->topo[i]);
		for (size_t j = 0; j < cls->topo[i]; j++) {
      if (j == cls->topo[i] - 1 && i != 0 && i != size - 1) {
        printf("BIAS ");
      } else {
        printf("     ");
      }
      printf("neuron %lu (activ: %.1f) weights:\n      ", j, cls->activs[FLAT2D(i, j, cls->topo[i])]);
      if (i != size - 1) {
        size_t loopLimit = (i == size - 2) ? cls->topo[i + 1] : cls->topo[i + 1] - 1;
				for (size_t k = 0; k < loopLimit; k++) {
					printf("[%.1f] ", cls->wgts[FLAT3D(i, j, k, cls->topo[i], cls->topo[i + 1])]);
				}
      }
			printf("\n");
		}
	}
}

void CNN_testClsfier(Classify_T *cls) {
  cuda_testClsfier<<<1,1>>>(cls);
  cudaDeviceSynchronize();
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

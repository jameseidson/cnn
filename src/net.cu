#include "net.h"

#define SIG(x) 1.0f/(1.0f + exp(-x))
#define DSIG(x) SIG(x) * (1 - SIG(X))

__global__ void CNN_prepNet(Net_T *net, double *imgs, size_t num, size_t hgt, size_t wid) {
  size_t i = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (i == gridDim.x * BLKS_1D - 1) { /* last thread is least likely to be doing any work */
    net->num = num;
    net->hgt = hgt;
    net->wid = wid;
  }

  if (i < num * hgt * wid * NUM_CHNL) {
    net->imgs[i] = imgs[i];
  }
}

__global__ void CNN_convolve(Net_T *net, Features_T *kern, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t kernPxls = kern->hgt * kern->wid;

  size_t outNum = net->num * kern->num;
  size_t outHgt = net->hgt - kern->hgt + 1;
  size_t outWid = net->wid - kern->wid + 1;
  
  if (x < outNum && y < outHgt && z < outWid) {
    size_t x_n = L_IDX(x, net->num);
    size_t x_k = K_IDX(x, x_n, kern->num, net->num); 

    double chnlTotal[NUM_CHNL] = { 0.0f, 0.0f, 0.0f };
    for (size_t i = y; i - y < kern->hgt; i++) {
      for (size_t j = z; j - z < kern->wid; j++) {
        for (size_t k = 0; k < NUM_CHNL; k++) {

          chnlTotal[k] += net->imgs[FLAT4D(x_n, i, j, k, net->hgt, net->wid, NUM_CHNL)] * 
                    kern->imgs[FLAT4D(x_k, (i - y), (j - z), k, kern->hgt, kern->wid, NUM_CHNL)];
        }
      }
    }

    for (size_t i = 0; i < NUM_CHNL; i++) {
      buf[FLAT4D(x, y, z, i, outHgt, outWid, NUM_CHNL)] = chnlTotal[i] / kernPxls;
    }
  }

  /* kernel launches are costly, so it's best to reiterate prepData rather than call it, especially since we
     already have the right amount of threads */
  __syncthreads();
  if (x == gridDim.x * BLKS_3D - 1 && y == gridDim.y * BLKS_3D - 1 && z == gridDim.z * BLKS_3D - 1) {
    net->num = outNum;
    net->hgt = outHgt;
    net->wid = outWid;
  }
  if (x < outNum && y < outHgt && z < outWid) {
    for (size_t i = 0; i < NUM_CHNL; i++) {
      size_t idx = FLAT4D(x, y, z, i, outHgt, outWid, NUM_CHNL);
      net->imgs[idx] = buf[idx];
    }
  }
}

__global__ void CNN_normalize(Net_T *net) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < net->num * net->hgt * net->wid * NUM_CHNL && net->imgs[x] < 0) {
    net->imgs[x] = 0;
  }
}

__global__ void CNN_pool(Net_T *net, Pool_T *pool, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);
  size_t z = FLAT2D(blockIdx.z, threadIdx.z, blockDim.z);

  size_t hgt = net->hgt;
  size_t wid = net->wid;
  size_t outHgt = ((hgt - pool->winDim) / pool->stride) + 1;
  size_t outWid = ((wid - pool->winDim) / pool->stride) + 1;

  if (x < net->num && y < outHgt && z < outWid) {
    double chnlMax[NUM_CHNL] = { DBL_MIN, DBL_MIN, DBL_MIN };

    for (size_t i = 0; i < pool->winDim; i++) {
      for (size_t j = 0; j < pool->winDim; j++) {
        size_t i_n = y * pool->stride + i;
        size_t j_n = z * pool->stride + j;
        for (size_t k = 0; k < NUM_CHNL; k++) {
          double curPxl = net->imgs[FLAT4D(x, i_n, j_n, k, hgt, wid, NUM_CHNL)];
          if (curPxl > chnlMax[k]) {
            chnlMax[k] = curPxl;
          }
        }
      }
    }

    for (size_t i = 0; i < NUM_CHNL; i++) {
      buf[FLAT4D(x, y, z, i, outHgt, outWid, NUM_CHNL)] = chnlMax[i];
    }
  }

  __syncthreads();
  if (x == gridDim.x * BLKS_3D - 1 && y == gridDim.y * BLKS_3D - 1 && z == gridDim.z * BLKS_3D - 1) {
    net->hgt = outHgt;
    net->wid = outWid;
  }
  if (x < net->num && y < outHgt && z < outWid) {
    for (size_t i = 0; i < NUM_CHNL; i++) {
      size_t idx = FLAT4D(x, y, z, i, outHgt, outWid, NUM_CHNL);
      net->imgs[idx] = buf[idx];
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
    printf("Printing convolved feature #%lu:\n", i);
    printf("Red Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.3f ", net->imgs[FLAT4D(i, j, k, 0, net->hgt, net->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Green Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.3f ", net->imgs[FLAT4D(i, j, k, 1, net->hgt, net->wid, NUM_CHNL)]);
      }
      printf("\n");
    }

    printf("Blue Channel:\n");
    for (size_t j = 0; j < net->hgt; j++) {
      for (size_t k = 0; k < net->wid; k++) {
        printf("%0.3f ", net->imgs[FLAT4D(i, j, k, 2, net->hgt, net->wid, NUM_CHNL)]);
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

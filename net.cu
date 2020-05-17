#include "net.h"

__global__ void CNN_normalize(Convlvd_T *conv) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < conv->num * conv->hgt * conv->wid * NUM_CHNL && conv->imgs[x] < 0) {
    conv->imgs[x] = 0;
  }
}

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

__global__ void cuda_testClsfier(GPUClassify_T *net, size_t size) {
	for (size_t i = 0; i < size; i++) {
		printf("  layer %lu:\n", i);
		printf("    numNrn %lu:\n", net->topo[i]);
		for (size_t j = 0; j < net->topo[i]; j++) {
      if (j == net->topo[i] - 1 && i != 0 && i != size - 1) {
        printf("BIAS ");
      } else {
        printf("     ");
      }
      printf("neuron %lu (activ: %.2f) weights:\n      ", j, net->activs[FLAT2D(i, j, net->topo[i])]);
      if (i != size - 1) {
        size_t loopLimit = (i == size - 2) ? net->topo[i + 1] : net->topo[i + 1] - 1;
				for (size_t k = 0; k < loopLimit; k++) {
					printf("[%.2f] ", net->wgts[FLAT3D(i, j, k, net->topo[i], net->topo[i + 1])]);
				}
      }
			printf("\n");
		}
	}
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


void CNN_testClsfier(Classify_T *net) {
  cuda_testClsfier<<<1,1>>>(net->dev, net->size);
  cudaDeviceSynchronize();
}

void CNN_testData(Data_T *data, size_t idx) {
  cuda_testData<<<1, 1>>>(data->imgs, data->lbls, idx, data->hgt, data->wid);
  cudaDeviceSynchronize();
}

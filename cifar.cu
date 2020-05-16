#include "cifar.h"

struct Cifar_Img {
  uint8_t lbl;
  uint8_t *r;
  uint8_t *g;
  uint8_t *b;
};

Cifar_Img_T *Cifar_readImg(FILE **batchBins) {
  Cifar_Img_T *cifar = (Cifar_Img_T *)malloc(NUM_IMG * sizeof(Cifar_Img_T));

  for (size_t i = 0; i < NUM_BATCH; i++) {
    FILE *curBin = batchBins[i];
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      Cifar_Img_T *curImg = &cifar[flat2d(i, j, BATCH_SIZE)];

      fread(&curImg->lbl, sizeof(uint8_t), 1, curBin);

      curImg->r = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->b = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->g = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));

      fread(curImg->r, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->g, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->b, sizeof(uint8_t), CHNL_SIZE, curBin);
    }
  }

  return cifar;
}

__global__ void cuda_prepData(double *imgs, size_t num, uint8_t *r, uint8_t *g, uint8_t *b) {
  size_t imgIdx = flat2d(blockIdx.x, threadIdx.x, blockDim.x);
  size_t imgSize = NUM_CHNL * CHNL_SIZE;

  if (imgIdx < num) {
    double *curImg = &imgs[imgIdx * imgSize];
    uint8_t *cur_r = &r[imgIdx * CHNL_SIZE];
    uint8_t *cur_g = &g[imgIdx * CHNL_SIZE];
    uint8_t *cur_b = &b[imgIdx * CHNL_SIZE];
    for (size_t j = 0; j < imgSize; j++) {
      size_t k = j / 3;
      switch(j % 3) {
        case 0:
          curImg[j] = (double)cur_r[k] / 255.0f;
          break;
        case 1:
          curImg[j] = (double)cur_g[k] / 255.0f;
          break;
        case 2:
          curImg[j] = (double)cur_b[k] / 255.0f;
          break;
      }
    }
  }
}

Data_T *Cifar_prepData(Cifar_Img_T *cifar, size_t idx, size_t num) {
  size_t totalPxls = num * DIM * DIM * NUM_CHNL;
  Data_T *data = (Data_T *)malloc(sizeof(Data_T));
  cudaMalloc((void **)&data->lbls, num * sizeof(size_t));
  cudaMalloc((void **)&data->imgs, totalPxls * sizeof(double));

  data->num = num;
  data->hgt = DIM;
  data->wid = DIM;

  size_t endIdx = idx + num;
  for (size_t i = idx; i < endIdx; i++) {
    size_t tmpLbl = (size_t)cifar[idx].lbl;
    cudaMemcpy(&data->lbls[i - idx], &tmpLbl, sizeof(size_t), cudaMemcpyHostToDevice);
  }

  size_t totalChnlBytes = num * CHNL_SIZE * sizeof(uint8_t);
  uint8_t *tmp_r;
  cudaMalloc((void **)&tmp_r, totalChnlBytes);
  uint8_t *tmp_g;
  cudaMalloc((void **)&tmp_g, totalChnlBytes);
  uint8_t *tmp_b;
  cudaMalloc((void **)&tmp_b, totalChnlBytes);

  size_t chnlBytes = CHNL_SIZE * sizeof(uint8_t);
  for (size_t i = 0; i < num; i++) {
    size_t chnlIdx = i * CHNL_SIZE;
    size_t cifarIdx = i + idx;
    cudaMemcpy(&tmp_r[chnlIdx], cifar[cifarIdx].r, chnlBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&tmp_g[chnlIdx], cifar[cifarIdx].g, chnlBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&tmp_b[chnlIdx], cifar[cifarIdx].b, chnlBytes, cudaMemcpyHostToDevice);
  }

  cuda_prepData<<<numBlk(num, BLKS_1D), BLKS_1D>>>(data->imgs, num, tmp_r, tmp_g, tmp_b);
  cudaDeviceSynchronize();

  cudaFree(tmp_r);
  cudaFree(tmp_g);
  cudaFree(tmp_b);

  return data;
}

void Cifar_exportPPM(Cifar_Img_T *cifar, size_t imgIdx, FILE *ppmOut) {
  fputc('P', ppmOut);
  fputc('6', ppmOut);
  fputc('\n', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut);
  fputc(' ', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut);
  fputc(' ', ppmOut);
  fputc('2', ppmOut);
  fputc('5', ppmOut);
  fputc('5', ppmOut);
  fputc('\n', ppmOut);

  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      size_t idx = flat2d(i, j, DIM);
      fputc(cifar[imgIdx].r[idx], ppmOut);
      fputc(cifar[imgIdx].g[idx], ppmOut);
      fputc(cifar[imgIdx].b[idx], ppmOut);
    }
  }
}

void Cifar_freeImg(Cifar_Img_T *cifar) {
    for (size_t i = 0; i < NUM_IMG; i++) {
      free(cifar[i].r);
      free(cifar[i].b);
      free(cifar[i].g);
    }

  free(cifar);
}

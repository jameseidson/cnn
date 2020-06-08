#include "cifar.h"

struct Cifar_Img {
  uint8_t lbl;
  uint8_t *r;
  uint8_t *g;
  uint8_t *b;
};

Cifar_Img_T *Cifar_readAll(FILE **batchBins) {
  Cifar_Img_T *cifar = (Cifar_Img_T *)malloc(NUM_IMG * sizeof(Cifar_Img_T));

  for (size_t i = 0; i < NUM_BATCH; i++) {
    FILE *curBin = batchBins[i];
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      Cifar_Img_T *curImg = &cifar[FLAT2D(i, j, BATCH_SIZE)];

      fread(&curImg->lbl, sizeof(uint8_t), 1, curBin);

      curImg->r = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->g = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->b = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));

      fread(curImg->r, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->g, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->b, sizeof(uint8_t), CHNL_SIZE, curBin);
    }
  }

  return cifar;
}

__global__ void test(uint8_t *r) {
  for (size_t i = 0; i < CHNL_SIZE; i++) {
    printf("%u", r[i]);
  }
}

__global__ void cuda_getImg(double *out, uint8_t *r, uint8_t *g, uint8_t *b) {
  size_t i = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t imgSize = NUM_CHNL * CHNL_SIZE;
  if (i < imgSize) {
    size_t j = i / 3;
    switch(i % 3) {
      case 0:
        out[i] = (double)r[j] / 255.0f;
        break;
      case 1:
        out[i] = (double)g[j] / 255.0f;
        break;
      case 2:
        out[i] = (double)b[j] / 255.0f;
        break;
    }
  }
}

double *Cifar_getImg(Cifar_Img_T *cifar, size_t idx) {
  size_t chnlBytes = CHNL_SIZE * sizeof(uint8_t);
  uint8_t *tmp_r;
  cudaMalloc((void **)&tmp_r, chnlBytes);
  uint8_t *tmp_g;
  cudaMalloc((void **)&tmp_g, chnlBytes);
  uint8_t *tmp_b;
  cudaMalloc((void **)&tmp_b, chnlBytes);

  cudaMemcpy(tmp_r, cifar[idx].r, chnlBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_g, cifar[idx].g, chnlBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_b, cifar[idx].b, chnlBytes, cudaMemcpyHostToDevice);

  size_t imgSize = NUM_CHNL * CHNL_SIZE;
  double *out;
  cudaMalloc((void **)&out, imgSize * sizeof(double));

  cuda_getImg<<<NUMBLK(imgSize, BLKS_1D), BLKS_1D>>>(out, tmp_r, tmp_g, tmp_b);
  cudaDeviceSynchronize();

  cudaFree(tmp_r);
  cudaFree(tmp_g);
  cudaFree(tmp_b);

  return out;
}

uint8_t Cifar_getLbl(Cifar_Img_T *cifar, size_t idx) {
  return cifar[idx].lbl;
}

__global__ void cuda_prepData(double *imgs, size_t num, uint8_t *r, uint8_t *g, uint8_t *b) {
  size_t idx = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t imgSize = NUM_CHNL * CHNL_SIZE;

  if (idx < num) {
    double *curImg = &imgs[idx * imgSize];
    uint8_t *cur_r = &r[idx * CHNL_SIZE];
    uint8_t *cur_g = &g[idx * CHNL_SIZE];
    uint8_t *cur_b = &b[idx * CHNL_SIZE];
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

  cuda_prepData<<<NUMBLK(num, BLKS_1D), BLKS_1D>>>(data->imgs, num, tmp_r, tmp_g, tmp_b);
  cudaDeviceSynchronize();

  cudaFree(tmp_r);
  cudaFree(tmp_g);
  cudaFree(tmp_b);

  return data;
}

void Cifar_exportPPM(Cifar_Img_T *cifar, size_t idx, FILE *ppmOut) {
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
      size_t idx = FLAT2D(i, j, DIM);
      fputc(cifar[idx].r[idx], ppmOut);
      fputc(cifar[idx].g[idx], ppmOut);
      fputc(cifar[idx].b[idx], ppmOut);
    }
  }
}

void Cifar_freeImg(double *img) {
  cudaFree(img);
}
void Cifar_freeAll(Cifar_Img_T *cifar) {
    for (size_t i = 0; i < NUM_IMG; i++) {
      free(cifar[i].r);
      free(cifar[i].b);
      free(cifar[i].g);
    }

  free(cifar);
}

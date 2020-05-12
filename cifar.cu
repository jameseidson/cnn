#include "cifar.h"

#define flat2d(i, j, wid) (i * wid) + j

struct Img {
  uint8_t lbl;
  uint8_t *r;
  uint8_t *g;
  uint8_t *b;
};

Img_T *readImg(FILE **batchBins) {
  Img_T *data = (Img_T *)malloc(NUM_IMG * sizeof(Img_T));

  for (int i = 0; i < NUM_BATCH; i++) {
    FILE *curBin = batchBins[i];
    for (int j = 0; j < BATCH_SIZE; j++) {
      Img_T *curImg = &data[flat2d(i, j, BATCH_SIZE)];

      fread(&curImg->lbl, sizeof(uint8_t), 1, curBin);

      curImg->r = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->b = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->g = (uint8_t *)malloc(CHNL_SIZE * sizeof(uint8_t));

      fread(curImg->r, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->g, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->b, sizeof(uint8_t), CHNL_SIZE, curBin);
    }
  }

  return data;
}

void copyChnl(double **chnl, Img_T *data, double *buffer, size_t imgBytes) {
  for (int i = 0; i < NUM_IMG; i++) {
    for (int j = 0; j < CHNL_SIZE; j++) {
      buffer[flat2d(i, j, CHNL_SIZE)] = data[i].r[j];
    }
  }
  cudaMalloc((void **)chnl, imgBytes);
  cudaMemcpy(*chnl, buffer, imgBytes, cudaMemcpyHostToDevice);
}

/* this is ABSURDLY inefficient and slow, but unfortunately i couldn't think of a better way to
do it since host FILE* cannot be dereferenced on the device. thankfully this func only gets
called once */
Data_T *prepData(Img_T *data) {
  Data_T *cifar = (Data_T *)malloc(sizeof(Data_T));
  cifar->imgDat.numImg = NUM_IMG;
  cifar->imgDat.wid = DIM;
  cifar->imgDat.hgt = DIM;
  size_t *newLbls = (size_t *)malloc(NUM_IMG * sizeof(size_t));
  for (int i = 0; i < NUM_IMG; i++) {
    newLbls[i] = (size_t)data[i].lbl;
  }
  free(newLbls);
  cudaMalloc((void **)&cifar->lbls, NUM_IMG * sizeof(size_t));
  cudaMemcpy(cifar->lbls, newLbls, NUM_IMG * sizeof(size_t), cudaMemcpyHostToDevice);

  size_t imgBytes = NUM_IMG * CHNL_SIZE * sizeof(double);
  double *buffer = (double *)malloc(imgBytes);
  
  copyChnl(&cifar->imgDat.r, data, buffer, imgBytes);
  copyChnl(&cifar->imgDat.g, data, buffer, imgBytes);
  copyChnl(&cifar->imgDat.b, data, buffer, imgBytes);

  free(buffer);
  freeCifarImg(data);
  return cifar;
}

void exportPPM(Img_T *data, size_t imgIdx, FILE *ppmOut) {
  fputc('P', ppmOut);
  fputc('6', ppmOut); /* magic num */
  fputc('\n', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut); /* wid */
  fputc(' ', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut); /* hgt */
  fputc(' ', ppmOut);
  fputc('2', ppmOut);
  fputc('5', ppmOut);
  fputc('5', ppmOut); /* denominator */
  fputc('\n', ppmOut);

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      size_t idx = flat2d(i, j, DIM);
      fputc(data[imgIdx].r[idx], ppmOut);
      fputc(data[imgIdx].g[idx], ppmOut);
      fputc(data[imgIdx].b[idx], ppmOut);
    }
  }
}

void freeCifarImg(Img_T *data) {
    for (int i = 0; i < NUM_IMG; i++) {
      free(data[i].r);
      free(data[i].b);
      free(data[i].g);
    }

  free(data);
}

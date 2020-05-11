#include "cifar.h"

#define flat2d(i, j, wid) (i * wid) + j

struct Img {
  uint8_t lbl;
  uint8_t *r;
  uint8_t *g;
  uint8_t *b;
};

Img_T *readImg(FILE **batchBins) {
  Img_T *data = malloc(NUM_IMG * sizeof(Img_T));

  for (int i = 0; i < NUM_BATCH; i++) {
    FILE *curBin = batchBins[i];
    for (int j = 0; j < BATCH_SIZE; j++) {
      Img_T *curImg = &data[flat2d(i, j, BATCH_SIZE)];

      fread(&curImg->lbl, sizeof(uint8_t), 1, curBin);

      curImg->r = malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->b = malloc(CHNL_SIZE * sizeof(uint8_t));
      curImg->g = malloc(CHNL_SIZE * sizeof(uint8_t));

      fread(curImg->r, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->g, sizeof(uint8_t), CHNL_SIZE, curBin);
      fread(curImg->b, sizeof(uint8_t), CHNL_SIZE, curBin);
    }
  }

  return data;
}

void exportPPM(Img_T *data, size_t imgIdx, FILE *ppmOut) {
  /* this is a little gross but i'm lazy */
  fputc('P', ppmOut);
  fputc('6', ppmOut); /* magic num */
  fputc(' ', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut); /* wid */
  fputc(' ', ppmOut);
  fputc('3', ppmOut);
  fputc('2', ppmOut); /* wid */
  fputc(' ', ppmOut);
  fputc('2', ppmOut);
  fputc('5', ppmOut);
  fputc('5', ppmOut); /* wid */
  fputc('\n', ppmOut);

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      fputc(data[imgIdx].r[flat2d(i, j, DIM)], ppmOut);
      fputc(data[imgIdx].g[flat2d(i, j, DIM)], ppmOut);
      fputc(data[imgIdx].b[flat2d(i, j, DIM)], ppmOut);
    }
  }
}

void freeImg(Img_T *data) {
    for (int i = 0; i < NUM_IMG; i++) {
      free(data[i].r);
      free(data[i].b);
      free(data[i].g);
    }

  free(data);
}

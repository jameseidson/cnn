#include "net.h"

__global__ void CNN_prepFwd(Forward_T *fwd, double *imgs, size_t num, size_t rows, size_t cols) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x == gridDim.x * BLKS_1D - 1) {
    fwd->num = num;
    fwd->rows = rows;
    fwd->cols = cols;
  }

  if (x < num) {
    size_t numElm = num * rows * cols * NUM_CHNL;
    size_t idx = x * rows * cols * NUM_CHNL;
    MAT_assign<<<NUMBLK((numElm), BLKS_1D), BLKS_1D>>>(&imgs[idx], &fwd->mats[idx], numElm);
  }
}

__global__ void CNN_convolve(Forward_T *fwd, Features_T *kern, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  size_t oRows = fwd->rows - kern->rows + 1;
  size_t oCols = fwd->cols - kern->cols + 1;
  size_t oMatSize = oRows * oCols * NUM_CHNL;

  if (x < fwd->num && y < kern->num) {
    size_t i = x * fwd->rows * fwd->cols * NUM_CHNL;
    size_t j = FLAT2D(x, y, kern->num) * oMatSize;
    size_t k = y * kern->rows * kern->cols * NUM_CHNL;

    dim3 grdSize(NUMBLK(oRows, BLKS_2D), NUMBLK(oCols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_convolve<<<grdSize, blkSize>>>(&fwd->mats[i], &buf[j], &kern->mats[k], fwd->rows, fwd->cols, kern->rows, kern->cols);
  }

  __syncthreads();
  if (x < fwd->num && y < kern->num) {
    size_t i = FLAT2D(x, y, kern->num) * oMatSize;
    MAT_assign<<<NUMBLK(oMatSize, BLKS_1D), BLKS_1D>>>(&buf[i], &fwd->mats[i], oMatSize);
  }
}

__global__ void CNN_pool(Pool_T *pool, Forward_T *fwd, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  size_t oCols = ((fwd->cols - pool->dim) / pool->stride) + 1;
  size_t oRows = ((fwd->cols - pool->dim) / pool->stride) + 1;

  if (x < fwd->num) {
    size_t i = x * fwd->rows * fwd->cols * NUM_CHNL;

    dim3 grdSize(NUMBLK(oRows, BLKS_2D), NUMBLK(oCols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_pool<<<grdSize, blkSize>>>(&fwd->mats[i], &buf[i], fwd->rows, fwd->cols, pool->dim, pool->stride);
  }

  __syncthreads();
  if (x < fwd->num) {
    size_t matSize = oRows * oCols * NUM_CHNL;
    size_t i = x * matSize;
    MAT_assign<<<NUMBLK(matSize, BLKS_1D), BLKS_1D>>>(&buf[i], &fwd->mats[i], matSize);
  }
}

__global__ void CNN_normalize(Forward_T *fwd, NonLin_T func) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  void (*nonLin)(double *, double *, size_t) = (func == RELU) ? &MAT_ReLU : &MAT_sigmoid;
  size_t matSize = fwd->rows * fwd->cols * NUM_CHNL;

  if (x < fwd->num) {
    size_t i = x * matSize;
    nonLin<<<NUMBLK(matSize, BLKS_1D), BLKS_1D>>>(&fwd->mats[i], &fwd->mats[i], matSize);
  }
}

__global__ void CNN_softmax_fwd(Softmax_T *sm, Forward_T *fwd) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<NUMBLK(sm->aTopo[0] - 1, BLKS_1D), BLKS_1D>>>(fwd->mats, sm->activs, sm->aTopo[0] - 1);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < lastLyr; i++) {
    MAT_fwdProp<<<NUMBLK(sm->wTopo[i], BLKS_1D), BLKS_1D>>>
      (&sm->wgts[sm->wIdx[i]], &sm->activs[sm->aIdx[i]], &sm->activs[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }
}

__global__ void CNN_softmax_loss(Softmax_T *sm, size_t lbl, double *loss) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_loss<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>(&sm->activs[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, loss);
}

__global__ void CNN_softmax_cpyOut(Softmax_T *sm, double *output) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>(&sm->activs[sm->aIdx[lastLyr]], output, sm->aTopo[lastLyr]);
}

__global__ void CNN_softmax_gradDescent(Softmax_T *sm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t lastLyr = sm->numLyr - 1;

  if (x < lastLyr) {
      dim3 grdSize(NUMBLK(sm->wTopo[x], BLKS_2D), NUMBLK(sm->aTopo[x], BLKS_2D));
      dim3 blkSize(BLKS_2D, BLKS_2D);
      MAT_applyGradient<<<grdSize, blkSize>>>(&sm->activs[sm->aIdx[x]], &sm->deltas[sm->aIdx[x + 1]], &sm->wgts[sm->wIdx[x]], sm->wTopo[x], sm->aTopo[x], sm->lrnRate);
  }
}

__global__ void CNN_softmax_back(Softmax_T *sm, size_t lbl) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_deltas_out<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>
    (&sm->activs[sm->aIdx[lastLyr]], &sm->deltas[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, sm->fType);

  for (size_t i = lastLyr; i-- > 0;) {
    MAT_deltas_hidden<<<NUMBLK(sm->aTopo[i], BLKS_1D), BLKS_1D>>>
      (&sm->activs[sm->aIdx[i]], &sm->deltas[sm->aIdx[i]], &sm->wgts[sm->wIdx[i]], &sm->deltas[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }

  CNN_softmax_gradDescent<<<NUMBLK(sm->numLyr, BLKS_1D), BLKS_1D>>>(sm);
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

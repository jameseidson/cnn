#include "layer.h"

__global__ void GPU_fwd_init(Forward_T *, size_t, size_t, size_t);
__global__ void GPU_fwd_free(Forward_T *fwd);

__global__ void GPU_conv_init(Features_T *, size_t, size_t, size_t);
__global__ void GPU_conv_free(Features_T *);

__global__ void GPU_softmax_init(Softmax_T *, size_t *, size_t, double, NonLin_T);
__global__ void GPU_softmax_free(Softmax_T *);

__global__ void LYR_softmax_gradDescent(Softmax_T *);

Forward_T *LYR_fwd_init(size_t maxNum, size_t hgt, size_t wid) {
  Forward_T *fwd;
  cudaMalloc((void **)&fwd, sizeof(Forward_T));
  GPU_fwd_init<<<1, 1>>>(fwd, maxNum, hgt, wid);
  cudaDeviceSynchronize();

  return fwd;
}

__global__ void LYR_fwd_prep(Forward_T *fwd, double *imgs, size_t num, size_t rows, size_t cols) {
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

void LYR_fwd_free(Forward_T *fwd) {
  GPU_fwd_free<<<1, 1>>>(fwd);
  cudaDeviceSynchronize();

  cudaFree(fwd);
}

Features_T *LYR_conv_init(size_t num, size_t hgt, size_t wid) {
  Features_T *kern;
  cudaMalloc((void **)&kern, sizeof(Features_T));
  GPU_conv_init<<<1, 1>>>(kern, num, hgt, wid);

  return kern;
}

__global__ void LYR_conv_fwd(Forward_T *fwd, Features_T *kern, double *buf) {
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

void LYR_conv_free(Features_T *kern) {
  GPU_conv_free<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

Pool_T *LYR_pool_init(size_t dim, size_t stride) {
  Pool_T *pool;
  cudaMalloc((void **)&pool, sizeof(Pool_T));
  cudaMemcpy(&pool->dim, &dim, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(&pool->stride, &stride, sizeof(size_t), cudaMemcpyHostToDevice);

  return pool;
}

__global__ void LYR_pool_fwd(Pool_T *pool, Forward_T *fwd, double *buf) {
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

void LYR_pool_free(Pool_T *pool) {
  cudaFree(pool);
}

__global__ void LYR_norm_fwd(Forward_T *fwd, NonLin_T func) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  void (*nonLin)(double *, double *, size_t) = (func == RELU) ? &MAT_ReLU : &MAT_sigmoid;
  size_t matSize = fwd->rows * fwd->cols * NUM_CHNL;

  if (x < fwd->num) {
    size_t i = x * matSize;
    nonLin<<<NUMBLK(matSize, BLKS_1D), BLKS_1D>>>(&fwd->mats[i], &fwd->mats[i], matSize);
  }
}

Softmax_T *LYR_softmax_init(size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType) {
  Softmax_T *sm;
  cudaMalloc((void **)&sm, sizeof(Softmax_T));

  size_t *topo_d;
  cudaMalloc((void **)&topo_d, numLyr * sizeof(size_t));
  cudaMemcpy(topo_d, topo, numLyr * sizeof(size_t), cudaMemcpyHostToDevice);

  GPU_softmax_init<<<1, 1>>>(sm, topo_d, numLyr, lrnRate, fType);
  cudaDeviceSynchronize();

  cudaFree(topo_d);
  return sm;
}

__global__ void LYR_softmax_fwd(Softmax_T *sm, Forward_T *fwd) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<NUMBLK(sm->aTopo[0] - 1, BLKS_1D), BLKS_1D>>>(fwd->mats, sm->activs, sm->aTopo[0] - 1);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < lastLyr; i++) {
    MAT_fwdProp<<<NUMBLK(sm->wTopo[i], BLKS_1D), BLKS_1D>>>
      (&sm->wgts[sm->wIdx[i]], &sm->activs[sm->aIdx[i]], &sm->activs[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }
}

__global__ void LYR_softmax_loss(Softmax_T *sm, size_t lbl, double *loss) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_loss<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>(&sm->activs[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, loss);
}

__global__ void LYR_softmax_cpyOut(Softmax_T *sm, double *output) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>(&sm->activs[sm->aIdx[lastLyr]], output, sm->aTopo[lastLyr]);
}

__global__ void LYR_softmax_back(Softmax_T *sm, size_t lbl) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_deltas_out<<<NUMBLK(sm->aTopo[lastLyr], BLKS_1D), BLKS_1D>>>
    (&sm->activs[sm->aIdx[lastLyr]], &sm->deltas[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, sm->fType);

  for (size_t i = lastLyr; i-- > 0;) {
    MAT_deltas_hidden<<<NUMBLK(sm->aTopo[i], BLKS_1D), BLKS_1D>>>
      (&sm->activs[sm->aIdx[i]], &sm->deltas[sm->aIdx[i]], &sm->wgts[sm->wIdx[i]], &sm->deltas[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }

  LYR_softmax_gradDescent<<<NUMBLK(sm->numLyr, BLKS_1D), BLKS_1D>>>(sm);
}

__global__ void LYR_softmax_gradDescent(Softmax_T *sm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t lastLyr = sm->numLyr - 1;

  if (x < lastLyr) {
      dim3 grdSize(NUMBLK(sm->wTopo[x], BLKS_2D), NUMBLK(sm->aTopo[x], BLKS_2D));
      dim3 blkSize(BLKS_2D, BLKS_2D);
      MAT_applyGradient<<<grdSize, blkSize>>>(&sm->activs[sm->aIdx[x]], &sm->deltas[sm->aIdx[x + 1]], &sm->wgts[sm->wIdx[x]], sm->wTopo[x], sm->aTopo[x], sm->lrnRate);
  }
}

void LYR_softmax_free(Softmax_T *sm) {
  GPU_softmax_free<<<1, 1>>>(sm);
  cudaDeviceSynchronize();

  cudaFree(sm);
}
__global__ void GPU_conv_init(Features_T *kern, size_t num, size_t hgt, size_t wid) {
  kern->num = num;
  kern->rows = hgt;
  kern->cols = wid;

  size_t numElm = num * hgt * wid * NUM_CHNL;
  cudaMalloc((void **)&kern->mats, numElm * sizeof(double));
  MAT_randomize<<<NUMBLK(numElm, BLKS_1D), BLKS_1D>>>(kern->mats, numElm);
}

__global__ void GPU_conv_free(Features_T *kern) {
  cudaFree(kern->mats);
}

__global__ void GPU_fwd_init(Forward_T *fwd, size_t maxNum, size_t hgt, size_t wid) {
  fwd->num = 1;
  fwd->rows = hgt;
  fwd->cols = wid;

  cudaMalloc((void **)&fwd->mats, maxNum * hgt * wid * NUM_CHNL * sizeof(double));
}

__global__ void GPU_fwd_free(Forward_T *fwd) {
  cudaFree(fwd->mats);
}

__global__ void GPU_softmax_init(Softmax_T *sm, size_t *topo, size_t numLyr, double lrnRate, NonLin_T fType) {
  sm->numLyr = numLyr;
  sm->lrnRate = lrnRate;
  sm->fType = fType;
  size_t lastLyr = numLyr - 1;
  cudaMalloc((void **)&sm->aTopo, numLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->aIdx, numLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->wTopo, lastLyr * sizeof(size_t));
  cudaMalloc((void **)&sm->wIdx, lastLyr * sizeof(size_t));

  size_t totalNrn = 0;
  size_t totalWgt = 0;
  for (size_t i = 0; i < lastLyr; i++) {
    sm->aTopo[i] = topo[i] + 1;
    sm->wTopo[i] = topo[i + 1];
    sm->aIdx[i] = totalNrn;
    sm->wIdx[i] = totalWgt;
    totalNrn += sm->aTopo[i];
    totalWgt += sm->aTopo[i] * sm->wTopo[i];
  }
  sm->aTopo[lastLyr] = topo[lastLyr];
  sm->aIdx[lastLyr] = totalNrn;
  totalNrn += topo[lastLyr];

  cudaMalloc((void **)&sm->activs, totalNrn * sizeof(double));
  cudaMalloc((void **)&sm->deltas, totalNrn * sizeof(double));
  cudaMalloc((void **)&sm->wgts, totalWgt * sizeof(double));
  MAT_setVal<<<NUMBLK(totalNrn, BLKS_1D), BLKS_1D>>>(sm->activs, totalNrn, 1);
  MAT_randomize<<<NUMBLK(totalWgt, BLKS_1D), BLKS_1D>>>(sm->wgts, totalWgt);
}

__global__ void GPU_softmax_free(Softmax_T *sm) {
  cudaFree(sm->aTopo);
  cudaFree(sm->wTopo);
  cudaFree(sm->aIdx);
  cudaFree(sm->wIdx);
  cudaFree(sm->deltas);
  cudaFree(sm->activs);
  cudaFree(sm->wgts);
}

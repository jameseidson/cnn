#include "layer.h"

static const unsigned DEFAULT_REDU_FAC = 1000;
static const unsigned RELU_REDU_FAC = 1000;
static const unsigned SIG_REDU_FAC = 100;

__global__ void GPU_net_init(Net_T *, size_t, size_t, size_t);
__global__ void GPU_net_free(Net_T *net);

__global__ void GPU_conv_init(Conv_T *, size_t, size_t, size_t, size_t, size_t, size_t);
__global__ void GPU_conv_free(Conv_T *);

__global__ void GPU_pool_init(Pool_T *, size_t, size_t, size_t, size_t, size_t);
__global__ void GPU_pool_free(Pool_T *);

__global__ void GPU_softmax_init(Softmax_T *, size_t *, size_t, double, NonLin_T);
__global__ void GPU_softmax_free(Softmax_T *);

__global__ void LYR_softmax_gradDescent(Softmax_T *);

Net_T *LYR_net_init(size_t maxNum, size_t maxHgt, size_t maxWid) {
  Net_T *net;
  cudaMalloc((void **)&net, sizeof(Net_T));
  GPU_net_init<<<1, 1>>>(net, maxNum, maxHgt, maxWid);
  cudaDeviceSynchronize();

  return net;
}

__global__ void LYR_net_update(Net_T *net, double *imgs, size_t num, size_t rows, size_t cols) {
  net->num = num;
  net->rows = rows;
  net->cols = cols;

  size_t numElm = num * rows * cols * NUM_CHNL;
  MAT_assign<<<LAUNCH1D(numElm)>>>(imgs, net->mats, numElm);
}

void LYR_net_free(Net_T *net) {
  GPU_net_free<<<1, 1>>>(net);
  cudaDeviceSynchronize();

  cudaFree(net);
}

Conv_T *LYR_conv_init(size_t fNum, size_t fHgt, size_t fWid, size_t iNum, size_t iHgt, size_t iWid) {
  Conv_T *kern;
  cudaMalloc((void **)&kern, sizeof(Conv_T));
  GPU_conv_init<<<1, 1>>>(kern, fNum, fHgt, fWid, iNum, iHgt, iWid);

  return kern;
}

__global__ void LYR_conv_fwd(Net_T *net, Conv_T *kern) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  if (x == 0 && y == 0 ) {
    size_t numInp = net->num * net->rows * net->cols * NUM_CHNL;
    MAT_assign<<<LAUNCH1D(numInp)>>>(net->mats, kern->inp.mats, numInp);
  }
  __syncthreads();

  size_t oNum = net->num * kern->fltrs.num;
  size_t oRows = CONV_OUT(net->rows, kern->fltrs.rows);
  size_t oCols = CONV_OUT(net->cols, kern->fltrs.cols);
  size_t oMatSize = oRows * oCols * NUM_CHNL;

  if (x < net->num && y < kern->fltrs.num) {
    size_t i = x * net->rows * net->cols * NUM_CHNL;
    size_t j = y * kern->fltrs.rows * kern->fltrs.cols * NUM_CHNL;
    size_t k = FLAT2D(x, y, kern->fltrs.num) * oMatSize;

    dim3 grdSize(NUMBLK(oRows, BLKS_2D), NUMBLK(oCols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_convolve<<<grdSize, blkSize>>>(&kern->inp.mats[i], &net->mats[k], &kern->fltrs.mats[j], net->rows, net->cols, kern->fltrs.rows, kern->fltrs.cols, false);
  }

  __syncthreads();
  if (x == 0 && y == 0) {
    net->num = oNum;
    net->rows = oRows;
    net->cols = oCols;
  }
}

__global__ void LYR_conv_back(Conv_T *kern, Net_T *net, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t y = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  size_t numElm_delt = net->rows * net->cols * NUM_CHNL;
  if (x < kern->inp.num && y < kern->fltrs.num) {
    size_t i = FLAT2D(x, y, kern->fltrs.num) * numElm_delt;
    size_t j = x * kern->inp.rows * kern->inp.cols * NUM_CHNL;
    size_t k = y * kern->fltrs.rows * kern->fltrs.cols * NUM_CHNL;

    dim3 grdSize(NUMBLK(kern->inp.rows, BLKS_2D), NUMBLK(kern->inp.cols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_inv_convolve<<<grdSize, blkSize>>>(&net->mats[i], &buf[j], &kern->fltrs.mats[k], net->rows, net->cols, kern->fltrs.rows, kern->fltrs.cols, true);
  }

  __syncthreads();
  if (x < kern->inp.num && y < kern->fltrs.num) {
    size_t i = FLAT2D(x, y, kern->fltrs.num) * numElm_delt;
    size_t j = x * kern->inp.rows * kern->inp.cols * NUM_CHNL;
    size_t k = y * kern->fltrs.rows * kern->fltrs.cols * NUM_CHNL;

    dim3 grdSize(NUMBLK(kern->fltrs.rows, BLKS_2D), NUMBLK(kern->fltrs.cols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_convolve<<<grdSize, blkSize>>>(&kern->inp.mats[j], &kern->fltrs.mats[k], &net->mats[i], kern->inp.rows, kern->inp.cols, net->rows, net->cols, true);
  }

  __syncthreads();
  if (x == 0 && y == 0) {
    net->num = kern->inp.num;
    net->rows = kern->inp.rows;
    net->cols = kern->inp.cols;

    size_t numElm = net->num * net->rows * net->cols * NUM_CHNL;
    MAT_assign<<<LAUNCH1D(numElm)>>>(buf, net->mats, numElm);
  }
}

void LYR_conv_free(Conv_T *kern) {
  GPU_conv_free<<<1, 1>>>(kern);
  cudaDeviceSynchronize();

  cudaFree(kern);
}

Pool_T *LYR_pool_init(size_t dim, size_t stride, size_t iNum, size_t iRows, size_t iCols) {
  Pool_T *pool;
  cudaMalloc((void **)&pool, sizeof(Pool_T));
  GPU_pool_init<<<1, 1>>>(pool, dim, stride, iNum, POOL_OUT(iRows, dim, stride), POOL_OUT(iCols, dim, stride));
  cudaDeviceSynchronize();

  return pool;
}

__global__ void LYR_pool_fwd(Pool_T *pool, Net_T *net, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t poolElms = pool->rows * pool->cols * NUM_CHNL;

  if (x < net->num) {
    size_t i = x * net->rows * net->cols * NUM_CHNL;
    size_t j = x * poolElms;

    dim3 grdSize(NUMBLK(pool->rows, BLKS_2D), NUMBLK(pool->cols, BLKS_2D));
    dim3 blkSize(BLKS_2D, BLKS_2D);
    MAT_pool<<<grdSize, blkSize>>>(&net->mats[i], &buf[j], &pool->idxs[j], net->rows, net->cols, pool->dim, pool->stride);
  }

  __syncthreads();
  if (x == 0) {
    net->rows = pool->rows;
    net->cols = pool->cols;

    size_t numElm = net->num * poolElms;
    MAT_assign<<<LAUNCH1D(numElm)>>>(buf, net->mats, numElm);
  }
}

__global__ void LYR_pool_back(Pool_T *pool, Net_T *net, double *buf) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  size_t oRows = ((pool->rows - 1) * pool->stride) + pool->dim;
  size_t oCols = ((pool->cols - 1) * pool->stride) + pool->dim;

  if (x < net->num) {
    size_t imgElm = pool->rows * pool->cols * NUM_CHNL;
    size_t i = x * imgElm;
    size_t j = x * oRows * oCols * NUM_CHNL;
    MAT_deltas_pool<<<LAUNCH1D(imgElm)>>>(&net->mats[i], &buf[j], &pool->idxs[i], net->rows, net->cols);
  }

  __syncthreads();

  if (x == 0) {
    net->rows = oRows;
    net->cols = oCols;
    size_t totalElm = net->num * oRows * oCols * NUM_CHNL;

    MAT_assign<<<LAUNCH1D(totalElm)>>>(buf, net->mats, totalElm);
  }
}

void LYR_pool_free(Pool_T *pool) {
  GPU_pool_free<<<1, 1>>>(pool);
  cudaDeviceSynchronize();

  cudaFree(pool);
}

__global__ void LYR_norm_fwd(Net_T *net, NonLin_T func) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  void (*nonLin)(double *, double *, size_t) = (func == RELU) ? &MAT_ReLU : &MAT_sigmoid;
  size_t matSize = net->rows * net->cols * NUM_CHNL;

  if (x < net->num) {
    size_t i = x * matSize;
    nonLin<<<LAUNCH1D(matSize)>>>(&net->mats[i], &net->mats[i], matSize);
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

__global__ void LYR_softmax_fwd(Softmax_T *sm, Net_T *net) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<LAUNCH1D(sm->aTopo[0] - 1)>>>(net->mats, sm->activs, sm->aTopo[0] - 1);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < lastLyr; i++) {
    MAT_fwdProp<<<LAUNCH1D(sm->wTopo[i])>>>(&sm->wgts[sm->wIdx[i]], &sm->activs[sm->aIdx[i]], &sm->activs[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }
}

__global__ void LYR_softmax_back(Softmax_T *sm, Net_T *net, size_t lbl) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_deltas_out<<<LAUNCH1D(sm->aTopo[lastLyr])>>>(&sm->activs[sm->aIdx[lastLyr]], &sm->deltas[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, sm->fType);
  cudaDeviceSynchronize();

  for (size_t i = lastLyr; i-- > 0;) {
    MAT_deltas_hidden<<<LAUNCH1D(sm->aTopo[i])>>>(&sm->activs[sm->aIdx[i]], &sm->deltas[sm->aIdx[i]], &sm->wgts[sm->wIdx[i]], &sm->deltas[sm->aIdx[i + 1]], sm->wTopo[i], sm->aTopo[i], sm->fType);
    cudaDeviceSynchronize();
  }

  size_t numElm = net->num * net->rows * net->cols * NUM_CHNL;
  MAT_assign<<<LAUNCH1D(numElm)>>>(sm->deltas, net->mats, numElm);
  LYR_softmax_gradDescent<<<LAUNCH1D(sm->numLyr)>>>(sm);
}

__global__ void LYR_softmax_loss(Softmax_T *sm, size_t lbl, double *loss) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_loss<<<LAUNCH1D(sm->aTopo[lastLyr])>>>(&sm->activs[sm->aIdx[lastLyr]], sm->aTopo[lastLyr], lbl, loss);
}

__global__ void LYR_softmax_cpyOut(Softmax_T *sm, double *output) {
  size_t lastLyr = sm->numLyr - 1;

  MAT_assign<<<LAUNCH1D(sm->aTopo[lastLyr])>>>(&sm->activs[sm->aIdx[lastLyr]], output, sm->aTopo[lastLyr]);
}

void LYR_softmax_free(Softmax_T *sm) {
  GPU_softmax_free<<<1, 1>>>(sm);
  cudaDeviceSynchronize();

  cudaFree(sm);
}

__global__ void GPU_net_init(Net_T *net, size_t maxNum, size_t maxHgt, size_t maxWid) {
  net->num = 1;
  net->rows = maxHgt;
  net->cols = maxWid;

  cudaMalloc((void **)&net->mats, maxNum * maxHgt * maxWid * NUM_CHNL * sizeof(double));
}

__global__ void GPU_net_free(Net_T *net) {
  cudaFree(net->mats);
}

__global__ void GPU_conv_init(Conv_T *kern, size_t fNum, size_t fHgt, size_t fWid, size_t iNum, size_t iHgt, size_t iWid) {
  MatList_T fltrs;
  MatList_T inp;
  fltrs.num = fNum;
  fltrs.rows = fHgt;
  fltrs.cols = fWid;
  inp.num = iNum;
  inp.rows = iHgt;
  inp.cols = iWid;

  size_t numElm_fltrs = fNum * fHgt * fWid * NUM_CHNL;
  size_t numElm_inp = iNum * iHgt * iWid * NUM_CHNL;
  cudaMalloc((void **)&fltrs.mats, numElm_fltrs * sizeof(double));
  cudaMalloc((void **)&inp.mats, numElm_inp  * sizeof(double));
  MAT_randomize<<<LAUNCH1D(numElm_fltrs)>>>(fltrs.mats, numElm_fltrs, DEFAULT_REDU_FAC);

  kern->fltrs = fltrs;
  kern->inp = inp;
}

__global__ void GPU_conv_free(Conv_T *kern) {
  cudaFree(kern->fltrs.mats);
  cudaFree(kern->inp.mats);
}

__global__ void GPU_pool_init(Pool_T *pool, size_t dim, size_t stride, size_t iNum, size_t oRows, size_t oCols) {
  pool->dim = dim;
  pool->stride = stride;
  pool->rows = oRows;
  pool->cols = oCols;

  cudaMalloc((void **)&pool->idxs, iNum * oRows * oCols * NUM_CHNL * sizeof(double));
}

__global__ void GPU_pool_free(Pool_T *pool) {
  cudaFree(pool->idxs);
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
  MAT_setVal<<<LAUNCH1D(totalNrn)>>>(sm->activs, totalNrn, 1);
  if (fType == RELU) {
    MAT_randomize<<<LAUNCH1D(totalWgt)>>>(sm->wgts, totalWgt, RELU_REDU_FAC);
  } else {
    MAT_randomize<<<LAUNCH1D(totalWgt)>>>(sm->wgts, totalWgt, SIG_REDU_FAC);
  }
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

__global__ void GPU_softmax_free(Softmax_T *sm) {
  cudaFree(sm->aTopo);
  cudaFree(sm->wTopo);
  cudaFree(sm->aIdx);
  cudaFree(sm->wIdx);
  cudaFree(sm->deltas);
  cudaFree(sm->activs);
  cudaFree(sm->wgts);
}

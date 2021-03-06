#include "cnn.h"

static inline void CNN_forward(CNN_T *);
static inline void CNN_backward(CNN_T *, size_t);

struct CNN {
  NetConfig_T *cfg;
  Net_T *net;
  double *buf;
};

CNN_T *CNN_init(FILE *cfgFile, Data_T *data) {
  NetConfig_T *cfg = CFG_parse(cfgFile, data);

  CNN_T *cnn = (CNN_T *)malloc(sizeof(CNN_T));
  cnn->cfg = cfg;
  cnn->net = LYR_net_init(cnn->cfg->numMat[FIN], data->hgt, data->wid);
  cudaMalloc((void **)&cnn->buf, cnn->cfg->numMat[FIN] * cnn->cfg->rows[INIT] * cnn->cfg->cols[INIT] * NUM_CHNL * sizeof(double));

  return cnn;
}

void CNN_train(CNN_T *cnn, Data_T *data) {
  size_t dataPxls = data->hgt * data->wid * NUM_CHNL;
  Softmax_T *sm = (Softmax_T *)cnn->cfg->lyrs[cnn->cfg->numLyr - 1];
  double *loss_d;
  cudaMalloc((void **)&loss_d, sizeof(double));

  for (size_t i = 0; i < data->numEpoch; i++) {
    cudaMemset(loss_d, 0.0, sizeof(double));
    for (size_t j = 0; j < data->num; j++) {
      LYR_net_update<<<1, 1>>>(cnn->net, &data->imgs[j * dataPxls], 1, data->hgt, data->wid);
      cudaDeviceSynchronize();

      CNN_forward(cnn);
      CNN_backward(cnn, data->lbls[j]);

      LYR_softmax_loss<<<1, 1>>>(sm, data->lbls[j], loss_d);
      cudaDeviceSynchronize();
    }
    double loss_h = 0.0;
    cudaMemcpy(&loss_h, loss_d, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Average Loss of Epoch %lu: %f\n", i, loss_h / (double)data->num);
  }

  cudaFree(loss_d);
}

void CNN_predict(CNN_T *cnn, double *image, double *output) {
  size_t outBytes = cnn->cfg->numOut * sizeof(double);
  double *out_d;
  cudaMalloc((void **)&out_d, outBytes);

  LYR_net_update<<<1, 1>>>(cnn->net, image, 1, cnn->cfg->rows[INIT], cnn->cfg->cols[INIT]);
  cudaDeviceSynchronize();

  CNN_forward(cnn);

  LYR_softmax_cpyOut<<<1, 1>>>((Softmax_T *)cnn->cfg->lyrs[cnn->cfg->numLyr - 1], out_d);
  cudaDeviceSynchronize();

  cudaMemcpy(output, out_d, outBytes, cudaMemcpyDeviceToHost);
  cudaFree(out_d);
}

void CNN_free(CNN_T *cnn) {
  CFG_free(cnn->cfg);
  LYR_net_free(cnn->net);
  cudaFree(cnn->buf);
  free(cnn);
}

static inline void CNN_forward(CNN_T *cnn) {
  for (size_t i = 0; i < cnn->cfg->numLyr; i++) {
    switch (cnn->cfg->lTypes[i]) {
      case CONVOLUTIONAL: {
        Conv_T *kern = (Conv_T *)cnn->cfg->lyrs[i];

        dim3 grdSize(NUMBLK(cnn->cfg->numMat[FIN], BLKS_2D), NUMBLK(cnn->cfg->numMat[FIN], BLKS_2D));
        dim3 blkSize(BLKS_2D, BLKS_2D);
        LYR_conv_fwd<<<grdSize, blkSize>>>(cnn->net, kern);
        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->cfg->lyrs[i];

        LYR_pool_fwd<<<LAUNCH1D(cnn->cfg->numMat[FIN])>>>(pool, cnn->net, cnn->buf);
        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {
        NonLin_T *nonLinearity = (NonLin_T *)cnn->cfg->lyrs[i];
        LYR_norm_fwd<<<LAUNCH1D(cnn->cfg->numMat[FIN])>>>(cnn->net, *nonLinearity);
        cudaDeviceSynchronize();
        break;
      } case FULLY_CONNECTED: {
        Softmax_T *sm = (Softmax_T *)cnn->cfg->lyrs[i];

        LYR_softmax_fwd<<<1, 1>>>(sm, cnn->net);
        cudaDeviceSynchronize();
        break;
      }
    }
  }
}

static inline void CNN_backward(CNN_T *cnn, size_t lbl) {
  for (size_t i = cnn->cfg->numLyr; i-- > 0;) {
    switch (cnn->cfg->lTypes[i]) {
      case CONVOLUTIONAL: {
        Conv_T *kern = (Conv_T *)cnn->cfg->lyrs[i];

        size_t maxElm = cnn->cfg->numMat[FIN] * cnn->cfg->rows[INIT] * cnn->cfg->cols[INIT] * NUM_CHNL;
        MAT_setVal<<<LAUNCH1D(maxElm)>>>(cnn->buf, maxElm, 0.0);
        cudaDeviceSynchronize();
        dim3 grdSize(NUMBLK(cnn->cfg->numMat[FIN], BLKS_2D), NUMBLK(cnn->cfg->numMat[FIN], BLKS_2D));
        dim3 blkSize(BLKS_2D, BLKS_2D);
        LYR_conv_back<<<grdSize, blkSize>>>(kern, cnn->net, cnn->buf);
        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->cfg->lyrs[i];
        size_t maxElm = cnn->cfg->numMat[FIN] * cnn->cfg->rows[INIT] * cnn->cfg->cols[INIT] * NUM_CHNL;
        MAT_setVal<<<LAUNCH1D(maxElm)>>>(cnn->buf, maxElm, 0.0);

        LYR_pool_back<<<LAUNCH1D(cnn->cfg->numMat[FIN])>>>(pool, cnn->net, cnn->buf);
        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {
        continue;
      } case FULLY_CONNECTED: {
        Softmax_T *sm = (Softmax_T *)cnn->cfg->lyrs[i];

        LYR_softmax_back<<<1, 1>>>(sm, cnn->net, lbl);
        cudaDeviceSynchronize();
        break;
      }
    }
  }
}

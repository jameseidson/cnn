#include "../include/cnn.h"

#define INIT 0
#define FIN 1

struct CNN {
  size_t numLyr;
  size_t numOut;

  Forward_T *fwd;
  Layer_T *ltypes;
  void **lyrs;

  size_t rows[2];
  size_t cols[2];
  size_t numMat[2];

  double *buf;
};

typedef enum NetErr {
  NET_FIRST,
  NET_REDEF,
  UNEXPECTED_KEY,
  INT_EXPECTED,
  FLOAT_EXPECTED,
  LYR_EXPECTED,
  KEY_EXPECTED,
  MISSING_KEY,
  ILLEGAL_VALUE,
  SIZE_FIRST,
  BAD_POOL_DIM,
  FC_LAST,
  NO_ERROR
} NetErr_T;

void forward(CNN_T *cnn);
void backward(CNN_T *cnn, size_t lbl);
static inline void haltErr(CNN_T *, TokenList_T *, NetErr_T, size_t);
void netErr(NetErr_T, size_t);
CNN_T *readNet(TokenList_T **, TokenList_T *);
NetErr_T readLayer(CNN_T *, TokenList **, size_t);
NetErr_T readSoftmax(CNN_T *, TokenList **, size_t, Data_T *);
size_t findMax(size_t *, size_t);

CNN_T *CNN_init(FILE *config, Data_T *data) {
  TokenList_T *head = lex(config);
  TokenList_T *i = head;
  CNN_T *cnn = readNet(&i, head);
  cnn->fwd = NULL;
  cnn->buf = NULL;
  cnn->rows[INIT] = cnn->rows[FIN] = data->hgt;
  cnn->cols[INIT] = cnn->cols[FIN] = data->wid;
  cnn->numMat[INIT] = cnn->numMat[FIN] = 1;

  size_t lyrIdx = 0;
  i = i->next;
  while(i->next) {
    NetErr_T errCode;
    if (i->token == LYR_TYPE) {
      cnn->ltypes[lyrIdx] = i->val.ltype;
      if (i->val.ltype == FULLY_CONNECTED) {
        if (lyrIdx != cnn->numLyr - 1) {
          haltErr(cnn, head, FC_LAST, i->lineNum);
        }
        errCode = readSoftmax(cnn, &i, lyrIdx, data);
        if (errCode != NO_ERROR) {
          haltErr(cnn, head, errCode, i->lineNum);
        }
        break;
      } else {
        errCode = readLayer(cnn, &i, lyrIdx);
      }
      lyrIdx++;
    } else {
      haltErr(cnn, head, LYR_EXPECTED, i->lineNum);
    }
    if (errCode != NO_ERROR) {
      haltErr(cnn, head, errCode, i->lineNum);
    }
  }
  freeTokens(head);

  cnn->fwd = CNN_initFwd(cnn->numMat[FIN], data->hgt, data->wid);
  cudaMalloc((void **)&cnn->buf, cnn->numMat[FIN] * cnn->rows[INIT] * cnn->cols[INIT] * NUM_CHNL * sizeof(double));
  return cnn;
}

void CNN_train(CNN_T *cnn, Data_T *data) {
  size_t dataPxls = data->hgt * data->wid * NUM_CHNL;
  Softmax_T *sm = (Softmax_T *)cnn->lyrs[cnn->numLyr - 1];
  double *loss_d;
  cudaMalloc((void **)&loss_d, sizeof(double));

  for (size_t i = 0; i < data->numEpoch; i++) {
    cudaMemset(loss_d, 0.0, sizeof(double));
    for (size_t j = 0; j < data->num; j++) {
      CNN_prepFwd<<<1, 1>>>(cnn->fwd, &data->imgs[j * dataPxls], 1, data->hgt, data->wid);
      cudaDeviceSynchronize();
      forward(cnn);
      backward(cnn, data->lbls[j]);

      CNN_softmax_loss<<<1, 1>>>(sm, data->lbls[j], loss_d);
      cudaDeviceSynchronize();
    }
    double loss_h = 0.0;
    cudaMemcpy(&loss_h, loss_d, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Average Loss of Epoch %lu: %f\n", i, loss_h / (double)data->num);
  }

  cudaFree(loss_d);
}

void CNN_predict(CNN_T *cnn, double *image, double *output) {
  size_t outBytes = cnn->numOut * sizeof(double);
  double *out_d;
  cudaMalloc((void **)&out_d, outBytes);

  CNN_prepFwd<<<1, 1>>>(cnn->fwd, image, 1, cnn->rows[INIT], cnn->cols[INIT]);

  cudaDeviceSynchronize();
  forward(cnn);

  CNN_softmax_cpyOut<<<1, 1>>>((Softmax_T *)cnn->lyrs[cnn->numLyr - 1], out_d);
  cudaDeviceSynchronize();

  cudaMemcpy(output, out_d, outBytes, cudaMemcpyDeviceToHost);
  cudaFree(out_d);
}

void CNN_free(CNN_T *cnn) {
  for (size_t i = 0; i < cnn->numLyr; i++) {
    if (cnn->lyrs[i]) {
      switch(cnn->ltypes[i]) {
        case CONVOLUTIONAL:
          CNN_freeFtrs((Features_T *)cnn->lyrs[i]);
          break;
        case POOLING:
          CNN_freePool((Pool_T *)cnn->lyrs[i]);
          break;
        case FULLY_CONNECTED:
          CNN_freeSoftmax((Softmax_T *)cnn->lyrs[i]);
          break;
      }
    }
  }

  if (cnn->fwd) {
    CNN_freeFwd(cnn->fwd);
  }
  if (cnn->buf) {
    cudaFree(cnn->buf);
  }

  free(cnn->lyrs);
  free(cnn->ltypes);
  free(cnn);
}

void forward(CNN_T *cnn) {
  for (size_t i = 0; i < cnn->numLyr; i++) {
    switch (cnn->ltypes[i]) {
      case CONVOLUTIONAL: {
        Features_T *kern = (Features_T *)cnn->lyrs[i];

        dim3 grdSize(NUMBLK(cnn->numMat[FIN], BLKS_2D), NUMBLK(cnn->numMat[FIN], BLKS_2D));
        dim3 blkSize(BLKS_2D, BLKS_2D);
        CNN_convolve<<<grdSize, blkSize>>>(cnn->fwd, kern, cnn->buf);
        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->lyrs[i];

        CNN_pool<<<NUMBLK(cnn->numMat[FIN], BLKS_1D), BLKS_1D>>>(pool, cnn->fwd, cnn->buf);
        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {
        /* TODO: add frontend support for multiple nonlinearities */

        CNN_normalize<<<NUMBLK(cnn->numMat[FIN], BLKS_1D), BLKS_1D>>>(cnn->fwd, RELU);
        cudaDeviceSynchronize();
        break;
      } case FULLY_CONNECTED: {
        Softmax_T *sm = (Softmax_T *)cnn->lyrs[i];

        CNN_softmax_fwd<<<1, 1>>>(sm, cnn->fwd);
        cudaDeviceSynchronize();
        break;
      }
    }
  }
}

void backward(CNN_T *cnn, size_t lbl) {
  for (size_t i = cnn->numLyr; i-- > 0;) {
    switch (cnn->ltypes[i]) {
      case CONVOLUTIONAL: {
        Features_T *kern = (Features_T *)cnn->lyrs[i];

        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->lyrs[i];

        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {

        cudaDeviceSynchronize();
        break;
      } case FULLY_CONNECTED: {
        Softmax_T *sm = (Softmax_T *)cnn->lyrs[i];

        CNN_softmax_back<<<1, 1>>>(sm, lbl);
        cudaDeviceSynchronize();
        break;
      }
    }
  }
}

static inline void haltErr(CNN_T *cnn, TokenList_T *head, NetErr_T errType, size_t lineNum) {
  freeTokens(head);
  CNN_free(cnn);
  netErr(errType, lineNum);
}

void netErr(NetErr_T errType, size_t lineNum) {
  fprintf(stderr, "Unable to construct network from config file (%lu): ", lineNum);
  switch (errType) {
    case NET_FIRST:
      printf("the first layer must be type net\n");
      break;
    case NET_REDEF:
      printf("the net layer can only be defined once\n");
      break;
    case UNEXPECTED_KEY:
      printf("configuration option is not a valid member of its parent layer\n");
      break;
    case INT_EXPECTED:
      printf("expected an integer\n");
      break;
    case FLOAT_EXPECTED:
      printf("expected a floating point value\n");
      break;
    case LYR_EXPECTED:
      printf("expected a layer type\n");
      break;
    case KEY_EXPECTED:
      printf("expected a configuration option\n");
      break;
    case MISSING_KEY:
      printf("required configuration option for layer not set\n");
      break;
    case ILLEGAL_VALUE:
      printf("illegal value for configuration option\n");
      break;
    case SIZE_FIRST:
      printf("numHidden must be set before hidden array is configured\n");
      break;
    case BAD_POOL_DIM:
      printf("pooling layers must satisfy ((inputDim - windowDim) %% stride == 0) for x and y dimensions\n");
      break;
    case FC_LAST:
      printf("the fully connected layer must be the last layer in the network\n");
      break;
    case NO_ERROR:
      printf("task failed successfully (yeah I'm not sure what happened here either)\n");
      break;
  }

  exit(EXIT_FAILURE);
}

CNN_T *readNet(TokenList_T **ip, TokenList_T *head) {
  TokenList_T *i = *ip;
  if (!i || i->token != LYR_TYPE || i->val.ltype != NET) {
    freeTokens(head);
    netErr(NET_FIRST, i->lineNum);
  }
  i = i->next;

  if (!i || i->token != KEY || i->val.key != NUM_LYR) {
    freeTokens(head);
    netErr(UNEXPECTED_KEY, i->lineNum);
  }
  i = i->next;

  if (!i || i->token != INT_VAL) {
    freeTokens(head);
    netErr(INT_EXPECTED, i->lineNum);
  }
  CNN_T* cnn = (CNN_T *)malloc(sizeof(CNN_T));
  assert(cnn);
  cnn->numLyr = i->val.ival;
  cnn->ltypes = (Layer_T *)malloc(cnn->numLyr * sizeof(Layer_T));
  cnn->lyrs = (void **)malloc(cnn->numLyr * sizeof(void *));
  for (size_t j = 0; j < cnn->numLyr; j++) {
    cnn->ltypes[j] = INVALID_LYR;
    cnn->lyrs[j] = NULL;
  }

  *ip = i;
  return cnn;
}

NetErr_T readLayer(CNN_T *cnn, TokenList_T **ip, size_t lyrIdx) {
  TokenList_T *i = *ip;
  switch(i->val.ltype) {
    case NET: {
      return NET_REDEF;
      break;
    } case CONVOLUTIONAL: {
      size_t numFeat = 0, featHgt = 0, featWid = 0;
      for (i = i->next; i && i->token == KEY; i = i->next) {
        switch (i->val.key) {
          case NUM_FEAT: 
            i = i->next; 
            if (!i || i->token != INT_VAL) {
              *ip = i;
              return INT_EXPECTED;
            }
            numFeat = i->val.ival;
            break;
          case FEAT_HGT: 
            i = i->next; 
            if (!i || i->token != INT_VAL) {
              *ip = i;
              return INT_EXPECTED;
            }
            featHgt = i->val.ival;
            break;
          case FEAT_WID: 
            i = i->next; 
            if (!i || i->token != INT_VAL) {
              *ip = i;
              return INT_EXPECTED;
            }
            featWid = i->val.ival;
            break;
          default:
            *ip = i;
            return UNEXPECTED_KEY;
            break;
        }
      }
      if (numFeat == 0 || featHgt == 0 || featWid == 0) {
        return MISSING_KEY;
      }

      cnn->lyrs[lyrIdx] = CNN_initFtrs(numFeat, featHgt, featWid);

      cnn->numMat[FIN] = cnn->numMat[FIN] * numFeat;
      cnn->rows[FIN] = (cnn->rows[FIN] - featHgt) + 1;
      cnn->cols[FIN] = (cnn->cols[FIN] - featWid) + 1;
      break;
    } case POOLING: {
      size_t winDim = 0, stride = 0;
      for (i = i->next; i->token == KEY; i = i->next) {
        switch (i->val.key) {
          case WIN_DIM:
            i = i->next; 
            if (!i || i->token != INT_VAL) {
              *ip = i;
              return INT_EXPECTED;
            }
            winDim = i->val.ival;
            break;
          case STRIDE: 
            i = i->next; 
            if (!i || i->token != INT_VAL) {
              *ip = i;
              return INT_EXPECTED;
            }
            stride = i->val.ival;
            break;
          default:
            *ip = i;
            return UNEXPECTED_KEY;
            break;
        }
      }

      if (winDim == 0 || stride == 0) {
        return MISSING_KEY;
      }
      if (((cnn->rows[FIN] - winDim) % stride) != 0 || ((cnn->cols[FIN] - winDim) % stride) != 0) {
        return BAD_POOL_DIM;
      }

      cnn->lyrs[lyrIdx] = CNN_initPool(winDim, stride);

      cnn->rows[FIN] = ((cnn->rows[FIN] - winDim) / stride) + 1;
      cnn->cols[FIN] = ((cnn->cols[FIN] - winDim) / stride) + 1;
      break;
    } case NORMALIZATION: {
      i = i->next;
      cnn->lyrs[lyrIdx] = NULL;
      break;
    }
  }

  *ip = i;
  return NO_ERROR;
}

NetErr_T readSoftmax(CNN_T *cnn, TokenList_T **ip, size_t lyrIdx, Data_T *data) {
  TokenList_T *i = *ip;
  double lrnRate = 0.0;
  size_t numHidden = 0;
  size_t numOut = 0;
  size_t *hiddens = NULL;
  bool allocHiddens = false;
  for (i = i->next; i && i->token == KEY; i = i->next) {
    switch (i->val.key) {
      case LRN_RATE:
        i = i->next; 
        if (!i || i->token != FLOAT_VAL) {
          if (allocHiddens) {
            free(hiddens);
          }
          *ip = i;
          return FLOAT_EXPECTED;
        }
        lrnRate = i->val.fval;
        break;
      case NUM_HIDDEN: 
        i = i->next; 
        if (!i || i->token != INT_VAL) {
          *ip = i;
          return INT_EXPECTED;
        }
        numHidden = i->val.ival;
        if (numHidden == 0) {
          return ILLEGAL_VALUE;
        }
        break;
      case HIDDENS: 
        if (numHidden == 0) {
          return SIZE_FIRST;
        }

        hiddens = (size_t *)malloc(numHidden * sizeof(size_t));
        allocHiddens = true;
        assert(hiddens);
        i = i->next;
        for (int j = 0; j < numHidden; j++) {
          if (!i || i->token != INT_VAL) {
            free(hiddens);
            *ip = i;
            return INT_EXPECTED;
          }
          hiddens[j] = i->val.ival;
          if (j != numHidden - 1) {
            i = i->next;
          }
        }
        break;
      case NUM_OUTPUT: 
        i = i->next; 
        if (!i || i->token != INT_VAL) {
          if (allocHiddens) {
            free(hiddens);
          }
          *ip = i;
          return INT_EXPECTED;
        }
        numOut = i->val.ival;
        break;
      default:
        if (allocHiddens) {
          free(hiddens);
        }
        *ip = i;
        return UNEXPECTED_KEY;
        break;
    }
  }
  if (!hiddens || lrnRate == 0.0 || numOut == 0) {
    return MISSING_KEY;
  }

  size_t numLyr = numHidden + 2;
  size_t *topo = (size_t *)malloc(numLyr * sizeof(size_t));
  assert(topo);


  for (size_t j = 0; j < numHidden; j++) {
    topo[j + 1] = hiddens[j];
  }
  free(hiddens);

  topo[0] = cnn->numMat[FIN] * cnn->rows[FIN] * cnn->cols[FIN] * NUM_CHNL;
  topo[numLyr - 1] = numOut;
  /* TODO: add frontend support for multiple nonlinearities */
  cnn->lyrs[lyrIdx] = CNN_initSoftmax(topo, numLyr, lrnRate, SIG);
  cnn->numOut = numOut;
  free(topo);

  *ip = i;
  return NO_ERROR;
}

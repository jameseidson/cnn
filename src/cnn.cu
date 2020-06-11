#include "../include/cnn.h"

#define INIT 0
#define FIN 1

struct CNN {
  size_t hgt[2];
  size_t wid[2];
  size_t numImg[2];
  size_t numOut;

  size_t maxPxls;
  dim3 grdSize;
  dim3 blkSize;

  double *imgBuf;
  double *outBuf;
  Net_T *net;

  size_t numLyr;
  LayerT_T *ltypes;
  void **lyrs;
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
NetErr_T readClsfier(CNN_T *, TokenList **, size_t, Data_T *);

CNN_T *CNN_init(FILE *config, Data_T *data) {
  TokenList_T *head = lex(config);
  TokenList_T *i = head;
  CNN_T *cnn = readNet(&i, head);
  cnn->net = NULL;
  cnn->imgBuf = NULL;
  cnn->hgt[INIT] = cnn->hgt[FIN] = data->hgt;
  cnn->wid[INIT] = cnn->wid[FIN] = data->wid;
  cnn->numImg[INIT] = cnn->numImg[FIN] = 1;

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
        errCode = readClsfier(cnn, &i, lyrIdx, data);
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

  cnn->net = CNN_initNet(cnn->numImg[FIN], data->hgt, data->wid);

  cnn->maxPxls = cnn->numImg[FIN] * cnn->hgt[INIT] * cnn->wid[INIT] * NUM_CHNL;
  cudaMalloc((void **)&cnn->imgBuf, cnn->maxPxls * sizeof(double));

  dim3 blkSize(BLKS_3D, BLKS_3D, BLKS_3D);
  dim3 grdSize(NUMBLK(cnn->numImg[FIN], BLKS_3D), NUMBLK(cnn->hgt[INIT], BLKS_3D), NUMBLK(cnn->wid[INIT], BLKS_3D));
  cnn->blkSize = blkSize;
  cnn->grdSize = grdSize;
  return cnn;
}

void CNN_classify(CNN_T *cnn, double *image, double *out) {
  Classify_T *cls = (Classify_T *)cnn->lyrs[cnn->numLyr - 1];
  size_t netPxls = cnn->numImg[INIT] * cnn->hgt[INIT] * cnn->wid[INIT] * NUM_CHNL;
  size_t outBytes = cnn->numOut * sizeof(double);
  double *outBuf;
  cudaMalloc((void **)&outBuf, outBytes);

  CNN_prepNet<<<NUMBLK(netPxls, BLKS_1D), BLKS_1D>>>(cnn->net, image, cnn->numImg[INIT], cnn->hgt[INIT], cnn->wid[INIT]);
  cudaDeviceSynchronize();

  forward(cnn);

  CNN_getOut<<<NUMBLK(cnn->numOut, BLKS_1D), BLKS_1D>>>(cls, outBuf);
  cudaMemcpy(out, outBuf, outBytes, cudaMemcpyDeviceToHost);
}


void CNN_train(CNN_T *cnn, Data_T *data) {
  size_t netPxls = cnn->numImg[INIT] * cnn->hgt[INIT] * cnn->wid[INIT] * NUM_CHNL;
  size_t dataPxls = data->hgt * data->wid * NUM_CHNL;
  Classify_T *cls = (Classify_T *)cnn->lyrs[cnn->numLyr - 1];
  double *err_d;
  cudaMalloc((void **)&err_d, sizeof(double));

  for (size_t i = 0; i < data->numEpoch; i++) {
    double errSum = 0;
    for (size_t j = 0; j < data->num; j++) {
      CNN_prepNet<<<NUMBLK(netPxls, BLKS_1D), BLKS_1D>>>(cnn->net, &data->imgs[j * dataPxls], 1, data->hgt, data->wid);
      cudaDeviceSynchronize();
      forward(cnn);
      backward(cnn, data->lbls[j]);

      double err_h = 0.0f;
      CNN_getErr<<<NUMBLK(cnn->numOut, BLKS_1D), BLKS_1D>>>(cls, data->lbls[j], err_d);
      cudaMemcpy(&err_h, err_d, sizeof(double), cudaMemcpyDeviceToHost);
      errSum += err_h;
      cudaDeviceSynchronize();
    }
    printf("Average Error for Epoch %lu: %f\n", i, errSum / (double)data->num);
  }

  cudaFree(err_d);
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
          CNN_freeClsfier((Classify_T *)cnn->lyrs[i]);
          break;
      }
    }
  }

  if (cnn->net) {
    CNN_freeNet(cnn->net);
  }
  if (cnn->imgBuf) {
    cudaFree(cnn->imgBuf);
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

        CNN_convolve<<<cnn->grdSize, cnn->blkSize>>>(cnn->net, kern, cnn->imgBuf);
        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->lyrs[i];

        CNN_pool<<<cnn->grdSize, cnn->blkSize>>>(cnn->net, pool, cnn->imgBuf);
        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {

        CNN_normalize<<<NUMBLK(cnn->maxPxls, BLKS_1D), BLKS_1D>>>(cnn->net);
        cudaDeviceSynchronize();
        break;
      } case FULLY_CONNECTED: {
        Classify_T *cls = (Classify_T *)cnn->lyrs[i];

        CNN_softmax_fwd<<<1, 1>>>(cls);
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
        /* TODO */

        cudaDeviceSynchronize();
        break;
      } case POOLING: {
        Pool_T *pool = (Pool_T *)cnn->lyrs[i];
        /* TODO */

        cudaDeviceSynchronize();
        break;
      } case NORMALIZATION: {
        /* TODO */

        cudaDeviceSynchronize();
        break;
      } case FULLY_CONNECTED: {
        Classify_T *cls = (Classify_T *)cnn->lyrs[i];

        CNN_softmax_bck<<<1, 1>>>(cls, lbl);
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
  cnn->ltypes = (LayerT_T *)malloc(cnn->numLyr * sizeof(LayerT_T));
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

      cnn->numImg[FIN] = cnn->numImg[FIN] * numFeat;
      cnn->hgt[FIN] = (cnn->hgt[FIN] - featHgt) + 1;
      cnn->wid[FIN] = (cnn->wid[FIN] - featWid) + 1;
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
      if (((cnn->hgt[FIN] - winDim) % stride) != 0 || ((cnn->wid[FIN] - winDim) % stride) != 0) {
        return BAD_POOL_DIM;
      }

      cnn->lyrs[lyrIdx] = CNN_initPool(winDim, stride);

      cnn->hgt[FIN] = ((cnn->hgt[FIN] - winDim) / stride) + 1;
      cnn->wid[FIN] = ((cnn->wid[FIN] - winDim) / stride) + 1;
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

NetErr_T readClsfier(CNN_T *cnn, TokenList_T **ip, size_t lyrIdx, Data_T *data) {
  TokenList_T *i = *ip;
  double lrnRate = 0.0f;
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
  if (!hiddens || lrnRate == 0.0f || numOut == 0) {
    return MISSING_KEY;
  }

  size_t numLyr = numHidden + 2;
  size_t *topo = (size_t *)malloc(numLyr * sizeof(size_t));
  assert(topo);

  for (size_t j = 0; j < numHidden; j++) {
    topo[j + 1] = hiddens[j];
  }
  free(hiddens);

  topo[0] = cnn->numImg[FIN] * cnn->hgt[FIN] * cnn->wid[FIN] * NUM_CHNL;
  topo[numLyr - 1] = numOut;
  cnn->lyrs[lyrIdx] = CNN_initClsfier(topo, numLyr, lrnRate);
  cnn->numOut = numOut;
  free(topo);

  *ip = i;
  return NO_ERROR;
}

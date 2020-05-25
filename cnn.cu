#include "cnn.h"

struct CNN {
  size_t numLyr;
  size_t clsHgt;
  size_t clsWid;
  size_t clsFeat;
  Net_T *net;
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

static inline void haltErr(CNN_T *, TokenList_T *, NetErr_T, size_t);
void netErr(NetErr_T, size_t);
CNN_T *readNet(TokenList_T **, TokenList_T *);
NetErr_T readLayer(CNN_T *, TokenList **, size_t);
NetErr_T readClsfier(CNN_T *, TokenList **, size_t, Data_T *);

CNN_T *CNN_init(FILE *config, Data_T *data) {
  TokenList_T *head = lex(config);
  TokenList_T *i = head;
  CNN_T *cnn = readNet(&i, head);
  cnn->clsHgt = data->hgt;
  cnn->clsWid = data->wid;

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

  cnn->net = NULL;
  return cnn;
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
  free(cnn->lyrs);
  free(cnn->ltypes);
  free(cnn);
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
      printf("expected a floating point valye\n");
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
  TokenList *i = *ip;
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

NetErr_T readLayer(CNN_T *cnn, TokenList **ip, size_t lyrIdx) {
  TokenList *i = *ip;
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

      cnn->clsFeat = numFeat;
      cnn->clsHgt = (cnn->clsHgt - featHgt) + 1;
      cnn->clsWid = (cnn->clsWid - featWid) + 1;
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
      if (((cnn->clsHgt - winDim) % stride) != 0 || ((cnn->clsWid - winDim) % stride) != 0) {
        return BAD_POOL_DIM;
      }

      cnn->lyrs[lyrIdx] = CNN_initPool(winDim, stride);

      cnn->clsHgt = ((cnn->clsHgt - winDim) / stride) + 1;
      cnn->clsWid = ((cnn->clsWid - winDim) / stride) + 1;
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

NetErr_T readClsfier(CNN_T *cnn, TokenList **ip, size_t lyrIdx, Data_T *data) {
  TokenList *i = *ip;
  double lrnRate = 0.0f;
  size_t numHidden = 0;
  size_t numOut = 0.0;
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
  if (hiddens == NULL || lrnRate == 0.0f || numOut == 0) {
    return MISSING_KEY;
  }

  size_t numLyr = numHidden + 2;
  size_t *topo = (size_t *)malloc(numLyr * sizeof(size_t));
  assert(topo);

  for (size_t j = 0; j < numHidden; j++) {
    topo[j + 1] = hiddens[j];
  }
  free(hiddens);


  topo[0] = cnn->clsFeat * cnn->clsHgt * cnn->clsWid * NUM_CHNL;
  topo[numLyr - 1] = numOut;
  cnn->lyrs[lyrIdx] = CNN_initClsfier(topo, numLyr, lrnRate);

  *ip = i;
  return NO_ERROR;
}

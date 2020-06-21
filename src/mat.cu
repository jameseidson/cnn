#include "mat.h"

#define SIG(x) (1.0/(1.0 + exp(-x)))
#define DSIG(x) (x * (1 - x))

#define RELU(x) ((x > 0) ? x : 0)
#define DRELU(x) ((double)(x > 0))

#define SE(obs, tru) (0.5 * (tru - obs) * (tru - obs))
#define DSE(obs, tru) (obs - tru)

#define LCG(seed) ((1103515245 * seed + 12345) % INT_MAX)

__global__ void MAT_randomize(double *m, size_t numElm, size_t reduFac) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < numElm) {
    m[x] = (double)(LCG(x) / (double)INT_MAX / reduFac);
  }
}

__global__ void MAT_setVal(double *m, size_t numElm, double val) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < numElm) {
    m[x] = val;
  }
}

__global__ void MAT_assign(double *src, double *dst, size_t numElm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  if (x < numElm) {
    dst[x] = src[x];
  }
}

__global__ void MAT_ewMul(double *mA, double *mB, double *mC, size_t numElm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < numElm) {
    mC[x] = mA[x] * mB[x];
  }
}

__global__ void MAT_mvMul(double *mA, double *vB, double *vC, size_t aRows, size_t aCols) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (r < aRows) {
    double dotProd = 0.0;
    for (size_t i = 0; i < aCols; i++) {
      dotProd += mA[FLAT2D(r, i, aCols)] * vB[i];
    }
    vC[r] = dotProd;
  }
}

__global__ void MAT_ReLU(double *mA, double *mB, size_t numElm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < numElm) {
    mB[x] = RELU(mA[x]);
  }
}

__global__ void MAT_sigmoid(double *mA, double *mB, size_t numElm) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < numElm) {
    mB[x] = SIG(mA[x]);
  }
}

__global__ void MAT_loss(double *m, size_t numElm, size_t lbl, double *loss) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < numElm) {
    *loss += SE(m[x], (double)(x == lbl));
  }
}

__global__ void MAT_fwdProp(double *mWgt, double *vAct, double *vNxtAct, size_t wRows, size_t wCols, NonLin_T fType) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (r < wRows) {
    double dotProd = 0.0;
    for (size_t i = 0; i < wCols; i++) {
      dotProd += mWgt[FLAT2D(r, i, wCols)] * vAct[i];
    }
    vNxtAct[r] = (fType == RELU) ? RELU(dotProd) : SIG(dotProd);
  }
}

__global__ void MAT_deltas_out(double *vOut, double *vDelt, size_t numElm, size_t lbl, NonLin_T fType) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < numElm) {
    vDelt[x] = DSE(vOut[x], (double)(x == lbl)) * ((fType == RELU) ? DRELU(vOut[x]) : DSIG(vOut[x]));
    __syncthreads();
  }
}

__global__ void MAT_deltas_hidden(double *vAct, double *vDelt, double *mWgt, double *vNxtDelt, size_t wRows, size_t wCols, NonLin_T fType) {
  size_t c = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (c < wCols) {
    double dotProd = 0.0;
    for (size_t i = 0; i < wRows; i++) {
      dotProd += mWgt[FLAT2D(c, i, wRows)] * vNxtDelt[i];
    
    }
    vDelt[c] = dotProd * ((fType == RELU) ? DRELU(vAct[c]) : DSIG(vAct[c]));
  }
}

__global__ void MAT_applyGradient(double *vAct, double *vNxtDelt, double *mWgt, size_t wRows, size_t wCols, double scaleFac) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t c = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  if (r < wRows) {
    if (c < wCols) {
      mWgt[FLAT2D(r, c, wCols)] -= scaleFac * vAct[c] * vNxtDelt[r];
    }
  }
}

__global__ void MAT_convolve(double *mA, double *mB, double *mFltr, size_t aRows, size_t aCols, size_t fRows, size_t fCols, bool applyGrad) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t c = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  size_t fElm = fRows * fCols * NUM_CHNL;
  size_t bRows = CONV_OUT(aRows, fRows);
  size_t bCols = CONV_OUT(aCols, fCols);

  if (r < bRows && c < bCols) {
    double chnlTotal[NUM_CHNL] = { 0.0, 0.0, 0.0 };
    for (size_t i = r; i - r < fRows; i++) {
      for (size_t j = c; j - c < fCols; j++) {
        for (size_t k = 0; k < NUM_CHNL; k++) {
          chnlTotal[k] += mA[FLAT3D(i, j, k, aCols, NUM_CHNL)] * mFltr[FLAT3D((i - r), (j - c), k, fCols, NUM_CHNL)];
        }
      }
    }
    for (size_t i = 0; i < NUM_CHNL; i++) {
      if (applyGrad) {
        mB[FLAT3D(r, c, i, bCols, NUM_CHNL)] -= (chnlTotal[i] / fElm);
      } else {
        mB[FLAT3D(r, c, i, bCols, NUM_CHNL)] = chnlTotal[i] / fElm;
      }
    }
  }
}

__global__ void MAT_inv_convolve(double *mA, double *mB, double *mFltr, size_t aRows, size_t aCols, size_t fRows, size_t fCols, bool updateDeltas) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t c = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  size_t fMaxChnlIdx = (fRows * fCols) - 1;
  size_t fElm = fRows * fCols * NUM_CHNL;
  size_t bRows = aRows + fRows - 1;
  size_t bCols = aRows + fRows - 1;
  size_t rPad = fRows - 1;
  size_t cPad = fCols - 1;

  if (r < bRows && c < bCols) {
    r -= rPad;
    c -= cPad;
    double chnlTotal[NUM_CHNL] = { 0.0, 0.0, 0.0 };
    for (size_t i = r; i - r < fRows; i++) {
      for (size_t j = c; j - c < fCols; j++) {
        if (i > 0 && i < aRows && j > 0 && j < aCols) {
          for (size_t k = 0; k < NUM_CHNL; k++) {
            chnlTotal[k] += mA[FLAT3D(i, j, k, aCols, NUM_CHNL)] * mFltr[(fMaxChnlIdx - FLAT2D((i - r), (j - c), fCols)) * NUM_CHNL + k];
          }
        }
      }
    }
    for (size_t i = 0; i < NUM_CHNL; i++) {
      if (updateDeltas) {
        mB[FLAT3D((r + rPad), (c + cPad), i, bCols, NUM_CHNL)] += chnlTotal[i] / fElm;
      } else {
        mB[FLAT3D((r + rPad), (c + cPad), i, bCols, NUM_CHNL)] = chnlTotal[i] / fElm;
      }
    }
  }
}

__global__ void MAT_pool(double *mA, double *mB, size_t *idxs, size_t aRows, size_t aCols, size_t wDim, size_t stride) {
  size_t r = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);
  size_t c = FLAT2D(blockIdx.y, threadIdx.y, blockDim.y);

  size_t bRows = POOL_OUT(aRows, wDim, stride);
  size_t bCols = POOL_OUT(aCols, wDim, stride);

  if (r < bRows && c < bCols) {
    double chnlMax[NUM_CHNL] = { DBL_MIN, DBL_MIN, DBL_MIN };
    size_t maxIdx[NUM_CHNL];

    for (size_t i = 0; i < wDim; i++) {
      for (size_t j = 0; j < wDim; j++) {
        for (size_t k = 0; k < NUM_CHNL; k++) {
          size_t idx = FLAT3D(FLAT2D(r, i, stride), FLAT2D(c, j, stride), k, bCols, NUM_CHNL);
          double curElm = mA[idx];
          if (curElm > chnlMax[k]) {
            chnlMax[k] = curElm;
            maxIdx[k] = idx;
          }
        }
      }
    }
    for (size_t i = 0; i < NUM_CHNL; i++) {
      size_t idx = FLAT3D(r, c, i, bCols, NUM_CHNL);
      mB[idx] = chnlMax[i];
      idxs[idx] = maxIdx[i];
    }
  }
}

__global__ void MAT_deltas_pool(double *mA, double *mB, size_t *idxs, size_t aRows, size_t aCols) {
  size_t x = FLAT2D(blockIdx.x, threadIdx.x, blockDim.x);

  if (x < aRows * aCols * NUM_CHNL) {
    mB[idxs[x]] = mA[x];
  }
}

__global__ void MAT_print(double *m, size_t rows, size_t cols, bool is3D) {
  if (is3D) {
    for (uint8_t k = 0; k < NUM_CHNL; k++) {
      printf("Channel %u:\n", k);
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
          printf("%1.1f ", m[FLAT3D(i, j, k, cols, NUM_CHNL)]);
        }
        printf("\n");
      }
    }
    printf("\n");
  } else if (cols == 1) {
    for (size_t i = 0; i < rows; i++) {
      printf("%1.1f", m[i]);
      printf("\n");
    }
  } else {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        printf("%1.1f ", m[FLAT2D(i, j, cols)]);
      }
      printf("\n");
    }
  }
}

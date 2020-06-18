#ifndef CONFIG_H
#define CONFIG_H

#include "def.h"
#include "net.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <stdbool.h>

static const unsigned INIT = 0;
static const unsigned FIN = 1;

typedef enum LayerT {
  NET,
  CONVOLUTIONAL,
  POOLING,
  NORMALIZATION,
  FULLY_CONNECTED
} LayerT_T;

typedef struct NetConfig {
  size_t numLyr;
  size_t numOut;

  size_t rows[2];
  size_t cols[2];
  size_t numMat[2];

  void **lyrs;
  LayerT_T *lTypes;
} NetConfig_T;

NetConfig_T *CFG_parse(FILE *cfgFile, Data_T *data);
void CFG_free(NetConfig_T *cfg);

#endif 

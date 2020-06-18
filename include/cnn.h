#ifndef CNN_H
#define CNN_H

#include "config.h"
#include "mat.h"
#include "def.h"
#include "net.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef struct CNN CNN_T;

CNN_T *CNN_init(FILE *config, Data_T *data);
void CNN_train(CNN_T *cnn, Data_T *data);
void CNN_predict(CNN_T *cnn, double *image, double *output);
void CNN_free(CNN_T *cnn);

#endif

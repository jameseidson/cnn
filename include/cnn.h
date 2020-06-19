#ifndef CNN_H
#define CNN_H

#include "data.h"
#include "config.h"
#include "layer.h"
#include "mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct CNN CNN_T;

CNN_T *CNN_init(FILE *config, Data_T *data);
void CNN_train(CNN_T *cnn, Data_T *data);
void CNN_predict(CNN_T *cnn, double *image, double *output);
void CNN_free(CNN_T *cnn);

#endif

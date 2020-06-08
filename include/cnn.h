#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "def.h"
#include "net.h"
#include "lex.h"

typedef struct CNN CNN_T;

CNN_T *CNN_init(FILE *config, Data_T *data);
double *CNN_feed(CNN_T *cnn, double *image);
void CNN_train(CNN_T *cnn, Data_T *data);
void CNN_free(CNN_T *cnn);

#endif

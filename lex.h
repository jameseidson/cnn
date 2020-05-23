#ifndef LEX_H
#define LEX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

const uint8_t NUM_LYR_TYPE = 5;
typedef enum {
  NET,
  CONVOLUTIONAL,
  POOLING,
  NORMALIZATION,
  FULLY_CONNECTED,
  INVALID_LYR
} Layer_T;

const uint8_t NUM_KEY = 10;
typedef enum {
  NUM_EPOCH,
  NUM_FEAT,
  FEAT_HGT,
  FEAT_WID,
  WIN_SIZE,
  STRIDE,
  LRN_RATE,
  NUM_HIDDEN,
  HIDDENS,
  NUM_OUTPUT,
  INVALID_KEY
} Key_T;

typedef enum {
  BAD_TOKEN,
  UNEXPECTED_EOF,
  BAD_LYR_TYPE,
  BAD_KEY,
  NUM_OVERFLOW
} Error_T;

typedef enum {
  LYR_TYPE,
  KEY,
  INT_VAL,
  FLOAT_VAL,
  VAL_LIST
} Token_T;

typedef union {
  size_t ival;
  double fval;
  Layer_T ltype;
  Key_T key;
} TokenVal_T;

typedef struct TokenList {
  Token_T token;
  TokenVal_T val;
  struct TokenList *next;
} TokenList_T;

TokenList_T *lex(FILE *f);
void freeTokens(TokenList_T *tokens);

#endif

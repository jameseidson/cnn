#include "config.h"

static const unsigned MAX_LYR_SIZE = 15;
static const unsigned MAX_KEY_SIZE = 12;
static const unsigned MAX_NUM_SIZE = 12;
static const unsigned MAX_STR_SIZE = 7;
static const unsigned NUM_LYR_TYPE = 5;

static const char OP_CHAR[] = { '=', '{', ',' , '}' };
static const char COMMENT_START = '#';

static const char *LYR_STR[] = { "net", "convolutional", "pooling", "normalization", "fully connected" };
static const unsigned NUM_KEY = 11;

static const char *KEY_STR[] = { "numLayer", "numFeature", "featureHgt", "featureWid", "windowDim", "stride", 
                                 "nonlinearity", "learnRate", "hiddens", "numOutput" };

static const unsigned NUM_NLIN = 2;
static const char *NLIN_STR[] = { "relu", "sigmoid" };

static const int INVALID = -1;

#define ADVANCE(p) (p->t = p->t->next)

typedef enum LexErr {
  UNEXP_TOK,
  BAD_LYR_TYPE,
  BAD_KEY,
  NUM_OVERFLOW,
  EMPTY_NUM,
  BAD_NLIN
} LexErr_T;

typedef enum ParseErr {
  NET_FIRST,
  SFTMX_LAST,
  NET_REDEF,
  EXTRA_LYR,
  INSUF_LYR,
  BAD_POOL_CFG,
  BAD_CONFIG,
  UNSET_CONFIG,
  EXP_KEY,
  EXP_VAL_INT,
  EXP_VAL_FLOAT,
  EXP_VAL_NLIN,
  EXP_VAL_LIST,
  EXP_OP_ASSIGN,
  EXP_OP_LDELIM,
  EXP_OP_LCLOSE,
  UNEXP_EOF
} ParseErr_T;

typedef enum Operator {
  ASSIGNMENT,
  LIST_OPEN,
  LIST_DELIMETER,
  LIST_CLOSE
} Operator_T;

typedef enum Key {
  NUM_LYR,
  NUM_FEAT,
  FEAT_HGT,
  FEAT_WID,
  WIN_DIM,
  STRIDE,
  NONLINEARITY,
  LRN_RATE,
  HIDDENS,
  NUM_OUTPUT,
} Key_T;

typedef enum Token {
  OPERATOR,
  LYR_TYPE,
  KEY,
  INT_VAL,
  FLOAT_VAL,
  NLIN
} Token_T;

typedef union TokenVal {
  Operator_T opCode;
  size_t iVal;
  double fVal;
  NonLin_T nLin;
  LayerT_T lType;
  Key_T key;
} TokenVal_T;

typedef struct TokenList {
  Token_T token;
  TokenVal_T val;
  size_t lineNum;
  struct TokenList *next;
} TokenList_T;

typedef struct Parser {
  TokenList_T *t;
  TokenList_T *head;
  NetConfig_T *cfg;
  size_t *listBuf;
  size_t lyrIdx;
} Parser_T;

typedef struct Lexer {
  FILE *f;
  size_t lineNum;
  TokenList_T *tokens;
} Lexer_T;

void CFG_parse_throw(Parser_T *, ParseErr_T, bool);
void CFG_parse_netLyr(Parser_T *);
void CFG_parse_convLyr(Parser_T *);
void CFG_parse_poolLyr(Parser_T *);
void CFG_parse_normLyr(Parser_T *);
void CFG_parse_sftmxLyr(Parser_T *);
static inline size_t CFG_parse_intAssign(Parser_T *);
static inline double CFG_parse_floatAssign(Parser_T *);
static inline NonLin_T CFG_parse_nLinAssign(Parser_T *);
static inline size_t CFG_parse_listAssign(Parser_T *);
void CFG_parse_free(Parser_T *);

TokenList_T *CFG_lex(FILE *);
void CFG_lex_getLyr(Lexer_T *);
void CFG_lex_getKey(Lexer_T *);
void CFG_lex_getVal_num(Lexer_T *);
void CFG_lex_getVal_list(Lexer_T *);
void CFG_lex_getVal_str(Lexer_T *);
void CFG_lex_throw(Lexer_T *, LexErr_T);
void CFG_lex_appendTok(TokenList_T **, Token_T, TokenVal_T, size_t);
static inline void CFG_lex_advUntil(Lexer_T *, int (*)(int), int (*)(int), char *, unsigned, LexErr_T);
static inline int CFG_lex_nxtValid(Lexer_T *);
static inline void CFG_lex_appendOp(Lexer_T *, Operator_T);
static inline int CFG_lex_isEnd_lyr(int);
static inline int CFG_lex_isValid_lyr(int);
static inline int CFG_lex_isEnd_key(int);
static inline int CFG_lex_isEnd_num(int);
static inline int CFG_lex_isValid_num(int);
static inline int CFG_lex_isEnd_str(int);
static inline int CFG_lex_strToEnum(char *, const char **, unsigned);
static inline int CFG_lex_peek(FILE *);
void CFG_lex_freeTok(TokenList_T *);
void CFG_lex_free(Lexer_T *);

NetConfig_T *CFG_parse(FILE *cfgFile, Data_T *data) {
  Parser_T *p = (Parser_T *)malloc(sizeof(Parser_T));
  p->head = CFG_lex(cfgFile);
  p->t = p->head;
  p->listBuf= NULL;
  p->lyrIdx = 0;

  NetConfig_T *cfg = (NetConfig_T *)malloc(sizeof(NetConfig_T));
  cfg->rows[INIT] = cfg->rows[FIN] = data->hgt;
  cfg->cols[INIT] = cfg->cols[FIN] = data->wid;
  cfg->numMat[INIT] = cfg->numMat[FIN] = 1;
  p->cfg = cfg;

  CFG_parse_netLyr(p);

  LayerT_T lastSet = NET;
  while (p->t) {
    if (p->lyrIdx == p->cfg->numLyr) {
      CFG_parse_throw(p, EXTRA_LYR, false);
    }
    switch(p->t->val.lType) {
      case NET:
        CFG_parse_throw(p, NET_REDEF, false);
        break;
      case CONVOLUTIONAL:
        CFG_parse_convLyr(p);
        lastSet = CONVOLUTIONAL;
        break;
      case POOLING:
        CFG_parse_poolLyr(p);
        lastSet = POOLING;
        break;
      case NORMALIZATION:
        CFG_parse_normLyr(p);
        lastSet = NORMALIZATION;
        break;
      case FULLY_CONNECTED:
        CFG_parse_sftmxLyr(p);
        lastSet = FULLY_CONNECTED;
        break;
    }
    p->cfg->lTypes[p->lyrIdx] = lastSet;
    ++p->lyrIdx;
  }

  if (p->lyrIdx != p->cfg->numLyr) {
    CFG_parse_throw(p, INSUF_LYR, true);
  }
  if (lastSet != FULLY_CONNECTED) {
    CFG_parse_throw(p, SFTMX_LAST, true);
  }

  CFG_lex_freeTok(p->head);
  free(p);
  free(p->listBuf);

  return cfg;
}

void CFG_free(NetConfig_T *cfg) {
  if (cfg->lyrs) {
    for (size_t i = 0; i < cfg->numLyr; i++) {
      if (cfg->lyrs[i] != NULL) {
        switch (cfg->lTypes[i]) {
          case CONVOLUTIONAL:
            LYR_conv_free((Conv_T *)cfg->lyrs[i]);
            break;
          case POOLING:
            LYR_pool_free((Pool_T *)cfg->lyrs[i]);
            break;
          case NORMALIZATION:
            free(cfg->lyrs[i]);
            break;
          case FULLY_CONNECTED:
            LYR_softmax_free((Softmax_T *)cfg->lyrs[i]);
            break;
        }
      }
    }
    free(cfg->lTypes);
    free(cfg->lyrs);
  }
  free(cfg);
}

void CFG_parse_netLyr(Parser_T *p) {
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != LYR_TYPE || p->t->val.lType != NET) {
    CFG_parse_throw(p, NET_FIRST, false);
  }
  ADVANCE(p);
  if (p->t->token != KEY || p->t->val.key != NUM_LYR) {
    CFG_parse_throw(p, BAD_CONFIG, false);
  }
  ADVANCE(p);
  p->cfg->numLyr = CFG_parse_intAssign(p);

  p->cfg->lTypes = (LayerT_T *)malloc(p->cfg->numLyr * sizeof(LayerT_T));
  p->cfg->lyrs = (void **)malloc(p->cfg->numLyr * sizeof(void *));
  for (size_t i = 0; i < p->cfg->numLyr; i++) {
    p->cfg->lyrs[i] = NULL;
  }
}

void CFG_parse_convLyr(Parser_T *p) {
  size_t numFeat = 0, featHgt = 0, featWid = 0;
  ADVANCE(p);
  while (p->t) {
    if (p->t->token == KEY) {
      switch(p->t->val.key) {
        case NUM_FEAT:
          ADVANCE(p);
          numFeat = CFG_parse_intAssign(p);
          break;
        case FEAT_HGT:
          ADVANCE(p);
          featHgt = CFG_parse_intAssign(p);
          break;
        case FEAT_WID:
          ADVANCE(p);
          featWid = CFG_parse_intAssign(p);
          break;
        default:
          CFG_parse_throw(p, BAD_CONFIG, false);
      }
    } else if (p->t->token == LYR_TYPE) {
      break;
    } else {
      CFG_parse_throw(p, EXP_KEY, false);
    }
  }

  if (numFeat == 0 || featHgt == 0 || featWid == 0) {
    CFG_parse_throw(p, UNSET_CONFIG, false);
  }
  p->cfg->lyrs[p->lyrIdx] = LYR_conv_init(numFeat, featHgt, featWid, p->cfg->numMat[FIN], p->cfg->rows[FIN], p->cfg->cols[FIN]);

  p->cfg->numMat[FIN] = p->cfg->numMat[FIN] * numFeat;
  p->cfg->rows[FIN] = CONV_OUT(p->cfg->rows[FIN], featHgt);
  p->cfg->cols[FIN] = CONV_OUT(p->cfg->cols[FIN], featWid);
}

void CFG_parse_poolLyr(Parser_T *p) {
  size_t winDim = 0, stride = 0;
  ADVANCE(p);
  while (p->t) {
    if (p->t->token == KEY) {
      switch(p->t->val.key) {
        case WIN_DIM:
          ADVANCE(p);
          winDim = CFG_parse_intAssign(p);
          break;
        case STRIDE:
          ADVANCE(p);
          stride = CFG_parse_intAssign(p);
          break;
        default:
          CFG_parse_throw(p, BAD_CONFIG, false);
      }
    } else if (p->t->token == LYR_TYPE) {
      break;
    } else {
      CFG_parse_throw(p, EXP_KEY, false);
    }
  }

  if (winDim == 0 || stride == 0) {
    CFG_parse_throw(p, UNSET_CONFIG, false);
  } 
	if (((p->cfg->rows[FIN] - winDim) % stride) != 0 || ((p->cfg->cols[FIN] - winDim) % stride) != 0) {
    CFG_parse_throw(p, BAD_POOL_CFG, false);
	}

  p->cfg->lyrs[p->lyrIdx] = LYR_pool_init(winDim, stride, p->cfg->numMat[FIN], p->cfg->rows[FIN], p->cfg->cols[FIN]);
  p->cfg->rows[FIN] = POOL_OUT(p->cfg->rows[FIN], winDim, stride);
  p->cfg->cols[FIN] = POOL_OUT(p->cfg->cols[FIN], winDim, stride);
}

void CFG_parse_normLyr(Parser_T *p) {
  ADVANCE(p);
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != KEY || p->t->val.key != NONLINEARITY) {
    CFG_parse_throw(p, BAD_CONFIG, false);
  }

  ADVANCE(p);
  p->cfg->lyrs[p->lyrIdx] = (NonLin_T *)malloc(sizeof(NonLin_T));
  NonLin_T *poolLyr = (NonLin_T *)p->cfg->lyrs[p->lyrIdx];
  *poolLyr = CFG_parse_nLinAssign(p);

  if (p->t && p->t->token != LYR_TYPE) {
    CFG_parse_throw(p, BAD_CONFIG, false);
  }
}

void CFG_parse_sftmxLyr(Parser_T *p) {
  int nLin = INVALID;
  double lrnRate = 0.0;
  size_t numHidden = 0, numOut = 0;
  ADVANCE(p);
  while (p->t) {
    if (p->t->token == KEY) {
      switch(p->t->val.key) {
        case NONLINEARITY:
          ADVANCE(p);
          nLin = CFG_parse_nLinAssign(p);
          break;
        case LRN_RATE:
          ADVANCE(p);
          lrnRate = CFG_parse_floatAssign(p);
          break;
        case HIDDENS:
          ADVANCE(p);
          numHidden = CFG_parse_listAssign(p);
          break;
        case NUM_OUTPUT:
          ADVANCE(p);
          numOut = CFG_parse_intAssign(p);
          break;
        default:
          CFG_parse_throw(p, BAD_CONFIG, false);
      }
    } else if (p->t->token == LYR_TYPE) {
      break;
    } else {
      CFG_parse_throw(p, EXP_KEY, false);
    }
  }

  if (nLin == INVALID || lrnRate == 0.0 || numOut == 0) {
    CFG_parse_throw(p, UNSET_CONFIG, false);
  }
  
  size_t numLyr = numHidden + 2;
  size_t *topo = (size_t *)malloc(numLyr * sizeof(size_t));

  for (size_t i = 0; i < numHidden; i++) {
    topo[i + 1] = p->listBuf[i];
  }
  topo[0] = p->cfg->numMat[FIN] * p->cfg->rows[FIN] * p->cfg->cols[FIN] * NUM_CHNL;
  topo[numLyr - 1] = numOut;
  p->cfg->lyrs[p->lyrIdx] = LYR_softmax_init(topo, numLyr, lrnRate, (NonLin_T)nLin);
  p->cfg->numOut = numOut;

  free(topo);
}

static inline size_t CFG_parse_intAssign(Parser_T *p) {
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != OPERATOR || p->t->val.opCode != ASSIGNMENT) {
    CFG_parse_throw(p, EXP_OP_ASSIGN, false);
  } 

  ADVANCE(p);
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != INT_VAL) {
    CFG_parse_throw(p, EXP_VAL_INT, false);
  } 
  
  size_t iVal = p->t->val.iVal;
  ADVANCE(p);
  return iVal;
}

static inline double CFG_parse_floatAssign(Parser_T *p) {
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != OPERATOR || p->t->val.opCode != ASSIGNMENT) {
    CFG_parse_throw(p, EXP_OP_ASSIGN, false);
  } 

  ADVANCE(p);
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != FLOAT_VAL) {
    CFG_parse_throw(p, EXP_VAL_FLOAT, false);
  } 
  
  double fVal = p->t->val.fVal;
  ADVANCE(p);
  return fVal;
}

static inline NonLin_T CFG_parse_nLinAssign(Parser_T *p) {
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != OPERATOR || p->t->val.opCode != ASSIGNMENT) {
    CFG_parse_throw(p, EXP_OP_ASSIGN, false);
  } 

  ADVANCE(p);
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != NLIN) {
    CFG_parse_throw(p, EXP_VAL_NLIN, false);
  } 

  NonLin_T nLin = p->t->val.nLin;
  ADVANCE(p);
  return nLin;
}

static inline size_t CFG_parse_listAssign(Parser_T *p) {
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != OPERATOR || p->t->val.opCode != ASSIGNMENT) {
    CFG_parse_throw(p, EXP_OP_ASSIGN, false);
  } 

  ADVANCE(p);
  if (!p->t) {
    CFG_parse_throw(p, UNEXP_EOF, true);
  } else if (p->t->token != OPERATOR || p->t->val.opCode != LIST_OPEN) {
    CFG_parse_throw(p, EXP_VAL_LIST, false);
  } 

  ADVANCE(p);
  TokenList_T *l = p->t;
  size_t numElm;
  for (numElm = 1; !(l->token == OPERATOR && l->val.opCode == LIST_CLOSE); l = l->next) {
    if (l->token == OPERATOR && l->val.opCode == LIST_DELIMETER) {
      ++numElm;
    } else if (!l || l->token != INT_VAL) {
      CFG_parse_throw(p, EXP_OP_LCLOSE, false);
    }
  }
  p->listBuf = (size_t *)malloc(numElm * sizeof(size_t));

  size_t lastElm = numElm - 1;
  for (size_t i = 0; i < numElm; i++) {
    if (!p->t) {
      CFG_parse_throw(p, UNEXP_EOF, true);
    } else if (p->t->token == INT_VAL) {
      p->listBuf[i] = p->t->val.iVal;
      ADVANCE(p);
      if (i != lastElm) {
        if (!p->t) {
          CFG_parse_throw(p, UNEXP_EOF, true);
        } else if (p->t->token == OPERATOR && p->t->val.opCode == LIST_DELIMETER) {
          ADVANCE(p);
        } else {
          CFG_parse_throw(p, EXP_OP_LDELIM, false);
        }
      }
    } else {
      CFG_parse_throw(p, EXP_VAL_INT, false);
    }
  }

  ADVANCE(p);
  return numElm;
}

void CFG_parse_throw(Parser_T *p, ParseErr_T errType, bool isNull) {
  if (isNull) {
    fprintf(stderr, "Logistical error in config file (EOF): ");
  } else {
    fprintf(stderr, "Logistical error in config file (%lu): ", p->t->lineNum);
  }
  switch (errType) {
    case NET_FIRST:
      fprintf(stderr, "the 'net' layer must be configured before anything else\n");
      break;
    case SFTMX_LAST:
      fprintf(stderr, "the 'fully connected' layer must be the last layer in the network\n");
      break;
    case NET_REDEF:
      fprintf(stderr, "the 'net' layer can only be defined once\n");
      break;
    case EXTRA_LYR:
      fprintf(stderr, "more layers were defined than specified\n");
      break;
    case INSUF_LYR:
      fprintf(stderr, "more layers were specified than defined\n");
      break;
    case BAD_POOL_CFG:
      fprintf(stderr, "pooling layers must satisfy ((inputDim - windowDim) %% stride == 0) for x and y dimensions\n");
      break;
    case BAD_CONFIG:
      fprintf(stderr, "configuration option is not a valid member of its parent layer\n");
      break;
    case UNSET_CONFIG:
      fprintf(stderr, "required configuration option for previous layer was left unset\n");
      break;
    case EXP_KEY:
      fprintf(stderr, "expected a configuration option\n");
      break;
    case EXP_VAL_INT:
      fprintf(stderr, "expected an integer value\n");
      break;
    case EXP_VAL_FLOAT:
      fprintf(stderr, "expected a floating point value\n");
      break;
    case EXP_VAL_NLIN:
      fprintf(stderr, "expected a nonlinearity function specifier\n");
      break;
    case EXP_VAL_LIST:
      fprintf(stderr, "expected a list\n");
      break;
    case EXP_OP_ASSIGN:
      fprintf(stderr, "expected '%c'\n", OP_CHAR[ASSIGNMENT]);
      break;
    case EXP_OP_LDELIM:
      fprintf(stderr, "expected '%c'\n", OP_CHAR[LIST_DELIMETER]);
      break;
    case EXP_OP_LCLOSE:
      fprintf(stderr, "expected '%c'\n", OP_CHAR[LIST_CLOSE]);
      break;
    case UNEXP_EOF:
      fprintf(stderr, "unexpected EOF\n");
      break;
  }
  CFG_parse_free(p);
  exit(EXIT_FAILURE);
}

void CFG_parse_free(Parser_T *p) {
  if (p->listBuf) {
    free(p->listBuf);
  }
  CFG_free(p->cfg);
  CFG_lex_freeTok(p->head);
  free(p);
}

TokenList_T *CFG_lex(FILE *f) {
  Lexer_T *lex = (Lexer_T *)malloc(sizeof(Lexer_T));
  lex->f = f;
  lex->tokens = NULL;
  lex->lineNum = 1;

  int c;
  do {
    c = CFG_lex_nxtValid(lex);
    if (c == '[') {
      CFG_lex_getLyr(lex);
    } else if (isalpha(c)) {
      CFG_lex_getKey(lex);
    } else if (isdigit(c) || c == '.') {
      CFG_lex_getVal_num(lex);
    } else if (c == '"') {
      CFG_lex_getVal_str(lex);
    } else if (c == OP_CHAR[ASSIGNMENT]) {
      fgetc(lex->f);
      CFG_lex_appendOp(lex, ASSIGNMENT);
    } else if (c == OP_CHAR[LIST_OPEN]) {
      fgetc(lex->f);
      CFG_lex_nxtValid(lex);
      CFG_lex_appendOp(lex, LIST_OPEN);

      CFG_lex_getVal_num(lex);
    } else if (c == OP_CHAR[LIST_DELIMETER]) {
      fgetc(lex->f);
      CFG_lex_nxtValid(lex);
      CFG_lex_appendOp(lex, LIST_DELIMETER);

      CFG_lex_getVal_num(lex);
    } else if (c == OP_CHAR[LIST_CLOSE]) {
      fgetc(lex->f);
      CFG_lex_appendOp(lex, LIST_CLOSE);
    } else if (c != EOF) {
      CFG_lex_throw(lex, UNEXP_TOK);
    }
  } while (c != EOF);

  TokenList_T *tokens = lex->tokens;
  free(lex);
  return tokens;
}

void CFG_lex_getLyr(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_LYR_SIZE + 1), sizeof(char));

  fgetc(lex->f);
  CFG_lex_advUntil(lex, CFG_lex_isValid_lyr, CFG_lex_isEnd_lyr, buf, MAX_LYR_SIZE, BAD_LYR_TYPE);
  fgetc(lex->f);

  int lyrT = CFG_lex_strToEnum(buf, LYR_STR, NUM_LYR_TYPE);
  free(buf);

  if (lyrT == INVALID) {
    CFG_lex_throw(lex, BAD_LYR_TYPE);
  }

  TokenVal_T val;
  val.lType = (LayerT_T)lyrT;
  CFG_lex_appendTok(&lex->tokens, LYR_TYPE, val, lex->lineNum);
}

void CFG_lex_getKey(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_KEY_SIZE + 1), sizeof(char));

  CFG_lex_advUntil(lex, isalpha, CFG_lex_isEnd_key, buf, MAX_KEY_SIZE, BAD_KEY);

  int key = CFG_lex_strToEnum(buf, KEY_STR, NUM_KEY);
  free(buf);

  if (key == INVALID) {
    CFG_lex_throw(lex, BAD_KEY);
  }

  TokenVal_T val;
  val.key = (Key_T)key;
  CFG_lex_appendTok(&lex->tokens, KEY, val, lex->lineNum);
}

void CFG_lex_getVal_num(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_NUM_SIZE + 1), sizeof(char));

  CFG_lex_advUntil(lex, CFG_lex_isValid_num, CFG_lex_isEnd_num, buf, MAX_NUM_SIZE, NUM_OVERFLOW);

  bool isFloat = false;
  for (size_t i = 0; buf[i] != '\0'; i++) {
    if (buf[i] == '.') {
      isFloat = true;
      break;
    }
  }

  if (isFloat) {
    double fVal = strtod(buf, NULL);
    free(buf);
    if (fVal == 0.0) {
      CFG_lex_throw(lex, EMPTY_NUM);
    }

    TokenVal_T val;
    val.fVal = fVal;
    CFG_lex_appendTok(&lex->tokens, FLOAT_VAL, val, lex->lineNum);
  } else {
    size_t iVal = strtoull(buf, NULL, 10);
    free(buf);
    if (iVal == 0) {
      CFG_lex_throw(lex, EMPTY_NUM);
    }

    TokenVal_T val;
    val.iVal = iVal;
    CFG_lex_appendTok(&lex->tokens, INT_VAL, val, lex->lineNum);
  }
}

void CFG_lex_getVal_str(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_STR_SIZE + 1), sizeof(char));
  fgetc(lex->f);

  CFG_lex_advUntil(lex, isalpha, CFG_lex_isEnd_str, buf, MAX_NUM_SIZE, BAD_NLIN);
  CFG_lex_nxtValid(lex);
  fgetc(lex->f);

  int nLin = CFG_lex_strToEnum(buf, NLIN_STR, NUM_NLIN);
  free(buf);

  if (nLin == INVALID) {
    CFG_lex_throw(lex, BAD_NLIN);
  }

  TokenVal_T val;
  val.nLin = (NonLin_T)nLin;
  CFG_lex_appendTok(&lex->tokens, NLIN, val, lex->lineNum);
}

void CFG_lex_throw(Lexer_T *lex, LexErr_T errType) {
  fprintf(stderr, "Syntax error in config file: (%lu) ", lex->lineNum);
  switch (errType) {
    case UNEXP_TOK:
      fprintf(stderr, "unexpected token\n");
      break;
    case BAD_LYR_TYPE:
      fprintf(stderr, "unknown layer type\n");
      break;
    case BAD_KEY:
      fprintf(stderr, "unknown configuration option\n");
      break;
    case NUM_OVERFLOW:
      fprintf(stderr, "value is too large\n");
      break;
    case EMPTY_NUM:
      fprintf(stderr, "numerical value cannot be empty or zero\n");
      break;
    case BAD_NLIN:
      fprintf(stderr, "unknown nonlinearity- valid options are \"relu\" or \"sigmoid\"\n");
      break;
  }

  CFG_lex_free(lex);
  exit(EXIT_FAILURE);
}

void CFG_lex_appendTok(TokenList_T **head, Token_T tok, TokenVal_T val, size_t lineNum) {
  TokenList_T *newTok = (TokenList_T *)malloc(sizeof(TokenList_T));
  newTok->token = tok;
  newTok->val = val;
  newTok->lineNum = lineNum;
  newTok->next = NULL;

  TokenList_T **i = head;
  while(*i) {
    i = &(*i)->next;
  }
  *i = newTok;
}

static inline void CFG_lex_advUntil(Lexer_T *lex, int (*isValid)(int), int (*isEnd)(int), char *buf, unsigned maxLen, LexErr_T ovrflwErr) {
  int c;
  for (unsigned i = 0; true; i++) {
    c = CFG_lex_peek(lex->f);
    if (isEnd(c)) {
      buf[i] = '\0';
      return;
    } else if (i == maxLen) {
      free(buf);
      CFG_lex_throw(lex, ovrflwErr);
    } else if (!isValid(c)) {
      free(buf);
      CFG_lex_throw(lex, UNEXP_TOK);
    }
    buf[i] = fgetc(lex->f);
  }
}

static inline int CFG_lex_nxtValid(Lexer_T *lex) {
  while (true) {
    int c = CFG_lex_peek(lex->f);
    if (isspace(c)) {
      if (c == '\n') {
        ++lex->lineNum;
      }
      fgetc(lex->f);
    } else if (c == COMMENT_START) {
      while (true) {
        c = fgetc(lex->f);
        if (c == '\n' || c == EOF) {
          break;
        }
      }
      ++lex->lineNum;
    } else {
      return c;
    }
  }
}

static inline void CFG_lex_appendOp(Lexer_T *lex, Operator_T opCode) {
  TokenVal_T val;
  val.opCode = opCode;
  CFG_lex_appendTok(&lex->tokens, OPERATOR, val, lex->lineNum);
}

static inline int CFG_lex_isEnd_lyr(int c) {
  return c == ']';
}

static inline int CFG_lex_isValid_lyr(int c) {
  return isalpha(c) || c == ' ';
}

static inline int CFG_lex_isEnd_key(int c) {
  return isspace(c) || c == '=';
}

static inline int CFG_lex_isEnd_num(int c) {
  return isspace(c) || c == ',' || c == '}';
}

static inline int CFG_lex_isValid_num(int c) {
  return isdigit(c) || c == '.';
}

static inline int CFG_lex_isEnd_str(int c) {
  return c == '"';
}

static inline int CFG_lex_strToEnum(char *inStr, const char **enumStr, unsigned numEnum) {
  for (int i = 0; i < numEnum; i++) {
    if (!strcmp(inStr, enumStr[i])) {
      return i;
    }
  }

  return INVALID;
}

static inline int CFG_lex_peek(FILE *f) {
  int c = fgetc(f);
  ungetc(c, f);

  return c;
}

void CFG_lex_freeTok(TokenList_T *head) {
  TokenList_T *t = head;
  while (t) {
    TokenList_T *next = t->next;
    free(t);
    t = next;
  }
}

void CFG_lex_free(Lexer_T *lex) {
  CFG_lex_freeTok(lex->tokens);
  free(lex);
}

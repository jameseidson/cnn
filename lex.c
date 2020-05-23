#include "lex.h"

static const uint8_t MAX_LYR_SIZE = 15;
static const uint8_t MAX_KEY_SIZE = 10;
static const uint8_t MAX_VAL_SIZE = 64;
static const char *LYR_STR[] = { "net", "convolutional", "pooling", "normalization", "fully connected" };
static const char *KEY_STR[] = { "numEpoch", "numFeature", "featureHgt", "featureWid", "windowSize", "stride", 
                                 "learnRate", "numHidden", "hiddens", "numOutput" };

typedef struct {
  FILE *f;
  size_t lineNum;
  TokenList_T *tokens;
} Lexer_T;

static inline TokenList_T *appendTok(TokenList_T *head, Token_T tok, TokenVal_T val) {
  TokenList_T *newTok = (TokenList_T *)malloc(sizeof(TokenList_T));
  assert(newTok);
  newTok->token = tok;
  newTok->val = val;

  newTok->next = head;
  return newTok;
}

static inline Key_T findKey(char *keyS) {
  for (int i = 0; i < NUM_KEY; i++) {
    if (!strcmp(KEY_STR[i], keyS)) {
      return (Key_T)i;
    }
  }

  return INVALID_KEY;
}

static inline int fpeek(FILE *f) {
  int c = fgetc(f);
  ungetc(c, f);

  return c;
}

void freeTokens(TokenList_T *tokens) {
  if (tokens == NULL) {
    return;
  } else {
    freeTokens(tokens->next);
    free(tokens);
  }
}

void lexThrow(Error_T errType, Lexer_T *lex) {
  fprintf(stderr, "Syntax error in config file- Line %lu: ", lex->lineNum);
  switch (errType) {
    case BAD_TOKEN:
      fprintf(stderr, "unknown token\n");
      break;
    case UNEXPECTED_EOF:
      fprintf(stderr, "unexpected EOF\n");
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
  }

  freeTokens(lex->tokens);
  exit(EXIT_FAILURE);
}

void readKey(Lexer_T *lex) {
  char *buf = (char *)malloc((MAX_KEY_SIZE + 1) * sizeof(char));
  int c;
  for (uint8_t i = 0; true; i++) {
    c = fgetc(lex->f);
    if (c == '=') {
      buf[i] = '\0';
      break;
    } else if (c == EOF) {
      free(buf);
      lexThrow(UNEXPECTED_EOF, lex);
      break;
    } else if (i == MAX_KEY_SIZE) {
      free(buf);
      lexThrow(BAD_KEY, lex);
      break;
    } else {
      buf[i] = c;
    }
  }

  Key_T foundKey = findKey(buf);
  free(buf);

  if (foundKey == INVALID_KEY) {
    lexThrow(BAD_KEY, lex);
  }

  TokenVal_T val;
  val.key = foundKey;
  lex->tokens = appendTok(lex->tokens, KEY, val);
}


void readNum(Lexer_T *lex) {
  char *buf = (char *)malloc((MAX_VAL_SIZE + 1) * sizeof(char));
  int c;
  bool isFloat = false;
  for (uint8_t i = 0; true; i++) {
    c = fgetc(lex->f);
    if (c == '\n') {
      lex->lineNum++;
      buf[i] = '\0';
      break;
    } else if (isspace(c) || c == EOF) {
      buf[i] = '\0';
      break;
    } else if (i == MAX_LYR_SIZE)  {
      free(buf);
      lexThrow(NUM_OVERFLOW, lex);
      break;
    } else if (c == '.') {
      isFloat = true;
      buf[i] = c;
    } else {
      buf[i] = c;
    }
  }

  if (isFloat) {
    double fval = strtod(buf, NULL);
    TokenVal_T val;
    val.fval = fval;
    lex->tokens = appendTok(lex->tokens, FLOAT_VAL, val);
  } else {
    size_t ival = strtoull(buf, NULL, 10);
    TokenVal_T val;
    val.ival = ival;
    lex->tokens = appendTok(lex->tokens, INT_VAL, val);
  }

  free(buf);
}

void readList(Lexer_T *lex) {
  fgetc(lex->f);
  int c;
  while (true) {
    c = fgetc(lex->f);
    if (c == '\n') {
      lex->lineNum++;
    } else if (isspace(c) || c == ',') {
      continue;
    } else if (c == EOF) {
      lexThrow(UNEXPECTED_EOF, lex);
      break;
    } else if (c == '}') {
      break;
    } else if (isdigit(c)) {
      char *buf = (char *)malloc((MAX_VAL_SIZE + 1) * sizeof(char));
      buf[0] = c;
      size_t i = 1;
      do {
        buf[i] = fgetc(lex->f);
        i++;
      } while(isdigit(fpeek(lex->f)));

      size_t ival = strtoull(buf, NULL, 10);
      TokenVal_T val;
      val.ival = ival;
      lex->tokens = appendTok(lex->tokens, INT_VAL, val);
      free(buf);
    } else {
      lexThrow(BAD_TOKEN, lex);
    }
  }
}

void readRValue(Lexer_T *lex) {
  int c = fpeek(lex->f);
  if (isdigit(c) || c == '.') {
    readNum(lex);
  } else if (c == '{') {
    readList(lex);
  }
}

Layer_T findLyr(char *layerS) {
  for (int i = 0; i < NUM_LYR_TYPE; i++) {
    if (!strcmp(LYR_STR[i], layerS)) {
      return (Layer_T)i;
    }
  }

  return INVALID_LYR;
}

void readLyrType(Lexer_T *lex) {
  char *buf = (char *)malloc((MAX_LYR_SIZE + 1) * sizeof(char));
  int c = fgetc(lex->f);
  for (uint8_t i = 0; true; i++) {
    c = fgetc(lex->f);
    if (c == ']') {
      buf[i] = '\0';
      break;
    } else if (c == EOF) {
      free(buf);
      lexThrow(UNEXPECTED_EOF, lex);
      break;
    } else if (i == MAX_LYR_SIZE)  {
      free(buf);
      lexThrow(BAD_LYR_TYPE, lex);
      break;
    } else {
      buf[i] = c;
    }
  }

  Layer_T foundLyr = findLyr(buf);
  free(buf);

  if (foundLyr == INVALID_LYR) {
    lexThrow(BAD_LYR_TYPE, lex);
  }

  TokenVal_T val;
  val.ltype = foundLyr;
  lex->tokens = appendTok(lex->tokens, LYR_TYPE, val);
}

TokenList_T *lex(FILE *f) {
  Lexer_T *lex = (Lexer_T *)malloc(sizeof(Lexer_T));
  assert(lex);
  lex->f = f;
  lex->tokens = NULL;
  lex->lineNum = 1;

  int c;
  while (true) {
    c = fpeek(lex->f);
    if (c == '\n') {
      fgetc(lex->f);
      lex->lineNum++;
    } else if (isspace(c)) {
      fgetc(f);
    } else if (isalpha(c)) {
      readKey(lex);
      readRValue(lex);
    } else if (c == '[') {
      readLyrType(lex);
    } else if (c == EOF) {
      break;
    } else {
      lexThrow(BAD_TOKEN, lex);
    }
  }

  TokenList_T *tokens = lex->tokens;
  free(lex);
  return tokens;
}

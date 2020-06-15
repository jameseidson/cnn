#include "lex.h"

static const uint8_t MAX_LYR_SIZE = 15;
static const uint8_t MAX_KEY_SIZE = 10;
static const uint8_t MAX_VAL_SIZE = 64;
static const char *LYR_STR[] = { "net", "convolutional", "pooling", "normalization", "fully connected" };
static const char *KEY_STR[] = { "numLayer", "numFeature", "featureHgt", "featureWid", "windowDim", "stride",
                                 "learnRate", "numHidden", "hiddens", "numOutput" };

typedef struct Lexer {
  FILE *f;
  size_t lineNum;
  TokenList_T *tokens;
} Lexer_T;

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

void appendTok(TokenList_T **head, Token_T tok, TokenVal_T val, size_t lineNum) {
  TokenList_T *newTok = (TokenList_T *)malloc(sizeof(TokenList_T));
  assert(newTok);
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

void freeTokens(TokenList_T *tokens) {
  if (!tokens) {
    return;
  }

  freeTokens(tokens->next);
  free(tokens);
}

void lexErr(LexErr_T errType, Lexer_T *lex) {
  fprintf(stderr, "Syntax error in config file: (%lu) ", lex->lineNum);
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
    case EXPECTED_VAL:
      fprintf(stderr, "expected a value\n");
      break;
    case NUM_OVERFLOW:
      fprintf(stderr, "value is too large\n");
      break;
  }

  freeTokens(lex->tokens);
  exit(EXIT_FAILURE);
}

void readKey(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_KEY_SIZE + 1), sizeof(char));
  int c;
  for (uint8_t i = 0; true; i++) {
    c = fgetc(lex->f);
    if (c == '=') {
      buf[i] = '\0';
      break;
    } else if (c == EOF) {
      free(buf);
      lexErr(UNEXPECTED_EOF, lex);
      break;
    } else if (i == MAX_KEY_SIZE) {
      free(buf);
      lexErr(BAD_KEY, lex);
      break;
    } else {
      buf[i] = c;
    }
  }

  Key_T foundKey = findKey(buf);
  free(buf);

  if (foundKey == INVALID_KEY) {
    lexErr(BAD_KEY, lex);
  }

  TokenVal_T val;
  val.key = foundKey;
  appendTok(&lex->tokens, KEY, val, lex->lineNum);
}


void readNum(Lexer_T *lex) {
  char *buf = (char *)calloc((MAX_VAL_SIZE + 1), sizeof(char));
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
      lexErr(NUM_OVERFLOW, lex);
      break;
    } else if (c == '.') {
      isFloat = true;
      buf[i] = c;
    } else if (isdigit(c)) {
      buf[i] = c;
    } else {
      lexErr(BAD_TOKEN, lex);
    }
  }

  if (isFloat) {
    double fval = strtod(buf, NULL);
    TokenVal_T val;
    val.fval = fval;
    appendTok(&lex->tokens, FLOAT_VAL, val, lex->lineNum);
  } else {
    size_t ival = strtoull(buf, NULL, 10);
    TokenVal_T val;
    val.ival = ival;
    appendTok(&lex->tokens, INT_VAL, val, lex->lineNum);
  }

  free(buf);
}

void readList(Lexer_T *lex) {
  fgetc(lex->f);
  int c;
  bool readNum = false;
  while (true) {
    c = fgetc(lex->f);
    if (c == '\n') {
      lex->lineNum++;
    } else if (isspace(c)) {
      continue;
    } else if (c == ',') {
      if (!readNum) {
        lexErr(EXPECTED_VAL, lex);
      }
      readNum = false;
      continue;
    } else if (c == EOF) {
      lexErr(UNEXPECTED_EOF, lex);
      break;
    } else if (c == '}') {
      break;
    } else if (isdigit(c)) {
      readNum = true;
      char *buf = (char *)calloc((MAX_VAL_SIZE + 1), sizeof(char));
      assert(buf);
      buf[0] = c;
      size_t i = 1;
      do {
        buf[i] = fgetc(lex->f);
        i++;
      } while(isdigit(fpeek(lex->f)));

      size_t ival = strtoull(buf, NULL, 10);
      TokenVal_T val;
      val.ival = ival;
      appendTok(&lex->tokens, INT_VAL, val, lex->lineNum);
      free(buf);
    } else {
      lexErr(BAD_TOKEN, lex);
    }
  }
  if (!readNum) {
    lexErr(EXPECTED_VAL, lex);
  }
}

void readRValue(Lexer_T *lex) {
  int c = fpeek(lex->f);
  if (isdigit(c) || c == '.') {
    readNum(lex);
  } else if (c == '{') {
    readList(lex);
  } else {
    lexErr(EXPECTED_VAL, lex);
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
  char *buf = (char *)calloc((MAX_LYR_SIZE + 1), sizeof(char));
  int c = fgetc(lex->f);
  for (uint8_t i = 0; true; i++) {
    c = fgetc(lex->f);
    if (c == ']') {
      buf[i] = '\0';
      break;
    } else if (c == EOF) {
      free(buf);
      lexErr(UNEXPECTED_EOF, lex);
      break;
    } else if (i == MAX_LYR_SIZE)  {
      free(buf);
      lexErr(BAD_LYR_TYPE, lex);
      break;
    } else {
      buf[i] = c;
    }
  }

  Layer_T foundLyr = findLyr(buf);
  free(buf);

  if (foundLyr == INVALID_LYR) {
    lexErr(BAD_LYR_TYPE, lex);
  }

  TokenVal_T val;
  val.ltype = foundLyr;
  appendTok(&lex->tokens, LYR_TYPE, val, lex->lineNum);
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
      lexErr(BAD_TOKEN, lex);
    }
  }

  TokenList_T *tokens = lex->tokens;
  free(lex);

  return tokens;
}

#include "net.h"

#include <stdio.h>
#include <stdlib.h>


int main() {
  size_t topology[] = {10, 10, 10};
  size_t netSize = 3;

  Classify_T *net = CNN_initC(topology, netSize);
  CNN_testC(net);
  CNN_freeC(net);

  return 0;
}

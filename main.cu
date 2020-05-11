#include "net.h"

#include <stdio.h>
#include <stdlib.h>


int main() {
  size_t topology[] = {10, 10, 10};
  size_t netSize = 3;

  Net_T *net = CNN_initFC(topology, netSize);
  CNN_testFC(net);
  CNN_freeFC(net);

  return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern layer_size;

load(netK,netC)
BPNN *netK,*netC;
{
  float *unitsK, *unitsC;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;
  
  imgsize = nr * nc;
  unitsK = netK->input_units;
  unitsC = netC->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
	  unitsK[k] = (float) rand()/RAND_MAX ;
	  unitsC[k] = unitsK[k];
	  k++;
    }
}

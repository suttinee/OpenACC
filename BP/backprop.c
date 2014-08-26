/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#define OPEN

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(x)
float x;
{
  float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
 return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(n)
int n;
{
  float *new;

  new = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (new == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new);
}


/*** Allocate 2d array of floats ***/
/*
float **alloc_2d_dbl(m, n)
int m, n;
{
  int i;
  float **new;

  new = (float **) malloc ((unsigned) (m * sizeof (float *)));
  if (new == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new[i] = alloc_1d_dbl(n);
  }

  return (new);
}
*/

void bpnn_copy_weights(float *w, int m,int n,float *original)
{
  int i, j;
  for (i = 0; i <= m; i++) {
   for (j = 0; j <= n; j++) {
     w[(i*n)+j] = original[(i*n)+j];
    }
  }
}

//bpnn_randomize_weights(w, m, n)
//float *w;
//int m, n;
void bpnn_randomize_weights(float *w, int m,int n)
{
  int i, j;
  for (i = 0; i <= m; i++) {
   for (j = 0; j <= n; j++) {
     w[(i*n)+j] = (float) rand()/RAND_MAX;
    }
  }
//printf("out\n");
}

bpnn_randomize_row(w, m)
float *w;
int m;
{
	int i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}


bpnn_zero_weights(w, m, n)
float *w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i*n+j] = 0.0;
    }
  }
}


void bpnn_initialize(seed)
{
  //printf("Random number generator seed: %d\n", seed);
  srand(seed);
}


BPNN *bpnn_internal_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

//printf("n_in = %d, n_hidden=%d, n_out=%d\n",n_in, n_hidden, n_out);
  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);
  
  newnet->input_weights = alloc_1d_dbl((n_in + 1)*(n_hidden + 1)+1);
  newnet->hidden_weights = alloc_1d_dbl((n_hidden + 1)*( n_out + 1)+1);

  newnet->input_prev_weights = alloc_1d_dbl((n_in + 1)*( n_hidden + 1)+1);
  newnet->hidden_prev_weights = alloc_1d_dbl((n_hidden + 1)*(n_out + 1)+1);

  return (newnet);
}


void bpnn_free(net)
BPNN *net;
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

//  for (i = 0; i <= n1; i++) {
    //free((char *) net->input_weights[i]);
    //free((char *) net->input_prev_weights[i]);
 // }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

//  for (i = 0; i <= n2; i++) {
    //free((char *) net->hidden_weights[i]);
    //free((char *) net->hidden_prev_weights[i]);
 // }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

//cpu will copy random weights from the kernel array must call kernel create first
BPNN *bpnn_create_cpu(n_in, n_hidden, n_out,kernelnet)
int n_in, n_hidden, n_out;
BPNN *kernelnet;
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);
#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
 // bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
   bpnn_copy_weights(newnet->input_weights, n_in, n_hidden,kernelnet->input_weights);
#endif
 // bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_copy_weights(newnet->hidden_weights, n_hidden,n_out,kernelnet->hidden_weights);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);


  return (newnet);
}

BPNN *bpnn_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);
#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);

  return (newnet);
}


void bpnn_layerforward_cpu(l1, l2, conn, n1, n2)
float *l1, *l2, *conn;
int n1, n2;
{
  float sum,m;
  int j, k;
l1[0]=1.0;

  for (j = 1; j <= n2; j++) {
   sum = 0.0;
    for (k = 0; k <= n1; k++) {	
 	sum += conn[k*n2+j]*l1[k];
    }
    l2[j] = squash(sum);
  }
}

void bpnn_layerforward(l1, l2, conn, n1, n2)
float *l1, *l2, *conn;
int n1, n2;
{
  float sum,m;
  int j, k;
  /*** Set up thresholding unit ***/
  l1[0] = 1.0;

#pragma acc data create(sum), copyin(l1[0:n1+1],conn[0:(n1+1)*(n2+1)]), copyout(l2[0:n2+1])
  /*** For each unit in second layer ***/
#pragma acc kernels
{
#ifdef INDEPENDENT
	#pragma acc loop independent
#else
	#pragma acc loop
#endif
 for (j = 1; j <= n2; j++) {
    /*** Compute weighted sum of its inputs ***/
   sum = 0.0;
#ifdef REDUCTION
    	#pragma acc loop vector reduction(+:sum)
#endif
#ifdef ILP
	#pragma hmppcg(OPENCL) unroll(8), jam
#endif
    for (k = 0; k <= n1; k++) {	
 	sum += conn[k*n2+j]*l1[k];
    }
    l2[j] = squash(sum);
  }
}//acc kernels
}
//extern "C"

void bpnn_output_error(delta, target, output, nj, err)  
float *delta, *target, *output, *err;
int nj;
{
  int j;
  float o, t, errsum;
  errsum = 0.0;
for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;

}


void bpnn_hidden_error(delta_h,   
					   nh, 
					   delta_o, 
					   no, 
					   who, 
					   hidden, 
					   err)
float *delta_h, *delta_o, *hidden, *who, *err;
int nh, no;
{
  int j, k;
  float h, sum, errsum;
  errsum = 0.0;
for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j*nh+k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_adjust_weights_cpu(delta, ndelta, ly, nly, w, oldw)
float *delta, *ly, *w, *oldw;
int nly,ndelta;
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  for (j = 1; j <= ndelta; j++) {
   for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k*ndelta+j]));
	  w[k*ndelta+j] += new_dw;
	  oldw[k*ndelta+j] = new_dw;
    }
  }
}


void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw)
float *delta, *ly, *w, *oldw;
int nly,ndelta;
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  //eta = 0.3;
  //momentum = 0.3;

/*#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for  \
      shared(oldw, w, delta) \
	  private(j, k, new_dw) \
	  firstprivate(ndelta, nly) 
#endif
printf("nly = %d ndelta=%d\n",nly,ndelta);
*/

#pragma acc data copy(oldw[0:(ndelta+1)*(nly+1)],w[0:(ndelta+1)*(nly+1)]), copyin(delta[0:(ndelta+1)],ly[0:nly+1])
 {
#ifdef INDEPENDENT
#pragma acc kernels loop independent, private(j, k, new_dw)
#else
#pragma acc kernels loop, private(j, k, new_dw)
#endif
  for (j = 1; j <= ndelta; j++) {
#ifdef INDEPENDENT
#pragma acc loop independent
#else
#pragma acc loop
#endif
   for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k*ndelta+j]));
	  w[k*ndelta+j] += new_dw;
	  oldw[k*ndelta+j] = new_dw;
    }
  }
 }

}

void bpnn_feedforward(net)
BPNN *net;
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}


void bpnn_train(net, eo, eh)
BPNN *net;
float *eo, *eh;
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights);

}




void bpnn_save(net, filename)
BPNN *net;
char *filename;
{
  int n1, n2, n3, i, j, memcnt;
  float dvalue, *w;
  char *mem;
  ///add//
  FILE *pFile;
  pFile = fopen( filename, "w+" );
  ///////
  /*
  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }
  */

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  //printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  //write(fd, (char *) &n1, sizeof(int));
  //write(fd, (char *) &n2, sizeof(int));
  //write(fd, (char *) &n3, sizeof(int));

  fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);

  

  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[(i*n1)+j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  fwrite( mem , (unsigned)(sizeof(float)), (unsigned) ((n1+1) * (n2+1) * sizeof(float)) , pFile);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  
mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[(i*n1)+j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  fwrite( mem , sizeof(float), (unsigned) ((n2+1) * (n3+1) * sizeof(float)) , pFile);
  free(mem);

  fclose(pFile);
  return;
}


BPNN *bpnn_read(filename)
char *filename;
{
  char *mem;
  BPNN *new;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  //printf("Reading '%s'\n", filename);  //fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));
  new = bpnn_internal_create(n1, n2, n3);

  //printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  //printf("Reading input weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new->input_weights[(i*n1)+j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);

  //printf("Done\nReading hidden weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new->hidden_weights[(i*n2)+j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  //printf("Done\n");  //fflush(stdout);

  bpnn_zero_weights(new->input_prev_weights, n1, n2);
  bpnn_zero_weights(new->hidden_prev_weights, n2, n3);

  return (new);
}
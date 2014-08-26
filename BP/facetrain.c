#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

void compare(BPNN *n1, BPNN *n2){

 int i;

  if((n1->input_n != n2->input_n)||(n1->hidden_n!=n2->hidden_n)||(n1->output_n!=n2->output_n))
	printf("Error\n");


  for(i=0;i< n1->input_n ;i++){
	if(n1->input_units[i]!=n2->input_units[i])
	{
		printf("%f  %f\n",n1->input_units[i],n2->input_units[i]);
		printf("Error input un\n");
		break;
	}
  }
 for(i=0;i< n1->hidden_n ;i++){
        if(n1->hidden_units[i]!=n2->hidden_units[i])
        {       
                printf("Error hidden\n");
                break;
        }
  }
for(i=1;i< n1->output_n ;i++){
        if(n1->output_units[i]!=n2->output_units[i])
        {
		printf("%f  %f\n",n1->output_units[i],n2->output_units[i]);
                printf("Error unit\n");
                break;
        }
  }
for(i=1;i< n1->output_n ;i++){
        if(n1->output_delta[i]!=n2->output_delta[i])
        {
                printf("Error outputdelta\n");
                break;
        }
  }

for(i=1;i< n1->hidden_n ;i++){
        if(n1->hidden_delta[i]!=n2->hidden_delta[i])
        {
                printf("Error hidden delta\n");
                break;
        }
  }

for(i=1;i< n1->output_n ;i++){
        if(n1->target[i]!=n2->target[i])
        {
                printf("Error target\n");
                break;
        }
  }
for(i=1;i< n1->input_n*n1->hidden_n ;i++){
        if(n1->input_weights[i]!=n2->input_weights[i])
        {
                printf("Error input_weights\n");
                break;
        }
  }

for(i=1;i< n1->output_n*n1->hidden_n ;i++){
        if(n1->hidden_weights[i]!=n2->hidden_weights[i])
        {
		 printf("%f  %f\n",n1->hidden_weights[i],n2->hidden_weights[i]);
                printf("Error hidden_weights\n");
                break;
        }
  }
for(i=1;i< n1->input_n*n1->hidden_n ;i++){
        if(n1->input_prev_weights[i]!=n2->input_prev_weights[i])
        {
                printf("Error input prev_weights\n");
                break;
        }
  }
for(i=1;i< n1->output_n*n1->hidden_n ;i++){
        if(n1->hidden_prev_weights[i]!=n2->hidden_prev_weights[i])
        {
		printf("%f  %f\n",n1->hidden_prev_weights[i],n2->hidden_prev_weights[i]);
                printf("Error hidden prev_weights\n");
                break;
        }
  }



}



backprop_face()
{
  BPNN *net_kernels,*net_cpu;
  int i;
  float out_err, hid_err;
  net_kernels = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
 net_cpu = bpnn_create_cpu(layer_size, 16, 1,net_kernels);  
 //printf("Input layer size : %d\n", layer_size);
  load(net_kernels,net_cpu);
  //printf("%f  %f\n",net_kernels->input_units[500],net_cpu->input_units[500]);

 struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL); 

  //entering the training kernel, only one iteration
 // printf("Starting training kernel\n");
  bpnn_train_kernel(net_kernels, &out_err, &hid_err);

  gettimeofday(&tv2, NULL);

 // printf("Starting Computing by CPU\n");
  bpnn_train_cpu(net_cpu, &out_err, &hid_err);

  //compare BPNN from Kernels and CPU 
  //uncomment to check the results
  compare(net_kernels,net_cpu);
 // printf("Compare result complete\n");

  bpnn_free(net_kernels);
  bpnn_free(net_cpu);
 // printf("Training done\n");

  //printf ("Total time = %f seconds\n",
    //     (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
      //   (double) (tv2.tv_sec - tv1.tv_sec));


}

int setup(argc, argv)
int argc;
char *argv[];
{
  if(argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }

  layer_size = atoi(argv[1]);
  
  int seed;

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}

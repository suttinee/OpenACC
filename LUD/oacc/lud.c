/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "../common/common.h"



void wul(){

  int data = 8192;
  double *arr_a = (double *)malloc(sizeof(double) * data);
  double *arr_b = (double *)malloc(sizeof(double) * data);
  int i = 0;

  if (arr_a==NULL||arr_b==NULL) {
      printf("Unable to allocate memory in wul.\n");
  }

  for (i=0;i<data;i++){
    arr_a[i] = (double) (rand()/(1.0+RAND_MAX));
  }

#pragma acc data copy(arr_b[0:data]), copyin(arr_a[0:data])
  {
#pragma acc parallel loop
    for (i=0;i<data;i++){
      arr_b[i] = arr_a[i] * 2;
    }
  }

  if (arr_a[0] < 0){
 printf("Error in WUL\n");
}

  free(arr_a);
  free(arr_b);

}




static int do_verify = 0;
int omp_num_threads = 1;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void
lud_oacc(float *m, int matrix_dim,int grid_x,int grid_y);

int
main ( int argc, char *argv[] )
{
//printf("Starting..\n");
  int matrix_dim = 32; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;
	int grid_x=0;
	int grid_y=0;

	
  while ((opt = getopt_long(argc, argv, "::vs:i:x:y:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      //printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case 'x':
	grid_x = atoi(optarg);
	break;
    case 'y':
	grid_y = atoi(optarg);
	break;

    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "1Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
 /* 
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "2Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
*/
  if (input_file) {
    //printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }
  else if (matrix_dim) {
    //printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }
 
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  if (do_verify){
    /* print_matrix(m, matrix_dim); */
    matrix_duplicate(m, &mm, matrix_dim);
  }
wul();
//printf("Starting. . . \n");
//lud_oacc(m, matrix_dim,grid_x,grid_y);

  stopwatch_start(&sw);
//  lud_omp(m, matrix_dim);
  lud_oacc(m, matrix_dim,grid_x,grid_y);
  stopwatch_stop(&sw);
  printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    /* print_matrix(m, matrix_dim); */
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */

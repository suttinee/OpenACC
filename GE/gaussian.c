/*-----------------------------------------------------------
 ** gaussian.c -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

int Size;
double *a, *b, *finalVec;
double *m;

FILE *fp;

#define REPS 1


void wul();

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
inline void Fan1(double *m, double *a, int Size, int t);
inline void Fan2(double *m, double *a, double *b,int Size, int j1, int t);
void InitMat(double *ary, int nrow, int ncol);
void InitAry(double *ary, int ary_size);
void PrintMat(double *ary, int nrow, int ncolumn);
void PrintAry(double *ary, int ary_size);

double totalKernelTime = 0;

int main(int argc, char *argv[])
{
    int verbose = 1;
    if (argc < 2) {
        printf("Usage: gaussian matrix.txt [-q]\n\n");
        printf("-q (quiet) suppresses printing the matrix and result values.\n");
        printf("The first line of the file contains the dimension of the matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf("The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. (optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6	-0.5	0.7	0.3\n");
        printf("-0.3	-0.9	0.3	0.7\n");
        printf("-0.4	-0.5	-0.3	-0.8\n");	
        printf("0.0	-0.1	0.2	0.9\n");
        printf("\n");
        printf("-0.85	-0.68	0.24	-0.53\n");	
        printf("\n");
        printf("0.7	0.0	-0.4	-0.5\n");
        exit(0);
    }
    
    //char filename[100];
    //sprintf(filename,"matrices/matrix%d.txt",size);

    // wake up device
    wul();

int rep = 0;

for (rep = 0; rep < REPS; rep++) {
    InitProblemOnce(argv[1]);
    if (argc > 2) {
        if (!strcmp(argv[2],"-q")) verbose = 0;
    }
    //InitProblemOnce(filename);
    InitPerRun();


    //begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);	
    
    // run kernels
    ForwardSub();
    
    //end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    double time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    
    if (verbose) {
        //printf("Matrix m is: \n");
    //    PrintMat(m, Size, Size);

       // printf("Matrix a is: \n");
      //  PrintMat(a, Size, Size);

        //printf("Array b is: \n");
        //PrintAry(b, Size);
    }
    BackSub();
    if (verbose) {
      //  printf("The final solution is: \n");
        PrintAry(finalVec,Size);
    }
    printf("Time total (including memory transfers)\t%lf s\n", time_total/1e6);
    //printf("Time for CUDA kernels:\t%lf s\n",totalKernelTime/1e6);
    
    /*printf("%d,%d\n",size,time_total);
    fprintf(stderr,"%d,%d\n",size,time_total);*/
/*
    long long nTmp1 = 0;
    long long nTmp2 = 0;
    
    int t = 0;
    for (t=0; t<(Size-1); t++) {
        nTmp1 += (Size-2-t) * (2*Size-2*t+1);
        nTmp2 += (Size-2-t) * (Size-t+1);
    }
*/
    free(m);
    free(a);
    free(b);
}    

return(0);
}

 
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename)
{
	//char *filename = argv[1];
	
	//printf("Enter the data file name: ");
	//scanf("%s", filename);
	//printf("The file name is: %s\n", filename);
	
	fp = fopen(filename, "r");
	
	fscanf(fp, "%d", &Size);	
	 
	a = (double *) malloc(Size * Size * sizeof(double));
	 
	InitMat(a, Size, Size);
	//printf("The input matrix a is:\n");
	//PrintMat(a, Size, Size);
	b = (double *) malloc(Size * sizeof(double));
	
	InitAry(b, Size);
	//printf("The input array b is:\n");
	//PrintAry(b, Size);
		
	 m = (double *) malloc(Size * Size * sizeof(double));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
/*
inline void Fan1(double *m, double *a, int Size, int t)
{   
	int k;
	#pragma acc kernels present(m[0:Size*Size],a[0:(Size*Size)])
	for (k=0; k<Size-1-t; k++)
		m[Size*(k+t+1)+t] = a[Size*(k+t+1)+t] / a[Size*t+t];
}
*/
/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 
/*
inline void Fan2(double *m, double *a, double *b,int Size, int j1, int t)
{
	int i,j;
	#pragma acc kernels present(m[0:Size*Size],a[0:(Size*Size)])
	for (i=0; i<Size-1-t; i++){
		for (j=0; j<Size-t; j++){
			a[Size*(i+1+t)+(j+t)] -= m[Size*(i+1+t)+t] * a[Size*t+(j+t)];
		}
	}
	#pragma acc kernels present(m[0:Size*Size],b[0:Size])//, async(1)
	for (i=0; i<Size-1-t; i++)
		b[i+1+t] -= m[Size*(i+1+t)+t] * b[t];
}
*/
/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub()
{
	int t;

    // begin timing kernels
    struct timeval time_start;
    // end timing kernels
    struct timeval time_end;
#pragma acc data copy(m[0:Size*Size],a[0:Size*Size],b[0:Size])
{
    gettimeofday(&time_start, NULL);
	#ifdef ILP_OPENCL
	 #pragma hmppcg(OPENCL) unroll(8), jam
	#endif
	#ifdef ILP_CUDA
	#pragma hmppcg(CUDA) unroll(8), jam
	#endif
	for (t=0; t<(Size-1); t++) {
		//Fan1(m,a,Size,t);
		//Fan2(m,a,b,Size,Size-t,t);
		int k;
		#ifdef INDEPENDENT
		#pragma acc kernels loop independent 
		#else
		#pragma acc kernels loop
		#endif
		#ifdef TILE
		#pragma hmppcg tile k:256
		#endif
		for (k=0; k<Size-1-t; k++)
			m[Size*(k+t+1)+t] = a[Size*(k+t+1)+t] / a[Size*t+t];

		int i,j;

		#pragma acc kernels
		{
	   	#ifdef INDEPENDENT
		#pragma acc loop independent
		#else
		#pragma acc loop
		#endif
		for (i=0; i<Size-1-t; i++){
			#ifdef INDEPENDENT
			#pragma acc loop independent
			#else
			#pragma acc loop
			#endif
			for (j=0; j<Size-t; j++){
				a[Size*(i+1+t)+(j+t)] -= m[Size*(i+1+t)+t] * a[Size*t+(j+t)];
			}
			b[i+1+t] -= m[Size*(i+1+t)+t] * b[t];
		}
        	}
	}
    gettimeofday(&time_end, NULL);
}
    totalKernelTime = (time_end.tv_sec * 1e6 + time_end.tv_usec) - (time_start.tv_sec * 1e6 + time_start.tv_usec);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
	// create a new vector to hold the final answer
	finalVec = (double *) malloc(Size * sizeof(double));
	// solve "bottom up"
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}

void InitMat(double *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%lf",  ary+Size*i+j);
		}
	}  
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(double *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2lf ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(double *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%lf",  &ary[i]);
	}
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(double *ary, int ary_size)
{
	int i;
	FILE *fpo = fopen("out.txt", "w");
	for (i=0; i<ary_size; i++) {
		fprintf(fpo,"%.2lf \n", ary[i]);
	}
        fclose(fpo);

}




/*
 * This function ensures the device is awake.
 * It is more portable than acc_init().
 */
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
    /*
     * This should never be called as rands should be in the range (0,1].
     * This stops clever optimizers.
     */
  }

  free(arr_a);
  free(arr_b);

}


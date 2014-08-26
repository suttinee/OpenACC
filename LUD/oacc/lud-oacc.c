#include <stdio.h>
//#include <openacc.h>


void lud_oacc(float *a, int size,int grid_x, int grid_y){
     int i,j,k;
     float sum;
#pragma acc data copy(a[0:size*size])
     for (i=0; i <size; i++){
	#ifdef GRID
        #pragma acc kernels loop private(j,k) gang(grid_x) worker(grid_y)
	#else
	#pragma acc kernels loop private(j,k)
	#endif
	 for (j=i; j <size; j++){
	    sum=0;
	#ifdef ILP_OPENCL
	#pragma hmppcg(OPENCL) unroll(8), jam
	#endif
	#ifdef ILP_CUDA
	#pragma hmppcg(CUDA) unroll(8), jam
	#endif
	#ifdef TILE
	#pragma hmppcg tile k:256
	#endif
		for (k=0; k<i; k++){
		 sum += a[i*size+k]*a[k*size+j];}
             a[i*size+j]=a[i*size+j]-sum;
	}
	#ifdef GRID
	#pragma acc kernels loop private(j,k) gang(grid_x) worker(grid_y)
	#else
	#pragma acc kernels loop private(j,k) 
	#endif 
         for (j=i+1;j<size; j++){ 
	    sum=0;
        #ifdef ILP_OPENCL
        #pragma hmppcg(OPENCL) unroll(8), jam
        #endif
        #ifdef ILP_CUDA
        #pragma hmppcg(CUDA) unroll(8), jam
        #endif
        #ifdef TILE
        #pragma hmppcg tile k:256
        #endif	     
		for (k=0; k<i; k++) {
			sum +=a[j*size+k]*a[k*size+i];}
             a[j*size+i]=(a[j*size+i]-sum)/a[i*size+i];
         }
     }
}

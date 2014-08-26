#!/bin/bash
INPUT=../data/bfs/graph1MW_6.txt 

export ACC_DEVICE_TYPE=acc_device_cuda

echo "============================== OpenACC taget CUDA on GPU================================"

	printf "Directive orgranization + Restructuring optimizations---" && ./bfs_organ_CUDA $INPUT 
        printf "Independent directive optimization----------------------" && ./bfs_inde_CUDA $INPUT
	printf "ILP optimizations---------------------------------------" && ./bfs_ilp_CUDA $INPUT
        printf "Grid thread optimization (Default=32x4)-----------------" && ./bfs_grid_CUDA $INPUT
        printf "Various Compiler----------------------------------------" && ./bfs_icc_CUDA $INPUT
        printf "Compiler flag options-----------------------------------" && ./bfs_fastmath_CUDA $INPUT

echo "========================================================================================"

export ACC_DEVICE_TYPE=acc_device_opencl

echo "============================ OpenACC taget OpenCL on GPU==============================="=
	printf "Directive orgranization + Restructuring optimizations---" && ./bfs_organ_OPENCL $INPUT 
        printf "Independent directive optimization----------------------" && ./bfs_inde_OPENCL $INPUT
	printf "ILP optimizations---------------------------------------" && ./bfs_ilp_OPENCL $INPUT
        printf "Grid thread optimization (Default=32x4)-----------------" && ./bfs_grid_OPENCL $INPUT
        printf "Various Compiler----------------------------------------" && ./bfs_icc_OPENCL $INPUT
        printf "Compiler flag options-----------------------------------" && ./bfs_fastmath_OPENCL $INPUT
echo "========================================================================================"

export OPENCL_INC_PATH=/opt/intel/opencl/include
export HMPPRT_NO_FALLBACK=1
export ACC_DEVICE_TYPE=acc_device_opencl
export HMPPRT_OPENCL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR

echo "============================ OpenACC taget OpenCL on MIC ==============================="
	printf "Directive orgranization + Restructuring optimizations---" && ./bfs_organ_OPENCL $INPUT 
        printf "Independent directive optimization----------------------" && ./bfs_inde_OPENCL $INPUT
	printf "ILP optimizations---------------------------------------" && ./bfs_ilp_OPENCL $INPUT
        printf "Grid thread optimization (Default=32x4)-----------------" && ./bfs_grid_OPENCL $INPUT
        printf "Various Compiler----------------------------------------" && ./bfs_icc_OPENCL $INPUT
        printf "Compiler flag options-----------------------------------" && ./bfs_fastmath_OPENCL $INPUT
echo "========================================================================================"



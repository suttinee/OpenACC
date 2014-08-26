export LANG=en_US.UTF-8
export LC_ALL=en_US

GRIDX=32
GRIDY=4

TARGET=OPENCL

FASTMATH=-fimf-precision=low -fimf-domain-exclusion=31 -fimf-accuracy-bits=11 -no-prec-div -no-prec-sqrt -fp-model fast=2
NVCC=--nvcc-options -Xptxas=-v,-prec-div=false,-arch=compute_35,-code=sm_35
GRIDOPT= -Xhmppcg -grid-block-size,$(GRIDX)x$(GRIDY)
OAC = capsmc
CFLAGS = 
LFLAGS = -fopenmp -lm
ARCH = GPU
CC = gcc

default:
	make organize && make independent && make ilp && make grid && make icc && make fastmath

organize: 
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -o bfs_organ_$(TARGET) bfs.c
	cp bfs_* ../bin/bfs/organ
	cp __h* ../bin/bfs/organ
	rm __h* bfs_*
independent: 
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -o bfs_inde_$(TARGET) bfs.c 
	cp bfs_* ../bin/bfs/inde
	cp __h* ../bin/bfs/inde
	rm __h* bfs_*
ilp:
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -DILP -o bfs_ilp_$(TARGET) bfs.c
	cp bfs_* ../bin/bfs/ilp
	cp __h* ../bin/bfs/ilp  
	rm __h* bfs_*
grid:
	$(OAC) $(GRIDOPT) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -DILP -o bfs_grid_$(TARGET) bfs.c
	cp bfs_* ../bin/bfs/grid
	cp __h* ../bin/bfs/grid
	rm __h* bfs_*
icc:
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc  $(CFLAGS) -DINDEPENDENT -DILP -o bfs_icc_$(TARGET) bfs.c
	cp bfs_* ../bin/bfs/icc
	cp __h* ../bin/bfs/icc 
	rm __h* bfs_*
fastmath:
	$(OAC) $(NVCC) $(GRIDOPT) --target=$(TARGET) icc  $(CFLAGS) -DINDEPENDENT $(FASTMATH) -DILP -o bfs_fastmath_$(TARGET) bfs.c
	cp bfs_* ../bin/bfs/fastmath
	cp __h* ../bin/bfs/fastmath
	rm __h* bfs_*

.PHONY: clean

clean:
	rm -f *.o bfs bfs_* __hmpp* *.lst *.cub *.ptx
	rm ../bin/bfs/organ/* ../bin/bfs/inde/* ../bin/bfs/ilp/* ../bin/bfs/grid/* ../bin/bfs/icc/* ../bin/bfs/fastmath/*

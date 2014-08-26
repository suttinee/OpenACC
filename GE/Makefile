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
	make organize && make independent && make ilp && make grid && make icc && make fastmath && make tile

organize: 
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -o ge_organ_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/organ
	cp __h* ../bin/ge/organ
	rm __h* ge_*
independent: 
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -o ge_inde_$(TARGET) gaussian.c 
	cp ge_* ../bin/ge/inde
	cp __h* ../bin/ge/inde
	rm __h* ge_*
ilp:
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -DILP_$(TARGET) -o ge_ilp_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/ilp
	cp __h* ../bin/ge/ilp  
	rm __h* ge_*
tile:
	$(OAC) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -DTILE -o ge_tile_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/tile
	cp __h* ../bin/ge/tile 
	rm __h* ge_*

grid:
	$(OAC) $(GRIDOPT) --target=$(TARGET) $(CC) $(CFLAGS) -DINDEPENDENT -DILP_$(TARGET) -o ge_grid_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/grid
	cp __h* ../bin/ge/grid
	rm __h* ge_*
icc:
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc  $(CFLAGS) -DINDEPENDENT -DILP_$(TARGET) -o ge_icc_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/icc
	cp __h* ../bin/ge/icc 
	rm __h* ge_*
fastmath:
	$(OAC) $(NVCC) $(GRIDOPT) --target=$(TARGET) icc  $(CFLAGS) -DINDEPENDENT $(FASTMATH) -DILP_$(TARGET) -o ge_fastmath_$(TARGET) gaussian.c
	cp ge_* ../bin/ge/fastmath
	cp __h* ../bin/ge/fastmath
	rm __h* ge_*

.PHONY: clean

clean:
	rm -f *.o bfs ge_* __hmpp* *.lst *.cub *.ptx
	rm ../bin/ge/organ/* ../bin/ge/inde/* ../bin/ge/ilp/* ../bin/ge/grid/* ../bin/ge/icc/* ../bin/ge/fastmath/* ../bin/ge/tile/*

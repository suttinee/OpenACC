export LANG=en_US.UTF-8
export LC_ALL=en_US

GRIDX=32
GRIDY=4

TARGET=OPENCL

FASTMATH=-fimf-precision=low -fimf-domain-exclusion=31 -fimf-accuracy-bits=11 -no-prec-div -no-prec-sqrt -fp-model fast=2
NVCC=--nvcc-options -Xptxas=-v,-prec-div=false,-arch=compute_35,-code=sm_35
GRIDOPT= -Xhmppcg -grid-block-size,$(GRIDX)x$(GRIDY)
OAC =hmpp 
CC_FLAGS = 
LFLAGS = -fopenmp -lm 
ARCH = GPU 
CC = gcc 


default:
	make organize && make independent && make redu && make ilp && make grid && make icc && make fastmath

organize: backprop_organ.o facetrain.o imagenet.o backprop_kernel.o 
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_organ_$(TARGET) -lm
	cp bp_* ../bin/bp/organ
	cp __h* ../bin/bp/organ
	rm *.o bp_* __h*
independent: backprop_inde.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_inde_$(TARGET) -lm
	cp bp_* ../bin/bp/inde
	cp __h* ../bin/bp/inde
	rm *.o bp_* __h*
ilp: backprop_ilp.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DILP backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_ilp_$(TARGET) -lm
	cp bp_* ../bin/bp/ilp
	cp __h* ../bin/bp/ilp
	rm *.o bp_* __h*
redu: backprop_redu.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DREDUCTION backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_redu_$(TARGET) -lm
	cp bp_* ../bin/bp/redu
	cp __h* ../bin/bp/redu
	rm *.o bp_* __h*
grid: backprop_grid.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) $(GRIDOPT) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DILP backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_grid_$(TARGET) -lm
	cp bp_* ../bin/bp/grid
	cp __h* ../bin/bp/grid
	rm *.o bp_* __h*
icc: backprop_icc.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc $(CC_FLAGS) -DINDEPENDENT -DILP backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_icc_$(TARGET) -lm
	cp bp_* ../bin/bp/icc
	cp __h* ../bin/bp/icc
	rm *.o bp_* __h*
fastmath:backprop_fastmath.o facetrain.o imagenet.o backprop_kernel.o
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc $(CC_FLAGS) -DINDEPENDENT -DILP $(FASTMATH) backprop.o facetrain.o imagenet.o backprop_kernel.o -o bp_fastmath_$(TARGET) -lm
	cp bp_* ../bin/bp/fastmath
	cp __h* ../bin/bp/fastmath
	rm *.o bp_* __h* 
	
#%.o: %.[ch]
#	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) facetrain.c -c
	
backprop_organ.o: backprop.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) backprop.c -c
backprop_inde.o: backprop.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT backprop.c -c
backprop_ilp.o: backprop.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DILP backprop.c -c
backprop_redu.o: backprop.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DREDUCTION backprop.c -c
backprop_grid.o: backprop.c backprop.h
	$(OAC) $(GRIDOPT) --target=$(TARGET) $(CC) $(CC_FLAGS) -DINDEPENDENT -DILP backprop.c -c
backprop_icc.o: backprop.c backprop.h
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc $(CC_FLAGS) -DINDEPENDENT -DILP backprop.c -c
backprop_fastmath.o:  backprop.c backprop.h
	$(OAC) $(GRIDOPT) --target=$(TARGET) icc $(CC_FLAGS) -DINDEPENDENT -DILP $(FASTMATH) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) imagenet.c -c

install:
	cp bp_* ../bin


clean:
	rm -f *.o *~ backprop bp_*  backprop_cuda.linkinfo *.hmf* *.i *.cl *.fatbin *.cu *.dpil
	rm ../bin/bp/fastmath/*
	rm ../bin/bp/icc/*
	rm ../bin/bp/grid/*
	rm ../bin/bp/redu/*
	rm ../bin/bp/ilp/*
	rm ../bin/bp/inde/*
	rm ../bin/bp/organ/* 

export LANG=en_US.UTF-8
export LC_ALL=en_US

GRIDX=192
GRIDY=15

TARGET=OPENCL

FASTMATH=-fimf-precision=low -fimf-domain-exclusion=31 -fimf-accuracy-bits=11 -no-prec-div -no-prec-sqrt -fp-model fast=2
NVCC=--nvcc-options -Xptxas=-v,-prec-div=false,-arch=compute_35,-code=sm_35
OAC =hmpp 
CC_FLAGS = 
LFLAGS = -fopenmp -lm 
ARCH = GPU 
CC = gcc 


default:
	make organize && make tile && make ilp && make grid && make icc && make fastmath

organize: lud-oacc_organ.o ../common/common.o lud.o 
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) ../common/common.o lud-oacc.o lud.o -o lud_organ_$(TARGET) -lm
	cp lud_* ../../bin/lud/organ
	cp __h* ../../bin/lud/organ
	rm *.o lud_* __h*
ilp: lud-oacc_ilp.o ../common/common.o lud.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS)  -DILP_$(TARGET) ../common/common.o lud-oacc.o lud.o -o lud_ilp_$(TARGET) -lm
	cp lud_* ../../bin/lud/ilp
	cp __h* ../../bin/lud/ilp
	rm *.o lud_* __h*
tile: lud-oacc_tile.o ../common/common.o lud.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DTILE ../common/common.o lud-oacc.o lud.o -o lud_tile_$(TARGET) -lm
	cp lud_* ../../bin/lud/tile
	cp __h* ../../bin/lud/tile
	rm *.o lud_* __h*
grid: lud-oacc_grid.o ../common/common.o lud.o
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) -DGRID   -DILP_$(TARGET) ../common/common.o lud-oacc.o lud.o -o lud_grid_$(TARGET) -lm
	cp lud_* ../../bin/lud/grid
	cp __h* ../../bin/lud/grid
	rm *.o lud_* __h*
icc: lud-oacc_icc.o ../common/common.o lud.o
	$(OAC)  --target=$(TARGET) icc $(CC_FLAGS) -DGRID   -DILP_$(TARGET) ../common/common.o lud-oacc.o lud.o -o lud_icc_$(TARGET) -lm
	cp lud_* ../../bin/lud/icc
	cp __h* ../../bin/lud/icc
	rm *.o lud_* __h*
fastmath:lud-oacc_fastmath.o ../common/common.o lud.o
	$(OAC)  --target=$(TARGET) icc $(CC_FLAGS) -DGRID  -DILP_$(TARGET) $(FASTMATH) ../common/common.o lud-oacc.o lud.o -o lud_fastmath_$(TARGET) -lm
	cp lud_* ../../bin/lud/fastmath
	cp __h* ../../bin/lud/fastmath
	rm *.o lud_* __h* 
	
#%.o: %.[ch]
#	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) $< -c

	
lud-oacc_organ.o: lud-oacc.c
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS) lud-oacc.c -c
lud-oacc_ilp.o: lud-oacc.c
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS)  -DILP_$(TARGET) lud-oacc.c -c
lud-oacc_tile.o: lud-oacc.c
	$(OAC) --target=$(TARGET) $(CC) $(CC_FLAGS)  -DTILE lud-oacc.c -c
lud-oacc_grid.o: lud-oacc.c
	$(OAC)  --target=$(TARGET) $(CC) $(CC_FLAGS) -DGRID  -DILP_$(TARGET) lud-oacc.c -c
lud-oacc_icc.o: lud-oacc.c
	$(OAC)  --target=$(TARGET) icc $(CC_FLAGS) -DGRID  -DILP_$(TARGET) lud-oacc.c -c
lud-oacc_fastmath.o:  lud-oacc.c
	$(OAC)  --target=$(TARGET) icc $(CC_FLAGS) -DGRID  -DILP_$(TARGET) $(FASTMATH) lud-oacc.c -c

lud.o: lud-oacc.c lud.c
	$(CC) $(CC_FLAGS) lud.c lud-oacc.c -c 
common.o: ../common/common.c ../common/common.h
	$(CC) $(CC_FLAGS) ../common/common.c ../common/common.h -c





clean:
	rm -f *.o *~ backprop lud_*  lud-oacc_cuda.linkinfo *.hmf* *.i *.cl *.fatbin *.cu *.dpil
	rm ../../bin/lud/fastmath/*
	rm ../../bin/lud/grid/*
	rm ../../bin/lud/icc/*
	rm ../../bin/lud/ilp/*
	rm ../../bin/lud/organ/*
	rm ../../bin/lud/tile/* 

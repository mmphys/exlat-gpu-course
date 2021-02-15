# 
# Makefile for reverse.cu exercise
#

MF= Makefile
CP= cp

EXE= reverse reduce

SRC= reverse.cu reduce.cu

ARCH=sm_60

CC=	nvcc
CFLAGS= -O3 -arch=$(ARCH)

.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)

.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

default: $(EXE)

reverse: reverse.o
	$(CC) $(CFLAGS) -o $@ $@.o

reduce: reduce.o
	$(CC) $(CFLAGS) -o $@ $@.o

$(EXE):	$(OBJ) 

$(OBJ):	$(SRC) $(MF) 

clean:
	rm -f $(OBJ) $(EXE) core

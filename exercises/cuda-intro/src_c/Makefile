# 
# Makefile for cuda-intro
#

MF= Makefile

EXE= intro intro_managed
SRC= intro.cu intro_managed.cu

ARCH=sm_60

CC=	nvcc
CFLAGS= -O3 -arch=$(ARCH)

default: $(EXE)


.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)

.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

$(EXE):	$(OBJ) 

intro: intro.o
	$(CC) $(CFLAGS) -o $@ $@.o

intro_managed: intro_managed.o
	$(CC) $(CFLAGS) -o $@ $@.o 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) core

NAME:=$(shell uname)

ifeq ($(NAME), Linux)
CC=g++
CCFLAGS=-Wall -I/opt/rocm/opencl/include
LDFLAGS=-L/opt/rocm/opencl/lib/x86_64 -lOpenCL -lfreeimage
else ifeq ($(NAME), Darwin)
CC=clang++
CCFLAGS=-g -framework OpenCL
LDFLAGS=-lfreeimage
endif
ifeq ($(DEBUG), 1)
CCFLAGS += -D_DEBUG
endif

SRC=$(wildcard *.cpp)
TARGETS=$(patsubst %.cpp, %, $(SRC))


all: $(TARGETS)

$(TARGETS):%:%.cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@rm -rf *.o $(TARGETS) *.dSYM .DS_Store

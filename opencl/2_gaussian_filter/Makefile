NAME:=$(shell uname)
TARGET=gaussian_filter

ifeq ($(NAME), Linux)
CC=g++
CCFLAGS=-g -Wall -I/opt/rocm/opencl/include
LDFLAGS=-L/opt/rocm/opencl/lib/x86_64 -lOpenCL -lfreeimage
else ifeq ($(NAME), Darwin)
CC=clang++
CCFLAGS=-g -framework OpenCL
LDFLAGS=-lfreeimage
endif

ifeq ($(DEBUG), 1)
CCFLAGS += -D_DEBUG
endif

$(TARGET): $(TARGET).cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@rm -f *.o $(TARGET) 2.png
run:
	./$(TARGET) 1.png 2.png

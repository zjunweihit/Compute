NAME:=$(shell uname)
TARGET=MatrixAdd

ifeq ($(NAME), Linux)
CC=g++
CCFLAGS=-Wall -I/opt/rocm/opencl/include
LDFLAGS=-L/opt/rocm/opencl/lib/x86_64 -lOpenCL
else ifeq ($(NAME), Darwin)
CC=clang++
CCFLAGS=-g -framework OpenCL
LDFLAGS=
endif

ifeq ($(DEBUG), 1)
CCFLAGS += -D_DEBUG
endif

$(TARGET): $(TARGET).cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@rm -f *.o $(TARGET)
run:
	./$(TARGET)


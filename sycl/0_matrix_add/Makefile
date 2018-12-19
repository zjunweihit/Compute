TARGET=MatrixAdd

CC=clang
CCFLAGS= -g -Wall -Wno-ignored-attributes -std=c++17
DEFINITION=-DTRISYCL_OPENCL -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION -DBOOST_COMPUTE_HAVE_THREAD_LOCAL -DBOOST_COMPUTE_THREAD_SAFE
INCLUDE=-I/opt/rocm/opencl/include/ -I../../../triSYCL/include -I../../../triSYCL/tests/common
LDFLAGS=-L/opt/rocm/opencl/lib/x86_64 -lOpenCL -lm -lstdc++ -lpthread -lboost_log

ifeq ($(DEBUG), 1)
CCFLAGS += -D_DEBUG
endif

$(TARGET): $(TARGET).cpp
	$(CC) $(INCLUDE) $(DEFINITION) $(CCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@rm -f *.o $(TARGET)
run:
	./$(TARGET)


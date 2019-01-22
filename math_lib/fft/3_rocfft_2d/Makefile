HIP_PATH?=$(wildcard /opt/rocm/hip)

HIPCC=$(HIP_PATH)/bin/hipcc
HIPCC_FLAGS=-g -lrocfft -I /opt/rocm/include/ -L /opt/rocm/lib/

TARGET_CC=2d_cpp
SRC_CC=rocfft_2d_hipcc.cpp

TARGET_C=2d_c
SRC_C=rocfft_2d_hipcc.c

TARGET_FFTW_CC=2d_fftw_cpp
SRC_FFTW_CC=rocfft_2d_fftw.cpp
CC=g++
CC_FLAGS=-std=c++11 -D__HIP_PLATFORM_HCC__
CC_INC=-I/opt/rocm/include
CC_LIBS=-L/opt/rocm/lib -lhip_hcc -lrocfft -lfftw3f

# gcc has no HCC backend
TARGET_FFTW_C=2d_fftw_c
SRC_FFTW_C=rocfft_2d_fftw.c

TARGETS=$(TARGET_CC) $(TARGET_C) $(TARGET_FFTW_CC) $(TARGET_FFTW_C)

all: $(TARGETS)

$(TARGET_CC): $(SRC_CC)
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(TARGET_C): $(SRC_C)
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(TARGET_FFTW_CC): $(SRC_FFTW_CC)
	$(CC) $(CC_FLAGS) $(CC_INC) $< -o $@ $(CC_LIBS)

$(TARGET_FFTW_C): $(SRC_FFTW_C)
	$(CC) $(CC_FLAGS) $(CC_INC) $< -o $@ $(CC_LIBS)

clean:
	rm -f *.o $(TARGETS)
run-c:
	./$(TARGET_C)
run-cpp:
	./$(TARGET_CC)

HIP_PATH?=$(wildcard /opt/rocm/hip)

HIPCC=$(HIP_PATH)/bin/hipcc
HIPCC_FLAGS=-g -lrocfft -I /opt/rocm/include/ -L /opt/rocm/lib/

TARGET_CC=3d_cpp
SRC_CC=rocfft_3d_hipcc.cpp

TARGET_FFTW_CC=3d_fftw_cpp
SRC_FFTW_CC=rocfft_3d_fftw.cpp
CC=g++
CC_FLAGS=-std=c++11 -D__HIP_PLATFORM_HCC__
CC_INC=-I/opt/rocm/include
CC_LIBS=-L/opt/rocm/lib -lhip_hcc -lrocfft -lfftw3f

TARGETS=$(TARGET_CC) $(TARGET_FFTW_CC)

all: $(TARGETS)

$(TARGET_CC): $(SRC_CC)
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(TARGET_FFTW_CC): $(SRC_FFTW_CC)
	$(CC) $(CC_FLAGS) $(CC_INC) $< -o $@ $(CC_LIBS)

clean:
	rm -f *.o $(TARGETS)
run-fftw:
	./$(TARGET_FFTW_CC)
run:
	./$(TARGET_CC)

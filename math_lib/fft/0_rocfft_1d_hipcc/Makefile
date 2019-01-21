TARGET=rocfft_1d_hipcc

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g -lrocfft -I /opt/rocm/include/ -L /opt/rocm/lib/

$(TARGET): rocfft_1d_hipcc.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

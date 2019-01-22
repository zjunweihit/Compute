TARGET=rocfft_3d_hipcc

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g -lrocfft -I /opt/rocm/include/ -L /opt/rocm/lib/

$(TARGET): $(TARGET).cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

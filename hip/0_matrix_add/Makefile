TARGET=MatrixAdd

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g

$(TARGET): MatrixAdd.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

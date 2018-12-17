TARGET=Stream

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g

$(TARGET): $(TARGET).cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

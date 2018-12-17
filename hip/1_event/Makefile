TARGET=Event

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g

$(TARGET): Event.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

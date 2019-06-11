TARGET=default_signal
C_FLAGS=-g
HSA_LIB_PATH=/opt/rocm/lib
HSA_INC_PATH=/opt/rocm/hsa/include/hsa

# Below is different header, seems old...
#HSA_INC_PATH=/opt/rocm/include

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	g++ $(C_FLAGS) $< -o $@ -I$(HSA_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64 -pthread

clean:
	rm -f $(TARGET) *.o

dev: clean $(TARGET)

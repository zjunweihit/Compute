CLANG=clang-7

TARGET=vector_add
C_FLAGS=-g
HSA_LIB_PATH=/opt/rocm/lib
HSA_INC_PATH=/opt/rocm/hsa/include/hsa

# Below is different header, seems old...
#HSA_INC_PATH=/opt/rocm/include

all: $(TARGET) $(TARGET).co

$(TARGET): $(TARGET).c
	gcc $(C_FLAGS) $< -o $@ -I$(HSA_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64

$(TARGET).co: $(TARGET).s
	$(CLANG) -target amdgcn--amdhsa -mcpu=gfx906 $< -o $@

clean:
	rm -f $(TARGET) *.o *.co

dev: clean $(TARGET)

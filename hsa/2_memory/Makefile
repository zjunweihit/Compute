TARGET=get_mem_info

C_FLAGS=-g
HSA_LIB_PATH=/opt/rocm/lib
HSA_INC_PATH=/opt/rocm/hsa/include/hsa
G_INC_PATH=../global
SRC=$(TARGET).c ../global/global.c

all: $(TARGET)

$(TARGET): $(SRC)
	gcc $(C_FLAGS) $(SRC) -o $@ -I$(HSA_INC_PATH) -I$(G_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64

clean:
	rm -f $(TARGET) *.o
run:
	./$(TARGET)

dev: clean $(TARGET) run

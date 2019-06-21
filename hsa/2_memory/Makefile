TARGET1=get_mem_info
TARGET2=alloc_mem

TARGET=$(TARGET1) $(TARGET2)

C_FLAGS=-g
HSA_LIB_PATH=/opt/rocm/lib
HSA_INC_PATH=/opt/rocm/hsa/include/hsa
G_INC_PATH=../global
SRC1=$(TARGET1).c ../global/global.c
SRC2=$(TARGET2).c ../global/global.c

all: $(TARGET)

$(TARGET1): $(SRC1)
	gcc $(C_FLAGS) $(SRC1) -o $@ -I$(HSA_INC_PATH) -I$(G_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64

$(TARGET2): $(SRC2)
	gcc $(C_FLAGS) $(SRC2) -o $@ -I$(HSA_INC_PATH) -I$(G_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64

clean:
	rm -f $(TARGET) *.o
run:
	./$(TARGET)

dev: clean $(TARGET) run

SRC=$(wildcard *.c)
TARGETS=$(patsubst %.c, %, $(SRC))

C_FLAGS=-g -Wall
HSA_LIB_PATH=/opt/rocm/lib
HSA_INC_PATH=/opt/rocm/hsa/include/hsa
G_INC_PATH=global
G_SRC=global/global.c

all: $(TARGETS)

$(TARGETS):%:%.c
	gcc $(C_FLAGS) $< $(G_SRC) -o $@ -I$(HSA_INC_PATH) -I$(G_INC_PATH) -L$(HSA_LIB_PATH) -lhsa-runtime64

clean:
	rm -f $(TARGETS) *.o
.PHONY: clean all

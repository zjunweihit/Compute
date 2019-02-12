OBJ=sgemm_tt
SRC=$(OBJ).cpp

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

ROCM_INCLUDE=/opt/rocm/include/
HIPBLAS_LIB_PATH=/opt/rocm/lib/
HIPBLAS_LIB_NAME=hipblas

DEBUG_OPTION=-g

HIPCC_FLAGS=$(DEBUG_OPTION) -l$(HIPBLAS_LIB_NAME) -I $(ROCM_INCLUDE) -L $(HIPBLAS_LIB_PATH)

$(OBJ): $(SRC)
	$(HIPCC) $< -o $@ $(HIPCC_FLAGS)

clean:
	@rm -f *.o $(OBJ)
run:
	./$(OBJ)

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipInfo

$(EXE): hipInfo.cpp
	$(HIPCC) hipInfo.cpp -o $@
clean:
	rm -f *.o $(EXE)

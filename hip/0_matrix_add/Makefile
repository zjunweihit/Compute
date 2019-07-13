SRC=$(wildcard *.cpp)
TARGETS=$(patsubst %.cpp, %, $(SRC))

HIP_PATH?=$(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS=-g

all: $(TARGETS)

$(TARGETS):%:%.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	@rm -f *.o $(TARGETS)

.PHONY: clean all

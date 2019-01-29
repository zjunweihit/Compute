TARGET=ring

MPI_NP=8
MPICC=mpicc
MPICC_FLAGS=-g

$(TARGET): $(TARGET).c
	$(MPICC) $(MPICC_FLAGS) $< -o $@

clean:
	@rm -f *.o $(TARGET)
run:
	mpirun -n $(MPI_NP) ./$(TARGET)
dev: clean $(TARGET) run

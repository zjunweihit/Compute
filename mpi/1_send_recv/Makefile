TARGET=send_recv

MPICC=mpicc
MPICC_FLAGS=-g

$(TARGET): $(TARGET).c
	$(MPICC) $(MPICC_FLAGS) $< -o $@

clean:
	@rm -f *.o $(TARGET)
run:
	mpirun -n 2 ./$(TARGET)
dev: clean $(TARGET) run

TARGET=hello

CC=mpicc
C_FLAGS= -g

$(TARGET): $(TARGET).c
	$(CC) $(CC_FLAGS) $< -o $@

clean:
	@rm -f *.o $(TARGET)
run:
	mpirun -n 2 ./$(TARGET)

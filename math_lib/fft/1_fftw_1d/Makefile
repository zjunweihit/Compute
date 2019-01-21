TARGET=fftw_1d

CXX=g++

CXX_FLAGS=-g -std=c++11
CXX_LIBS=-g -lfftw3f

$(TARGET): $(TARGET).cpp
	$(CXX) $(CXX_FLAGS) $< -o $@ $(CXX_LIBS)

clean:
	rm -f *.o $(TARGET)
run:
	./$(TARGET)

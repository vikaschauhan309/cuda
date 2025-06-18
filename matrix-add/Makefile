CUDA_FLAGS = -Xcompiler -fPIC
CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart
GTEST_FLAGS = -lgtest -lgtest_main -lpthread

all: test_matrix_add

matrix_add.o: src/matrix_add.cu src/matrix_add.hpp
	nvcc $(CUDA_FLAGS) -c src/matrix_add.cu -o matrix_add.o

test_matrix_add: tests/test_matrix_add.cpp matrix_add.o
	g++ -std=c++17 -I./src tests/test_matrix_add.cpp matrix_add.o -o test_matrix_add $(GTEST_FLAGS) $(CUDA_LIB)

clean:
	rm -f *.o test_matrix_add

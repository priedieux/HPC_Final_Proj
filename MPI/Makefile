MPICC = mpicc
CFLAGS = -O3 -Wall -lm
TARGET = image_proc_mpi.exe

all: $(TARGET)

$(TARGET): image_proc_mpi.c
	$(MPICC) $(CFLAGS) -o $(TARGET) image_proc_mpi.c

test: $(TARGET)
	@echo "Testing MPI implementation..."
	mpirun -np 4 ./$(TARGET) test_gradient_medium.ppm output_mpi.ppm grayscale

clean:
	rm -f $(TARGET) *.o

.PHONY: all test clean
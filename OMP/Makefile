CC = gcc
CFLAGS = -O3 -fopenmp -Wall -Wextra -lm
TARGET = image_proc.exe

all: $(TARGET)

$(TARGET): image_proc_omp.c
	$(CC) $(CFLAGS) -o $(TARGET) image_proc_omp.c

test: $(TARGET)
	@echo "Generating test images..."
	python3 gen_img.py
	@echo "\nRunning tests..."
	./$(TARGET) test_gradient_small.ppm out_gray_1t.ppm grayscale 1
	./$(TARGET) test_gradient_small.ppm out_gray_4t.ppm grayscale 4
	./$(TARGET) test_gradient_small.ppm out_blur_1t.ppm blur 1
	./$(TARGET) test_gradient_small.ppm out_blur_4t.ppm blur 4

benchmark: $(TARGET)
	@echo "Running benchmark with different thread counts..."
	@for threads in 1 2 4 8 16; do \
		echo "\n=== $$threads threads ==="; \
		./$(TARGET) test_gradient_large.ppm output_$$threads.ppm blur $$threads; \
	done

clean:
	rm -f $(TARGET) *.o *.ppm

.PHONY: all test benchmark clean
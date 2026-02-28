# Makefile for Image Blur CUDA programs

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_50

# Targets
ORIGINAL = blur
OPTIMIZED = blur_optimized

# Source files
ORIGINAL_SRC = blur.cu
OPTIMIZED_SRC = blur_optimized.cu

# Default target - build both
all: $(ORIGINAL) $(OPTIMIZED)

# Build original version
$(ORIGINAL): $(ORIGINAL_SRC)
	$(NVCC) $(NVCCFLAGS) $(ORIGINAL_SRC) -o $(ORIGINAL)

# Build optimized version
$(OPTIMIZED): $(OPTIMIZED_SRC)
	$(NVCC) $(NVCCFLAGS) $(OPTIMIZED_SRC) -o $(OPTIMIZED)

# Clean build artifacts
clean:
	rm -f $(ORIGINAL) $(OPTIMIZED) blurred.bmp blurred_optimized.bmp

# Test original version
test-original: $(ORIGINAL)
	./$(ORIGINAL) grumpy.bmp

# Test optimized version  
test-optimized: $(OPTIMIZED)
	./$(OPTIMIZED) grumpy.bmp

# Test both versions for comparison
test-both: $(ORIGINAL) $(OPTIMIZED)
	@echo "=== Testing Original Version ==="
	./$(ORIGINAL) grumpy.bmp
	@echo ""
	@echo "=== Testing Optimized Version ==="
	./$(OPTIMIZED) grumpy.bmp

.PHONY: all clean test-original test-optimized test-both

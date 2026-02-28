# Project 2 Part 2: Image Blur Optimization

## Optimization Strategy

### Problem with Original Implementation
The original blur kernel accesses **global memory** repeatedly for each pixel's blur computation. For a blur size of 16, each pixel requires **(2×16+1)² = 1089 global memory accesses**. Global memory is slow (hundreds of cycles latency).

### Shared Memory Optimization Strategy

**Key Idea:** Load image data into fast **shared memory** once, then reuse it for all blur computations.

#### Strategy Details:

1. **Tile-Based Processing**
   - Divide the image into tiles (16×16 pixels per block)
   - Each tile is loaded into shared memory

2. **Halo Region Loading**
   - For blur computation, each pixel needs neighboring pixels within BLUR_SIZE radius
   - Load extra "halo" pixels around the tile boundaries
   - Total shared memory tile size: (16 + 2×16)² = 48×48 pixels

3. **Parallel Loading**
   - Each thread loads:
     - Its primary pixel
     - Halo pixels (top, bottom, left, right, corners) if on tile edge
   - All threads synchronize after loading (__syncthreads())

4. **Fast Computation**
   - After sync, each thread computes blur using **shared memory** instead of global memory
   - Shared memory is ~100x faster than global memory
   - Reduces global memory accesses from 1089 per pixel to just 1

5. **Edge Handling**
   - Pixels outside image boundaries are set to black (0,0,0)
   - Ensures valid blur computation at image edges

### Expected Performance Improvement

**Memory Access Reduction:**
- Original: 1089 global memory accesses per pixel
- Optimized: 1 global memory load + fast shared memory accesses

**Theoretical Speedup:**
- Depends on memory bandwidth vs. compute ratio
- Expected: 2-5x speedup for typical images
- Larger images benefit more (better memory reuse)

### Implementation Challenges

1. **Complex Halo Loading Logic**
   - Must load top, bottom, left, right halos
   - Must load 4 corner halos
   - Each requires boundary checks

2. **Shared Memory Size Limits**
   - Shared memory per block is limited (~48KB)
   - Our tile uses: 48×48×3 bytes = 6.9 KB (well within limits)

3. **Synchronization**
   - Must sync after loading before computation
   - Adds small overhead but necessary for correctness

### Performance Analysis

The program outputs timing for both versions:
- **Original (Global Memory):** Baseline time
- **Optimized (Shared Memory):** Improved time
- **Speedup:** Ratio of improvement

Tested on images of different sizes to show how performance scales.

## Building and Running
```bash
# Compile both versions
make

# Test optimized version
./blur_optimized grumpy.bmp

# Compare both versions
make test-both
```

## Performance Testing

Test with images of different sizes:
1. Small (200×200)
2. Medium (800×800) - grumpy.bmp
3. Large (2000×2000)

Record times and create graphs showing:
- Time vs. Image Size
- Speedup vs. Image Size

## Results

[Add your performance graphs and analysis here]

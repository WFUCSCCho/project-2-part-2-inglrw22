#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>

#define BLUR_SIZE 16 // size of surrounding image is 2X this
#define TILE_SIZE 16 // block dimension
#define SHARED_SIZE (TILE_SIZE + 2 * BLUR_SIZE) // tile size + halo region

#include "bitmap_image.hpp"

using namespace std;

/**
 * Optimized Blur Kernel using Shared Memory
 * 
 * Strategy:
 * 1. Load a tile of the image into shared memory (including halo/border pixels)
 * 2. Synchronize threads to ensure all data is loaded
 * 3. Each thread computes blur using fast shared memory instead of slow global memory
 * 4. This reduces global memory accesses from (2*BLUR_SIZE+1)^2 per pixel to just 1
 */
__global__ void blurKernelOptimized(uchar3 *in, uchar3 *out, int width, int height) {
    
    // Shared memory tile with halo region for neighboring pixels
    __shared__ uchar3 sharedTile[SHARED_SIZE][SHARED_SIZE];
    
    // Global position of this thread's output pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local position within the block
    int localCol = threadIdx.x;
    int localRow = threadIdx.y;
    
    // Load main tile data into shared memory
    // Each thread loads one pixel from global memory to shared memory
    int sharedCol = localCol + BLUR_SIZE;
    int sharedRow = localRow + BLUR_SIZE;
    
    if (col < width && row < height) {
        sharedTile[sharedRow][sharedCol] = in[row * width + col];
    } else {
        // Out of bounds - set to black
        sharedTile[sharedRow][sharedCol] = {0, 0, 0};
    }
    
    // Load halo region (border pixels needed for blur computation)
    // Top halo
    if (localRow < BLUR_SIZE) {
        int haloRow = row - BLUR_SIZE;
        if (haloRow >= 0 && col < width) {
            sharedTile[localRow][sharedCol] = in[haloRow * width + col];
        } else {
            sharedTile[localRow][sharedCol] = {0, 0, 0};
        }
    }
    
    // Bottom halo
    if (localRow >= blockDim.y - BLUR_SIZE) {
        int haloRow = row + BLUR_SIZE;
        int sharedHaloRow = localRow + 2 * BLUR_SIZE;
        if (haloRow < height && col < width) {
            sharedTile[sharedHaloRow][sharedCol] = in[haloRow * width + col];
        } else {
            sharedTile[sharedHaloRow][sharedCol] = {0, 0, 0};
        }
    }
    
    // Left halo
    if (localCol < BLUR_SIZE) {
        int haloCol = col - BLUR_SIZE;
        if (haloCol >= 0 && row < height) {
            sharedTile[sharedRow][localCol] = in[row * width + haloCol];
        } else {
            sharedTile[sharedRow][localCol] = {0, 0, 0};
        }
    }
    
    // Right halo
    if (localCol >= blockDim.x - BLUR_SIZE) {
        int haloCol = col + BLUR_SIZE;
        int sharedHaloCol = localCol + 2 * BLUR_SIZE;
        if (haloCol < width && row < height) {
            sharedTile[sharedRow][sharedHaloCol] = in[row * width + haloCol];
        } else {
            sharedTile[sharedRow][sharedHaloCol] = {0, 0, 0};
        }
    }
    
    // Corner halos (4 corners)
    // Top-left
    if (localRow < BLUR_SIZE && localCol < BLUR_SIZE) {
        int haloRow = row - BLUR_SIZE;
        int haloCol = col - BLUR_SIZE;
        if (haloRow >= 0 && haloCol >= 0) {
            sharedTile[localRow][localCol] = in[haloRow * width + haloCol];
        } else {
            sharedTile[localRow][localCol] = {0, 0, 0};
        }
    }
    
    // Top-right
    if (localRow < BLUR_SIZE && localCol >= blockDim.x - BLUR_SIZE) {
        int haloRow = row - BLUR_SIZE;
        int haloCol = col + BLUR_SIZE;
        int sharedHaloCol = localCol + 2 * BLUR_SIZE;
        if (haloRow >= 0 && haloCol < width) {
            sharedTile[localRow][sharedHaloCol] = in[haloRow * width + haloCol];
        } else {
            sharedTile[localRow][sharedHaloCol] = {0, 0, 0};
        }
    }
    
    // Bottom-left
    if (localRow >= blockDim.y - BLUR_SIZE && localCol < BLUR_SIZE) {
        int haloRow = row + BLUR_SIZE;
        int haloCol = col - BLUR_SIZE;
        int sharedHaloRow = localRow + 2 * BLUR_SIZE;
        if (haloRow < height && haloCol >= 0) {
            sharedTile[sharedHaloRow][localCol] = in[haloRow * width + haloCol];
        } else {
            sharedTile[sharedHaloRow][localCol] = {0, 0, 0};
        }
    }
    
    // Bottom-right
    if (localRow >= blockDim.y - BLUR_SIZE && localCol >= blockDim.x - BLUR_SIZE) {
        int haloRow = row + BLUR_SIZE;
        int haloCol = col + BLUR_SIZE;
        int sharedHaloRow = localRow + 2 * BLUR_SIZE;
        int sharedHaloCol = localCol + 2 * BLUR_SIZE;
        if (haloRow < height && haloCol < width) {
            sharedTile[sharedHaloRow][sharedHaloCol] = in[haloRow * width + haloCol];
        } else {
            sharedTile[sharedHaloRow][sharedHaloCol] = {0, 0, 0};
        }
    }
    
    // Synchronize to ensure all shared memory is loaded before computation
    __syncthreads();
    
    // Compute blur using shared memory (much faster than global memory)
    if (col < width && row < height) {
        int3 pixVal;
        pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
        int pixels = 0;
        
        // Get the average of the surrounding box from SHARED MEMORY
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {
                
                int sharedCurRow = sharedRow + blurRow;
                int sharedCurCol = sharedCol + blurCol;
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // Verify that we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    // Access from SHARED MEMORY instead of global memory
                    pixVal.x += sharedTile[sharedCurRow][sharedCurCol].x;
                    pixVal.y += sharedTile[sharedCurRow][sharedCurCol].y;
                    pixVal.z += sharedTile[sharedCurRow][sharedCurCol].z;
                    pixels++;
                }
            }
        }
        
        // Write output pixel
        out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
        out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
        out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
    }
}

/**
 * Original Blur Kernel (no shared memory optimization)
 * Kept for performance comparison
 */
__global__ void blurKernel(uchar3 *in, uchar3 *out, int width, int height) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int3 pixVal;
        pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
        int pixels = 0;
        
        // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {
                
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // Verify that we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixVal.x += in[curRow * width + curCol].x;
                    pixVal.y += in[curRow * width + curCol].y;
                    pixVal.z += in[curRow * width + curCol].z;
                    pixels++;
                }
            }
        }
        
        // Write our new pixel value out
        out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
        out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
        out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);
    
    if(!bmp) {
        cerr << "Image not found" << endl;
        exit(1);
    }
    
    int height = bmp.height();
    int width = bmp.width();
    
    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;
    cout << "Total pixels: " << (width * height) << endl;
    
    // Transform image into vector
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++) {
        for(int y = 0; y < height; y++) {
            bmp.get_pixel(x, y, color);
            input_image.push_back({color.red, color.green, color.blue});
        }
    }
    
    vector<uchar3> output_image(input_image.size());
    
    // Allocate device memory
    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    
    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, 
                 (height + TILE_SIZE - 1) / TILE_SIZE, 1);
    
    cout << "\nGrid dimensions: " << dimGrid.x << " x " << dimGrid.y << endl;
    cout << "Block dimensions: " << dimBlock.x << " x " << dimBlock.y << endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ===== Test Original Kernel (Global Memory) =====
    cout << "\n--- Testing Original Kernel (Global Memory) ---" << endl;
    
    // Warm-up run (GPU initialization)
    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float originalTime = 0;
    cudaEventElapsedTime(&originalTime, start, stop);
    cout << "Time: " << originalTime << " ms" << endl;
    
    // ===== Test Optimized Kernel (Shared Memory) =====
    cout << "\n--- Testing Optimized Kernel (Shared Memory) ---" << endl;
    
    // Warm-up run
    blurKernelOptimized<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    blurKernelOptimized<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float optimizedTime = 0;
    cudaEventElapsedTime(&optimizedTime, start, stop);
    cout << "Time: " << optimizedTime << " ms" << endl;
    
    // Calculate speedup
    float speedup = originalTime / optimizedTime;
    cout << "\n=== Performance Summary ===" << endl;
    cout << "Original (Global Memory): " << originalTime << " ms" << endl;
    cout << "Optimized (Shared Memory): " << optimizedTime << " ms" << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    
    // Copy result back to host (using optimized version)
    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    // Set updated pixels
    for(int x = 0; x < width; x++) {
        for(int y = 0; y < height; y++) {
            int pos = x * height + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }
    
    cout << "\nSaving blurred image..." << endl;
    bmp.save_image("./blurred_optimized.bmp");
    cout << "Saved as: blurred_optimized.bmp" << endl;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}

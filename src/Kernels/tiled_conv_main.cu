#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define TILE_DIM 32
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

// Declare constant memory for the filter
__constant__ float F_c[FILTER_SIZE][FILTER_SIZE];

// Tiled convolution kernel
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float* N, float* P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float N_s[TILE_DIM][TILE_DIM];

    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                int shared_y = threadIdx.y - FILTER_RADIUS + fRow;
                int shared_x = threadIdx.x - FILTER_RADIUS + fCol;

                if (shared_y >= 0 && shared_y < TILE_DIM &&
                    shared_x >= 0 && shared_x < TILE_DIM) {
                    Pvalue += F_c[fRow][fCol] * N_s[shared_y][shared_x];
                } else {
                    int global_y = row - FILTER_RADIUS + fRow;
                    int global_x = col - FILTER_RADIUS + fCol;
                    if (global_y >= 0 && global_y < height &&
                        global_x >= 0 && global_x < width) {
                        Pvalue += F_c[fRow][fCol] * N[global_y * width + global_x];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

// Host main function
int main() {
    const int width = 64;
    const int height = 64;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    std::vector<float> h_N(size);
    std::vector<float> h_P(size);
    float h_filter[FILTER_SIZE][FILTER_SIZE];

    // Initialize input and filter with random data
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < size; ++i)
        h_N[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < FILTER_SIZE; ++i)
        for (int j = 0; j < FILTER_SIZE; ++j)
            h_filter[i][j] = 1.0f / (FILTER_SIZE * FILTER_SIZE); // simple averaging filter

    // Copy filter to constant memory
    cudaMemcpyToSymbol(F_c, h_filter, sizeof(h_filter));

    // Allocate device memory
    float *d_N, *d_P;
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);

    // Copy input to device
    cudaMemcpy(d_N, h_N.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    convolution_cached_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_N, d_P, width, height);

    // Copy result back to host
    cudaMemcpy(h_P.data(), d_P, bytes, cudaMemcpyDeviceToHost);
     
    /* Print part of the result
    std::cout << "Sample output:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_P[i] << " ";
    }
    std::cout << std::endl;
    */

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}


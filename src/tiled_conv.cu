
#define TILE_DIM 32
#define FILTER_RADIUS 2  // these can be changed

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float* N, float* P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Declare shared memory tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];

    // Load shared tile with boundary check
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Output computation
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
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


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define Mask_width 5
#define Mask_radius (Mask_width / 2)
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0f), 1.0f))

__global__ void convolution(float *I, const float *M, float *P, int C, int W, int H) {
    __shared__ float N_ds[w][w];

    int k;
    for (k = 0; k < C; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        int src = (srcY * W + srcX) * C + k;

        if (srcY >= 0 && srcY < H && srcX >= 0 && srcX < W)
            N_ds[destY][destX] = I[src];
        else
            N_ds[destY][destX] = 0;

        __syncthreads();

        float accum = 0;
        for (int y = 0; y < Mask_width; y++)
            for (int x = 0; x < Mask_width; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];

        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < H && x < W)
            P[(y * W + x) * C + k] = clamp(accum);

        __syncthreads();
    }
}

void vanilla_convolve(torch::Tensor input, torch::Tensor mask, torch::Tensor output) {
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(0);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH);

    convolution<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(), mask.data_ptr<float>(), output.data_ptr<float>(),
        C, W, H
    );
}


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
#define TILE_WIDTH 16
#define W (TILE_WIDTH + MASK_WIDTH - 1)
#define clamp(x) (fminf(fmaxf((x), 0.0f), 1.0f))

__global__ void convolution(float *I, const float *__restrict__ M, float *P, 
                            int channels, int width, int height) {
    __shared__ float N_ds[W][W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    int dest = ty * TILE_WIDTH + tx;
    int destY = dest / W;
    int destX = dest % W;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;

    for (int k = 0; k < channels; ++k) {
        int srcIdx = ((srcY * width + srcX) * channels) + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = I[srcIdx];
        } else {
            N_ds[destY][destX] = 0.0f;
        }

        __syncthreads();

        float accum = 0.0f;
        for (int i = 0; i < MASK_WIDTH; ++i)
            for (int j = 0; j < MASK_WIDTH; ++j)
                accum += N_ds[ty + i][tx + j] * M[i * MASK_WIDTH + j];

        if (row_o < height && col_o < width)
            P[((row_o * width + col_o) * channels) + k] = clamp(accum);

        __syncthreads();
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <image_height> <image_width> <image_channels>\n", argv[0]);
        return 0;
    }

    int imageHeight = atoi(argv[1]);
    int imageWidth = atoi(argv[2]);
    int imageChannels = atoi(argv[3]);
    int maskRows = MASK_WIDTH;
    int maskColumns = MASK_WIDTH;

    float *hostInputImageData = (float *)malloc(imageHeight * imageWidth * imageChannels * sizeof(float));
    float *hostOutputImageData = (float *)malloc(imageHeight * imageWidth * imageChannels * sizeof(float));
    float *hostMaskData = (float *)malloc(maskRows * maskColumns * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < imageHeight * imageWidth * imageChannels; ++i)
        hostInputImageData[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < maskRows * maskColumns; ++i)
        hostMaskData[i] = ((float)(rand() % 256) / 255.0f) / (MASK_WIDTH * MASK_WIDTH / 4.0f);

    float *deviceInputImageData, *deviceOutputImageData, *deviceMaskData;
    cudaMalloc((void **)&deviceInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));

    cudaMemcpy(deviceInputImageData, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);

    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);
    free(hostMaskData);

    return 0;
}


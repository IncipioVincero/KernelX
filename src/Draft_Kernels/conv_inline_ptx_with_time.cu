#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define Mask_width 5
#define Mask_radius (Mask_width / 2)
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0f), 1.0f))

__global__ void convolution_with_ptx(float *I, const float *__restrict__ M, float *P, 
                                     int channels, int width, int height) {

    __shared__ float N_ds[w][w];

    for (int k = 0; k < channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        int src = (srcY * width + srcX) * channels + k;

        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = I[src];
        } else {
            N_ds[destY][destX] = 0;
        }

        __syncthreads();

        float accum = 0.0f;

        for (int y = 0; y < Mask_width; y++) {
            for (int x = 0; x < Mask_width; x++) {
                float val;

                // Inline PTX to load from shared memory
                asm volatile("ld.shared.f32 %0, [%1];"
                             : "=f"(val)
                             : "l"(&N_ds[threadIdx.y + y][threadIdx.x + x]));

                accum += val * M[y * Mask_width + x];
            }
        }

        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width) {
            P[(y * width + x) * channels + k] = clamp(accum);
        }

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
    int maskRows = Mask_width;
    int maskColumns = Mask_width;

    size_t imageSize = imageHeight * imageWidth * imageChannels * sizeof(float);
    size_t maskSize = maskRows * maskColumns * sizeof(float);

    float *hostInputImageData = (float *)malloc(imageSize);
    float *hostOutputImageData = (float *)malloc(imageSize);
    float *hostMaskData = (float *)malloc(maskSize);

    for (int i = 0; i < imageHeight * imageWidth * imageChannels; ++i)
        hostInputImageData[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < maskRows * maskColumns; ++i)
        hostMaskData[i] = ((float)(rand() % 256) / 255.0f) / (Mask_width * Mask_width / 4.0f);

    float *deviceInputImageData, *deviceOutputImageData, *deviceMaskData;
    cudaMalloc((void **)&deviceInputImageData, imageSize);
    cudaMalloc((void **)&deviceOutputImageData, imageSize);
    cudaMalloc((void **)&deviceMaskData, maskSize);

    cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);

    dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // CUDA timing events
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch and time kernel
    cudaEventRecord(start);
    convolution_with_ptx<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                                deviceOutputImageData, imageChannels,
                                                imageWidth, imageHeight);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time (PTX version): %.4f ms\n", milliseconds);

    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);
    free(hostMaskData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


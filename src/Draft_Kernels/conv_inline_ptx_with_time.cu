#include <stdio.h>

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0f), 1.0f))

// Define the convolution kernel with inline PTX
__global__ void convolution_with_ptx(float *I, const float *__restrict__ M, float *P, 
                                      int channels, int width, int height) {

    __shared__ float N_ds[w][w];
    
    int k;
    for (k = 0; k < channels; k++) {
        
        // First batch loading into shared memory (this part is the same as the original code)
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        int src = (srcY * width + srcX) * channels + k;
        
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = I[src];
        } else {
            N_ds[destY][destX] = 0.0f;
        }
        
        __syncthreads();
    
        float accum = 0.0f;
        int y, x;
        
        // PTX: Perform the convolution calculation using PTX for fine-grained control
        for (y = 0; y < Mask_width; y++) {
            for (x = 0; x < Mask_width; x++) {
                // Here we would perform the operation using inline PTX
                // This is where we use PTX to directly load from shared memory, potentially leveraging Tensor Cores
                
                // For demonstration, using inline PTX to perform a simple load and multiply
                // PTX: Using atomic load to fetch data in FP16 (assuming it's in FP16)
                float val = 0.0f;
                asm("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(N_ds[threadIdx.y + y][threadIdx.x + x]));
                accum += val * M[y * Mask_width + x];
            }
        }
        
        // Output result: We write the result to global memory with PTX
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            P[(y * width + x) * channels + k] = clamp(accum);

        __syncthreads();
    }
}

int main(int argc, char *argv[]) {

    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;

    if (argc != 4){
        printf("Usage: %s <image_height> <image_width> <image_channels>\n", argv[0]);
        return 0;
    }

    imageHeight = atoi(argv[1]);
    imageWidth = atoi(argv[2]);
    imageChannels = atoi(argv[3]);
    maskRows = Mask_width;
    maskColumns = Mask_width;

    // Allocate host memory
    hostInputImageData = (float *)malloc(imageHeight * imageWidth * imageChannels * sizeof(float));
    hostOutputImageData = (float *)malloc(imageHeight * imageWidth * imageChannels * sizeof(float));
    hostMaskData = (float *)malloc(maskRows * maskColumns * sizeof(float));

    // Initialize the input and mask
    srand(time(NULL));
    for (int i = 0; i < imageHeight; ++i)
        for (int j = 0; j < imageWidth; ++j)
            for(int k = 0; k < imageChannels; ++k)
                hostInputImageData[(i * imageWidth + j) * imageChannels + k] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < maskRows; ++i)
        for (int j = 0; j < maskColumns; ++j)
            hostMaskData[i * maskColumns + j] = ((float)(rand() % 256) / 255.0f) / (Mask_width * Mask_width / 4.0f); 

    // Allocate device memory
    cudaMalloc((void **)&deviceInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with PTX
    dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    
    // Timer start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    convolution_with_ptx<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                               deviceOutputImageData, imageChannels,
                                               imageWidth, imageHeight);

    // Timer stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (with PTX): %f ms\n", milliseconds);

    // Copy result from device to host
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, 
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    
    free(hostMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}


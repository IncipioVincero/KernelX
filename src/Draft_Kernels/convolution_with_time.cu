//Source: Programming Massively Parallel Processors Fourth Edition
//Link: https://github.com/R100001/Programming-Massively-Parallel-Processors

// Define constants used for the convolution operation
#define Mask_width 5                           // Width and height of the square convolution mask
#define Mask_radius (Mask_width / 2)           // Radius is half of mask width, for padding logic
#define TILE_WIDTH 16                          // Width/height of each thread block (tile)
#define w (TILE_WIDTH + Mask_width - 1)        // Shared memory tile width (including halo for convolution)

// Clamp function to ensure pixel values are in [0.0, 1.0] range
#define clamp(x) (min(max((x), 0.0f), 1.0f))

// Mark this function as a CUDA kernel visible to host code
extern "C"
__global__ void convolution(float *I,               // Input image data
                            const float *__restrict__ M, // Convolution mask (read-only)
                            float *P,               // Output image data
                            int channels,           // Number of color channels (e.g., 3 for RGB)
                            int width,              // Image width in pixels
                            int height)             // Image height in pixels
{
    // Declare shared memory tile with halo, per block
    __shared__ float N_ds[w][w];

    // Loop over each channel separately (R, G, B, etc.)
    for (int k = 0; k < channels; k++) {

        // Calculate linear thread ID in 2D block
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;     // Destination Y index in shared memory
        int destX = dest % w;     // Destination X index in shared memory

        // Map destination back to global image coordinates (with halo offset)
        int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;

        // Compute 1D index into the input image for the given channel
        int src = (srcY * width + srcX) * channels + k;

        // Load pixel value into shared memory, with zero-padding for out-of-bounds
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = I[src];
        } else {
            N_ds[destY][destX] = 0.0f;
        }

        // Synchronize threads to make sure all shared memory loads complete
        __syncthreads();

        // Perform convolution
        float accum = 0.0f;
        for (int y = 0; y < Mask_width; y++) {
            for (int x = 0; x < Mask_width; x++) {
                // Multiply mask value with corresponding pixel in shared memory
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
            }
        }

        // Compute global output coordinates
        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

        // Write result to global output image, if within bounds
        if (y < height && x < width) {
            P[(y * width + x) * channels + k] = clamp(accum);
        }

        // Sync threads before the next channel iteration
        __syncthreads();
    }
}

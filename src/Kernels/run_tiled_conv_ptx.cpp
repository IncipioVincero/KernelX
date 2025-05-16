#include <cuda.h>
#include <iostream>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* func, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " code=" << err << " \"" << func << "\"" << std::endl;
        exit(1);
    }
}

int main() {
    // Initialize CUDA
    checkCudaErrors(cuInit(0));

    // Get device and create context
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));

    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));

    // Load PTX
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, "tiled_conv.ptx"));

    // Get kernel function
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module,
        "_Z44convolution_cached_tiled_2D_const_mem_kernelPfS_ii"));

    // Allocate memory
    int width = 32;
    int height = 32;
    size_t size = width * height * sizeof(float);

    float* h_input = new float[width * height];
    float* h_output = new float[width * height];
    for (int i = 0; i < width * height; ++i) h_input[i] = i;

    CUdeviceptr d_input, d_output;
    checkCudaErrors(cuMemAlloc(&d_input, size));
    checkCudaErrors(cuMemAlloc(&d_output, size));

    checkCudaErrors(cuMemcpyHtoD(d_input, h_input, size));

    // Set kernel args
    void* args[] = { &d_input, &d_output, &width, &height };

    // Launch
    int blockSize = 32;
    checkCudaErrors(cuLaunchKernel(kernel,
        width / blockSize, height / blockSize, 1, // grid dim
        blockSize, blockSize, 1,                  // block dim
        0, 0, args, 0));

    checkCudaErrors(cuCtxSynchronize());

    // Copy back and print
    checkCudaErrors(cuMemcpyDtoH(h_output, d_output, size));
    std::cout << "Output[0]: " << h_output[0] << std::endl;

    // Clean up
    cuMemFree(d_input);
    cuMemFree(d_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    delete[] h_input;
    delete[] h_output;

    return 0;
}


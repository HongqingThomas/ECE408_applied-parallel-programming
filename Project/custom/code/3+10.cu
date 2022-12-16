// import lxr as love
// kernel fusion for unroll and matrix multiplication + Fp16

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#define TILE_WIDTH 16


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ __half subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int n = bz;
    int m = Row;
    int h = Col / Width_out;
    int w = Col % Width_out;
    int numARows = Map_out;
    int numACols = Channel * K * K;
    int numBRows = numACols;
    int numBCols = Height_out * Width_out;
    int numCRows = numARows;
    int numCCols = numBCols;

    __half Pvalue = 0 ;
    for (int i = 0; i < ceil(numACols/float(TILE_WIDTH)); i++) {
        int currentCol = i * TILE_WIDTH + tx;
        int currentRow = i * TILE_WIDTH + ty;
        if (Row < numARows && currentCol < numACols) {
            subTileM[ty][tx] = __float2half(mask[m * Channel * K * K + currentCol]);
        }
        else {
            subTileM[ty][tx] = 0;
        }
        int _c = currentRow / (K * K);
        int _h = Col / Width_out;
        int _w = Col % Width_out;
        int _p = currentRow % (K * K) / K;
        int _q = (currentRow % (K * K)) % K;
        if (currentRow < numBRows && Col < numBCols) {
            subTileN[ty][tx] = __float2half(in_4d(n, _c, _h + _p, _w + _q));
        }
        else {
            subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += subTileM[ty][i] * subTileN[i][tx];
        }
        __syncthreads();
    }
    if (Row < numCRows && Col < numCCols) {
        out_4d(n, m, h, w) = Pvalue;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    
}



	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Function paramters:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps (for each input image!)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)  
    */
    // Allocate memory and copy over the relevant data structures to the GPU
    int input_size = Batch * Channel * Height * Width;
    int output_size =  Batch * Map_out * (Height-K+1) * (Width-K+1);
    int map_size = Map_out * Channel * K * K;

    cudaMalloc((void **) &*device_input_ptr, sizeof(float) * input_size);
    cudaMalloc((void **) &*device_output_ptr, sizeof(float) * output_size);
    cudaMalloc((void **) &*device_mask_ptr, sizeof(float) * map_size);

    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * input_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * map_size, cudaMemcpyHostToDevice); 
    // cudaMemcpyToSymbol(Mask, host_mask, sizeof(float) * map_size);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int x_grid = ceil(1.0 * Height_out * Width_out / TILE_WIDTH);
    int y_grid = ceil(1.0 *Map_out /TILE_WIDTH);
    int z_grid = Batch;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(x_grid,y_grid,z_grid);
    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

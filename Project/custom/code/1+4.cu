// import lxr as love
// const + shared memory
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16 // 16
__constant__ float Mask[6000];

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

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int shared_width = TILE_WIDTH + K - 1;
    extern __shared__ float SM[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]// b,map_out,h,m(row,col)
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0] //i3:which image; i2:which channel; i1:which height(row); i0:which width(col);
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0] //for each picture: i3:which feature map; i2:which channel; i1:mask row; i0:mask col;
    #define sm3d(i2, i1, i0) SM[(i2) * (shared_width * shared_width) + (i1) * shared_width + i0]
    //Insert your GPU convolution kernel code here
    int b = blockIdx.x;
    int m = blockIdx.y;

    int numblock_row = (Width_out - 1)/TILE_WIDTH + 1;
    // int numblock_row = ceil((Width - K + 1) / (1.0*TILE_WIDTH));

    int w_out = TILE_WIDTH * (blockIdx.z % numblock_row) + threadIdx.x;
    int h_out = TILE_WIDTH * (blockIdx.z/numblock_row) + threadIdx.y;

    int TILE_h = TILE_WIDTH * (blockIdx.z/numblock_row);
    int TILE_w = TILE_WIDTH * (blockIdx.z%numblock_row);
    int c = threadIdx.z;
        for(int i = threadIdx.y; i < shared_width; i += TILE_WIDTH)
        {
            for(int j = threadIdx.x; j < shared_width; j += TILE_WIDTH)
            {
                if (TILE_h + i < Height && TILE_w + j < Width)
                {
                    sm3d(c, i, j) = in_4d(b, c, TILE_h + i, TILE_w + j);
                }
                
            }
        } 
    
    
    __syncthreads();

    if (h_out < Height_out && w_out < Width_out)
    {
        float result = 0;
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    //result += x4d(b_out, c, h_out + p, w_out + q) * k4d(m_out, c, p, q);
                    result += sm3d(c, threadIdx.y + p, threadIdx.x + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h_out, w_out) = result;
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
    // cudaMalloc((void **) &*device_mask_ptr, sizeof(float) * map_size);

    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * input_size, cudaMemcpyHostToDevice); 
    // cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * map_size, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(Mask, host_mask, sizeof(float) * map_size);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    //input: batch images(b,c,x,y); a mask(map_out,channel,x,y) aim to each picture; 
    //output: batch * map_out images:(b,map_out,x,y)
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int x_grid = Batch; //the first dimension (X) in the grid corresponds to samples (N) in the batch
    int y_grid = Map_out; //the second dimension (Y) corresponds to the (M) output features maps
    int grid_h = ceil(Height_out/(1.0*TILE_WIDTH));
    int grid_w = ceil(Width_out/(1.0*TILE_WIDTH));
    int z_grid = grid_h * grid_w;//the last dimension (Z) will define the location of the output tile inside the output feature map

    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,Channel);
    dim3 DimGrid(x_grid,y_grid,z_grid);

    conv_forward_kernel<<<DimGrid,DimBlock,Channel*(TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)*sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input,float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size =  Batch * Map_out * (Height-K+1) * (Width-K+1);
    cudaMemcpy(host_output, device_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost); 
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask);
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


// import lxr as love
// unroll
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void forward_kernel(const float *device_mask, const float *X_unrolled, float *device_output, int H_unroll, const int Map_out, int W_unroll)
{
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Cols = blockIdx.y * blockDim.y + ty;
  int Rows = blockIdx.x * blockDim.x + tx;
  float result = 0;
  for (int i = 0; i < ceil(1.0*H_unroll/TILE_WIDTH); i++) {
    int col = i * TILE_WIDTH + tx;
    if (Cols < Map_out && col < H_unroll){
        subTileM[ty][tx] = device_mask[Cols * H_unroll + col];
    }
    else{
        subTileM[ty][tx] = 0;
    }
    int row = i * TILE_WIDTH + ty;
    if (Rows < W_unroll && row < H_unroll){
        subTileN[ty][tx] = X_unrolled[row * W_unroll + Rows];
    }
    else{
        subTileN[ty][tx] = 0;
    }
    __syncthreads();
    if ((Cols < Map_out) && (Rows < W_unroll)) {
        for (int k = 0; k < TILE_WIDTH; k++)
            result += subTileM[ty][k] * subTileN[k][tx];
    }
    __syncthreads();
  }
  if ((Cols < Map_out) && (Rows < W_unroll)) {
    device_output[Cols * W_unroll + Rows] = result;
  }
}

void unroll_launch(const float* device_mask, const float* X_unrolled, float* device_output, int H_unroll, const int Map_out, int W_unroll) {
    dim3 gridDim (ceil(1.0 * W_unroll / TILE_WIDTH),  ceil(1.0 *  Map_out/ TILE_WIDTH), 1);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
    forward_kernel<<<gridDim, blockDim>>>(device_mask, X_unrolled, device_output, H_unroll, Map_out, W_unroll);
}

__global__ void unroll_forwardkernel(const int Channel, const int Height, const int Width, const int K,  float* X_unroll,const float* X) {
    int Rows = blockDim.x * blockIdx.x + threadIdx.x;
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int W_unroll = Height_out * Width_out;
    int H_unroll = Channel*K*K;
    if (Rows < Channel * W_unroll) {
        int c = Rows / W_unroll;
        int s = Rows % W_unroll;
        int h_out = s / Width_out;
        int w_out = s % Width_out;
        int w_unroll = h_out * Width_out + w_out;
        int w_base = c * K * K;
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q;
                X_unroll[w_unroll + h_unroll * W_unroll ] = X[c * Height * Width + (h_out+p) * Width + w_out+q];
            }
        }
    }
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
    // int x_grid = Batch; //the first dimension (X) in the grid corresponds to samples (N) in the batch
    // int y_grid = Map_out; //the second dimension (Y) corresponds to the (M) output features maps
    // int grid_h = ceil(Height_out/(1.0*TILE_WIDTH));
    // int grid_w = ceil(Width_out/(1.0*TILE_WIDTH));
    // int z_grid = grid_h * grid_w;//the last dimension (Z) will define the location of the output tile inside the output feature map

    int H_unroll = Channel * K * K;
    int W_unroll = Height_out * Width_out; 
    float * X_unrolled;
    int BLOCK_SIZE = 1000;
    cudaMalloc((void **)&X_unrolled, H_unroll * W_unroll * sizeof(float));
    int num_blocks = ceil((1.0 * Channel * Height_out * Width_out) / BLOCK_SIZE);
    for (int i = Batch - 1; i >= 0; i--) {
        unroll_forwardkernel<<<num_blocks, BLOCK_SIZE>>>(Channel, Height, Width, K, X_unrolled, device_input + i * Channel * Height * Width);
        unroll_launch(device_mask,  X_unrolled,  device_output + i * Map_out * Height_out * Width_out,  H_unroll,  Map_out,  W_unroll);
    }
    // // for reduction tree, we need to change DimBlock's threadIdx.z
    // dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,Channel);
    // dim3 DimGrid(x_grid,y_grid,z_grid);

    // conv_forward_kernel<<<DimGrid,DimBlock,Channel*(TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)*sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input,float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size =  Batch * Map_out * (Height-K+1) * (Width-K+1);
    cudaMemcpy(host_output, device_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost); 
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

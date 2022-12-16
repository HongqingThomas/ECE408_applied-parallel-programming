
//import lxr as love

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
__constant__ float Mc[3][3][3];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int mask_width = 3;
  int tile_width_o = 8;
  int tile_width_i =tile_width_o+mask_width-1;
  __shared__ float N_ds[10][10][10];
  // int radius = mask_width/2;
  // //upload the matrix

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int y_o = blockIdx.y * tile_width_o + ty;
  int x_o = blockIdx.x * tile_width_o + tx;
  int z_o = blockIdx.z * tile_width_o + tz;
  int x_i = x_o - (mask_width/2);
  int y_i = y_o - (mask_width/2);
  int z_i = z_o - (mask_width/2);
  // float N_ds[tile_width_i][tile_width_i][tile_width_i];
  if ((y_i>=0)&&(y_i<y_size)&&(x_i>=0)&&(x_i<x_size)&&(z_i>=0)&&(z_i<z_size)){
    N_ds[tz][ty][tx] = input[z_i*x_size*y_size+y_i*x_size+x_i];
  }
  else{
    N_ds[tz][ty][tx] = 0;
  }
  __syncthreads();
  //do the convolution part 
  float Pvalue = 0;
  if((tx<tile_width_o) && (ty<tile_width_o) && (tz<tile_width_o)){
    for(int i=0;i<mask_width;i++){
      for(int j=0;j<mask_width;j++){
        for(int k=0;k<mask_width;k++){
          Pvalue += Mc[i][j][k]* N_ds[i+tz][j+ty][k+tx];
        }
      }
    }
    if(y_o<y_size&&x_o<x_size&&z_o<z_size){
      output[z_o*x_size*y_size+y_o*x_size+x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  // float Mask[3][3][3];
  // for(int i=1; i<4; i++){
  //   for(int k =1;k<4; k++){
  //     for (int j=1;j<4;j++){
  //       Mask[j-1][k-1][i-1] = hostKernel[i*k*j-1];
  //     }
  //   }
  // }

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, sizeof(float) * (inputLength - 3));
  cudaMalloc((void **) &deviceOutput, sizeof(float) * (inputLength - 3));
  // cudaMalloc((void **) &deviceB, sizeof(float) * (kernelLength - 3));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  float hostInputValue[inputLength-3];
  for(int i = 3;i<inputLength;i++){
    hostInputValue[i-3] = hostInput[i];
  }
  // cudaMemcpy(deviceInput, hostInput, sizeof(float) * (inputLength - 3), cudaMemcpyHostToDevice);  
  cudaMemcpy(deviceInput, hostInputValue, sizeof(float) * (inputLength - 3), cudaMemcpyHostToDevice);  
  cudaMemcpyToSymbol(Mc,hostKernel,kernelLength*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  int tile_width = 8;
  int mask_width = 3;
  dim3 DimGrid(ceil(1.0*x_size/tile_width),ceil(1.0*y_size/tile_width),ceil(1.0*z_size/tile_width));
  dim3 DimBlock(tile_width+mask_width-1,tile_width+mask_width-1,tile_width+mask_width-1);
  
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  float  hostOutputValue[inputLength-3];
  // cudaMemcpy(hostOutput, deviceOutput, sizeof(float) *(inputLength - 3),cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutputValue, deviceOutput, sizeof(float) *(inputLength - 3),cudaMemcpyDeviceToHost);
  for(int k = 3;k<inputLength;k++){
    hostOutput[k]=hostOutputValue[k-3];
  }
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

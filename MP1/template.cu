// MP 1
#include <wb.h>

__global__ void vecAdd(float *A_d, float *B_d, float *C_d, int n) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
  C_d[i] = A_d[i] + B_d[i];
  }
  //C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char **argv) {

  wbArg_t args;
  int inputLength;
  int size;
  //float *hostInput1;
  //float *hostInput2;
  //float *hostOutput;
  //float *deviceInput1;
  //float *deviceInput2;
  //float *deviceOutput;
  float *A;
  float *B;
  float *C;
  float *A_d;
  float *B_d;
  float *C_d;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  A = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  B = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  C = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);


  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  size = sizeof(float) * inputLength;
  cudaMalloc((void **) &A_d, size);
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 DimGrid(ceil(inputLength/256.0),1,1);
  dim3 DimBlock(256,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  vecAdd<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  wbTime_stop(GPU, "Freeing GPU Memory");

  //wbSolution(args, hostOutput, inputLength);

  wbSolution(args, C, inputLength);

  free(A);
  free(B);
  free(C);

  //free(hostInput1);
  //free(hostInput2);
  //free(hostOutput);

  return 0;
}

//import lxr as love

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockDim.x*blockIdx.x;
  unsigned int bx = blockIdx.x;
  if(start+t<len){
    partialSum[t] = input[start+t];
  }
  else{
    partialSum[t] = 0.0f;
  }
  if(start+t+blockDim.x<len){
    partialSum[t+blockDim.x] = input[start+t+blockDim.x];
  }
  else{
    partialSum[t+blockDim.x] = 0.0f;
  }
  
  int stride = 1;
  while(stride<2*BLOCK_SIZE){
    __syncthreads();
    int index = (t+1)*stride*2-1;//1 3 5 7 9 11 13 15;3 7 11 15;7 15;15
    if (index<2*BLOCK_SIZE && (index-stride)>=0){
      partialSum[index] += partialSum[index-stride];
    }
    stride = stride * 2;
  }

  int _stride = BLOCK_SIZE/2;
  while(_stride>0){
    __syncthreads();
    int _index = (t+1)*_stride*2-1;
    if((_index + _stride)<2*BLOCK_SIZE){
      partialSum[_index+_stride] += partialSum[_index];
    }
    _stride = _stride/2;
  }
  //update shared memory back to output list
  __syncthreads();
  output[start+t] = partialSum[t];
  output[start+t+blockDim.x] = partialSum[t+blockDim.x];  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0*numElements/BLOCK_SIZE),1,1);
  dim3 DimBlock(BLOCK_SIZE,1,1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  for(int i = 0; i < numElements; i++){
    int _blockIdx = i/(BLOCK_SIZE*2);
    if(_blockIdx>0){
      hostOutput[i] += hostOutput[_blockIdx*BLOCK_SIZE*2-1];
    } 
  }

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

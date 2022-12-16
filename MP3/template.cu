
//import lxr as love

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int TILE_WIDTH = 16;
  //__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  //__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileA[16][16];
  __shared__ float subTileB[16][16];
  //each block uses tile_width2 threads to compute
//  int bx = blockIdx.x;
//  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float Cvalue = 0.0;
  //int Width = numAColumns;

  for (int q = 0; q < (int)(ceil((float)numAColumns/TILE_WIDTH)); q++) 
  {
    if (Row < numARows && (q * TILE_WIDTH + tx) < numAColumns) {
      subTileA[ty][tx] = A[Row * numAColumns + q * TILE_WIDTH + tx];
    }
    else {
      subTileA[ty][tx] = 0.0;
    }

    if (Col < numBColumns && (q * TILE_WIDTH + ty) < numBRows){
      subTileB[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
    }
    else{
      subTileB[ty][tx] = 0.0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++){
      Cvalue += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }

  if (Row < numCRows && Col < numCColumns) 
    C[Row * numCColumns + Col] = Cvalue;

  }  

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  //numCRows = 0;
  //numCColumns = 0;
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows* numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // TILE_WIDTH = 16;
  // dim3 DimGrid(ceil(1.0*numCColumns/TILE_WIDTH),ceil(1.0*numCRows/TILE_WIDTH),1);
  // dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);

  dim3 DimGrid(ceil(1.0*numCColumns/16),ceil(1.0*numCRows/16),1);
  dim3 DimBlock(16,16,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) *numCRows * numCColumns, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

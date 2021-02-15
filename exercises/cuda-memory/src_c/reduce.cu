/*

 Reduction exercise

 */

#include <stdio.h>
#include <stdlib.h>

#define scalar float

/* Forward Declaration*/
/* Utility function to check for and report CUDA errors */

void checkCUDAError(const char*);

/*
 * The number of CUDA threads per block to use.
 */

#define THREADS_PER_BLOCK 128

/* The number of elements in the array */

static __constant__ int constArraySize;

/* Add all the elements of the array d_in
   Put the results from each block in d_out */

__global__ void ReduceAdd(scalar * d_in, scalar * d_out)
{
  __shared__ scalar sh[THREADS_PER_BLOCK];
  int read_idx = blockIdx.x * ( 2 * blockDim.x ) + threadIdx.x;
  sh[threadIdx.x] = d_in[read_idx] + d_in[read_idx + blockDim.x];
  __syncthreads();

  for( unsigned int i = blockDim.x / 2; i; i >>= 1 )
  {
    if( threadIdx.x < i )
      sh[threadIdx.x] += sh[threadIdx.x + i];
    __syncthreads();
  }

  if( !threadIdx.x )
    d_out[blockIdx.x] = sh[threadIdx.x];
}


/* Main routine */
void TestArray(int ARRAY_SIZE)
{
    printf("\n\nTesting array size %d\n\n", ARRAY_SIZE);
    cudaMemcpyToSymbol(constArraySize, &ARRAY_SIZE, sizeof(ARRAY_SIZE)/*,
			0, cudaMemcpyHostToDevice*/);
    checkCUDAError("Copying constant ARRAY_SIZE to symbol table");

    scalar *h_in, *h_out;
    scalar *d_in, *d_out;

    int i;
    const size_t sz  = ARRAY_SIZE * sizeof(scalar);
    const size_t NumBlocks = ARRAY_SIZE / ( 2 * THREADS_PER_BLOCK );
    const size_t szR = NumBlocks * sizeof(scalar);

    /*
     * allocate memory on host
     * h_in holds the input array, h_out holds the result
     */
    h_in = (scalar *) malloc(sz);
    h_out = (scalar *) malloc(szR);

    /*
     * allocate memory on device
     */
    cudaMalloc(&d_in, sz);
    cudaMalloc(&d_out, szR);

    /* initialise host arrays */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = i;
    }

    /* copy input array from host to GPU */

    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */

    dim3 blocksPerGrid(NumBlocks, 1, 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);

    ReduceAdd<<< blocksPerGrid, threadsPerBlock >>>(d_in, d_out);

    /* wait for all threads to complete and check for errors */

    cudaDeviceSynchronize();
    checkCUDAError("kernel invocation");

    /* copy the result array back to the host */

    cudaMemcpy(h_out, d_out, szR, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpyDeviceToHost");

    /* print out the result */
    for (i = 1; i < NumBlocks; ++i)
      h_out[0] += h_out[i];

    printf("Sum = %f\n", h_out[0]);

    /* free device buffers */

    cudaFree(d_out);
    cudaFree(d_in);

    /* free host buffers */
    free(h_in);
    free(h_out);
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

/* Main routine */
int main(int argc, char *argv[])
{
  /* Print device details */
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("  Device name: %s\n", prop.name);

  TestArray(65536);
  TestArray(0x20000);
}

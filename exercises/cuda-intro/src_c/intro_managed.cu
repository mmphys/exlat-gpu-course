/*
 * This is a simple CUDA code that negates an array of integers.
 * It introduces the concepts of device memory management, and
 * kernel invocation.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010 
 */

#include <stdio.h>
#include <stdlib.h>

void ShowResults(const char * pszMessage, int *h_out);

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The number of integer elements in the array */
#define ARRAY_SIZE 300

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */

#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS  ( ( ARRAY_SIZE - 1 ) / THREADS_PER_BLOCK + 1 )

/* The actual array negation kernel (basic single block version) */

__global__ void negate(int * d_a) {

  /* Part 2B: negate an element of d_a */

}

/* Multi-block version of kernel for part 2C */

__global__ void negate_multiblock(int *d_a) {

  /* Part 2C: negate an element of d_a, using multiple blocks this time */

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < ARRAY_SIZE )
    d_a[i] = - d_a[i];
}

/* Main routine */

int main(int argc, char *argv[]) {

  int *d_a;

  int i;
  size_t sz = ARRAY_SIZE * sizeof(int);

  printf("\n\nMike's intro program\n");
  printf("  Using %d blocks x %d threads per block = %d elements\n\n",
    NUM_BLOCKS, THREADS_PER_BLOCK, ARRAY_SIZE );

  /* Print device details */

  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("  Device name: %s\n", prop.name);

  /*
   * allocate memory on device
   */
  /* Part 1A: allocate device memory */

  cudaMallocManaged(&d_a, sz);
  checkCUDAError("Allocating device memory");

  /* initialise host arrays */
  for (i = 0; i < ARRAY_SIZE; i++) {
    d_a[i] = i;
  }
  ShowResults("Input", d_a);

  /* copy input array from host to GPU */
  /* Part 1B: copy host array h_a to device array d_a */
  /*cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
  checkCUDAError("Copy host->device");*/

  /* run the kernel on the GPU */
  /* Part 2A: configure and launch kernel (un-comment and complete) */
  dim3 blocksPerGrid(NUM_BLOCKS,1,1);
  dim3 threadsPerBlock(THREADS_PER_BLOCK,1,1);
  /* negate<<< blocksPerGrid, threadsPerBlock >>>(d_a); */
  negate_multiblock<<< blocksPerGrid, threadsPerBlock >>>(d_a);

  /* wait for all threads to complete and check for errors */

  cudaDeviceSynchronize();
  checkCUDAError("kernel invocation");

  /* copy the result array back to the host */
  /* Part 1C: copy device array d_a to host array h_out */

  /*cudaMemcpy(h_out, d_a, sz, cudaMemcpyDeviceToHost);
  checkCUDAError("Copy host<-device");*/

  /* print out the result */
  ShowResults("Results", d_a);

  /* free device buffer */
  /* Part 1D: free d_a */
  cudaFree(d_a);
  checkCUDAError("Free device memory");

  return 0;
}

void ShowResults(const char * pszMessage, int *h_out)
{
  printf("\n\n");
  printf(pszMessage);
  printf(":\n\n");
  for (int i = 0; i < ARRAY_SIZE; i++) {
    printf("  %d, ", h_out[i]);
  }
  printf("\n\n");
}

/* Utility function to check for and report CUDA errors */

void checkCUDAError(const char * msg) {

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

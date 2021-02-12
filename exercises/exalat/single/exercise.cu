/*
 * Skeleton for Very Basic Linear Solver Development
 *
 * README
 * This template should compile, but is missing relevant functionality.
 * The exercise is to add, step-by-step, the necessary code.
 * STEPS are indicated by comments in the code, e.g.,
 *
 * STEP 1.1(a) Create a vector dot product kernel.
 *
 * Nick Johnson, EPCC && ExaLAT.
 */

#include <stdio.h>
#include <stdlib.h>

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The number of integer elements in the array */
#define ARRAY_SIZE 32

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */
#define NUM_BLOCKS 32
#define THREADS_PER_BLOCK 32

/* Define max number of devices we expect per node.
 * It's currently 8 on Cirrus, so we keep to that for now. */

#define MAX_DEVICES 8


/*
 * Vector Vector product (dot product)
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: result, pointer to a previously allocated scalar which will
 * contain the dot product A.B
 */

__global__ void vector_vector(float *vectorA, float *vectorB, float *result) {

  /*
   * STEP 1.1(a) Implement your vector dot product here.
   * STEP 1.1(b) Invoke the kernel from the main loop.
   */

}

/*
 * Matrix Vector product
 * input:  matrix, pointer to matrix with flattened 1-d addressing
 *         A_ij = matrix[j*ARRAY_SIZE + i]
 * input:  vector, pointer to a previously allocated vector
 * output: result, pointer to a previously allocated vector
 */
 
__global__ void matrix_vector(float *matrix, float *vector, float *result) {


  /* STEP 1.2(a) Implement you matrix vector product here.
   * STEP 1.2(b) Invoke the kernel from the main code below and
   * check the result. */
  
}

/*
 * Vector plus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector
 * which will contain the elementwise sum a_i + b_i.
 */
 
__global__ void vector_add(float *vectorA, float *vectorB, float *resvector) {

  /* STEP 1.3(a) Implement your vector addition here */
  /* STEP 1.3(b) Implement the kernel launch in the main code and check
   * your result for known input */
  
}

/*
 * Vector plus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * input:  factor, a scalar which elementwise multiplies the second vector
 * output: resscalar, pointer to a previously allocated vector which
 * contain the result elementwise a_i + f b_i
 */
 
__global__ void vector_add_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  /* STEP 1.4(a) implement the kernel here, and check the kernel
   * invocation in the main code. */
}


/*
 * Function which seeds a square matrix of ARRAY_SIZE x ARRAY_SIZE
 * with positive values on the diagonal.
 */

__host__ int seedmatrix(float *matrix) {

  int i = 0;
  int j = 0;

  for (j = 0; j < ARRAY_SIZE; j++) {
    for (i = 0; i< ARRAY_SIZE; i++) {
      if (i == j) {
        matrix[j*ARRAY_SIZE +i] = 1.0;
      }
      else{
        matrix[j*ARRAY_SIZE +i] = 0.0;
      }

    }
  }

  return 0;
}



/* Main function */

int main(int argc, char *argv[]) {

  /*
   * This is pre-amble code to deal with multiple GPUs, please do not edit.
   */
  
  /*
   * Check that there are some GPUs, but not too many
   */

   int cuda_device_count = 0;
   cudaGetDeviceCount(&cuda_device_count);

   if (cuda_device_count == 0) {
     printf("No GPU devices found!\n");
     return -1;
   }

  /*
   * We print out the properties of each CUDA device for information.
   */

  int i = 0;
  cudaDeviceProp prop;

  printf("Number of CUDA Devices = %d\n", cuda_device_count);
  for (i = 0; i < cuda_device_count; i++) {
    cudaGetDeviceProperties(&prop, i);
    printf("\tDevice %d : Device name: %s\n", i, prop.name);
  }
  printf("\n");

  /*
   * End pre-amble
   */


  /*
   * Begin main code
   */


  /*
   * Some useful helper sizes and variables
   */
  int j = 0;
  size_t matrix_sz = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);
  size_t vector_sz = ARRAY_SIZE * sizeof(float);
  size_t scalar_sz = 1 * sizeof(float);
  float scalar = 0;

  

  /*
   * Create pointers to hold data on the host
   */
  float *matrixA = NULL;
  float *vectorR = NULL;
  float *vectorB = NULL;
  float *vectorX = NULL;
  float *vectorP = NULL;

  float *vectorRnew = NULL;
  float *vectorXnew = NULL;
  float *vectorPnew = NULL;

  /*
   * Allocate memory on host & test it was successful
   * This is an often missed step and can catch you out
   * We use heap allocations rather than stack for two reasons
   * 1. It makes everything a pointer which marries nicely with cudaMalloc
   * 2. It would be easy to fill the stack space and we cannot use ulimit on all systems to increase it
   */
  matrixA = (float *) calloc(matrix_sz, 1);
  vectorR = (float *) calloc(vector_sz, 1);
  vectorB = (float *) calloc(vector_sz, 1);
  vectorP = (float *) calloc(vector_sz, 1);
  vectorX = (float *) calloc(vector_sz, 1);
  vectorRnew = (float *) calloc(vector_sz, 1);
  vectorXnew = (float *) calloc(vector_sz, 1);
  vectorPnew = (float *) calloc(vector_sz, 1);
  
  if (matrixA == NULL ||\
      vectorR == NULL ||\
      vectorB == NULL ||\
      vectorP == NULL ||\
      vectorX == NULL ||\
      vectorRnew == NULL ||\
      vectorXnew == NULL ||\
      vectorPnew == NULL){
    printf("Error allocating host memory.\n");
    return 1;
  }

  /*
   * Initialise host arrays
   * Calloc should push these to be 0, but using this method we can pick anything.
   * Having a non-zero initialiser for the output array can help spot problems if we never expect a 0 in the output
   */
  seedmatrix(matrixA);
  for (j = 0; j < ARRAY_SIZE; j++){
    vectorP[j] = 0.0;
    vectorB[j] = 1.0;
    vectorX[j] = 2.0;
    vectorRnew[j] = 0.0;
    vectorXnew[j] = 0.0;
    vectorPnew[j] = 0.0; 
  }



  
  /*
   * Create pointers to hold data on the device
   */  
  float *device_matrixA = NULL;
  float *device_vectorR = NULL;
  float *device_vectorB = NULL;
  float *device_vectorX = NULL;
  float *device_vectorP = NULL;
  
  float *device_vectorRnew = NULL;
  float *device_vectorXnew = NULL;
  float *device_vectorPnew = NULL;
  float *device_vectorAP = NULL;
    
  float *device_scalar = NULL;



  /*
   * Set up some useful values
   * threadsPerBlock is as in previous exercises and simply a cast from the macro define
   * nBlocks is split across the number of devices we have
   */
  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  dim3 nBlocks(NUM_BLOCKS/cuda_device_count);


  printf("numBlocks: %d\n", (NUM_BLOCKS/cuda_device_count));
  printf("threadsPerBlock: %d\n", THREADS_PER_BLOCK);

  /*
   * The compiler ignores pragmas statements which it cannot parse, so this can live outside the guard
   * The value of cuda_k will be 0 for a serial case so we get a single iteration of this loop
   * and ergo a single thread of execution.
   */

  for (int cuda_device = 0; cuda_device < cuda_device_count; cuda_device++) {
   
    cudaSetDevice(cuda_device);
    cudaGetDeviceProperties(&prop, cuda_device);
    
    printf("Allocate for device: %d %s\n", cuda_device, prop.name);
  
    /*
     * Allocate device memory
     * This is done inside the loop to give us some flexibility in a multi-GPU case
     */
    cudaMalloc(&device_matrixA, matrix_sz/cuda_device_count);
    checkCUDAError("Device matrixA allocation");
    cudaMalloc(&device_vectorR, vector_sz);
    checkCUDAError("Device vectorR allocation");
    cudaMalloc(&device_vectorB, vector_sz);
    checkCUDAError("Device vectorB allocation");
    cudaMalloc(&device_vectorP, vector_sz);
    checkCUDAError("Device vectorP allocation");
    cudaMalloc(&device_vectorX, vector_sz);
    checkCUDAError("Device vectorX allocation");

    cudaMalloc(&device_vectorRnew, vector_sz);
    checkCUDAError("Device vectorRnew allocation");
    cudaMalloc(&device_vectorPnew, vector_sz);
    checkCUDAError("Device vectorPnew allocation");
    cudaMalloc(&device_vectorXnew, vector_sz);
    checkCUDAError("Device vectorXnew allocation");

    cudaMalloc(&device_vectorAP, vector_sz);
    checkCUDAError("Device vectorAP allocation");

    cudaMalloc(&device_scalar, scalar_sz);
    checkCUDAError("Device vectorXnew allocation");



    /*
     * This is the start of the initialisation step
     * We must compute an initial residual r_0, and set p = r_0
     */

    
    /*
     * Copy arrays and matrices to device(s)
     * The offset arrangement helps with >1 GPU
     */
    cudaMemcpy(device_matrixA, matrixA, matrix_sz, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy: H2D matrix");
    cudaMemcpy(device_vectorX, vectorX, vector_sz, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy: H2D vectorX");

    /*
     * Compute Ax_0 and keep the result vector in device memory
     */

    /* STEP 1.2(b) Use matrix_vector<<<>>>(); */
   
    /*
     * Compute the initial residual r_0 = b - (Ax_0) 
     */
    cudaMemcpy(device_vectorB, vectorB, vector_sz, cudaMemcpyHostToDevice);

    /* STEP 1.4(b) use (with f = -1.0)  vector_add_factor<<<>>>(); */

    /*
     * Copy the initial residual vector back to the host
     */
    cudaMemcpy(vectorR, device_vectorR, vector_sz, cudaMemcpyDeviceToHost);
    
    /* Set p_0 = r_0, copy this initial r to device p host side only! */
    memcpy(vectorP, vectorR, vector_sz);


    scalar = 0;
    cudaMemcpy(device_scalar, &scalar, scalar_sz, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy: H2D scalar");
    cudaDeviceSynchronize();

    /*
     * Compute r_0 r_0 and store as device_scalar
     */
    /* STEP 1.1(b) Implement the appropriate kernel configuration and
     * for the initial values given, check you have the correct result.
     * Remember to copy result back to host.

     vector_vector<<<>>>(); */
    
    
    float initial_rs = scalar;
    printf("Initial Rs = %f\n", initial_rs);
    float rsold = initial_rs;
    float beta = 0;
    float alpha = 0;
    float rsnew = 0;

    /*
     * This is the end of the initialisation step
     * We have derived an initial R_0, computed Rs and set P = R_0
     */


    /*
     * This is the start of the main loop
     * We now need to compute alpha, then R_k+1, beta, P_k+1 etc.
     * Once we have computed the value of (R_k+1)s, ie the updated residual, we can stop.
     */   
    int k = 0;
    for (k = 0; k < ARRAY_SIZE; k++){
    
      /*
       * Compute vector Ap_k and store, temporarily, in Pnew
       */
      /* STEP 1.2(b) Use matrix_vector<<<>>>(); */

      /*
       * Compute Ap_k dot p_k
       */

      /* STEP 1.1(b) vector_vector<<<>>>(); */


      /*
       * Compute Alpha
       */
      alpha = 0;
      alpha = rsold / scalar;

      /*
       * Compute x_k+1 = x_k + alpha p_k
       * Store in Xnew
       */
      /* STEP 1.4(b) Use vector_add_factor<<<>>>(): */

      /*
       * Compute r_k+1 = r_k - alpha Ap_k
       * Store in Rnew
       */
      /* STEP 1.4(b) Use vector_add_factor<<<>>>(): */
  

      /* Calculate beta = r_k+1 r_k+1 / r_k r_k
       * Recall that we have the denominator r_k r_k as "rsold" */

      scalar = 0;
      /* STEP 1.1(b) vector_vector<<<>>>(); */
      rsnew = 0;

      beta = rsnew / rsold;

      /* Compute  p_k+1 = r_k+1 + beta p_k  and store in "Pnew" */

      /* STEP 1.4(b) vector_add_factor<<<>>>(); */

      /*
       * Set up for next iteration; copy host vectors.
       */
      rsold = rsnew;
      memcpy(vectorP, vectorPnew, vector_sz);
      memcpy(vectorR, vectorRnew, vector_sz);
      memcpy(vectorX, vectorXnew, vector_sz);
      
    }

    /* STEP 1.5 Recover solution x vector to host */

    /*
     * Free the device memory
     */
    cudaFree(device_matrixA);
    cudaFree(device_vectorR);
    cudaFree(device_vectorB);
    cudaFree(device_vectorP);
    cudaFree(device_vectorX);
    cudaFree(device_vectorRnew);
    cudaFree(device_vectorPnew);
    cudaFree(device_vectorXnew);
    cudaFree(device_vectorAP);
    cudaFree(device_scalar);
    
  }


  /*
   * Print the output vector and then free the host memory
   */

  for (i = 0; i < ARRAY_SIZE; i++) {
    printf("%d ", vectorX[i]);
  }
  printf("\n\n");

  free(matrixA);
  free(vectorR);
  free(vectorB);
  free(vectorP);
  free(vectorX);
  free(vectorRnew);
  free(vectorPnew);
  free(vectorXnew);

  return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char * msg) {

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

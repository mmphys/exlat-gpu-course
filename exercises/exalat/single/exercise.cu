/*
 * Skeleton for Very Basic Linear Solver Development
 *
 * Nick Johnson, EPCC && ExaLAT.
 */

#include <stdio.h>
#include <stdlib.h>

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* Utility Function to enumerate devices from SLURM/PBS etc */
void deviceEnumerator(int *cuda_devices, int *cuda_device_count);

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

/* Define max number of devices we expect per node. It's currently 8 on Cirrus, so we keep to that for now. */
#define MAX_DEVICES 8


/*
 * Matrix Vector product
 * input:  matrix, pointer to a previously allocated 1D matrix
 * input:  vector, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void matrix_vector(float *matrix, float *vector, float *resvector) {

  /*
   * Implement Matrix Vector here
   */
  
}

/*
 * Vector Vector product (dot product)
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resscalar, pointer to a previously allocated scalar which will contain the result
 */
__global__ void vector_vector(float *vectorA, float *vectorB, float *resscalar) {

  /*
   * Implement your vector dot product here
   */  

}

/*
 * Vector plus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_add(float *vectorA, float *vectorB, float *resvector) {

  /*
   * Implement your vector add here
   */   
  
}

/*
 * Vector plus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * input:  factor, a scalar which elementwise multiplies the second vector
 * output: resscalar, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_add_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  /*
   * Implement your vector multi with factor here
   */
  
}




/*
 * Vector minus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resscalar, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus(float *vectorA, float *vectorB, float *resvector) {

  /*
   * Implement your vector subtraction here
   */
  
}

/*
 * Vector minus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * input:  factor, a scalar which multiplies the second vector elementwise
 * output: resscalar, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  /*
   * Implement your vector subtraction with factor here
   */
  
}

/*
 * Function which seeds a square matrix of ARRAY_SIZE x ARRAY_SIZE with positive values on the leading diag.
 */
int seedvectors(float *matrix){

  int i = 0;
  int j = 0;

  for (j = 0; j < ARRAY_SIZE; j++){
    for (i = 0; i< ARRAY_SIZE; i++){
      if (i == j){
        matrix[j*ARRAY_SIZE +i] = (float)rand()/RAND_MAX;
      }
      else{
        matrix[j*ARRAY_SIZE +i] = 0;
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
   * cuda_devices holds the handles of the cuda devices we need to pass around in a parallel case
   */
  int *cuda_devices = (int*) calloc(MAX_DEVICES, sizeof(int));
  int cuda_device_count = 0;

  
  /*
   * Check that there are some GPUs, but not too many
   */
  deviceEnumerator(cuda_devices, &cuda_device_count);
  if (cuda_device_count == -1 || cuda_device_count > MAX_DEVICES){
    printf("Error enumerating CUDA devices - found %d.\n Exiting.\n", cuda_device_count);
    return 1;
  }


  /*
   * We print out the properties of the CUDA devices (GPUs in this case, but could be CPUs etc)
   * This is useful to know, but also servies as a quick check we can access the devices
   */
  int i = 0;
  int cuda_k = 0;
  int deviceNum = 0;
  cudaDeviceProp prop;

  printf("Number of CUDA Devices = %d\n", cuda_device_count);
  for (i = 0; i < cuda_device_count; i++){
    deviceNum = cuda_devices[i];
    cudaGetDeviceProperties(&prop, deviceNum);
    printf("\tDevice %d : Device name: %s\n", deviceNum, prop.name);
    
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
  seedvectors(matrixA);
  for (j = 0; j < ARRAY_SIZE; j++){
    vectorP[j] = 0;
    vectorB[j] = 1.0;
    vectorX[j] = 0;
    vectorRnew[j] = 0;
    vectorXnew[j] = 0;
    vectorPnew[j] = 0; 
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

  for (cuda_k = 0; cuda_k < cuda_device_count; cuda_k++) {
   
    deviceNum = cuda_devices[cuda_k];
    cudaSetDevice(deviceNum);
    cudaGetDeviceProperties(&prop, deviceNum);
    
    printf("%d:  Device name: %s\n", deviceNum, prop.name);
  
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
     * We must derive an initial R_0, compute Rs and set P = R_0
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
     * Compute the first step Ax and save somewhere (Xnew??)
     */
    matrix_vector<<<>>>();
   
    /*
     * Compute the second step b - (Ax) and put in r
     */
    cudaMemcpy(device_vectorB, vectorB, vector_sz, cudaMemcpyHostToDevice);
    vector_minus<<<>>>();

    /*
     * Copy this back to the host
     */
    cudaMemcpy(vectorR, device_vectorR, vector_sz, cudaMemcpyDeviceToHost);
   
  
    
    // Since p = r, copy this initial r to device p host side only!
    memcpy(vectorP, vectorR, vector_sz);

    

    
    scalar = 0;
    cudaMemcpy(device_scalar, &scalar, scalar_sz, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy: H2D scalar");
    cudaDeviceSynchronize();

    /*
     * Compute R . R into device_scalar
     */
    vector_vector<<<>>>();
    
    
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
    int mainloop = 0;
    for (mainloop = 0; mainloop < ARRAY_SIZE; mainloop++){
    
      /*
       * Compute A times P and store, temporarily in Pnew
       */
      matrix_vector<<<>>>();

            /*
       * Compute Ap dot P
       */
      vector_vector<<<>>>();



      /*
       * Compute Alpha
       */
      alpha = 0;
      alpha = rsold / scalar;

      /*
       * Compute x_k+1 = x_k + alpha.*P_k
       * Store in Xnew
       */
      vector_add_factor<<<>>>():

      /*
       * Compute R_k+1 = R_k - alpha.*(AP_k)
       * Store in Rnew
       */
      vector_minus_factor<<<>>>():
  

      // Calculate Beta
      // Rnew dot Rnew / R dot R
      scalar = 0;
      vector_vector<<<>>>();
      rsnew = 0;

    
    
      beta = rsnew / rsold;

      // Make Pnew = Rnew + Beta P   
      vector_add_factor<<<>>>();
   
 
    
      /*
       * Set up for next iteration
       */
      rsold = rsnew;
      memcpy(vectorP, vectorPnew, vector_sz);
      memcpy(vectorR, vectorRnew, vector_sz);
      memcpy(vectorX, vectorXnew, vector_sz);
      
    }
    
    /*
     * Herein lies the end of the first iteration.
     */
    
    

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
    
  } // end of cuda_k loop


  /*
   * Print the output vector and then free the host memory
   */
  // for (i = 0; i < ARRAY_SIZE; i++) {
  //   printf("%d ", host_result_vector[i]);
  // }
  // printf("\n\n");

  // int sum = 0;
  // for (i = 0; i < ARRAY_SIZE; i++){
  //   sum += vectorR[i] * vectorR[i];
  // }
  

  // printf("Scalar = %d\n", scalar);
  // printf("Sum = %d\n", sum);
  

  free(matrixA);
  free(vectorR);
  free(vectorB);
  free(vectorP);
  free(vectorX);
  free(vectorRnew);
  free(vectorPnew);
  free(vectorXnew);


  // Fin.
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


/*
 * Print device details
 *
 * Based on code available at:
 * https://wiki.sei.cmu.edu/confluence/display/c/STR06-C.+Do+not+assume+that+strtok%28%29+leaves+the+parse+string+unchanged
 */
void deviceEnumerator(int *cuda_devices, int *cuda_device_count){

  char * tokenized = NULL;

  /*
   * CUDA_VISIBLE_DEVICES may not be present on all systems
   * Or, another env. variable might be used
   * This works for Cirrus at EPCC
   */
  const char* s = getenv("CUDA_VISIBLE_DEVICES");

  // If s is NULL, CUDA_VISIBLE_DEVICES was empty, ie unset and, on productions systems, means no GPUs available
  if (s == NULL){
    *cuda_device_count = -1; // Flag as error
    return;
  }

  // If we cannot allocate enough memory here, something has gone wrong
  char * copy = (char *) malloc(strlen(s) + 1);
  if (copy == NULL) {
    *cuda_device_count = -1; // Flag as error
    return;
  }


  /*
   * Iterate over a copy of s (called copy) and look for device handles
   * Store those in the cuda_devices array, and increment the device count local_cdc
   * Return local_cdc as cuda_device_count
   */
  int local_cdc = 0; 
  strcpy(copy, s);
  
  tokenized = strtok(copy, ",");
  *cuda_devices++ = atoi(tokenized);
  local_cdc++;

  while (tokenized = strtok(0, ",")) {
    *cuda_devices++ = atoi(tokenized);
    local_cdc++;
  }

  *cuda_device_count = local_cdc;

}
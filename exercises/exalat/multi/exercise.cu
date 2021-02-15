/*
 * Skeleton for Very Basic Linear Solver Development
 * Multi-GPU Version
 *
 * Nick Johnson, EPCC && ExaLAT.
 */


/*
 * Instructions
 * Step 1 - Add the code for the relevant kernels (matrix vector, vector dot product, vector +/- vector etc.
 * Step 2 - Copy Data to the Devices, thinking carefully about whether each device needs a copy of the complete data, or only part of it.
 * Step 3 - Call the kernels to compute the initial residual
 * Step 4 - Uncomment the main loop and call the necessary memcpy's, and insert the relevant kernels
 * Step 5 - Check your answer by observing the output, it should match that of the single GPU example for an indentical input
 *
 * Please do not edit the OpenMP lines, unless you are experienced with OpenMP programming (or wish to spend a lot of time debugging!)
 */

#include <stdio.h>
#include <stdlib.h>
#if defined(_OPENMP)
#include <omp.h> // Needed for parallel GPU access
#endif

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
   * Step 1 - Add the matrix vector kernel here
   * Verify it works with any number of blocks and threads
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
   * Step 1 - Add the vector dor product kernel here
   * Verify it works with any number of blocks and threads
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
   * Step 1 - Add the vector add kernel here
   * Verify it works with any number of blocks and threads
   */  
}

/*
 * Vector plus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_add_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  /*
   * Step 1 - Add the vector add factor kernel here
   * Verify it works with any number of blocks and threads
   */
  
}




/*
 * Vector minus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus(float *vectorA, float *vectorB, float *resvector) {

  /*
   * Step 1 - Add the vector minus kernel here
   * Verify it works with any number of blocks and threads
   */  
}

/*
 * Vector minus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  /*
   * Step 1 - Add the vector minus factor kernel here
   * Verify it works with any number of blocks and threads
   */
  
}


int seedmatrix(float *matrix){

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


  for (j = 0; j < ARRAY_SIZE; j++){
    for (i = 0; i < ARRAY_SIZE; i++){
      if (matrix[j*ARRAY_SIZE +i] != 0){
  	printf("%f ", matrix[j*ARRAY_SIZE +i]);
      }
    }
    printf("\n");
  }
  printf("\n\n");


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
  float initial_rs = 0;
  float rsold = 0;
  float beta = 0;
  float alpha = 0;
  float rsnew = 0;
  

  

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
  float *vectorAP = NULL;
  float *scalar = NULL;
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
  vectorAP = (float *) calloc(vector_sz, 1);
  scalar = (float*) calloc(scalar_sz, cuda_device_count);

  
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
    vectorP[j] = 0;
    vectorB[j] = 1.0;
    vectorX[j] = 0;
    vectorRnew[j] = 0;
    vectorXnew[j] = 0;
    vectorPnew[j] = 0;
    vectorAP[j] = 0;
  }


  int mainloop = 0;
  
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
   * offset and v_offset are as for nBlocks
   */
  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  dim3 threadsPerBlockover2(THREADS_PER_BLOCK/2);
  dim3 nBlocks(NUM_BLOCKS/cuda_device_count);
  int m_offset = NUM_BLOCKS*ARRAY_SIZE/cuda_device_count;
  int v_offset = ARRAY_SIZE/cuda_device_count;



  printf("numBlocks: %d\n", (NUM_BLOCKS/cuda_device_count));
  printf("threadsPerBlock: %d\n", threadsPerBlock);
  printf("threadsPerBlockover2: %d\n", threadsPerBlockover2);
  printf("v_offset: %d\n", v_offset);
  printf("m_offset: %d\n", m_offset);
  printf("\n\n");

  /*
   * This tricky input guard combination allows the parallelisation code to live here when run in serial without compilation issues
   */
#if defined(_OPENMP)
  omp_set_num_threads(cuda_device_count);
#endif


 

  /*
   * The compiler ignores pragmas statements which it cannot parse, so this can live outside the guard
   * The value of cuda_k will be 0 for a serial case so we get a single iteration of this loop
   * and ergo a single thread of execution.
   */
#pragma omp parallel default(shared) private(deviceNum, prop, device_matrixA, device_vectorR, device_vectorB, device_vectorP, device_vectorX, device_vectorRnew, device_vectorPnew, device_vectorXnew, device_vectorAP, cuda_k, device_scalar, i, j)
  {
    printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    

    /*
     * This loop is serialized and allocated the memory on each GPU.
     * It could be parallelised, but probably not to much effect except at large scale.
     */
    cuda_k = omp_get_thread_num();
      deviceNum = cuda_devices[cuda_k];
      cudaSetDevice(deviceNum);
      cudaGetDeviceProperties(&prop, deviceNum);    
      //printf("Thread: %d\t\tDevice Num: %d\t\t Device name: %s\n", omp_get_thread_num(), deviceNum, prop.name);
      scalar[cuda_k] = 0;

     /*
       * We save a slightly confusing case in the 1 GPU situation, but otherwise check our offsets
       */
      if (cuda_device_count > 1){
	printf("Device = %d : v_offset = %d\n", deviceNum, v_offset*deviceNum);
	printf("Device = %d : m_offset = %d\n", deviceNum, m_offset*deviceNum);
      }


    
      /*
       * Allocate device memory
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
    
#pragma omp barrier    
   

      /*
       * This is the start of the initialisation step
       * We must derive an initial R_0, compute Rs and set P = R_0
       */

    
      /*
       * Copy arrays and matrices to device(s)
       * The offset arrangement helps with >1 GPU
       */
      // Step 2 - ensure the following cudaMemcpy's place data in the right places, and think about the sizes of vectors
      // The first HostToDevice and DeviceToHost are given to show examples of copying partial vectors/matrices
      cudaMemcpy(device_matrixA, matrixA+(deviceNum * m_offset), matrix_sz/cuda_device_count, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D matrix");
      cudaMemcpy(); // Vector X
      checkCUDAError("Memcpy: H2D vectorX");

      /*
       * Compute the first step Ax and save somewhere
       */
      // Step 3 Call the following kernels with appropriate parameters
      matrix_vector<<<nBlocks,threadsPerBlock>>>(); // Perform Xnew = AX
      cudaMemcpy(vectorXnew+(deviceNum * v_offset), device_vectorXnew, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      checkCUDAError("kernel invocation");

      

      /*
       * Compute the second step b - (Ax) and store in r
       */
      cudaMemcpy(); // Copy b to each device
      checkCUDAError("Memcpy: H2D Rnew");
      cudaDeviceSynchronize();
      vector_minus<<<>>>(); // Step 3 - think carefully about exactly how many threads are needed here!
      checkCUDAError("Kernel: Rnew");
      cudaDeviceSynchronize();
      /*
       * Copy this back to the host
       */
      // Step 2 - ensure the host has a complete copy of vector r
      cudaMemcpy(); 
      cudaDeviceSynchronize();
      checkCUDAError("Memcpy: D2H Rnew");
      
    
      scalar[cuda_k] = 0;
      cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D scalar");
      cudaDeviceSynchronize();


      // Step 3 - computing the initial residual using a vector dot product, one half on each GPU
      // Think carefully about the number of threads/GPU and the sizes of any vectors you supply
      vector_vector<<<>>>(); // Remember the vector is 1 block long!
      checkCUDAError("Kernel: vectorR (init)");
      cudaDeviceSynchronize();

      // An example of copying back two parts of the scalar to the host
      cudaMemcpy(scalar+deviceNum, device_scalar, scalar_sz, cudaMemcpyDeviceToHost);
      checkCUDAError("Memcpy: D2H scalar (init)");
      cudaDeviceSynchronize();

    

      /*
       * This strange construct ensures that one host thread computes the initial residual from parts
       * And then updates the variable "rsnew" on BOTH threads.
       * With the flush statements, this would be a race condition
       */
#pragma omp barrier    
#pragma omp single
    {
      memcpy(vectorP, vectorR, vector_sz);
      float s_scalar = 0;
      for (i = 0; i < cuda_device_count; i++){
      	s_scalar += scalar[i];
      }
      initial_rs = s_scalar;
      printf("Initial Rs = %f\n", initial_rs);
      rsold = initial_rs;
      rsnew = rsold;
#pragma omp flush(rsnew)
#pragma omp flush(initial_rs)
#pragma omp flush(rsold)
      

    }

    /*
     * This is the end of the initialisation step
     * We have derived an initial R_0, computed Rs and set P = R_0
     */

    

    // Step 4 - uncomment the below and try to complete the rest of the algorithm
    // Use your single-GPU code as a guide
    // As a hint, I print out the values of rsnew, alpha and beta each iteration so I can quickly compare


//     /*
//      * This is the start of the primary loop. We need to use a while loop since OpenMP gets upset about non-parallel for loops.
//      */
//     while (mainloop < ARRAY_SIZE && sqrt(rsnew) > 1e-5){
   
      
//       cudaMemcpy(); // Push P to device
// 	cudaDeviceSynchronize();
// 	checkCUDAError("Memcpy: D2H P");

// 	// Zero out device_vectorAP, may not be strictly needed
// 	cudaMemcpy();    
// 	matrix_vector<<<>>>(); // Compute AP and store in device_vectorAP
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");
// 	cudaMemcpy(vectorAP+(deviceNum * v_offset), device_vectorAP, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost); // Keep a copy of the compete AP on the host - is it needed ?
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");


// 	/*
// 	 * Compute Ap dot P
// 	 */
// 	scalar[cuda_k] = 0;
// 	cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
// 	vector_vector<<<>>>(); // P dot Ap 
// 	cudaDeviceSynchronize();
// 	cudaMemcpy(&scalar[cuda_k], device_scalar, scalar_sz, cudaMemcpyDeviceToHost);
// 	checkCUDAError("Memcpy: D2H vector");

// #pragma omp barrier

	
// #pragma omp single
//       {
// 	float s_scalar = 0;
// 	for (i = 0; i < cuda_device_count; i++){
// 	  s_scalar += scalar[i];
// 	}
      
// 	/*
// 	 * Compute Alpha
// 	 */
      
// 	alpha = rsold / s_scalar;
// #pragma omp flush(alpha)
      
//       }

// #pragma omp barier


// 	/*
// 	 * Compute x_k+1 = x_k + alpha.*P_k
// 	 * Store in Xnew
// 	 */
//       cudaMemcpy(); // Push X to device?
// 	vector_add_factor<<<>>>();
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");
// 	cudaMemcpy(); // Keep vectorXnew on the host updated with a full copy of Xnew
// 	cudaDeviceSynchronize();    
// 	checkCUDAError("Memcpy: D2H Xnew");

// 	/*
// 	 * Compute R_k+1 = R_k - alpha.*(AP_k)
// 	 * Store in Rnew
// 	 */
// 	cudaMemcpy(); // What goes here?
// 	vector_minus_factor<<<>>>();
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");

// 	cudaMemcpy(); // Keep Rnew on the host up to date
// 	cudaDeviceSynchronize();    
// 	checkCUDAError("Memcpy: D2H Rnew");


// #pragma omp barrier
// 	// Calculate Beta
// 	// Rnew dot Rnew / R dot R
// 	scalar[cuda_k] = 0;
// 	cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
// 	cudaDeviceSynchronize();
// 	vector_vector<<<>>>();    // Rnew dot Rnew
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");
// 	cudaMemcpy(scalar+deviceNum, device_scalar, scalar_sz, cudaMemcpyDeviceToHost);

// #pragma omp barrier


// #pragma omp single
//       {
// 	rsnew = scalar[0] + scalar[1];
// 	beta = rsnew / rsold;
// #pragma omp flush(beta)
// #pragma omp flush(rsnew)
//       }
      
      
// #pragma omp barrier

// 	// Make Pnew = Rnew + Beta P   
//       vector_add_factor<<<>>>(); // Think carefully here about how much of vector P is on each GPU and set the pointer arg appropriately
// 	cudaDeviceSynchronize();
// 	checkCUDAError("kernel invocation");
// 	cudaMemcpy(); // Pnew to host
// 	cudaDeviceSynchronize();    
// 	checkCUDAError("Memcpy: D2H Pnew");

// #pragma omp barrier
   
// #pragma omp single
//       {
    
// 	printf("Thread %d\t\tML: %d\t\tRsnew = %f\t\tBeta = %f\t\tAlpha = %f\n", omp_get_thread_num(), mainloop, rsnew, beta, alpha);
// 	/*
// 	 * Set up for next iteration
// 	 */
// 	rsold = rsnew;
// 	memcpy(vectorP, vectorPnew, vector_sz);
// 	memcpy(vectorR, vectorRnew, vector_sz);
// 	memcpy(vectorX, vectorXnew, vector_sz);
// #pragma omp flush(rsold)
      
//       }
// #pragma omp barrier
    
//       /*
//        * Herein lies the end of the first iteration.
//        */
      
// #pragma omp single
//       {
// 	// printf("T: %d \t\tMainloop %d\n", omp_get_thread_num(), mainloop);
// 	mainloop++;
// #pragma omp flush(mainloop)
	
//       }


      
      
//     } // end of mainloop construct...  


// Don't delete this part - or you'll leak memory on the device.
    
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


  // This would be a good place to print out the final value of vectorX since it's what you set out to find!
    
    

  /*
   * Free the device memory
   */

  free(matrixA);
  free(vectorR);
  free(vectorB);
  free(vectorP);
  free(vectorX);
  free(vectorRnew);
  free(vectorPnew);
  free(vectorXnew);
  free(vectorAP);


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
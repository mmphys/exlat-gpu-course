/*
 * Skeleton for Very Basic Linear Solver Development
 *
 * Nick Johnson, EPCC && ExaLAT.
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

  int m_idx, v_idx, b_idx;
  __shared__ float temp;
  __shared__ float temp_array[THREADS_PER_BLOCK];
  m_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // matrix index
  b_idx = blockIdx.x; // block index
  v_idx = threadIdx.x; //vector index

  temp = 0;
  temp_array[v_idx] = matrix[m_idx] * vector[v_idx];
  __syncthreads();
  
  atomicAdd(&temp, temp_array[v_idx]);
  if (v_idx == 0){
    resvector[b_idx] = temp;
  }

}

/*
 * Vector Vector product (dot product)
 * input:  vectorB, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resscalar, pointer to a previously allocated scalar which will contain the result
 */
__global__ void vector_vector(float *vectorA, float *vectorB, float *resscalar) {

  __shared__ float temp_array[THREADS_PER_BLOCK];
  int t_idx;
  t_idx = threadIdx.x; // thread index in each threadblock
  int v_idx;
  v_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // vector index
  

  temp_array[t_idx] = vectorA[v_idx] * vectorB[v_idx];
  *resscalar = 0;
  __syncthreads();

  if (t_idx == 0){
    float sum = 0;
    for (int i = 0; i < blockDim.x; i++){
      sum += temp_array[i];
    }
    atomicAdd(resscalar, sum);
  }
  

}

/*
 * Vector plus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_add(float *vectorA, float *vectorB, float *resvector) {

  int v_idx;
  v_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // vector index
  resvector[v_idx] = vectorA[v_idx] + vectorB[v_idx];
  
}

/*
 * Vector plus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_add_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  int v_idx;
  v_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // vector index
  resvector[v_idx] = vectorA[v_idx] + (factor * vectorB[v_idx]);
  
}




/*
 * Vector minus Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus(float *vectorA, float *vectorB, float *resvector) {

  int v_idx;
  v_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // vector index
  resvector[v_idx] = vectorA[v_idx] - vectorB[v_idx];
  
}

/*
 * Vector minus Factor * Vector
 * input:  vectorA, pointer to a previously allocated vector
 * input:  vectorB, pointer to a previously allocated vector
 * output: resvector, pointer to a previously allocated vector which will contain the result
 */
__global__ void vector_minus_factor(float *vectorA, float *vectorB, float factor, float *resvector) {

  int v_idx;
  v_idx = threadIdx.x + ((blockIdx.x) * blockDim.x); // vector index
  resvector[v_idx] = vectorA[v_idx] - (factor * vectorB[v_idx]);
  
}


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

#if defined(DEBUG)
  for (j = 0; j < ARRAY_SIZE; j++){
    for (i = 0; i < ARRAY_SIZE; i++){
      if (matrix[j*ARRAY_SIZE +i] != 0){
  	printf("%f ", matrix[j*ARRAY_SIZE +i]);
      }
    }
    printf("\n");
  }
  printf("\n\n");
#endif

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
  seedvectors(matrixA);
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



#if defined(DEBUG)
  printf("Host Debug:\n");
  printf("numBlocks: %d\n", (NUM_BLOCKS/cuda_device_count));
  printf("threadsPerBlock: %d\n", threadsPerBlock);
  printf("threadsPerBlockover2: %d\n", threadsPerBlockover2);
  printf("v_offset: %d\n", v_offset);
  printf("m_offset: %d\n", m_offset);
  printf("\n\n");
#endif

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
      printf("Thread: %d\t\tDevice Num: %d\t\t Device name: %s\n", omp_get_thread_num(), deviceNum, prop.name);
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
    
      cudaMemcpy(device_matrixA, matrixA+(deviceNum * m_offset), matrix_sz/cuda_device_count, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D matrix");
      cudaMemcpy(device_vectorX, vectorX, vector_sz, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D vectorX");

      /*
       * Compute the first step Ax and save somewhere
       */
      matrix_vector<<<nBlocks,threadsPerBlock>>>(device_matrixA, device_vectorX, device_vectorXnew); // Stash the result in Xnew
      cudaMemcpy(vectorXnew+(deviceNum * v_offset), device_vectorXnew, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      checkCUDAError("kernel invocation");

      

      /*
       * Compute the second step b - (Ax) and put in r
       */
      cudaMemcpy(device_vectorB, vectorB+(deviceNum * v_offset), vector_sz/cuda_device_count, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D Rnew");
      cudaDeviceSynchronize();
      vector_minus<<<1, threadsPerBlockover2>>>(device_vectorB, device_vectorXnew, device_vectorR);
      checkCUDAError("Kernel: Rnew");
      cudaDeviceSynchronize();
      /*
       * Copy this back to the host
       */
      cudaMemcpy(vectorR+(deviceNum * v_offset), device_vectorR, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      checkCUDAError("Memcpy: D2H Rnew");
      
    
      scalar[cuda_k] = 0;
      cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
      checkCUDAError("Memcpy: H2D scalar");
      cudaDeviceSynchronize();


      // This is double done, should be half per GPU!
      //cudaMemcpy(device_vectorR, vectorR, vector_sz, cudaMemcpyHostToDevice);
      vector_vector<<<1, threadsPerBlockover2>>>(device_vectorR, device_vectorR, device_scalar); // Remember the vector is 1 block long!
      checkCUDAError("Kernel: vectorR (init)");
      cudaDeviceSynchronize();

      // Both scalars should be equal here
      cudaMemcpy(scalar+deviceNum, device_scalar, scalar_sz, cudaMemcpyDeviceToHost);
      checkCUDAError("Memcpy: D2H scalar (init)");
      cudaDeviceSynchronize();

    
  
#pragma omp barrier    
#pragma omp single
    {
      memcpy(vectorP, vectorR, vector_sz);
      float s_scalar = 0;
      for (i = 0; i < cuda_device_count; i++){
      	s_scalar += scalar[i];
      }
      //initial_rs = scalar[0];
      // for (i = 0; i < cuda_device_count; i++){
      // 	scalar[i] = s_scalar;
      // }
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




    /*
     * This is the start of the primary loop. We need to use a while loop since OpenMP gets upset about non-parallel for loops.
     */
    while (mainloop < ARRAY_SIZE && sqrt(rsnew) > 1e-5){
   
      
	cudaMemcpy(device_vectorP, vectorP, vector_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCUDAError("Memcpy: D2H P");

	// Zero out device_vectorAP, may not be strictly needed
	cudaMemcpy(device_vectorAP, vectorAP, vector_sz, cudaMemcpyHostToDevice);    
	matrix_vector<<<nBlocks,threadsPerBlock>>>(device_matrixA, device_vectorP, device_vectorAP);
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");
	cudaMemcpy(vectorAP+(deviceNum * v_offset), device_vectorAP, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");


	/*
	 * Compute Ap dot P
	 */
	scalar[cuda_k] = 0;
	cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
	//cudaMemcpy(device_vectorAP, vectorAP, vector_sz, cudaMemcpyHostToDevice);
	vector_vector<<<1, threadsPerBlockover2>>>(device_vectorP+(deviceNum * v_offset), device_vectorAP, device_scalar); // P dot Ap - both vectors are 1 block long !
	//vector_vector<<<1, threadsPerBlock>>>(device_vectorP, device_vectorAP, device_scalar); // P dot Ap - both vectors are 1 block long !
	cudaDeviceSynchronize();
	cudaMemcpy(&scalar[cuda_k], device_scalar, scalar_sz, cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy: D2H vector");

#pragma omp barrier

	
#pragma omp single
      {
	float s_scalar = 0;
	for (i = 0; i < cuda_device_count; i++){
	  s_scalar += scalar[i];
	}
      
	/*
	 * Compute Alpha
	 */
      
	//alpha = rsold / scalar[0];
	alpha = rsold / s_scalar;
#pragma omp flush(alpha)
      
      }

#pragma omp barier


	/*
	 * Compute x_k+1 = x_k + alpha.*P_k
	 * Store in Xnew
	 */
	cudaMemcpy(device_vectorX, vectorX, vector_sz, cudaMemcpyHostToDevice);
	vector_add_factor<<<1, threadsPerBlockover2>>>(device_vectorX+(deviceNum * v_offset), device_vectorP+(deviceNum * v_offset), alpha, device_vectorXnew);
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");
	cudaMemcpy(vectorXnew+(deviceNum * v_offset), device_vectorXnew, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();    
	checkCUDAError("Memcpy: D2H Xnew");

	/*
	 * Compute R_k+1 = R_k - alpha.*(AP_k)
	 * Store in Rnew
	 */
	cudaMemcpy(device_vectorR, vectorR+(deviceNum * v_offset), vector_sz/cuda_device_count, cudaMemcpyHostToDevice);
	vector_minus_factor<<<1, threadsPerBlockover2>>>(device_vectorR, device_vectorAP, alpha, device_vectorRnew);
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");

	cudaMemcpy(vectorRnew+(deviceNum * v_offset), device_vectorRnew, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();    
	checkCUDAError("Memcpy: D2H Rnew");


#pragma omp barrier
	// Calculate Beta
	// Rnew dot Rnew / R dot R
	scalar[cuda_k] = 0;
	cudaMemcpy(device_scalar, &scalar[cuda_k], scalar_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	vector_vector<<<1, threadsPerBlockover2>>>(device_vectorRnew, device_vectorRnew, device_scalar);    // Rnew dot Rnew
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");
	cudaMemcpy(scalar+deviceNum, device_scalar, scalar_sz, cudaMemcpyDeviceToHost);

#pragma omp barrier


#pragma omp single
      {
	rsnew = scalar[0] + scalar[1];
	beta = rsnew / rsold;
#pragma omp flush(beta)
#pragma omp flush(rsnew)
      }
      
      
#pragma omp barrier

	// Make Pnew = Rnew + Beta P   
	vector_add_factor<<<1, threadsPerBlockover2>>>(device_vectorRnew, device_vectorP+(deviceNum * v_offset), beta, device_vectorPnew);
	cudaDeviceSynchronize();
	checkCUDAError("kernel invocation");
	cudaMemcpy(vectorPnew+(deviceNum * v_offset), device_vectorPnew, vector_sz/cuda_device_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();    
	checkCUDAError("Memcpy: D2H Pnew");

#pragma omp barrier
   
#pragma omp single
      {
    
	printf("Thread %d\t\tML: %d\t\tRsnew = %f\t\tBeta = %f\t\tAlpha = %f\n", omp_get_thread_num(), mainloop, rsnew, beta, alpha);
	/*
	 * Set up for next iteration
	 */
	rsold = rsnew;
	memcpy(vectorP, vectorPnew, vector_sz);
	memcpy(vectorR, vectorRnew, vector_sz);
	memcpy(vectorX, vectorXnew, vector_sz);
#pragma omp flush(rsold)
      
      }
#pragma omp barrier
    
      /*
       * Herein lies the end of the first iteration.
       */
      
#pragma omp single
      {
	// printf("T: %d \t\tMainloop %d\n", omp_get_thread_num(), mainloop);
	mainloop++;
#pragma omp flush(mainloop)
	
      }


      
      
    } // end of mainloop construct...  



    
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 54 * 1024
#define threadsPerBlock 256
#define blocksPerGrid (N + threadsPerBlock - 1) / threadsPerBlock

__global__ void addKernel(const float *a, float *result)
{
	__shared__ float cache_result[threadsPerBlock];
	float temp = 0;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// Computes the sum of squares each thread, maybe more than one time
	while (tid < N)
	{
		temp += a[tid] * a[tid];
		tid += blockDim.x * gridDim.x;
	}
	// Saves the result from each thread within each block 
	// to corresponding position in cache_result which is in
	// shared memory
	cache_result[threadIdx.x] = temp;

	// Uses Reduction to get the sum of array cache_result 
	// corresponding to each block
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (threadIdx.x < i)
		{
			cache_result[threadIdx.x] += cache_result[i + threadIdx.x];
		}
		__syncthreads();
		i /= 2;
	}
	__syncthreads();
	// Saves the sum to the corresponding position of array result
	// using a thread whose id is 0
	if (threadIdx.x == 0)
	{
		result[blockIdx.x] = cache_result[0];
	}
}

int main()
{
	float *arr, *result;
	float *dev_arr, *dev_result;

	arr = (float *)malloc(N * sizeof(float));
	result = (float *)malloc(blocksPerGrid * sizeof(float));
	cudaMalloc(&dev_result, blocksPerGrid * sizeof(float));
	cudaMalloc(&dev_arr, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		arr[i] = i + 1;
	}
	cudaMemcpy(dev_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

	addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_arr, dev_result);

	cudaMemcpy(result, dev_result, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < blocksPerGrid; i++)
	{
		result[0] += result[i];
	}

	printf("The result of computer is: %lf\n", result[0]);
	printf("The correct result is: %lf\n", ((float)N) * ((float)(N + 1)) * ((float)(2 * N + 1)) / 6);

	free(result);
	free(arr);
	cudaFree(dev_result);
	cudaFree(arr);
	return 0;
}
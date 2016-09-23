#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>

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
	__syncthreads();
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
	// Get the time of computing in GPU
	cudaEvent_t g_start, g_stop;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	// Get the time of computing in CPU
	clock_t c_start, c_stop;

	arr = (float *)malloc(N * sizeof(float));
	result = (float *)malloc(blocksPerGrid * sizeof(float));
	cudaMalloc(&dev_result, blocksPerGrid * sizeof(float));
	cudaMalloc(&dev_arr, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		arr[i] = (float)i + 1;
	}

	//cudaEventRecord(g_start, 0);
	cudaMemcpy(dev_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(g_start, 0);
	addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_arr, dev_result);

	cudaMemcpy(result, dev_result, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < blocksPerGrid; i++)
	{
		result[0] += result[i];
	}
	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float GPUelapsedTime;
	cudaEventElapsedTime(&GPUelapsedTime, g_start, g_stop);
	// Get the result using CPU
	double CPUelapsedTime;
	c_start = clock();
	float c = 0;
	for (int i = 0; i < N; i++)
	{
		c += arr[i] * arr[i];
	}
	c_stop = clock();
	CPUelapsedTime = ((double)(c_stop - c_start)) / (CLOCKS_PER_SEC * 1000);
	float correct_result = ((float)N) * ((float)(N + 1)) * ((float)(2 * N + 1)) / 6;
	printf("The result of GPU is:\t%lf\n", result[0]);
	printf("The result of CPU is:\t%lf\n", c);
	printf("The correct result is:\t%lf\n", correct_result);
	printf("The deviation produced by CPU & GPU separately is %lf, %lf\n", correct_result - c, correct_result - result[0]);
	printf("Time in GPU: %lfms\n", GPUelapsedTime);
	printf("Time in CPU: %fms\n", CPUelapsedTime);

	cudaEventDestroy(g_start);
	cudaEventDestroy(g_stop);
	free(result);
	free(arr);
	cudaFree(dev_result);
	cudaFree(arr);
	return 0;
}
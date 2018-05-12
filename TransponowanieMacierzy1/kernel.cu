
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstdlib>

using namespace std;

cudaError_t addWithCuda(unsigned int *tablicaBeforeT, unsigned int *tablicaAfterT, double d1, double d2);
__global__ void countOnGPU(unsigned int *tablica, unsigned int *tablicaT, int iteration);
void generateArrayOnCPU(unsigned int **&tablica);
void showArrayOnCPU(unsigned int **tablica);
void transpositionOfMatrixOnCPU(unsigned int **&tablica, unsigned int **&tablicaT);
void showArrayAfterTranspositionOnCPU(unsigned int **tablica);
void changeTheMatrixDimensionToArrayOnCPU(unsigned int **tablica, unsigned int *tablica2);
void showArrayInOneDimension(unsigned int *tablica);
void showArrayAterTranspositionInOneDimension(unsigned int *tablica);
void generateArrayOnCPU2(unsigned int *tablica);
void transpositionOfMatrixOnCPU2(unsigned int *tablica, unsigned int *tablicaT);

#define numberOfLines 10000
#define numberOfColums 10000
#define size 100000000
#define threads 1024
#define blocks 1024

unsigned int ** matrix = new unsigned int *[numberOfLines];
unsigned int ** matrixT = new unsigned int*[numberOfColums];

unsigned int matrixCPUBeforeT[size];
unsigned int matrixCPUAfterT[size];

unsigned int matrixGPUBeforeT[size];
unsigned int matrixGPUAfterT[size];

int main()
{
	clock_t start,start2;
	srand(time(NULL));
	
	//on CPU
	generateArrayOnCPU(matrix);
	//showArrayOnCPU(matrix);

	start = clock();
	transpositionOfMatrixOnCPU(matrix, matrixT);
	//showArrayAfterTranspositionOnCPU(matrixT);

	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

	//on CPU2
	//generateArrayOnCPU2(matrixCPUBeforeT);
	changeTheMatrixDimensionToArrayOnCPU(matrix, matrixCPUBeforeT);
	//showArrayInOneDimension(matrixCPUBeforeT);

	start2 = clock();
	transpositionOfMatrixOnCPU2(matrixCPUBeforeT, matrixCPUAfterT);
	//showArrayAterTranspositionInOneDimension(matrixCPUAfterT);

	double duration2 = (clock() - start2) / (double)CLOCKS_PER_SEC;

	//on GPU
	changeTheMatrixDimensionToArrayOnCPU(matrix, matrixGPUBeforeT);
	//showArrayInOneDimension(matrixGPUBeforeT);
	
	cudaError_t cudaStatus = addWithCuda(matrixGPUBeforeT, matrixGPUAfterT, duration, duration2);
	
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t addWithCuda(unsigned int *tablicaBeforeT, unsigned int *tablicaAfterT, double d1, double d2)
{
	cudaEvent_t poczatek, koniec;
	cudaError_t cudaStatus;

	unsigned int *dev_tablicaBeforeT = 0;
	unsigned int *dev_tablicaAfterT = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_tablicaBeforeT, size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tablicaAfterT, size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_tablicaBeforeT, tablicaBeforeT, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int amountOfIteration = blocks * threads;

	float timeDurationOnGPU = 0;

	cudaEventCreate(&poczatek);
	cudaEventCreate(&koniec);

	cudaEventRecord(poczatek, 0);

	// Launch a kernel on the GPU with one thread for each element.
	countOnGPU << <blocks, threads >> >(dev_tablicaBeforeT, dev_tablicaAfterT, amountOfIteration);
	
	cudaEventRecord(koniec, 0);
	cudaEventSynchronize(koniec);

	cudaEventElapsedTime(&timeDurationOnGPU, poczatek, koniec);

	cudaEventDestroy(poczatek);
	(cudaEventDestroy(koniec));

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(tablicaAfterT, dev_tablicaAfterT, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//showArrayAterTranspositionInOneDimension(tablicaAfterT);

	cout << "Time on CPU: " << d1 << " s" << endl;
	cout << "Time on CPU2: " << d2 << " s" << endl;
	cout << "Time on GPU: " << timeDurationOnGPU << " ms" << endl;

	cout << " Time on GPU is " << (d1 * 1000) / timeDurationOnGPU << " times faster than CPU" << endl;
	cout << " Time on GPU is " << (d2 * 1000) / timeDurationOnGPU << " times faster than CPU2" << endl;

Error:
	cudaFree(dev_tablicaBeforeT);
	cudaFree(dev_tablicaAfterT);

	return cudaStatus;
}

__global__ void countOnGPU(unsigned int *tablica, unsigned int *tablicaT, int iteration)
{
	int i;
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	for (int j = 0; j <size; j += iteration)
	{

		i = pos + j;
		if (i<size)
			tablicaT[i] = tablica[(i%numberOfLines)*numberOfLines + i / numberOfLines];
	}
}

void generateArrayOnCPU(unsigned int **&tablica)
{
	for (int i = 0; i < numberOfLines; i++)
	{
		tablica[i] = new unsigned int[numberOfColums];
		for (int j = 0; j < numberOfColums; j++)
			tablica[i][j] = (rand() % 10);
	}
}

void transpositionOfMatrixOnCPU(unsigned int **&tablica, unsigned int **&tablicaT)
{
	//gennete the matrixT
	for (int i = 0; i < numberOfColums; i++)
	{
		tablicaT[i] = new unsigned int[numberOfLines];
	}

	for (int i = 0; i < numberOfLines; i++)
	{
		for (int j = 0; j < numberOfColums; j++)
			tablicaT[j][i] = tablica[i][j];
	}
}

void showArrayOnCPU(unsigned int **tablica)
{
	for (int i = 0; i < numberOfLines; i++)
	{
		for (int j = 0; j < numberOfColums; j++)
		{
			cout << tablica[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void showArrayAfterTranspositionOnCPU(unsigned int **tablica)
{
	for (int i = 0; i < numberOfColums; i++)
	{
		for (int j = 0; j < numberOfLines; j++)
		{
			cout << tablica[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void changeTheMatrixDimensionToArrayOnCPU(unsigned int **tablica, unsigned int *tablica2)
{
	for(int i = 0, int a = 0; i < numberOfLines; i++)
	{
		for (int j = 0; j < numberOfColums; j++)
		{
			tablica2[a] = tablica[i][j];
			a++;
		}
	}
}

void showArrayInOneDimension(unsigned int *tablica)
{
	for (int i = 0, int a = 0; i < numberOfLines; i++)
	{
		for (int j = 0; j < numberOfColums; j++)
		{
			cout << tablica[a] << " ";
			a++;
		}
		cout << endl;
	}
	cout << "\n";
}

void showArrayAterTranspositionInOneDimension(unsigned int *tablica) 
{
	for (int i = 0, int a = 0; i < numberOfColums; i++)
	{
		for (int j = 0; j < numberOfLines; j++)
		{
			cout << tablica[a] << " ";
			a++;
		}
		cout << endl;
	}
	cout << "\n";
}

void transpositionOfMatrixOnCPU2(unsigned int *tablica, unsigned int *tablicaT)
{
	for (int i = 0; i < numberOfLines; i++)
	{
		for (int j = 0; j < numberOfColums; j++)
		{
			tablicaT[j + i*numberOfColums] = tablica[j*numberOfLines + i];
		}
	}
}

void generateArrayOnCPU2(unsigned int *tablica)
{
	for (int i = 0; i < size; i++)
	{
		tablica[i] = (rand() % 10);
	}
}
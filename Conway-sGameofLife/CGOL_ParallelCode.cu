/*Name: Brijesh Mavani
CWID: A20406960
University: Illinois Institute of Technology 
Course: Parallel and Distributed Processing
Assignment: 5
*/

/*Conway's Game of Life parallel implementation using CUDA. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <unistd.h>

int iterations,size,i,j,n;
typedef unsigned char ubyte;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void iterate(int n, ubyte* cBoard, ubyte* nBoard)
{
   // Assigning starting position for each thread.
	int x = blockIdx.x * 16 + threadIdx.x;
	int y = blockIdx.y * 16 + threadIdx.y;

	int k = x * n + y;
	int num = 0;
	nBoard[k] = cBoard[k];
	
	
    num+=(x-1 >= 0 && x-1 < n && y >= 0 && y <n )?cBoard[(x-1)*n+y]:0;  // check neighbor at top
	num+=(x+1 >= 0 && x+1 < n && y >= 0 && y <n )?cBoard[(x+1)*n+y]:0; // check neighbor at bottom
	num+=(x >= 0 && x < n && y-1 >= 0 && y-1 <n )?cBoard[x*n+(y-1)]:0;  // check neighbor at left
	num+=(x >= 0 && x < n && y+1 >= 0 && y+1 <n )?cBoard[x*n+(y+1)]:0;  // check neighbor at right
	num+=(x-1 >= 0 && x-1 < n && y-1 >= 0 && y-1 <n )?cBoard[(x-1)*n+(y-1)]:0;  // check neighbor at top-left
	num+=(x-1 >= 0 && x-1 < n && y+1 >= 0 && y+1 <n )?cBoard[(x-1)*n+(y+1)]:0;  // check neighbor at top-right
	num+=(x+1 >= 0 && x+1 < n && y-1 >= 0 && y-1 <n )?cBoard[(x+1)*n+(y-1)]:0;  // check neighbor at bottom-left
	num+=(x+1 >= 0 && x+1 < n && y+1 >= 0 && y+1 <n )?cBoard[(x+1)*n+(y+1)]:0;  // check neighbor at bottom-right

	// Apply game rules:
		
	//Any live cell with fewer than two live neighbors dies, as if caused by under-population	
	//Any live cell with two or three live neighbors lives on to the next generation.
	//Any live cell with more than three live neighbors dies, as if by overcrowding	
     if (num<2||num>3)
	 {
		nBoard[k] =0;
	 }	
	 
	//Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
	if (num==3&&!cBoard[k])
	 {
		nBoard[k] =1; 
	 }
}


void parameters(int argc, char **argv) 
{
	 /* Read command-line arguments */
	if (argc > 1)
	{
		iterations = atoi(argv[1]);
		size = atoi(argv[2]);
		printf("Interations: %d Size: %d\n",iterations,size);
	}
	else
	{
	  printf("Please provide number of interations in terms of 10/100/1000 and size for matrix.\n"); 
	}  

}
void initialize_inputs(ubyte* matrix)
{
	for (i = 0; i < size; i++)
	{
	   for (j = 0; j < size; j++)  
	   {
	     matrix[i * size + j] = rand() % 2;
	   } 
	}
}

void print_inputs(ubyte* matrix)
{
	printf("Legends: a: Alive cell, d: Dead cell.\n");
	printf("Initial Matrix:\n");
    for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			printf("%s ", matrix[i * size + j]? "a|" : "d|");
		}
		printf("\n");
	}

}
int main(int argc, char* argv[])
{
 	/* Process program parameters */
	parameters(argc, argv);
    ubyte* matrix = (ubyte *)malloc(size * size * sizeof(ubyte));
	//Initialize the Matrix
	initialize_inputs(matrix);
	
	//Print the initialized Matrix
	print_inputs(matrix);
		
	printf("Running for total %d iterations.\n", iterations);
	srand(time(0));
	
	//Creating matrix in GPU to load initial data from CPU	
	ubyte* currentmatrix;
	cudaMalloc((void **)&currentmatrix, size * size * sizeof(int));
	if (currentmatrix == NULL)
    {
       printf( "Memory allocation issue for current matrix.\n");
	   return false;
	}
	// Copy initial matrix to GPU currentmatrix.
	cudaMemcpy(currentmatrix, matrix, size * size * sizeof(ubyte), cudaMemcpyHostToDevice);
	cudaCheckErrors("Error when copying the initial matrix to the GPU.\n");

	//Creating matrix in GPU to save intermediate result between each iterations
	ubyte* nextmatrix;
	cudaMalloc((void **)&nextmatrix, size * size * sizeof(ubyte));
	if (nextmatrix == NULL)
    {
       printf( "Memory allocation issue for next matrix.\n");
	   return false;
	}
	
	//initialize nextmatrix as all 0.
	cudaMemset(nextmatrix, 0, size * size * sizeof(ubyte));
	cudaCheckErrors("Error when copying the next matrix to the GPU.\n");
	
	//Defining num of threads and block to execute in GPU
	dim3 threadsPerBlock(16, 16); 
	dim3 numBlocks(size/threadsPerBlock.x,size/threadsPerBlock.y);
	
	struct timeval starttime;
	gettimeofday(&starttime, NULL); //Initial time before computing starts. 

	ubyte* cmatrix;
	ubyte* nmatrix;
	int ite;
	 
	for (ite = 0; ite < iterations; ite++)
	{
		// Swap pointers every iterations to make sure next iteration uses solved matrix of previous iteration.
		if ((ite % 2) == 0)
		{
			cmatrix = currentmatrix;
			nmatrix = nextmatrix;
		}
		else
		{
			cmatrix = nextmatrix;
			nmatrix = currentmatrix;
		}

		iterate<<<numBlocks, threadsPerBlock>>>(size, cmatrix, nmatrix);
		/*if(ite <3) // for debugging the results. Checked the computed result manually to make sure code is working as expected.
		{
		  // copy the results after above mentioned iteration to CPU for printing.
		  cudaMemcpy(matrix, cmatrix, size * size * sizeof(ubyte), cudaMemcpyDeviceToHost);		
		  printf("Printing matrix after iterations number :%d \n",ite+1);
		  for (i = 0; i < 10; i++)
			{
				for (j = 0; j < 10; j++)
				{
					printf("%s ", matrix[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}
		}*/
		
		
		if(ite ==9||ite ==99||ite ==999)
		{
		  // copy the results after above mentioned iteration to CPU for printing.
		  cudaMemcpy(matrix, cmatrix, size * size * sizeof(ubyte), cudaMemcpyDeviceToHost);		
		  printf("Printing matrix after iterations number :%d \n",ite+1);
		  for (i = 0; i < 10; i++)
			{
				for (j = 0; j < 10; j++)
				{
					printf("%s ", matrix[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}
		}		
	}
	
	// copy the final result after N iterations 
	cudaMemcpy(matrix, cmatrix, size * size * sizeof(ubyte), cudaMemcpyDeviceToHost);
	
	struct timeval endtime;
	gettimeofday(&endtime, NULL);  //End time After computing ends. 
	double t = ((endtime.tv_sec - starttime.tv_sec) * 1000.0) + ((endtime.tv_usec - starttime.tv_usec) / 1000.0);

		
	printf("Final Matrix after %d iterations: \n",iterations);
		  for (i = 0; i < size; i++)
			{
				for (j = 0; j < size; j++)
				{
					printf("%s ", matrix[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}

	cudaFree(nextmatrix);
	cudaFree(currentmatrix);
	free(matrix);

	printf("%d iterations in %f milliseconds\n", iterations, t);

	return 0;
}

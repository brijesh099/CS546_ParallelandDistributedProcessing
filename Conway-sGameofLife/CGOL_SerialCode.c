/*Name: Brijesh Mavani
CWID: A20406960
University: Illinois Institute of Technology 
Course: Parallel and Distributed Processing
Assignment: 5
*/

/*Conway's Game of Life Serial implementation using C. */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

int iterations,size,i,j,n;
typedef unsigned char ubyte;

void iterate(int n, ubyte* cBoard, ubyte* nBoard)
{
	int k,x,y;
	for (x=0;x<size;x++)
	{
		for (y =0;y<size;y++)
		{
			k = x * n + y;	
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
	
	//Creating matrix in CPU to save the result after each iterations	
	
	ubyte* matrixN = (ubyte *)malloc(size * size * sizeof(ubyte));
		
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
			cmatrix = matrix; 
			nmatrix = matrixN;
		}
		else
		{
			cmatrix = matrixN;
			nmatrix = matrix;
		}

		iterate(size, cmatrix, nmatrix);
		/*if(ite <3) // for debugging the results. Checked the computed result manually to make sure code is working as expected. 
		{
		  printf("Printing matrix after iterations number :%d \n",ite+1);
		  for (i = 0; i < 10; i++)
			{
				for (j = 0; j < 10; j++)
				{
					printf("%s ", matrixN[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}
		}*/
		
		if(ite ==9||ite ==99||ite ==999)
		{
		  printf("Printing matrix after iterations number :%d \n",ite+1);
		  for (i = 0; i < 10; i++)
			{
				for (j = 0; j < 10; j++)
				{
					printf("%s ", matrixN[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}
		}		
	}
		
	struct timeval endtime;
	gettimeofday(&endtime, NULL);  //End time After computing ends. 
	double t = ((endtime.tv_sec - starttime.tv_sec) * 1000.0) + ((endtime.tv_usec - starttime.tv_usec) / 1000.0);

		
	printf("Final Matrix after %d iterations: \n",iterations);
		  for (i = 0; i < size; i++)
			{
				for (j = 0; j < size; j++)
				{
					printf("%s ", matrixN[i * size + j]? "a|" : "d|");
				}
				printf("\n");
			}

	free(matrix);
	free(matrixN);

	printf("%d iterations in %f milliseconds\n", iterations, t);

	return 0;
}
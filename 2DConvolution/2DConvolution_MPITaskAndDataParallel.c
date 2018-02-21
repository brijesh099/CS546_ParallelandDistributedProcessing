/* Name: Brijesh Mavani
CWID: A20406960
University: Illinois Institute of Technology
Course: Parallel and Distributed Processing
Final Project: 2-D Convolution using MPI Point to point communication by task parallelism */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

typedef struct {
    float r;
    float i;
} complex;
static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

#define N 512

// provided fft 1d code
void c_fft1d(complex *r, int n, int isign)
{
    int     m,i,i1,j,k,i2,l,l1,l2;
    float   c1,c2,z;
    complex t, u;

    if (isign == 0) return;

    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;
    for (i=0;i<n-1;i++) {
        if (i < j)
            C_SWAP(r[i], r[j]);
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* m = (int) log2((double)n); */
    for (i=n,m=0; i>1; m++,i/=2);

    /* Compute the FFT */
    c1 = -1.0;
    c2 =  0.0;
    l2 =  1;
    for (l=0;l<m;l++) {
        l1   = l2;
        l2 <<= 1;
        u.r = 1.0;
        u.i = 0.0;
        for (j=0;j<l1;j++) {
            for (i=j;i<n;i+=l2) {
                i1 = i + l1;

                /* t = u * r[i1] */
                t.r = u.r * r[i1].r - u.i * r[i1].i;
                t.i = u.r * r[i1].i + u.i * r[i1].r;

                /* r[i1] = r[i] - t */
                r[i1].r = r[i].r - t.r;
                r[i1].i = r[i].i - t.i;

                /* r[i] = r[i] + t */
                r[i].r += t.r;
                r[i].i += t.i;
            }
            z =  u.r * c1 - u.i * c2;

            u.i = u.r * c2 + u.i * c1;
            u.r = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (isign == -1) /* FWD FFT */
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for inverse transform */
    if (isign == 1) {       /* IFFT*/
        for (i=0;i<n;i++) {
            r[i].r /= n;
            r[i].i /= n;
        }
    }
}

int main(int argc, char **argv) {
    int Procid, nproc;
    int i ,j, offset,result;
    float real, img;
    double startTime, stopTime,time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12,time13,time14,time15;
    complex temp;
    /* Initialize MPI */
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        fprintf(stderr, "Unable to initialize MPI!\n");
        return -1;
    }
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &Procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    //printf("nproc: %d",nproc);
    int group_size = nproc / 4;
    int Procidgrp;
    int P1[group_size], P2[group_size], P3[group_size], P4[group_size];
    complex A[N][N], B[N][N], C[N][N];

    // Divide processors into 4 groups

    for (i=0; i<nproc; i++)
    {
        int pgrp = i / group_size;
        if (pgrp == 0)
        {
            P1[ i%group_size ] = i;
        }
        else if (pgrp == 1)
        {
            P2[ i%group_size ] = i;
        }
        else if (pgrp == 2)
        {
            P3[ i%group_size ] = i;
        }
        else if (pgrp == 3)
        {
            P4[ i%group_size ] = i;
        }
    }
    // printf("Grp\n");

    // Create 4 groups
    MPI_Group world_group, P1grp, P2grp, P3grp, P4grp;
    MPI_Comm P1Comm, P2Comm, P3Comm, P4Comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    
    int mygrp = Procid / group_size;
    switch (mygrp)
    {
    case 0:
        MPI_Group_incl(world_group, nproc/4, P1, &P1grp);
        MPI_Comm_create( MPI_COMM_WORLD, P1grp, &P1Comm);
        MPI_Group_rank(P1grp, &Procidgrp);
        break;
    case 1:
        MPI_Group_incl(world_group, nproc/4, P2, &P2grp);
        MPI_Comm_create( MPI_COMM_WORLD, P2grp, &P2Comm);
        MPI_Group_rank(P2grp, &Procidgrp);
        break;
    case 2:
        MPI_Group_incl(world_group, nproc/4, P3, &P3grp);
        MPI_Comm_create( MPI_COMM_WORLD, P3grp, &P3Comm);
        MPI_Group_rank(P3grp, &Procidgrp);
        break;
    case 3:
        MPI_Group_incl(world_group, nproc/4, P4, &P4grp);
        MPI_Comm_create( MPI_COMM_WORLD, P4grp, &P4Comm);
        MPI_Group_rank(P4grp, &Procidgrp);
        break;

    }
//printf("Comm\n");
    //Starting and send rows of A, B
    offset = N/group_size;

    if (Procid == 0)
    {
        // read 1st file
        FILE *f = fopen("im1", "r");
        for (i=0;i<N;i++)
        {
            for (j=0;j<N;j++)
            {
                result = fscanf(f,"%g",&A[i][j].r);
                A[i][j].i = 0.00;
            }
        }
        fclose(f);
        // read 2nd file
        f = fopen("im2", "r");
        for (i=0;i<N;i++)
        {
            for (j=0;j<N;j++)
            {
                result = fscanf(f,"%g",&B[i][j].r);
                B[i][j].i = 0.00;
            }
        }
        fclose(f);
        startTime = MPI_Wtime();
        // Send Matrix A to the all processors in P1 
        for ( i=0; i<group_size; i++ )
        {
            if ( P1[i]==0 ) continue;
            MPI_Send( &A[offset*i][0], offset*N, MPI_COMPLEX, P1[i], 0, MPI_COMM_WORLD );
        }
        // Send Matrix B to the all processors in P2 
        for ( i=0; i<group_size; i++ ) {
            if ( P2[i]==0 ) continue;
            MPI_Send( &B[offset*i][0], offset*N, MPI_COMPLEX, P2[i], 0, MPI_COMM_WORLD );
        }
    }
    else
    {
        // If processors group P1 receive matrix A
        if ( mygrp == 0 )
            MPI_Recv( &A[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status );
        // If processors group P2 receive matrix B
		
        if ( mygrp == 1 )
            MPI_Recv( &B[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status );
    }
//printf("1st send receive\n");
    MPI_Barrier(MPI_COMM_WORLD); 

    if ( Procid == 0 )
    {
        time1 = MPI_Wtime();
    }

    if ( mygrp == 0 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
			//Perform 1d fft forward for A rows in processors P1
            c_fft1d(A[i], N, -1);
        }
    }

      
    if ( mygrp == 1 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
			//Perform 1d fft forward for B rows in processors P2
            c_fft1d(B[i], N, -1);
        }
    }
    //printf("after 1st fft\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time2 = MPI_Wtime();
    }

    

    if ( mygrp == 0 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
				    //Receving rows of matrix A
                MPI_Recv( &A[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P1Comm, &status );
            }
        }
        else
        {
            MPI_Send( &A[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P1Comm );
        }
    }

 
    if ( mygrp == 1 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
				//Receving rows of matrix B
                MPI_Recv( &B[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P2Comm, &status );
            }
        }
        else
        {
            MPI_Send( &B[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P2Comm );
        }
    }
    //printf("2nd send receive\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time3 = MPI_Wtime();
    }

    if ( mygrp == 0 && Procidgrp == 0 )
    {
        for (i=0;i<N;i++)
        {
            for (j=i;j<N;j++)
            {
				//Transpose Matrix A
                temp = A[i][j];
                A[i][j] = A[j][i];
                A[j][i] = temp;
            }
        }
    }

    if ( mygrp == 1 && Procidgrp == 0 )
    {
        for (i=0;i<N;i++)
        {
            for (j=i;j<N;j++)
            {
				//Transpose Matrix B
                temp = B[i][j];
                B[i][j] = B[j][i];
                B[j][i] = temp;
            }
        }
    }
    //printf("transpose a and b \n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time4 = MPI_Wtime();
    }

    if ( mygrp == 0 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
                MPI_Send( &A[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P1Comm );
            }
        }
        else
        {
            MPI_Recv( &A[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P1Comm, &status );
        }
    }


    if ( mygrp == 1 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
                MPI_Send( &B[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P2Comm );
            }
        }
        else
        {
            MPI_Recv( &B[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P2Comm, &status );
        }
    }
    //printf("send receive a and b\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time5 = MPI_Wtime();
    }

    

    if ( mygrp == 0 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
			//Perform 1d fft forward for A rows in processors P1
            c_fft1d(A[i], N, -1);
        }
    }

    if ( mygrp == 1 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
			//Perform 1d fft forward for B rows in processors P2
            c_fft1d(B[i], N, -1);
        }
    }
    //printf("after 2nd fft\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time6 = MPI_Wtime();
    }


    if ( mygrp == 0 )
    {
        MPI_Send ( &A[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P3[Procidgrp], 0, MPI_COMM_WORLD );
    }
    else if ( mygrp == 1 )
    {
        MPI_Send ( &B[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P3[Procidgrp], 0, MPI_COMM_WORLD );
    }
    else if ( mygrp == 2 )
    {
		//Receive Matrix A from P1 and matrix B from P2 in processors P3
        MPI_Recv( &A[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P1[Procidgrp], 0, MPI_COMM_WORLD, &status );
        MPI_Recv( &B[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P2[Procidgrp], 0, MPI_COMM_WORLD, &status );
    }
//printf("send data to p3\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time7 = MPI_Wtime();
    }

    if ( mygrp == 2 )
    {
        for (i= offset*Procidgrp ;i< offset*(Procidgrp+1);i++)
        {
            for (j=0;j<N;j++)
            {
				/* Perform Point to point multiplication */
                C[i][j].r = A[i][j].r*B[i][j].r - A[i][j].i*B[i][j].i;
                C[i][j].i = A[i][j].r*B[i][j].i + A[i][j].i*B[i][j].r;
            }
        }
    }
    //printf("mm\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time8 = MPI_Wtime();
    }

    if ( mygrp == 2 )
    {
        MPI_Send ( &C[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P4[Procidgrp], 0, MPI_COMM_WORLD );
    }
    else if ( mygrp == 3 )
    {
		//P4 will receive result from P3
        MPI_Recv( &C[offset*Procidgrp][0], offset*N, MPI_COMPLEX, P3[Procidgrp], 0, MPI_COMM_WORLD, &status );
    }
    //printf("send to P3\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time9 = MPI_Wtime();
    }


    if ( mygrp == 3 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
			//Perform inverse 1d fft on rows of C
            c_fft1d(C[i], N, 1);
        }
    }
    //printf("c fft\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time10 = MPI_Wtime();
    }

    if ( mygrp == 3 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
                MPI_Recv( &C[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P4Comm, &status );
            }
        }
        else
        {
            MPI_Send( &C[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P4Comm );
        }
    }
    //printf("send C to P4\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time11 = MPI_Wtime();
    }


    if ( mygrp == 3 && Procidgrp == 0 )
    {
        for (i=0;i<N;i++)
        {
            for (j=i;j<N;j++)
            {
				//Transpose matrix C
                temp = C[i][j];
                C[i][j] = C[j][i];
                C[j][i] = temp;
            }
        }
    }
    //printf("c transpose\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time12 = MPI_Wtime();
    }


    if ( mygrp == 3 )
    {
        if ( Procidgrp == 0 )
        {
            for ( i=1; i<group_size; i++ )
            {
                MPI_Send( &C[offset*i][0], offset*N, MPI_COMPLEX, i, 0, P4Comm );
            }
        }
        else
        {
            MPI_Recv( &C[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, P4Comm, &status );
        }
    }
    //printf("send c after transpose\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time13 = MPI_Wtime();
    }


    if ( mygrp == 3 )
    {
        for ( i=offset*Procidgrp; i<offset*(Procidgrp+1); i++ )
        {
		    //Perform inverse 1d fft on rows of C
            c_fft1d(C[i], N, 1);
        }
    }
    //printf("c fft col\n");

    MPI_Barrier(MPI_COMM_WORLD); 
    if ( Procid == 0 )
    {
        time14 = MPI_Wtime();
    }

// Final result will be sent to source processor
    if ( Procid == 0 )
    {
        for ( i=0; i<group_size; i++ )
        {
            if ( P4[i]== 0 ) continue; 

            MPI_Recv( &C[offset*i][0], offset*N, MPI_COMPLEX, P4[i], 0, MPI_COMM_WORLD, &status );
        }
    }
    else if ( mygrp == 3 )
    {
        MPI_Send( &C[offset*Procidgrp][0], offset*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD );
    }
    //printf("final c send\n");
    MPI_Barrier(MPI_COMM_WORLD);


    if (Procid == 0)
    {
        /* Stop Clock */
        stopTime = MPI_Wtime();

        // Write output in a file.
        FILE *f = fopen("mpi_task_output", "w");
        for (i=0;i<N;i++)
        {
            for (j=0;j<N;j++)
            {
                fprintf(f,"   %.6e",C[i][j].r); //Change value of X in .<X>e to change the numbers after the decimal.
            }
            fprintf(f,"\n");
        }
        fclose(f);

        printf("\nTotal Elapsed time is: %lf sec.\n",(stopTime - startTime));

        printf("\nTotal Computation time is: %1f sec.\n",((time2-time1)+(time4-time3)+(time6-time5)+(time8-time7)+(time10-time9)+(time12-time11)+(time14-time13)));

        printf("\nTotal Communication time is: %1f sec.\n",((time1-startTime)+(time3-time2)+(time5-time4)+(time7-time6)+(time9-time8)+(time11-time10)+(time13-time12)+(stopTime-time14)));
    }

    MPI_Finalize();
    return 0;
}

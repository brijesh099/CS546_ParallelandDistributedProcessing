/* Name: Brijesh Mavani
CWID: A20406960
University: Illinois Institute of Technology
Course: Parallel and Distributed Processing
Final Project: 2-D Convolution using MPI Point to point communication */

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
    int i ,j, offset,nrows,lb,hb,result;
    float real, img;
    double startTime, stopTime,time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12,time13,time14,time15;

    /* Initialize MPI */
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        fprintf(stderr, "Unable to initialize MPI!\n");
        return -1;
    }
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &Procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    complex **A, **B, **C, **D;
    A = malloc(N * sizeof(complex *));
    B = malloc(N * sizeof(complex *));
    C = malloc(N * sizeof(complex *));
    D = malloc(N * sizeof(complex *));
    for (i = 0; i < N; i++)
    {
        A[i] = malloc(N * sizeof(complex *));
        B[i] = malloc(N * sizeof(complex *));
        C[i] = malloc(N * sizeof(complex *));
        D[i] = malloc(N * sizeof(complex *));
    }
    complex *vec,*vec1;


    /* Setup description of the 4 MPI_FLOAT fields x, y, z, velocity */
    MPI_Datatype mystruct;
    int          length[2] = { 1, 1 };
    MPI_Aint     index[2] = { 0, sizeof(float) };
    MPI_Datatype old_type[2] = { MPI_FLOAT, MPI_FLOAT };

    /* Make relative */
    MPI_Type_struct( 2, length, index, old_type, &mystruct );
    MPI_Type_commit( &mystruct );    

    nrows = N/nproc;
    lb = Procid*nrows;
    hb = lb + nrows;

    printf("Processor %d have lower bound = %d and  upper bound = %d\n", Procid, lb, hb);

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

        /* Start Clock */
        printf("\nStarting clock in processor :%d.\n",Procid);
        startTime = MPI_Wtime();

        //Send rows of A and B to other processors

        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Send(&A[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
                MPI_Send(&B[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Recv(A[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(B[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    if (Procid == 0)

    {

        time1 = MPI_Wtime();

    }

    //Perform 1d fft forward for A and B rows

    vec = (complex *)malloc(N * sizeof(complex));
    vec1 = (complex *)malloc(N * sizeof(complex));

    for (i=lb;i<hb;i++)
    {
        for (j=0;j<N;j++)
        {
            vec[j] = A[i][j];
            vec1[j] = B[i][j];
        }
        c_fft1d(vec, N, -1);
        c_fft1d(vec1, N, -1);
        for (j=0;j<N;j++)
        {
            A[i][j] = vec[j];
            B[i][j] = vec1[j];
        }
    }

    free(vec);
    free(vec1);

    if (Procid == 0)

    {

        time2 = MPI_Wtime();

    }

    //Receving rows of A, B
    if (Procid == 0)
    {
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Recv(A[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(B[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++) {
            MPI_Send(&A[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&B[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (Procid == 0)
    {
        time3 = MPI_Wtime();
    }

    // perform 1d fft on columns of resulted matrix of A and B. We will transpose the row result to obtain the new data corresponding to columns.
    if (Procid == 0)
    {
        //Transpose A and B. Resulting matrix will be stored in C and D respectively.
        for (i = 0; i < N; i++)
        {
            for (j = 0 ; j < N ; j++)
            {
                C[j][i] = A[i][j];
                D[j][i] = B[i][j];
            }
        }
        time4 = MPI_Wtime();
        //Send rows of C and D to other processors

        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Send(&C[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
                MPI_Send(&D[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Recv(C[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(D[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    if (Procid == 0)
    {
        time5 = MPI_Wtime();
    }
    //Perform 1d fft forward for A and B columns

    vec = (complex *)malloc(N * sizeof(complex));
    vec1 = (complex *)malloc(N * sizeof(complex));

    for (i=lb;i<hb;i++)
    {
        for (j=0;j<N;j++)
        {
            vec[j] = C[i][j];
            vec1[j] = D[i][j];
        }
        c_fft1d(vec, N, -1);
        c_fft1d(vec1, N, -1);
        for (j=0;j<N;j++)
        {
            C[i][j] = vec[j];
            D[i][j] = vec1[j];
        }
    }

    free(vec);
    free(vec1);
    if (Procid == 0)
    {
        time6 = MPI_Wtime();
    }

    //Receving columns of A, B

    if (Procid == 0)
    {
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Recv(C[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(D[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Send(&C[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&D[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (Procid == 0)
    {
        time7 = MPI_Wtime();
    }

    if (Procid == 0)
    {
		//Transpose C and D. Resulting matrix will be stored in A and B respectively.
        for (i = 0; i < N; i++)
        {
            for (j = 0 ; j < N ; j++)
            {
                A[j][i] = C[i][j];
                B[j][i] = D[i][j];
            }
        }
		/* Perform Point to point multiplication */
        for (i=0;i<N;i++)
        {
            for (j=0;j<N;j++)
            {
                C[i][j].r = (A[i][j].r * B[i][j].r) - (A[i][j].i * B[i][j].i);
                C[i][j].i = (A[i][j].r * B[i][j].i) + (A[i][j].i * B[i][j].r);
            }
        }

        time8 = MPI_Wtime();

        //Send rows of C to other processors
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Send(&C[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Recv(C[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
        }
    }
    if (Procid == 0)
    {
        time9 = MPI_Wtime();
    }
    //Perform inverse 1d fft on rows of C
    vec = (complex *)malloc(N * sizeof(complex));

    for (i=lb;i<hb;i++)
    {
        for (j=0;j<N;j++)
        {
            vec[j] = C[i][j];
        }
        c_fft1d(vec, N, 1);
        for (j=0;j<N;j++)
        {
            C[i][j] = vec[j];
        }
    }
    free(vec);

    if (Procid == 0)
    {
        time10 = MPI_Wtime();
    }

    //Receving rows of 1d fft result of C

    if (Procid == 0)
    {
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Recv(C[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Send(&C[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (Procid == 0)
    {
        time11 = MPI_Wtime();
    }
   
    if (Procid == 0)
    {
		//Transpose C
        for (i = 0; i < N; i++)
        {
            for (j = 0 ; j < N ; j++)
            {
                D[j][i] = C[i][j];
            }
        }

        time12 = MPI_Wtime();
		//Send C to all other processors
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Send(&D[j][0], N, mystruct, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Recv(D[j], N, mystruct, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    if (Procid == 0)
    {
        time13 = MPI_Wtime();
    }

    //Perform inverse 1d fft on columns of C
    vec = (complex *)malloc(N * sizeof(complex));

    for (i=lb;i<hb;i++)
    {
        for (j=0;j<N;j++)
        {
            vec[j] = D[i][j];
        }
        c_fft1d(vec, N, 1);
        for (j=0;j<N;j++)
        {
            D[i][j] = vec[j];
        }
    }

    free(vec);
    if (Procid == 0)
    {
        time14 = MPI_Wtime();
    }
    //Send result back to processor 0
    if (Procid == 0)
    {
        for (i=1;i<nproc;i++)
        {
            offset=i*nrows;
            for (j = offset; j < (offset+nrows); j++)
            {
                MPI_Recv(D[j], N, mystruct, i, 0, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        for (j = lb; j < hb; j++)
        {
            MPI_Send(&D[j][0], N, mystruct, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (Procid == 0)
    {
        time15 = MPI_Wtime();
    }
    if (Procid == 0)
    {
		//Final transpose
        for (i = 0; i < N; i++)
        {
            for (j = 0 ; j < N ; j++)
            {
                C[j][i] = D[i][j];
            }
        }

        /* Stop Clock */
        stopTime = MPI_Wtime();

        // Write output in a file.
        FILE *f = fopen("mpi_sendrecv_output", "w");
        for (i=0;i<N;i++)
        {
            for (j=0;j<N;j++)
            {
                fprintf(f,"   %.6e",C[i][j].r);  //Change value of X in .<X>e to change the numbers after the decimal.
            }
            fprintf(f,"\n");
        }
        fclose(f);

        printf("\nTotal Elapsed time is: %lf sec.\n",(stopTime - startTime));

        printf("\nTotal Computation time is: %1f sec.\n",((time2-time1)+(time4-time3)+(time6-time5)+(time8-time7)+(time10-time9)+(time12-time11)+(time14-time13)+(stopTime-time15)));

        printf("\nTotal Communication time is: %1f sec.\n",((time1-startTime)+(time3-time2)+(time5-time4)+(time7-time6)+(time9-time8)+(time11-time10)+(time13-time12)+(time15-time14)));
    }

    MPI_Finalize();

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}


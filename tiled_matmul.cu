
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define matdim 1024
#define matsize (matdim*matdim*sizeof(int))
#define TILESIZE 32

__global__ void init(int *v1,int *v2)
{
    int x = threadIdx.x + blockIdx.x*TILESIZE;
    int y = threadIdx.y + blockIdx.y*TILESIZE;

    if ((x>=matdim)||(y>=matdim))
        return;

    v1[y*matdim+x] = y*matdim+x+1;
    v2[y*matdim+x] = y*matdim+x+1;
}

__global__ void matmul(int *v1, int *v2, int*v3)
{
    int x = threadIdx.x + blockIdx.x*TILESIZE;
    int y = threadIdx.y + blockIdx.y*TILESIZE;

    int thx = threadIdx.x;
    int thy = threadIdx.y;

    if ((x>=matdim)||(y>=matdim))
        return;

    __shared__ int atile[TILESIZE][TILESIZE];
    __shared__ int btile[TILESIZE][TILESIZE];

    int count = 0;
    for(int i=0;i<matdim;i+= TILESIZE)
    {
        atile[thy][thx] = v1[y*matdim + i + thx];
        btile[thy][thx] = v2[(thy+i)*matdim + x];

        __syncthreads();

        for(int j=0;j<TILESIZE;j++)
        {
            count += atile[thy][j]*btile[j][thx];
        }
        __syncthreads();
    }
    v3[y*matdim + x] = count;

}

int main()
{
    int *h_v3;
    int *d_v1, *d_v2, *d_v3;

    h_v3 = (int*)malloc(matsize);
    cudaMalloc(&d_v1, matsize);
    cudaMalloc(&d_v2, matsize);
    cudaMalloc(&d_v3, matsize);

    dim3 blocksize(TILESIZE, TILESIZE);
    dim3 gridsize((matdim+TILESIZE-1)/TILESIZE,(matdim+TILESIZE-1)/TILESIZE);

    init<<<gridsize,blocksize>>>(d_v1,d_v2);

    matmul<<<gridsize, blocksize>>>(d_v1, d_v2, d_v3);

    cudaMemcpy(h_v3, d_v3, matsize, cudaMemcpyDeviceToHost);

    free(h_v3);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);

    return 0;
}

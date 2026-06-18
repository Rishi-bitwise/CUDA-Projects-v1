#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define TILE_SIZE 32
#define MAT_DIM 1024



__device__ void form_matrix(float *input_arr)
{
    int xcoord = blockIdx.x*TILE_SIZE + threadIdx.x;
    int ycoord = blockIdx.y*TILE_SIZE + threadIdx.y;

    int total_tilesx = MAT_DIM/TILE_SIZE;

    for(int i=0;i<TILE_SIZE; i+= blockDim.y)
    {
        input_arr[(ycoord+i)*MAT_DIM + xcoord] = (xcoord+ycoord+i)*10;
    }
}

__global__ void copymat(float *input_arr, float *output_arr)
{
    form_matrix(input_arr);
    __syncthreads();
    int xcoord = blockIdx.x*TILE_SIZE + threadIdx.x;
    int ycoord = blockIdx.y*TILE_SIZE + threadIdx.y;

    int total_tilesx = (int)(MAT_DIM/TILE_SIZE);

    for (int i=0;i<TILE_SIZE; i+= blockDim.y)
    {
        output_arr[(ycoord+i)*MAT_DIM + xcoord] = input_arr[(ycoord+i)*MAT_DIM + xcoord];
    }
}


int main()
{
    dim3 blocksize(32,8);
    dim3 gridsize(32,32);

    float h_output_arr[MAT_DIM][MAT_DIM];
    float *d_input_arr;
    float *d_output_arr;

    cudaMalloc(&d_input_arr, sizeof(h_output_arr));
    cudaMalloc(&d_output_arr, sizeof(h_output_arr));

    copymat<<<gridsize, blocksize>>>(d_input_arr, d_output_arr);

    cudaMemcpy(h_output_arr, d_output_arr, sizeof(h_output_arr), cudaMemcpyDeviceToHost);

    for(int i=0;i<MAT_DIM;i++)
    {
        for(int j=0;j<MAT_DIM;j++)
        printf("%f ", h_output_arr[i][j]);
        printf("\n");
    }
    
    cudaFree(d_input_arr);
    cudaFree(d_output_arr);
    return 0;
}


// __global__ void copymatt(float *output, float *input)
// {
//     int xcoord = blockIdx.x*TILE_SIZE + threadIdx.x;
//     int ycoord = blockIdx.y*TILE_SIZE + threadIdx.y;

//     for(int i=0;i<TILE_SIZE; i+= blockDim.y)
//     {
//         output[(ycoord+i)*]
//     }
// }
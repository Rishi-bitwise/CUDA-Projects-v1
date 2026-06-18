#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define MAT_DIM 128
#define TILE_SIZE 32

__device__ void form_mat(float *input_arr)
{
    int xcoord = blockIdx.x*TILE_SIZE + threadIdx.x;
    int ycoord = blockIdx.y*TILE_SIZE + threadIdx.y;

    for(int i=0;i<TILE_SIZE;i+= blockDim.y)
    {
        input_arr[(ycoord+i)*MAT_DIM + xcoord] = (xcoord+ycoord+i)*10;
    }
}

__global__ void transpose(float *input_arr, float *output_arr)
{

    int xcoord = threadIdx.x + blockIdx.x*TILE_SIZE;
    int ycoord = threadIdx.y + blockIdx.y*TILE_SIZE;

    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    if ((xcoord<MAT_DIM) && (ycoord<MAT_DIM))
    {

        form_mat(input_arr);
        __syncthreads();

        for(int j=0;j< TILE_SIZE; j+= blockDim.y)
        {
            tile[threadIdx.y+j][threadIdx.x] = input_arr[(ycoord+j)*MAT_DIM + xcoord];
        }

        __syncthreads();

        // transpose the entire tile to a separate place, but its contents are not transposed here:
        xcoord = blockIdx.y*TILE_SIZE + threadIdx.x;
        ycoord = blockIdx.x*TILE_SIZE + threadIdx.y;
        // now these coordinates are for th output, the input is already captured in the tile.

        for(int j=0;j<TILE_SIZE; j+= blockDim.y)
        {
            output_arr[(ycoord+j)*MAT_DIM + xcoord] = tile[threadIdx.x][threadIdx.y + j];
        }

    }

}

int main()
{
    float h_output_arr[MAT_DIM][MAT_DIM];
    float *d_input_arr;
    float *d_output_arr;

    dim3 blocksize(32,16);
    dim3 gridsize(MAT_DIM/TILE_SIZE, MAT_DIM/TILE_SIZE);

    cudaMalloc(&d_input_arr, MAT_DIM*MAT_DIM*sizeof(float));
    cudaMalloc(&d_output_arr, MAT_DIM*MAT_DIM*sizeof(float));

    transpose<<<gridsize, blocksize>>>(d_input_arr, d_output_arr);

    cudaMemcpy(h_output_arr, d_output_arr, sizeof(h_output_arr), cudaMemcpyDeviceToHost);

    for(int i=0;i<MAT_DIM; i++)
    {
        for(int j=0;j<MAT_DIM; j++)
        printf("%f ", h_output_arr[i][j]);
        printf("\n");
    }

    cudaFree(d_input_arr);
    cudaFree(d_output_arr);

    return 0;
}

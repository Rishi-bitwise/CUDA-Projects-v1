#include<stdio.h>
#include<cuda.h>
#include<math.h>

/*
REALITY : 
A little bit of calculation is needed. We are trying to get 10000 pendulums, with 1000 timesteps. 
So each thread needs to store 1000 x float 32(theta). that is 1M*1k*4 bytes = 4GB. This was a problem,
I did not want to use global memory all that much, nor did i want to keep looping through the cpu. so,
we take a middle ground, of registermaxxing, and cpu looping. 

CALCULATION:
I dont know the hardware so the numbers maybe differ. change it according to the hardware:
We wanna run all the threads in an SM. registers per SM is around 64k, 
and we have probably 2048 threads per SM = 2 blocks . 

so each thread gets 32 registers. so we should safely be able to loop and record 15 float values,
write it to global memory and get that into our RAM. then we call the kernel again and get the next 15.

REGISTER BEHAVIOUR:
Register usage is tight and needs to be studied. it is not a one-to-one for varibales and registers.
Register is used based on lifetime of a variable.
(find the opcode and compilation nature and verifying the overlapping nature of variables)
If variables exist and overlap then we need separate registers. 

NOTE : TODO I HAVENT USED SHARED MEMORY TO SQUEEZE OUT MORE LOOPS/ COORDINATES PER THREAD. WE CAN DO THAT LATER.

*/

// #define ARR_SZ_PER_THREAD 15
#define TOTAL_THREADS (8*1024)                         // MORE THAN THIS AND CONSTANT GLOBL MEMORY LIMIT(64k) IS EXCEEDED FOR INIT_LENGTHS_LIST 
#define DT 0.125f
#define TOTAL_TIME 64
#define TOTAL_STEPS ((int)(TOTAL_TIME/DT))
#define BLOCKSIZE 1024                  //this is fine, SM will have 2 blocks, so total 2k threads, with 32 registers each, for 64k total registers
#define GRIDSIZE ((int)(TOTAL_THREADS+BLOCKSIZE-1)/BLOCKSIZE)

struct Params
{
    float *init_theta_list;
    float *theta_arr_final;
    float *init_omega_list;
};

__constant__ float length_list[TOTAL_THREADS];

__device__ void getderivatives(float theta, float omega, float L, float *theta_d, float *omega_d)
{
    const float g = 9.81f;
    *theta_d = omega;
    *omega_d = -(g/L)*sinf(theta);
}

__global__ void transpose(float *input_mat, float *output_mat, int rows, int cols)
{
    int tilesize = 32;        //shared memory capacity limited, cant have big tile size. also blocks can accomodate 1024 threads 
    // assert(blockDim.x == 32);        //  whether assert is slow in kernel ??
    // assert(blockDim.y == 32);                       
    /* we are doing a simple transpose, no learning needed here, just get it working */

    __shared__ float tile[tilesize][tilesize+1];

    int xcoord = threadIdx.x + blockIdx.x*tilesize;
    int ycoord = threadIdx.y + blockIdx.y*tilesize;

    if ((xcoord<cols) && (ycoord<rows))
    {
        tile[threadIdx.x][threadIdx.y] = input_mat[ycoord*cols + xcoord];
    }
    __syncthreads();

    int nxcoord = blockIdx.y*tilesize + threadIdx.x;
    int nycoord = blockIdx.x*tilesize + threadIdx.y;

    output_mat[nycoord*rows + nxcoord] = tile[threadIdx.x][threadIdx.y];         //IMP Here shape has changed, so ycoord*rows instead of cols.

}

__global__ void rk4_step(Params P)
{
    
    // theta_list, omega_list, length_list are the starting values for EACH THREAD.
    //We still have to make the array for each thread.

    int xcoord = threadIdx.x + blockDim.x*blockIdx.x;

    if (xcoord<TOTAL_THREADS)
    {
        float theta = P.init_theta_list[xcoord];
        float omega = P.init_omega_list[xcoord];
        float length = length_list[xcoord];     //length_list is const global mem.

        //TODO WE HAVE TOO MANY VARIABLES TAKING REGISTER SPACE.
        float d_th1, d_w1, d_th2, d_w2, d_th3, d_w3, d_th4, d_w4;

        for(int i = 0;i<TOTAL_STEPS;i++)
        {
            getderivatives(theta, omega, length, &d_th1, &d_w1);
            getderivatives(theta + (DT/2.0f)*d_th1, omega + (DT/2.0f)*d_w1, length, &d_th2, &d_w2);
            getderivatives(theta + (DT/2.0f)*d_th2, omega + (DT/2.0f)*d_w2, length, &d_th3, &d_w3);
            getderivatives(theta + DT*d_th3, omega + DT*d_w3, length, &d_th4, &d_w4);

            theta = theta + (DT/6.0f)*(d_th1 + 2*d_th2 + 2*d_th3 + d_th4);
            omega = omega + (DT/6.0f)*(d_w1 + 2*d_w2 + 2*d_w3 + d_w4);   //ready omega for next iter
            
            P.theta_arr_final[i*TOTAL_THREADS + xcoord] = theta;
        }

        //memcpy(P.theta_arr_final[xcoord],theta_arr, sizeof(theta_arr));       
            //massive struggle here,                                                                         
            //had to change a lot of stuff, still sad about the time I lost here trying to fix this.
            //memory coalescing was planned poorly :(  now its so obvious...

        //UPDATE THE INIT VALS HERE !!!!

    }
}

void form_initial_values(float *arr, float start, float step_size, int total_steps)
{
    for (int i=0; i<total_steps;i++)
    {
        arr[i] = start + i*step_size;
    }
}

int main()
{

    float h_init_theta_list[TOTAL_THREADS];
    float h_init_omega_list[TOTAL_THREADS];
    //float h_theta_arr_final[TOTAL_STEPS][TOTAL_THREADS];        //not enough memory for stack
    //float h_theta_arr_final_arranged[TOTAL_THREADS][TOTAL_STEPS];
    float * h_theta_arr_final;
    float * h_theta_arr_final_arranged;

    dim3 blocksize_transpose(32,32);
    dim3 gridsize_transpose(TOTAL_THREADS/32, TOTAL_STEPS/32);

    h_theta_arr_final = (float*)malloc(sizeof(float)*TOTAL_THREADS*TOTAL_STEPS);
    h_theta_arr_final_arranged = (float*)malloc(sizeof(float)*TOTAL_THREADS*TOTAL_STEPS);

    float h_length_list[TOTAL_THREADS];

    form_initial_values(h_init_theta_list, 0.1f, 0.01f, TOTAL_THREADS);      // form inital theta values for 1M threads
    form_initial_values(h_init_omega_list, 0.5f, 0.0f, TOTAL_THREADS);           // form init omega vals for 1M threads
    form_initial_values(h_length_list, 1.0f, 0.00001f, TOTAL_THREADS);            // form init length vals for 1M threads

    cudaMemcpyToSymbol(length_list, h_length_list, sizeof(h_length_list));

    float *d_init_theta_list;
    float *d_theta_arr_final;
    float *d_init_omega_list;
    float *d_theta_arr_final_arranged;

    cudaMalloc(&d_init_theta_list, sizeof(h_init_theta_list));
    cudaMalloc(&d_theta_arr_final, TOTAL_THREADS*TOTAL_STEPS*sizeof(float));
    cudaMalloc(&d_init_omega_list, sizeof(h_init_omega_list));
    cudaMalloc(&d_theta_arr_final_arranged, TOTAL_THREADS*TOTAL_STEPS*sizeof(float));

    cudaMemcpy(d_init_theta_list, h_init_theta_list, sizeof(h_init_theta_list), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init_omega_list, h_init_omega_list, sizeof(h_init_omega_list), cudaMemcpyHostToDevice);

    Params P;

    P.init_theta_list = d_init_theta_list;
    P.init_omega_list = d_init_omega_list;
    P.theta_arr_final = d_theta_arr_final;

    rk4_step<<<GRIDSIZE, BLOCKSIZE>>>(P);

    cudaMemcpy(h_theta_arr_final, d_theta_arr_final, TOTAL_STEPS*TOTAL_THREADS*sizeof(float), 
                cudaMemcpyDeviceToHost);

    transpose<<<gridsize_transpose, blocksize_transpose>>>(d_theta_arr_final, d_theta_arr_final_arranged, TOTAL_STEPS, TOTAL_THREADS);
    
    cudaMemcpy(h_theta_arr_final_arranged, d_theta_arr_final_arranged,TOTAL_STEPS*TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_init_theta_list);
    cudaFree(d_init_omega_list);
    cudaFree(d_theta_arr_final);

    free(h_theta_arr_final);
    free(h_theta_arr_final_arranged);

    return 0;
}

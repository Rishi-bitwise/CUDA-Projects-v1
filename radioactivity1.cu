#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<limits.h>
#include<curand_kernel.h>

#define ATOMS (1024*1024)
#define p 0.02f
#define TOTAL_TIME_STEPS (64*8)    
#define P_UINT32 ((int)(UINT32_MAX*p)) 

__global__ void decay(int *alive_count, unsigned long seed)
{
    int xcoord = threadIdx.x + blockIdx.x*blockDim.x;
    int count = 0;

    __shared__ int blockcount[1024]; 
    blockcount[threadIdx.x] = 0;

    bool active = xcoord<ATOMS/32;      //one thread handles 32 atoms, so each xcoord will limit to ATOMS/32. Not more than that.
    curandState state;
    if (active)
    {
        curand_init(seed, xcoord, 0, &state);           
    }

    uint32_t status = UINT32_MAX;       //Initially all the atoms are alive

    for(int t=0; t<TOTAL_TIME_STEPS; t++)
    {
        count = 0;                      // If statement messes up __syncthreads(), so we need count=0 for all other threads
        uint32_t new_state = 0;         //new_state contains the bits after the simulation.
        if (active)
        {
            for(int i=0;i<32;i++)
            {
                int b = (status>>i)&1;      //We can save b in int, since it is being saved in register, and we have plenty of those.
                int tval = ((int)(curand(&state) > P_UINT32)) & b;          // IF rand greater than prob, then alive(1) else dead(0)
                new_state = new_state | (tval<<i);
            }
        }
        
        // __syncthreads();         // never use syncthreads inside an if block. not all threads are running cuz xccord<ATOMS/32;
        status = new_state;
        count = __popc(status);
        //atomicAdd(alive_count[t],count);      //important but it locks for each thread so everything gets serealized.
        blockcount[threadIdx.x] = count;        //each thread places the number of alive atoms into its blockcount slot.
        __syncthreads();
        for(int stride = blockDim.x/2; stride>0; stride=stride>>1)              
        {
            if(threadIdx.x < stride)
            {
                blockcount[threadIdx.x] += blockcount[threadIdx.x+stride];   
            }
            __syncthreads();
        }
        if(threadIdx.x ==0)
            atomicAdd(&alive_count[t], blockcount[0]);
    }
    
}

int main()
{
    //int  h_alive_count[TOTAL_TIME_STEPS][ATOMS/32];         
    int *d_alive_count;

    int *h_alive_count;
    h_alive_count = (int*)malloc(TOTAL_TIME_STEPS*sizeof(int));

    cudaMalloc(&d_alive_count, TOTAL_TIME_STEPS*sizeof(int));
    cudaMemset(d_alive_count, 0, TOTAL_TIME_STEPS*sizeof(int));

    dim3 gridsize((ATOMS/32)/1024);
    dim3 blocksize(1024);

    unsigned long seed = time(NULL);

    decay<<<gridsize, blocksize>>>(d_alive_count, seed);

    cudaMemcpy(h_alive_count, d_alive_count, TOTAL_TIME_STEPS*sizeof(int), cudaMemcpyDeviceToHost);

    FILE *f = fopen("decay_output.csv", "w");
    fprintf(f, "TIME-STEP,ATOMS-ALIVE\n");
    for(int i=0;i<TOTAL_TIME_STEPS;i++)
    {
        fprintf(f,"%d,%d\n", i, h_alive_count[i]);
    }
    fclose(f);

    cudaFree(d_alive_count);
    free(h_alive_count);

    return 0;
}


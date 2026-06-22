#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

#define N 1024
#define TOTAL_TIME_STEPS 1024
#define dt 0.0125f
#define G 1
#define m 1
#define simdim 1024
#define rk4mode 0
#define verletmode 1
#define epsilon 1.0f
#define tilesize 1024

struct Params
{
    float *posx, *posy, *velx, *vely;         // this is only for a single timestep, so dimension will be N
    float *accx0, *accy0;
    float *accx1, *accy1;                     
    float *energy, *cofm;        
};

__global__ void simulate(Params P)
{
    int total_tiles = (N + tilesize -1)/tilesize;
    int xid = threadIdx.x + blockIdx.x*blockDim.x;
    if (xid>=N)                  
        return;
    
    float dx, dy, rsq;

    float ax=0, ay=0;
    float xcomp, ycomp;

    __shared__ float s_posx[tilesize];
    __shared__ float s_posy[tilesize];

    for(int tile=0; tile<total_tiles; tile++)
    {
        if (tile*tilesize + threadIdx.x < N)                //note - tilesize and blocksize are the same here
        {
            s_posx[threadIdx.x] = P.posx[tile*tilesize + threadIdx.x];
            s_posy[threadIdx.x] = P.posy[tile*tilesize + threadIdx.x];
        }

        else                    //tiles maybe incomplete...
        {
            s_posx[threadIdx.x] = 0;
            s_posy[threadIdx.x] = 0;
        }
        
        __syncthreads();        // Let the entire block load the data into shared memory
        
        for(int k=0; k<tilesize; k++)
        {
            if (xid == tile*tilesize + k)           //ignore the particle itself.
                continue;
            if (tile*tilesize + k >= N)             //ignore particles not within the N=1024 limit
                continue;
            dx = s_posx[k] - P.posx[xid] ;
            dy = s_posy[k] - P.posy[xid] ;

            rsq = sqrtf(dx*dx + dy*dy + epsilon*epsilon) ;   
            xcomp = dx/rsq;
            ycomp = dy/rsq;

            atomicAdd(P.energy, -0.5*G*m*m/rsq); // half because potential energy involves pairs of threads....

            ax = ax+(G*m/(rsq*rsq))*xcomp;
            ay = ay+(G*m/(rsq*rsq))*ycomp;

        }

        __syncthreads();
    }
    float ke = 0.5*m*(P.velx[xid]*P.velx[xid] + P.vely[xid]*P.vely[xid]);

    atomicAdd(P.energy, ke);
    atomicAdd(P.cofm, (P.posx[xid])/N);         // m is constant

    P.accx1[xid] = ax;
    P.accy1[xid] = ay;

    //update(P);

}

__global__ void updatepos(Params P)
{
    int xid = blockIdx.x*blockDim.x + threadIdx.x;
    if (xid >= N)
        return;
    P.posx[xid] = P.posx[xid] + P.velx[xid]*dt + 0.5*P.accx0[xid]*dt*dt;
    P.posy[xid] = P.posy[xid] + P.vely[xid]*dt + 0.5*P.accy0[xid]*dt*dt;

}

__global__ void updatevel(Params P)
{
    int xid = blockIdx.x*blockDim.x + threadIdx.x;
    if (xid>=N)
        return;

    P.velx[xid] = P.velx[xid] + P.accx0[xid]*dt/2 + P.accx1[xid]*dt/2;
    P.vely[xid] = P.vely[xid] + P.accy0[xid]*dt/2 + P.accy1[xid]*dt/2;
}

void init(float *arr, int l)
{

    srand(time(NULL));
    for(int i=0; i<l; i++)
    {
        arr[i] = rand() % simdim/2;
    }
}

int main()
{
    float *h_posx, *h_posy;

    float *h_energy, *h_cofm;

    float *d_posx, *d_posy, *d_velx, *d_vely;
    float *d_accx0, *d_accy0;
    float *d_accx1, *d_accy1;
    float *d_energy, *d_cofm;

    h_posx = (float*)malloc(N*TOTAL_TIME_STEPS*sizeof(float));
    h_posy = (float*)malloc(N*TOTAL_TIME_STEPS*sizeof(float));

    h_energy = (float*)malloc(TOTAL_TIME_STEPS*sizeof(float));
    h_cofm = (float*)malloc(TOTAL_TIME_STEPS*sizeof(float));

    init(h_posx, N);
    init(h_posy, N);

    cudaMalloc(&d_posx, N*sizeof(float));
    cudaMalloc(&d_posy, N*sizeof(float));
    cudaMalloc(&d_velx, N*sizeof(float));
    cudaMalloc(&d_vely, N*sizeof(float));
    cudaMalloc(&d_accx0, N*sizeof(float));
    cudaMalloc(&d_accy0, N*sizeof(float));
    cudaMalloc(&d_accx1, N*sizeof(float));
    cudaMalloc(&d_accy1, N*sizeof(float));
    cudaMalloc(&d_energy, sizeof(float));
    cudaMalloc(&d_cofm, sizeof(float));

    cudaMemcpy(d_posx, h_posx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posy, h_posy, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_velx, 0, N*sizeof(float));
    cudaMemset(d_vely, 0, N*sizeof(float));
    
    cudaMemset(d_accx0, 0, N*sizeof(float));
    cudaMemset(d_accy0, 0, N*sizeof(float));
    cudaMemset(d_accx1, 0, N*sizeof(float));
    cudaMemset(d_accy1, 0, N*sizeof(float));

    Params P;
    P.posx = d_posx;
    P.posy = d_posy;
    P.velx = d_velx;
    P.vely = d_vely;
    P.accx0 = d_accx0;
    P.accy0 = d_accy0;
    P.accx1 = d_accx1;
    P.accy1 = d_accy1;
    P.energy = d_energy;
    P.cofm = d_cofm;

    int blocksize = 1024;
    int gridsize = (N+blocksize-1)/blocksize;

    for(int i=0; i<TOTAL_TIME_STEPS; i++)
    {
        cudaMemset(d_energy, 0, sizeof(float));
        cudaMemset(d_cofm, 0, sizeof(float));

        simulate<<<gridsize, blocksize>>>(P);
        updatepos<<<gridsize, blocksize>>>(P);
        updatevel<<<gridsize, blocksize>>>(P);

        cudaMemcpy(d_accx0, d_accx1, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accy0, d_accy1, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&h_energy[i], d_energy, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_cofm[i], d_cofm, sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(&h_posx[i*N], d_posx, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_posy[i*N], d_posy, N*sizeof(float), cudaMemcpyDeviceToHost);

    }


    FILE *f;
    f = fopen("energy.csv", "w");
    fprintf(f, "TIME-STEP,ENERGY\n");
    for(int i=0; i<TOTAL_TIME_STEPS;i++)
    {
        fprintf(f,"%d,%f\n",i,h_energy[i]);
    }
    fclose(f);

    f = fopen("cofm.csv", "w");
    fprintf(f, "TIME-STEP,C-of-M\n");
    for(int i=0;i<TOTAL_TIME_STEPS; i++)
    {
        fprintf(f,"%d,%f\n",i,h_cofm[i]);
    }
    fclose(f);

    free(h_posx);
    free(h_posy);
    free(h_energy);
    free(h_cofm);

    cudaFree(d_posx);
    cudaFree(d_posy);
    cudaFree(d_velx);
    cudaFree(d_vely);
    cudaFree(d_accx0);
    cudaFree(d_accy0);
    cudaFree(d_accx1);
    cudaFree(d_accy1);
    cudaFree(d_energy);
    cudaFree(d_cofm);

}


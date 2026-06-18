#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

#define N (1024)                // We can go higher but its difficult to animate.
#define TOTAL_TIME_STEPS (1024)
#define dt 0.0625f
#define G 1
#define M 1
#define epsilon 0.01
#define SIM_DIM 1000


struct Params
{
    float *posx;
    float *posy;
    float *energy;
    float *cofm;
};

__device__ void getderivatives(float v, float a, float *x_d, float *v_d)
{
    *x_d = v;
    *v_d = a;
}

__device__ void rk4(float x, float v, float a, float *x_next, float *v_next)
{
    float x1, v1, x2, v2, x3, v3, x4, v4;

    getderivatives(v, a, &x1, &v1);
    getderivatives(v + ((dt/2.0f)*v1), a, &x2, &v2 );
    getderivatives(v + ((dt/2.0f)*v2), a, &x3, &v3);
    getderivatives(v + (dt*v3), a, &x4, &v4);

    *x_next = x + dt*(x1+2*x2+2*x3+x4)/6.0f;
    *v_next = v + dt*(v1+2*v2+2*v3+v4)/6.0f;

}

__global__ void simulator(Params P)
{
    int xid = threadIdx.x + blockIdx.x*blockDim.x;
    int tilesize = blockDim.x;
    int total_tiles = gridDim.x/tilesize;

    __shared__ float s_posx[tilesize];
    __shared__ float s_posy[tilesize];

    float curr_posx = P.posx[xid];
    float curr_posy = P.posy[xid];
    float curr_velx = 0;
    float curr_vely = 0;

    float dx=0;
    float dy=0;
    float rsq;

    float ax = 0;
    float ay = 0;

    float pcos=0;
    float psin=0;

    float tile_cofm=0;
    float tile_energy=0;

    for(int i=0; i<TOTAL_TIME_STEPS; i++)
    {
        for(int j=0; j<total_tiles; j++)
        {
            s_posx[threadIdx.x] = P.posx[threadIdx.x + j];
            s_posy[threadIdx.x] = P.posy[threadIdx.x + j];
            __syncthreads();

            for(k=0; k<tilesize; k++)
            {
                if (xid == j*tilesize + k)
                    continue;
                dx = s_posx[k] - curr_posx;
                dy = s_posy[k] - curr_posy;

                rsq = dx*dx + dy*dy + epsilon;
                pcos = dx/sqrtf(rsq);
                psin = dy/sqrtf(rsq);
                ax = ax + (G*m/rsq)*pcos;
                ay = ay + (G*m/rsq)*psin;

                atomicAdd(&tile_energy, 1/sqrtf(rsq));
                
            }
        }

        // all tiles loaded,all accelerations summed up, now we step through time
        rk4(curr_posx, curr_velx, ax, &curr_posx, &curr_velx);
        rk4(curr_posy, curr_vely, ay, &curr_posy, &curr_vely);
        //the rk4 also updates the curr_pos, and curr_vel.

        atomicAdd(&tile_cofm, curr_posx*curr_posx + curr_psy*curr_posy);        // get center of mass
        tile_cofm = tile_cofm/N;

        atomicAdd(&tile_energy, curr_velx*curr_velx + curr_vely*curr_vely);     // get kinetic energy

        int xfact =  (int)((curr_posx > SIM_DIM) | (curr_posx < 0));          //if true need to reflect.
        int yfact =  (int)((curr_posy > SIM_DIM) | (curr_posy < 0));

        xfact = 1- xfact*2;
        yfact = 1- yfact*2;

        curr_velx *= xfact;
        curr_vely *= yfact;

        P.posx[i][xid] = curr_posx;
        P.posy[i][xid] = curr_posy;

        P.energy[i] = tile_energy;
        P.cofm[i] = tile_cofm;

        tile_energy = 0;
        tile_cofm = 0;

    }
}

void initialize(float *arr)
{
    long seed = time(NULL);
    for (int i=0; i<N; i++)
    {
            arr[i] = rand(seed+i) % SIM_DIM/2;   
    }
}

int main()
{
    dim3 blocksize(1024);
    dim3 gridsize((N+1024-1)/1024);

    int memsize2d = N * TOTAL_TIME_STEPS* sizeof(float);
    int memsize1d = TOTAL_TIME_STEPS*sizeof(float);

    float * h_posx, *h_posy, *h_energy, *h_cofm;
    float * d_posx, *d_posy, *d_energy, *d_cofm;

    h_posx = (float*)malloc(memsize2d);
    h_posy = (float*)malloc(memsize2d);
    h_energy = (float*)malloc(memsize1d);
    h_cofm = (float*)malloc(memsize1d);

    initialize(h_posx);
    initialize(h_posy);

    cudaMalloc(&d_posx, memsize2d);
    cudaMalloc(&d_posy, memsize2d);
    cudaMalloc(&d_energy, memsize1d);
    cudaMalloc(&d_cofm, memsize1d);

    cudaMemcpy(d_posx, h_posx, memsize2d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posy, h_posy, memsize2d, cudaMemcpyHostToDevice);

    Params P;
    P.posx = d_posx;
    P.posy = d_posy;
    P.energy = d_energy;
    P.cofm = d_cofm;

    simulator<<<gridsize, blocksize>>>(P);

    
    return 0;
}
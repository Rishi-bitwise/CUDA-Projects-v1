#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>

#define bin_size 256
#define THRESH 5 

struct cell
{
    float pressure;
    float density;
    float velx;
    float vely;
    float temp;
};

struct Patch_data
{
    int cols;
    int rows;
    int level;
    cell *matrix;
    Patch_data * parent;
    Patch_data * children;
    bool *mask;

};

void init_original(Patch_data* original)
{
    original->children = nullptr;

}


void get_mask(Patch_data* mypatch)
{
    int rows = mypatch->rows;
    int cols = mypatch->cols;
    int indx = 0;
    int indy = 0;
    float grad= 0;
    cell * cellmat;

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            indx = i*cols + j;
            cellmat = mypatch->matrix;
            if ((j>0) && (j<cols-1))
            {
                grad = (cellmat[indx+1].pressure - 2*cellmat[indx].pressure + cellmat[indx-1].pressure)/cellmat[indx].pressure;
                if (grad>THRESH)
                    mypatch->mask[indx] = true;
                else
                    mypatch->mask[indx] = false;
            }

            if ((i>0) && (i<rows-1))
            {
                
                grad = (cellmat[indx+cols].pressure - 2*cellmat[indx].pressure + cellmat[indx-cols].pressure)/cellmat[indx].pressure;
                if (grad>THRESH)
                    mypatch->mask[indx] = true;
                else
                    mypatch->mask[indx] = false;
                
            }
            
        }
    }

}


int main()
{
    Patch_data original;
    init_original(&original);

    Patch_data *temp = &(original);

    while(temp != nullptr)
    {

    }


    return 0;
}




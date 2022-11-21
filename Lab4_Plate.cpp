/* 
 * Author : Aakash Venkataraman
 * Class: ECE 6122 A
 * Last Date Modified: 12th Novemeber 2022
 * File Description: This program calculates the steady state temperature of the hot plate and creates the new file which contains the temperature values. It also prints the execution time.
 */

#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "cuda_runtime.h"

using namespace std;

void steadystatetemperature(double *, double *, int);     //Function to calculate steady state temperature of the plate

inline cudaError_t HANDLE_ERROR(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) 
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


int main(int argc, char** argv)
{
 int matrixvalid = 0, Iterationvalid = 0,k =0;
 int matrix_i=0, Iteration_number =0;
 while((k = getopt(argc, argv, "n:I:")) !=-1)
 {
  switch(k)
  {
   case 'n':   
   {
      int Matrix_length = 0;
      for(int i =0; i<strlen(optarg);i++)
      { 
      
         if(isdigit(optarg[i]) && (optarg[i]>=0))
         {
             Matrix_length++;}
      }
         if(Matrix_length == strlen(optarg))
         { 
             matrixvalid =1;
             matrix_i = stoi(optarg);
         }
         else
         { 
            cout<<"Input values of n and I are invalid.";
            matrixvalid = 0;
         } 
    break;
   }
    case 'I':
    { 
       int Iteration_count = 0;
       for(int i =0; i<strlen(optarg);i++)
       {
          if(isdigit(optarg[i]) && optarg[i] > 0)
          {
              Iteration_count++;}
       }
          if(Iteration_count == strlen(optarg))
          {
             Iteration_number = stoi(optarg);
             Iterationvalid = 1;
          }
          else
          {
             cout<<"Input values of n and I are invalid";
             Iterationvalid = 0;
          }
     break;
   }
    default:
       cout<<"enter integer values of n and I";
       exit(1);
   }
 } 

if(matrixvalid && Iterationvalid)
{
   int width = matrix_i + 2;
            
   int size = (width) * (width) *sizeof(double);
   double *g, *h;      
   ofstream outfile;
   outfile.open("finalTemperatures.csv");

   //Creating Cuda Event
   cudaEvent_t     start, stop;
   HANDLE_ERROR(cudaEventCreate(&start));
   HANDLE_ERROR(cudaEventCreate(&stop));
   HANDLE_ERROR(cudaEventRecord(start, 0));

   //Allocating shared memory
   cudaMallocManaged(&g, size);
   cudaMallocManaged(&h, size);
      
       
    //Initializing boundary values and other matrices values during initial stage
        
   for (int i = 0; i < width; i++)
   {
      for (int j = 0; j < width; j++)
      {
          int index = i*width + j;
          if (i == 0)
          {
             if ((j >= 0.3*width) && (j < (0.7*width - 1)))
             {
                g[index] = 100.0;
                continue;
             }
             else
             {
                g[index] = 20.0;
                continue;
             }
          } 
          if (i == width - 1)
          {
             g[index] = 20.0;
             continue;
          }
          if (j == 0 || j == width - 1)
          {
             g[index] = 20.0;
             continue;
          }
     g[index] = 0.0;
       } 
   }         
    
     
   

  for(int b =0; b< Iteration_number;b = b+2)
  {
     steadystatetemperature(g,h,width);
     cudaDeviceSynchronize();
     steadystatetemperature(h,g,width);
     cudaDeviceSynchronize();
  }
   // Printing values in file
  outfile<<setprecision(15);
  for (int y=0; y<width; y++)
  { 
     for (int x =0; x< width; x++)
     {
         int index = y*width +x;
         outfile<<h[index];
         if( x<width-1)
             outfile<<",";
     }
     outfile<<endl;
  }

  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf( "Thin plate calculation took %3.3f milliseconds.\n", elapsedTime );

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  cudaFree(g);
  cudaFree(h);

  outfile.close();
 } 
return 0;
}


__global__ void calculateshValue(double *Gd, double *Hd, int m) 
{
  //calculate the row index, denote by y 
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  // Calculate the column index of the Pd element, denote by x
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  double Pvalue =0;
  if(i< m && j<m)
  {
     if( i == 0 || i == m -1)
        Pvalue = Gd[i*m + j];
     else if (j == 0 || j == m-1)
        Pvalue = Gd[i*m +j];
     else
        Pvalue = 0.25*(Gd[(i-1)*m + j]+ Gd[(i+1)* m +j]+ Gd[i*m + (j-1)] + Gd[i*m + (j+1)]);
         
     Hd[i*m +j] = Pvalue;
  }
}


void steadystatetemperature(double *g, double *h, int t)
{
  int threadsize, blocksize;
 

  if(t >32)
  {
      threadsize = 32;
      blocksize = (t +31)/32;
  }
  else
  {
      threadsize = t;
      blocksize = 1;
  }

  dim3 dimBlock(threadsize, threadsize, 1);
  dim3 dimGrid(blocksize, blocksize, 1);
  
  calculateshValue<<<dimGrid, dimBlock>>>(g, h, t);
}



    

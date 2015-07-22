/*******************************************************************
************PROGRAM FOR NUMERICAL INTEGRATION IN CUDA***************
********************************************************************
* CODE BY: Justin Smith						   *
*								   *
* REQUIRED LIBRARIES:					       	   *
* cuda_reduction.h (For array summation of approximation methods)  *
* 						    		   *
* SPECIAL USE INSTRUCTIONS:                         		   *
*                                                   		   * 
* Requires device constant memory to be defined prior to calling   *
* the integral function/s as well as the working array. This is	   *
* accomplished by running the set_indi() function prior to any	   *
* integration runs. Program takes full advantage of GPU 	   *
* acceleration when running many integrals with the same set_indi()*
* call. 	   						   *
*						    		   *
* AccN in the accuracy number. It is defined as:    		   *
* AccN = 0 --> 16384				    		   *
* AccN = 1 --> 1048576				    		   *
* AccN = 2 --> 16777216				    		   *
* AccN = 3 --> 1073741824                           		   *
*						    		   *
* This is predefined for speed optimization of code.		   *
*						    		   *
* EXTERNAL PROGRAMS INCLUDED:                       		   *
*                                                   		   *
* Call Functions: 				    		   *
* simpsons_float_1D(float *d_I,float d_data_arr,int AccN, int idx) *
* set_indi(float x0,float xN,unsigned int AccN,float **d_data_arr) *
*******************************************************************/

// *** INCLUDED LIBRARIES*** //
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cmath>
#include "cuda_reduction.h"

// ***CUDA ERROR HANDLER*** //
#define CUDA_MEMORY_CALL(x) do { if((x) != cudaSuccess) { \
printf("Cuda Memory Error at %s:%d\n",__FILE__,__LINE__);\
exit(EXIT_FAILURE);}} while(0)

// ********************************************************************* //
// ********************CONSTANT MEMORY DECLARAION*********************** //
// ********************************************************************* //
__constant__ float dc_int_params[3];

extern void set_indi(float x0,float xN,unsigned int AccN,float **d_data_arr)
{
	int idsx;
	switch (AccN)
	{
		case 0: {idsx = 16384; break;}
		case 1: {idsx = 1048576; break;}
		case 2: {idsx = 16777216; break;}
		case 3: {idsx = 1073741824; break;}
		default: {
                std::cout << "!!! Error, invalid accuracy selected for integration !!!\n";
                exit (EXIT_FAILURE);
		break;}
	}
	float dx = fabs(xN - x0) / idsx;
	std::cout << "dx= " << dx << "\n";

	float *trans_arr;	
	trans_arr = (float *)malloc(3 * sizeof(float));

	trans_arr[0] = x0;
	trans_arr[1] = dx;
	trans_arr[2] = dx/3;
	
	CUDA_MEMORY_CALL(cudaMalloc((float **)d_data_arr, idsx * sizeof(float)));
	CUDA_MEMORY_CALL(cudaMemcpyToSymbol(dc_int_params,trans_arr,3*sizeof(float)));
}

// ********************************************************************* //
// ********************FUNCTIONS FOR EVALUTATION************************ //
// ********************************************************************* //
__device__ float cuda_function_test(float x)
{
        float eval;

        eval = powf(x,2);//expf(-powf(x,2));

        return eval;
}


// ********************************************************************* //
// *********************SIMPSONS RULE FOR FLOATS************************ //
// ********************************************************************* //

// ****************************************** //
// *****FLOATS DEVICE SIDE KERNEL PROGRAM**** //
// ****************************************** //
template <unsigned int N>
__global__ void cuda_simps_flt_1D(float *d_idat)
{
        //LOAD ELEMENT FROM GLOBAL TO SHARED MEM
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
	float tmp_eval;
	float x = dc_int_params[0] + i * dc_int_params[1];

        //EXECUTE SIMPSONS RULE
	switch(i)
	{
		case 0://FIRST TERM 
			tmp_eval = cuda_function_test(x);
		break;

		case N-1://LAST TERM
			tmp_eval = cuda_function_test(x);
		break;
		
		default://ALL OTHER TERMS
			tmp_eval = 2 * ((i % 2) + 1) * cuda_function_test(x);
		break;
        }
	//WRITE RESULT TO GLOBAL DEVICE MEMORY
        d_idat[i] = dc_int_params[2] * tmp_eval;
}
// ****************************************** //
// ***********KERNEL LAUNCHER PROGRAM******** //
// ****************************************** //
template <const unsigned int N>
void simpsons_float_klauncher(float *d_data_arr,int MAX_THREADS)
{
        int BLOCKS = N/MAX_THREADS;
	int THREADS = MAX_THREADS;

        /* Launch Integration Kernel */
        cuda_simps_flt_1D<N><<<BLOCKS,THREADS>>>(d_data_arr);

        /* Launch Floats Reduction Program */
        cuda_asum_fltm(d_data_arr,N,MAX_THREADS);
}

// ********************************************** //
// ***SIMPs RULE FLTS HOST SIDE KERNEL PROGRAM*** //
// ********************************************** //

// d_I is declared in the main program
// PRE-DEFINED ACCURACY (0 = 2^14,1 = 2^20, 2 = 2 ^ 24)
// PRE-DEFINED FUNCTIONS
extern void simpsons_float_1D(float *d_I,float *d_data_arr,int AccN, int idx)
{
	/*WANT TO SET THESE PROPERTIES AS A GLOBAL VARIABLE*/
	int device, MAX_THREADS;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaError_t (cudaGetDeviceProperties(&props,device));
        cudaError_t devpcode = cudaGetLastError();
        if (devpcode != cudaSuccess)
                printf("Cuda device setting error -- %s\n",cudaGetErrorString(devpcode));
        MAX_THREADS =  props.maxThreadsPerBlock;

        //std::cout << "DEVICE: " << device << "\n";
	//std::cout << "MAX THREADS: " << MAX_THREADS << "\n";

	switch (AccN)
	{
		case 0:
		{
		        const unsigned int N = 16384;
			simpsons_float_klauncher<N>(d_data_arr,MAX_THREADS);
			break;
		}
                case 1:
		{
                	const unsigned int N = 1048576;
			simpsons_float_klauncher<N>(d_data_arr,MAX_THREADS);
                	break;
		}
                case 2:
		{
                	const unsigned int N = 16777216;
			simpsons_float_klauncher<N>(d_data_arr,MAX_THREADS);
                	break;
		}
                case 3:
                {
                        const unsigned int N = 1073741824;
			simpsons_float_klauncher<N>(d_data_arr,MAX_THREADS);
                        break;
                }
		default:
		{
			std::cout << "!!! Error, invalid accuracy selected for integration !!!\n";
			exit (EXIT_FAILURE);
		}
                break;
	}

        /* Copy integral values to given array */
        CUDA_MEMORY_CALL(cudaMemcpy(&d_I[idx], &d_data_arr[0], sizeof(float), cudaMemcpyDeviceToDevice));
}


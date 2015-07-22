/**************************************************
*******PROGRAM FOR ARRAY REDUCTION IN CUDA*********
***************************************************
* EXTERNAL PROGRAMS INCLUDED:			  *
*						  *
* SUMS UPTO 2^24 INTS BEFORE REACHING BLOCK LIMIT *
* ARRAY MUST CONTAIN POWER OF TWO ELEMENTS	  *
* PROGRAM INCLUDES INTS,FLOATS AND DOUBS          *
*						  *
* Call INTS: cuda_asum_intm(ipt/opt array, N#s)   *
* Call FLTS: cuda_asum_fltm(ipt/opt array, N#s)   *
* Call DOUBS: cuda_asum_doubm(ipt/opt array, N#s) *
*						  *
***************************************************/

// *** INCLUDED LIBRARIES*** //
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

// ***CUDA ERROR HANDLER*** //
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)

// ********************************************************************* //
// **********************INTEGER ARRAY SUMMATION************************ //
// ********************************************************************* //

// ****************************************** //
// ***INTEGERS DEVICE SIDE KERNEL PROGRAM**** //
// ****************************************** //
template <unsigned int blockSize>
__global__ void cuda_asum_int(int *d_idat, int *d_odat)
{
	extern __shared__ int sdata[];

	//LOAD ELEMENT FROM GLOBAL TO SHARED MEM
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
	
	sdata[tid] = d_idat[i] + d_idat[i + blockDim.x];
	__syncthreads();

	//REDUCTION
	if (blockSize >= 512) 
	{
		if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();
	}
        if (blockSize >= 256)
        {
                if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();
        }
        if (blockSize >= 128)
        {
                if (tid < 64) {sdata[tid] += sdata[tid + 64];} __syncthreads();
        }
     
	if (tid < 32)
	{
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; 
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; 
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; 
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; 
	}

	//WRITE RESULT TO GLOB MEM
	if(tid == 0) d_odat[blockIdx.x] = sdata[0];
}

// ****************************************** //
// ***INTS HOST SIDE KERNEL CALLER PROGRAM*** //
// ****************************************** //

//DEVICE ARRAY (d_idat) NEEDS TO BE DEFINED IN PARENT PROGRAM
extern void cuda_asum_intm(int *d_idat, int N_elem) //sqrt(N_elem) must be an int
{

	//THREAD AND BLOCK SIZES
	unsigned int THREAD_SIZE = 512/2;
	unsigned int BLOCK_SIZE = N_elem/512;
	unsigned int FINAL_THREAD;
	
        if (N_elem > 512)
        {
	//SUM THE ARRAY THROUGH MULTIPLE KERNEL CALLS
	while(BLOCK_SIZE > 1)
	{
		//RUN REDUCTIONS
		switch (THREAD_SIZE)
			{
			case 512:
				cuda_asum_int<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat); break;
                        case 256:
                                cuda_asum_int<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(int)>>>(d_idat,d_idat); break;
                        case 128:
                                cuda_asum_int<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(int)>>>(d_idat,d_idat); break;
                        case 64:
                                cuda_asum_int< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(int)>>>(d_idat,d_idat); break;
                        case 32:
                                cuda_asum_int< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(int)>>>(d_idat,d_idat); break;
                        case 16:
                                cuda_asum_int< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(int)>>>(d_idat,d_idat); break;
                        case 8:
                                cuda_asum_int<  8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(int)>>>(d_idat,d_idat); break;
                        case 4:
                                cuda_asum_int<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(int)>>>(d_idat,d_idat); break;
                        case 2:
                                cuda_asum_int<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(int)>>>(d_idat,d_idat); break;
                        case 1:
                                cuda_asum_int<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(int)>>>(d_idat,d_idat); break;
			}
		//cuda_asum_int<<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat);
		FINAL_THREAD = BLOCK_SIZE;
		BLOCK_SIZE = BLOCK_SIZE/512;
	}
        } else {
                FINAL_THREAD = (unsigned int)N_elem;
        }

	THREAD_SIZE = FINAL_THREAD/2;
	BLOCK_SIZE = 1;
        //RUN REDUCTIONS
        switch (THREAD_SIZE)
        	{
                case 512:
                	cuda_asum_int<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat); break;
                case 256:
                        cuda_asum_int<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(int)>>>(d_idat,d_idat); break;
                case 128:
                        cuda_asum_int<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(int)>>>(d_idat,d_idat); break;
                case 64:
                        cuda_asum_int< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(int)>>>(d_idat,d_idat); break;
                case 32:
                        cuda_asum_int< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(int)>>>(d_idat,d_idat); break;
                case 16:
                        cuda_asum_int< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(int)>>>(d_idat,d_idat); break;
                case 8:
                        cuda_asum_int<  8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(int)>>>(d_idat,d_idat); break;
                case 4:
                        cuda_asum_int<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(int)>>>(d_idat,d_idat); break;
                case 2:
                        cuda_asum_int<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(int)>>>(d_idat,d_idat); break;
                case 1:
                        cuda_asum_int<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(int)>>>(d_idat,d_idat); break;
                }

}

// ********************************************************************* //
// ************************FLOAT ARRAY SUMMATION************************ //
// ********************************************************************* //

// ****************************************** //
// ****FLOATS DEVICE SIDE KERNEL PROGRAM***** //
// ****************************************** //
template <unsigned int blockSizeflt>
__global__ void cuda_asum_flt(float *d_idat, float *d_odat)
{
        extern __shared__ float sdataf[];

        //LOAD ELEMENT FROM GLOBAL TO SHARED MEM
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

        sdataf[tid] = d_idat[i] + d_idat[i + blockDim.x];
        __syncthreads();

        //REDUCTION
        if (blockSizeflt >= 1024)
        {
                if (tid < 512) {sdataf[tid] += sdataf[tid + 512];} __syncthreads();
        }
        if (blockSizeflt >= 512)
        {
                if (tid < 256) {sdataf[tid] += sdataf[tid + 256];} __syncthreads();
        }	
        if (blockSizeflt >= 256)
        {
                if (tid < 128) {sdataf[tid] += sdataf[tid + 128];} __syncthreads();
        }

        if (tid < 64)
        {
		if (blockSizeflt >= 128) sdataf[tid] += sdataf[tid + 64];__syncthreads();
                if (blockSizeflt >= 64) sdataf[tid] += sdataf[tid + 32];__syncthreads();
                if (blockSizeflt >= 32) sdataf[tid] += sdataf[tid + 16];__syncthreads();
                if (blockSizeflt >= 16) sdataf[tid] += sdataf[tid + 8];__syncthreads();
                if (blockSizeflt >= 8) sdataf[tid] += sdataf[tid + 4];__syncthreads();
                if (blockSizeflt >= 4) sdataf[tid] += sdataf[tid + 2];__syncthreads();
		if (blockSizeflt >= 2) sdataf[tid] += sdataf[tid + 1];__syncthreads();
        }
        //WRITE RESULT TO GLOB MEM
        if(tid == 0) d_odat[blockIdx.x] = sdataf[0];
}

// ****************************************** //
// **FLOATS HOST SIDE KERNEL CALLER PROGRAM** //
// ****************************************** //

//DEVICE ARRAY (d_idat) NEEDS TO BE DEFINED IN PARENT PROGRAM
extern void cuda_asum_fltm(float *d_idat, int N_elem,int MAX_THREADS) //sqrt(N_elem) must be an int
{
	
        //THREAD AND BLOCK SIZES
        unsigned int THREAD_SIZE = MAX_THREADS;
        unsigned int BLOCK_SIZE = N_elem/(MAX_THREADS*2);
        unsigned int FINAL_THREAD;

        if (N_elem > THREAD_SIZE)
        {
        	//SUM THE ARRAY THROUGH MULTIPLE KERNEL CALLS
        	while(BLOCK_SIZE > 1)
        	{
        	        //RUN REDUCTIONS
        	        switch (THREAD_SIZE)
        	                {
        	                case 1024:
        	                        cuda_asum_flt<1024><<<BLOCK_SIZE,THREAD_SIZE,1024*sizeof(int)>>>(d_idat,d_idat); break;
				case 512:
        	                        cuda_asum_flt<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 256:
        	                        cuda_asum_flt<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 128:
        	                        cuda_asum_flt<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 64:
        	                        cuda_asum_flt< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 32:
        	                        cuda_asum_flt< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 16:
        	                        cuda_asum_flt< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 8:
        	                        cuda_asum_flt<  8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 4:
        	                        cuda_asum_flt<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 2:
        	                        cuda_asum_flt<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(int)>>>(d_idat,d_idat); break;
        	                case 1:
        	                        cuda_asum_flt<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(int)>>>(d_idat,d_idat); break;
        	                }
        	        FINAL_THREAD = BLOCK_SIZE;
        	        BLOCK_SIZE = BLOCK_SIZE/1024;
        	}
	} else {
		FINAL_THREAD = (unsigned int)N_elem/2;
	}
        THREAD_SIZE = FINAL_THREAD/2;
        BLOCK_SIZE = 1;
        //RUN REDUCTIONS
        switch (THREAD_SIZE)
                {
                case 1024:
                        cuda_asum_flt<1024><<<BLOCK_SIZE,THREAD_SIZE,1024*sizeof(int)>>>(d_idat,d_idat); break;
                case 512:
                        cuda_asum_flt<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat); break;
                case 256:
                        cuda_asum_flt<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(int)>>>(d_idat,d_idat); break;
                case 128:
                        cuda_asum_flt<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(int)>>>(d_idat,d_idat); break;
                case 64:
                        cuda_asum_flt< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(int)>>>(d_idat,d_idat); break;
                case 32:
                        cuda_asum_flt< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(int)>>>(d_idat,d_idat); break;
                case 16:
                        cuda_asum_flt< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(int)>>>(d_idat,d_idat); break;
                case 8:
                        cuda_asum_flt< 8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(int)>>>(d_idat,d_idat); break;
                case 4:
                        cuda_asum_flt<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(int)>>>(d_idat,d_idat); break;
                case 2:
                        cuda_asum_flt<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(int)>>>(d_idat,d_idat); break;
                case 1:
                        cuda_asum_flt<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(int)>>>(d_idat,d_idat); break;
                }
}

// ********************************************************************* //
// ***********************DOUBLE ARRAY SUMMATION************************ //
// ********************************************************************* //

// ****************************************** //
// ****DOUBLES DEVICE SIDE KERNEL PROGRAM**** //
// ****************************************** //

template <unsigned int blockSizedoub>
__global__ void cuda_asum_doub(double *d_idat, double *d_odat)
{
        extern __shared__ double sdatad[];

        //LOAD ELEMENT FROM GLOBAL TO SHARED MEM
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

        sdatad[tid] = d_idat[i] + d_idat[i + blockDim.x];
        __syncthreads();

        //REDUCTION
        if (blockSizedoub >= 512)
        {
                if (tid < 256) {sdatad[tid] += sdatad[tid + 256];} __syncthreads();
        }
        if (blockSizedoub >= 256)
        {
                if (tid < 128) {sdatad[tid] += sdatad[tid + 128];} __syncthreads();
        }
        if (blockSizedoub >= 128)
        {
                if (tid < 64) {sdatad[tid] += sdatad[tid + 64];} __syncthreads();
        }

        if (tid < 32)
        {
                if (blockSizedoub >= 64) sdatad[tid] += sdatad[tid + 32];
                if (blockSizedoub >= 32) sdatad[tid] += sdatad[tid + 16];
                if (blockSizedoub >= 16) sdatad[tid] += sdatad[tid + 8];
                if (blockSizedoub >= 8) sdatad[tid] += sdatad[tid + 4];
                if (blockSizedoub >= 4) sdatad[tid] += sdatad[tid + 2];
                if (blockSizedoub >= 2) sdatad[tid] += sdatad[tid + 1];
        }

        //WRITE RESULT TO GLOB MEM
        if(tid == 0) d_odat[blockIdx.x] = sdatad[0];
}

// ****************************************** //
// *DOUBLES HOST SIDE KERNEL CALLER PROGRAM** //
// ****************************************** //

//DEVICE ARRAY (d_idat) NEEDS TO BE DEFINED IN PARENT PROGRAM
extern void cuda_asum_doubm(double *d_idat, double N_elem) //sqrt(N_elem) must be an int
{

        //THREAD AND BLOCK SIZES
        unsigned int THREAD_SIZE = 512/2;
        unsigned int BLOCK_SIZE = N_elem/512;
        unsigned int FINAL_THREAD;

        if (N_elem > 512)
        {
        	//SUM THE ARRAY THROUGH MULTIPLE KERNEL CALLS
        	while(BLOCK_SIZE > 1)
        	{
        	        //RUN REDUCTIONS
        	        switch (THREAD_SIZE)
        	                {
        	                case 512:
               		                cuda_asum_doub<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 256:
                	                cuda_asum_doub<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 128:
                	                cuda_asum_doub<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 64:
                	                cuda_asum_doub< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 32:
                	                cuda_asum_doub< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 16:
                	                cuda_asum_doub< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 8:
                	                cuda_asum_doub<  8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 4:
                	                cuda_asum_doub<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 2:
                	                cuda_asum_doub<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(double)>>>(d_idat,d_idat); break;
                	        case 1:
                	                cuda_asum_doub<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(double)>>>(d_idat,d_idat); break;
                	        }
                	//cuda_asum_int<<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(int)>>>(d_idat,d_idat);
                	FINAL_THREAD = BLOCK_SIZE;
                	BLOCK_SIZE = BLOCK_SIZE/512;
        	}
        } else {
                FINAL_THREAD = (unsigned int)N_elem;
        }

	// THIS SECTION BEGINS WITH THREADS NEEDED ARE LESS THAN MAXIMUM BLOCK SIZE
        THREAD_SIZE = FINAL_THREAD/2;
        BLOCK_SIZE = 1;
        //RUN REDUCTIONS
        switch (THREAD_SIZE)
                {
                case 512:
                        cuda_asum_doub<512><<<BLOCK_SIZE,THREAD_SIZE,512*sizeof(double)>>>(d_idat,d_idat); break;
                case 256:
                        cuda_asum_doub<256><<<BLOCK_SIZE,THREAD_SIZE,256*sizeof(double)>>>(d_idat,d_idat); break;
                case 128:
                        cuda_asum_doub<128><<<BLOCK_SIZE,THREAD_SIZE,128*sizeof(double)>>>(d_idat,d_idat); break;
                case 64:
                        cuda_asum_doub< 64><<<BLOCK_SIZE,THREAD_SIZE,64*sizeof(double)>>>(d_idat,d_idat); break;
                case 32:
                        cuda_asum_doub< 32><<<BLOCK_SIZE,THREAD_SIZE,32*sizeof(double)>>>(d_idat,d_idat); break;
                case 16:
                        cuda_asum_doub< 16><<<BLOCK_SIZE,THREAD_SIZE,16*sizeof(double)>>>(d_idat,d_idat); break;
                case 8:
                        cuda_asum_doub<  8><<<BLOCK_SIZE,THREAD_SIZE,8*sizeof(double)>>>(d_idat,d_idat); break;
                case 4:
                        cuda_asum_doub<  4><<<BLOCK_SIZE,THREAD_SIZE,4*sizeof(double)>>>(d_idat,d_idat); break;
                case 2:
                        cuda_asum_doub<  2><<<BLOCK_SIZE,THREAD_SIZE,2*sizeof(double)>>>(d_idat,d_idat); break;
                case 1:
                        cuda_asum_doub<  1><<<BLOCK_SIZE,THREAD_SIZE,1*sizeof(double)>>>(d_idat,d_idat); break;
                }

}


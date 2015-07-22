#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include "cuda_lib_numer_int.h"

int main (int argc, char *argv[])
{
	unsigned int AccN = 2;
	float x0 = 0,xN = 4.0;
	float *d_I,*I;
	float *d_data_arr;

	/* Allocate n floats on host */
        I = (float *)malloc(1 * sizeof(float));

        /* Allocate n ints on device */
        cudaMalloc((float **)&d_I, 1 * sizeof(float));

	/* launch program with kernel executions */
	set_indi(x0,xN,AccN,&d_data_arr);

	for (int i = 0; i < 1; ++i)
	{
		simpsons_float_1D(d_I, d_data_arr, AccN, 0);
	}

	/* Copy array back to the host */
	cudaMemcpy(I, &d_I[0],sizeof(float), cudaMemcpyDeviceToHost);

	float act_val = 21.3333333333333333;
        std::cout << "Integral Value = ";
	std::cout << std::setprecision(20) << I[0] << "\n";

        std::cout << "Error = ";
	float tval = abs(I[0] - act_val);
        std::cout << std::setprecision(20) << tval << "\n";

	free(I);
	cudaFree(d_I);cudaFree(&d_data_arr);cudaFree(d_data_arr);
	return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main (int argc, char *argv[])
{
	long int N = 4194304*2*2,i,j,sum;
	int *test_arr;

	/* Allocate n floats on host */
        test_arr = (int *)malloc(N * sizeof(int));

	/* Define array */
	for(i = 0; i < N; ++i)
	{
		test_arr[i] = 1;
		//std::cout << 1 << "+";
	}
	std::cout << "SUM = ";	
	sum = 0;
        /* REDUCE ARRAY */
	for (j = 0; j < 100; ++j)
	{
		for (i = 0; i < N; ++i)
		{
			sum += test_arr[i];
		}
	}
	std::cout << sum << "\n";

	free(test_arr);
	return 0;
}


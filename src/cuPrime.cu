#include "cuPrime.hu"
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include <math.h>

__global__ void _millerRabinTest(curandState* globalState, unsigned int n, int s, int d, volatile bool * isPrime)
{
    curandState localState = globalState[threadIdx.x];
    unsigned int a = getRandomInt(localState, n);

    unsigned int p = powerMod(a, d, n);
    if (p != 1 && *isPrime) {
        bool isComposite = true;
        for (int i = 0; i < s; ++i) {
            int powerOf2 = pow((float) 2, (float) i);
            unsigned int k = powerMod(a, d*powerOf2, n);

            if (k == n-1) {
                isComposite = false;
            }
        }

        if (isComposite) {
            *isPrime = false;
        }
    }
}

__device__ int getRandomInt(curandState localState, int max) {
    float myrandf = curand_uniform(&localState);
    myrandf *= max - 1;
    myrandf += 1;
    return (int)truncf(myrandf);
}

__device__ int powerMod(int a, int b, int m) {
    long result = 1;
    long x = a%m;

    for (int i=1; i<=b; i<<=1)
    {
        x %= m;
        if ((b&i) != 0)
        {
            result *= x;
            result %= m;
        }
        x *= x;
    }
    return result;
}


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}
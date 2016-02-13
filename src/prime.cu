#include "prime.hu"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#include "handleError.h"
#include "cuPrime.hu"

#include <curand.h>
#include <curand_kernel.h>


using namespace std;


void PrimeController::readFromFile(char* path){
    fstream stream(path, std::ios_base::in);
    long a;
    while (stream >> a) {
        numbers.push_back(a);
    }
}

PrimeController::PrimeController(char* path)
{
    readFromFile(path);
    timer = 0;
}

int PrimeController::findBiggestPowerOf2(unsigned int n) {
    unsigned int s = n - 1;
    int i = 0;
    while (s % 2 == 0)
    {
        s /= 2;
        i++;
    }
    return i;
};

bool PrimeController::isPrime(unsigned int n) {
    int N = 50;

    //Measure Time Taken
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Miller-Rabin algorithm
    int s = findBiggestPowerOf2(n);
    int d = (int) floor(n / pow(2, s));

    curandState* devStates;
    bool* d_isPrime;
    bool* h_isPrime = (bool*)malloc(sizeof(bool));
    *h_isPrime = true;

    HANDLE_ERROR(cudaMalloc ( &devStates, N*sizeof( curandState ) ));
    HANDLE_ERROR(cudaMalloc ( &d_isPrime, sizeof( bool ) ));

    dim3 dimBlock(N);
    HANDLE_ERROR(cudaMemcpy(d_isPrime, h_isPrime, sizeof(bool), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    setup_kernel <<<1, dimBlock>>> ( devStates, time(NULL) );
    _millerRabinTest <<<1, dimBlock>>>(devStates, n, s, d, d_isPrime);

    HANDLE_ERROR(cudaMemcpy(h_isPrime, d_isPrime, sizeof(bool), cudaMemcpyDeviceToHost));

    results.insert( pair<unsigned int, bool>(n, *h_isPrime));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timer +=milliseconds;

    cudaFree(devStates);
    cudaFree(d_isPrime);
    free(h_isPrime);
    return true;
}

void PrimeController::startPrimesCheck() {
    for (vector<unsigned int>::iterator it = numbers.begin() ; it != numbers.end(); ++it) {
        unsigned int n = *it;
        if ( n % 2 == 0) {
            results.insert( pair<unsigned int, bool>(n, false));
        } else {
            bool result = isPrime(n);
            results.insert( pair<unsigned int, bool>(n, result));
        }
    }
}

void PrimeController::printResults() {
    for(map<unsigned int, bool>::iterator it = results.begin(); it != results.end(); ++it) {
        if (it->second) {
            printf("%d - prime\n", it->first);
        } else {
            printf("%d - composite\n", it->first);
        }
    }
};

void PrimeController::printTime() {
    printf("Time:\t%fms\n",timer);
}

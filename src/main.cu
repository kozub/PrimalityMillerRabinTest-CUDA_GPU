#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

#include "prime.hu"
#include "cuda_runtime.h"

bool validateCudaDevice();
bool validateArgs(int argc, char *argv[]);
bool resetCudaDevice();



int main(int argc, char *argv[])
{
    if((!validateArgs(argc, argv)) || (!validateCudaDevice())) {
        return -1;
    }

    char* path = argv[1];
    PrimeController *pC = new PrimeController(path);
    pC->startPrimesCheck();
    pC->printTime();
    pC->printResults();
    delete pC;

    if(!resetCudaDevice()) {
        return 1;
    }

    return 0;
}


bool validateCudaDevice() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA device validation - failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }
    return true;
}

bool resetCudaDevice() {
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return false;
    }
    return true;
}

bool validateArgs(int argc, char *argv[]){
    if(argc!=2) {
        fprintf(stderr, "Usage: %s <prime_numbers_file> \n", argv[0]);
        return false;
    }

    struct stat st;
    int result = stat(argv[1], &st);
    if (result != 0) {
        fprintf(stderr, "Program can not open input file \n");
        return false;
    }
    return true;
}

# PrimalityMillerRabinTest-CUDA_GPU

Testing given prime numbers using Miller Rabin's test and NVIDA's CUDA architecture.

Requirements:

    * NVIDIA GPU
    * installed: nvcc

Compilation:

    make

Running:

    ./primes_gpu f

    where:
         * f - text file contains integers (each in new line)

Example:

     ./primes_gpu input

Result:
	
	Time:	4.659104ms
	12344 - composite
	15487271 - prime
	15487291 - prime
	15487309 - prime
	15487313 - prime
	15487317 - composite
	961757501 - prime
	961757633 - prime
	961757637 - composite
	961757809 - prime


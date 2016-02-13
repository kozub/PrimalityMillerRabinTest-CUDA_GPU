default: all

all:
	nvcc -o primes_gpu src/main.cu src/prime.cu src/cuPrime.cu

clean:
	rm primes_gpu

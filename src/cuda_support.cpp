#include "cuda_support.h"
#include <cstdio>
#include <stdexcept>

unsigned int ceil_div(const unsigned int a, const unsigned int b)
{
    return (a + b - 1) / b;
}


void cuda_check(cudaError error, const char *operation)
{
    if (error != cudaSuccess) {
        char message[512];
        (void)sprintf_s(message, 512, "%s error : %d (%s)", operation, error, cudaGetErrorString(error));
        throw std::runtime_error(message);
    }
}
#include "cuda_support.h"
#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <cublas_api.h>
#include <curand.h>

#include <cuda_gl_interop.h>

cudaGraphicsGLRegisterImage();

void testFunction(int (*callback)(int a, void*)) {
    callback(1, nullptr);
}

void cuda_check(cudaError error, const char *operation)
{
    if (error != cudaSuccess) {
        char message[512];
        (void)sprintf_s(message, 512, "%s error : %d (%s)", operation, error, cudaGetErrorString(error));
        throw std::runtime_error(message);
    }
}
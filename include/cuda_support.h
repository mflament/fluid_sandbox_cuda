#pragma once

#include "cuda_runtime.h"

void cuda_check(cudaError error, const char* operation);

inline unsigned ceil_div(unsigned int a, unsigned int b)
{
    return (a + b - 1) / b;
}

// only to avoid syntax error on CUDA C++ syntax and types if not compiling  with NVCC
#ifndef __CUDACC__

template <typename T>
// ReSharper disable once CppFunctionIsNotImplemented
// ReSharper disable once CppInconsistentNaming
static void surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y,
                        cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

// ReSharper disable once CppFunctionIsNotImplemented
// ReSharper disable once CppInconsistentNaming
template <typename T>
static cudaError_t cudaMemcpyToSymbol(T symbol, const void* src, size_t count, size_t offset = 0,
                                      cudaMemcpyKind kind = cudaMemcpyHostToDevice);

// ReSharper disable once CppFunctionIsNotImplemented
// ReSharper disable once CppInconsistentNaming
static void __syncthreads();

#endif

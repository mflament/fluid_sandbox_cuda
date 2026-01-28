#pragma once

#include "cuda_runtime.h"

int ceil_div(int a, int b);

unsigned int ceil_div(unsigned int a, unsigned int b);

void cuda_check(cudaError error, const char* operation);

#if defined(__cplusplus) && defined(__CUDACC__)
#define TEST 42
#endif
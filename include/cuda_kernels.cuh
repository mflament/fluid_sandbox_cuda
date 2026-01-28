#pragma once

#include <cuda_runtime.h>

void update_texture(cudaGraphicsResource_t dst, float4* src, unsigned int nx, unsigned int ny);

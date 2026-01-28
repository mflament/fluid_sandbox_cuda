#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_support.h"

template <class Type>
static __global__ void update_texture_kernel(cudaSurfaceObject_t dst, Type* src, const unsigned int nx,
                                      const unsigned int ny)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny)
    {
        surf2Dwrite(src[y * nx + x], dst, x * sizeof(Type), y);
    }
}


template <typename Type>
static void update_texture(cudaGraphicsResource_t dst, Type* src, unsigned int nx, unsigned int ny)
{
    cuda_check(cudaGraphicsMapResources(1, &dst), "cudaGraphicsMapResources");
    cudaArray_t array;
    cuda_check(cudaGraphicsSubResourceGetMappedArray(&array, dst, 0, 0), "cudaGraphicsSubResourceGetMappedArray");

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface{};
    cuda_check(cudaCreateSurfaceObject(&surface, &resDesc), "cudaCreateSurfaceObject");

    dim3 block{32, 32, 1};
    dim3 grid{ceil_div(nx, block.x), ceil_div(ny, block.y), 1};
    update_texture_kernel<<< grid, block >>>(surface, src, nx, ny);
    cuda_check(cudaGetLastError(), "update_texture_kernel");
    cuda_check(cudaStreamSynchronize(0), "cudaStreamSynchronize");

    cudaDestroySurfaceObject(surface);
    cuda_check(cudaGraphicsUnmapResources(1, &dst), "cudaGraphicsUnmapResources");
}

void update_texture(cudaGraphicsResource_t dst, float4* src, unsigned int nx, unsigned int ny)
{
    update_texture<float4>(dst, src, nx, ny);
}

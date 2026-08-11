// ReSharper disable CppClangTidyMiscUseAnonymousNamespace
#include "cuda_fluid_solver.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "cuda_support.h"
#include <nvrtc.h>

#pragma region CUDA Kernels declarations
static __device__ __constant__ int N = 0;

#define IX(i,j) ((i)+(N+2)*(j))

static __global__ void update_texture_kernel(cudaSurfaceObject_t dst, const float* src);

static __global__ void add_input_kernel(float* dst, int index, float input);

static __global__ void add_source_kernel(float* x, const float* x0, float dt);

static __global__ void lin_solve_kernel(float* x, const float* x0, float a, float c, int color);

static __global__ void set_bnd_kernel(int b, float* x);

static __global__ void init_div_kernel(float* div, const float* u, const float* v);
static __global__ void project_kernel(float* x, const float* p, int dx, int dy);
static __global__ void advect_kernel(float* d, const float* d0, const float* u, const float* v, float dt0);

#pragma endregion

#pragma region CUDA solver definitions

static float* cuda_allocate(int count)
{
    float* ptr{};
    cuda_check(cudaMalloc(&ptr, sizeof(float) * count), "cuda_allocate");
    cuda_check(cudaMemset(ptr, 0, sizeof(float) * count), "cuda_memset");
    return ptr;
}

cuda_fluid_solver::cuda_fluid_solver(const fluid_solver_config& cfg,
                                     original_fluid_solver* reference) : fluid_solver(cfg), reference_(reference)
{
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp device_prop;
    cuda_check(cudaGetDeviceProperties(&device_prop, 0), "cudaGetDeviceProperties");
    printf("Using CUDA device %s\n", device_prop.name);

    const auto count = get_pixel_count();
    x_ = cuda_allocate(count);
    x0_ = cuda_allocate(count);
    u_ = cuda_allocate(count);
    u0_ = cuda_allocate(count);
    v_ = cuda_allocate(count);
    v0_ = cuda_allocate(count);

    const auto n = config_.n;
    add_source_block_size_ = dim3(256, 1, 1);
    add_source_grid_size_ = dim3(ceil_div((n + 2) * (n + 2), add_source_block_size_.x), 1, 1);

    view_block_size_ = dim3(6, 16, 1);
    view_grid_size_ = ceil_div(dim3(n, n, 1), view_block_size_);

    setbnd_block_size_ = dim3(64, 4, 1);
    setbnd_grid_size_ = dim3(ceil_div(n , setbnd_block_size_.x), 1, 1);

    update_texture_block_size_ = dim3(16,16, 1);
    update_texture_grid_size_ = ceil_div(dim3(n+2, n+2, 1), update_texture_block_size_);

    cuda_check(cudaMemcpyToSymbol(N, &config_.n, sizeof(int)), "initialize::cudaMemcpyToSymbol(N)");
    for (int i = 0; i < 3; ++i)
    {
        cuda_check(cudaStreamCreate(&streams_[i]), "initialize::cudaStreamCreate");
        cuda_check(cudaEventCreateWithFlags(&uv_events_[i], cudaEventDisableTiming), "cudaEventCreate");
    }

    if (reference)
        solver_state_ = new float[get_pixel_count()];

    host_x = new float[get_pixel_count()];
    host_x0 = new float[get_pixel_count()];
}

cuda_fluid_solver::~cuda_fluid_solver()
{
    cudaFree(x_);
    cudaFree(x0_);
    cudaFree(u_);
    cudaFree(u0_);
    cudaFree(v_);
    cudaFree(v0_);

    for (int i = 0; i < 3; ++i)
    {
        cuda_check(cudaStreamDestroy(streams_[i]), "~cuda_fluid_solver::cudaStreamDestroy");
        cuda_check(cudaEventDestroy(uv_events_[i]), "~cuda_fluid_solver::cudaEventDestroy");
    }

    if (reference_)
        delete []solver_state_;

    fluid_solver::~fluid_solver();
}

void cuda_fluid_solver::initialize(GLuint den_texture, GLuint u_texture, GLuint v_texture)
{
    fluid_solver::initialize(den_texture, u_texture, v_texture);
    constexpr auto store_flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
    cuda_check(cudaGraphicsGLRegisterImage(&cuda_dens_texture_, den_texture, GL_TEXTURE_2D,
                                           store_flags), "cudaGraphicsGLRegisterImage(den_texture)");
    cuda_check(cudaGraphicsGLRegisterImage(&cuda_u_texture_, u_texture, GL_TEXTURE_2D, store_flags),
               "cudaGraphicsGLRegisterImage(u_texture)");
    cuda_check(cudaGraphicsGLRegisterImage(&cuda_v_texture_, v_texture, GL_TEXTURE_2D, store_flags),
               "cudaGraphicsGLRegisterImage(v_texture)");
}

void cuda_fluid_solver::clear() const
{
    clear_sources();
    const auto size = get_pixel_count() * sizeof(float);
    cuda_check(cudaMemset(x_, 0, size), "cudaMemset(x0)");
    cuda_check(cudaMemset(u_, 0, size), "cudaMemset(u0)");
    cuda_check(cudaMemset(v_, 0, size), "cudaMemset(v0)");

    if (reference_)
        reference_->clear();
}

void cuda_fluid_solver::add_density(const int2 grid_pos, const float density)
{
    add_input_kernel<<<1 , 1, 0, streams_[0]>>>(x0_, idx(grid_pos.x, grid_pos.y), density);
    if (reference_)
    {
        reference_->add_density(grid_pos, density);
        stopped_ |= compare_state("input density", "x0", reference_->x0(), x0_, streams_[0]);
    }
    hasInput_ = true;
}

void cuda_fluid_solver::add_velocity(const int2 grid_pos, const float2 velocity)
{
    const auto i = idx(grid_pos);
    add_input_kernel<<<1 , 1, 0, streams_[1]>>>(u0_, i, velocity.x);
    add_input_kernel<<<1 , 1, 0, streams_[2]>>>(v0_, i, velocity.y);
    if (reference_)
    {
        reference_->add_velocity(grid_pos, velocity);
        stopped_ |= compare_state("input velocity", "u0", reference_->u0(), u0_, streams_[1]);
        stopped_ |= compare_state("input velocity", "v0", reference_->v0(), v0_, streams_[1]);
    }
    hasInput_ = true;
}

void cuda_fluid_solver::solve(const render_state& render_state)
{
    if (stopped_)
        return;

    // wait add_input
    cuda_check(cudaStreamSynchronize(streams_[0]), "solve::cudaStreamSynchronize(0)");
    cuda_check(cudaStreamSynchronize(streams_[1]), "solve::cudaStreamSynchronize(1)");
    cuda_check(cudaStreamSynchronize(streams_[2]), "solve::cudaStreamSynchronize(2)");

    if (reference_)
    {
        reference_->add_source(reference_->x(), reference_->x0());
        reference_->add_source(reference_->u(), reference_->u0());
        reference_->add_source(reference_->v(), reference_->v0());

        add_source(x_, x0_, streams_[0]);
        add_source(u_, u0_, streams_[1]);
        add_source(v_, v0_, streams_[2]);

        stopped_ |= compare_state("add_source");

        reference_->diffuse(0, reference_->x0(), reference_->x(), config_.diff);
        reference_->diffuse(1, reference_->u0(), reference_->u(), config_.visc);
        reference_->diffuse(2, reference_->v0(), reference_->v(), config_.visc);

        diffuse(0, x0_, x_, config_.diff);
        diffuse(1, u0_, u_, config_.visc);
        diffuse(2, v0_, v_, config_.visc);

        stopped_ |= compare_state("diffuse");

        reference_->project(reference_->u0(), reference_->v0(), reference_->u(), reference_->v());
        project(u0_, v0_, u_, v_);
        stopped_ |= compare_state("project");


        advect(1, u_, u0_, u0_, v0_);
        advect(2, v_, v0_, u0_, v0_);
        reference_->advect(1, reference_->u(), reference_->u0(), reference_->u0(), reference_->v0());
        reference_->advect(2, reference_->v(), reference_->v0(), reference_->u0(), reference_->v0());

        stopped_ |= compare_state("advect");

        if (hasInput_ && !stopped_)
        {
            printf("frame %d had input and no errors\n", render_state.frame);
        }
    }
    else
    {
        add_source(u_, u0_, streams_[1]);
        diffuse(1, u0_, u_, config_.visc);

        add_source(v_, v0_, streams_[2]);
        diffuse(2, v0_, v_, config_.visc);

        add_source(x_, x0_, streams_[0]);
        
        diffuse(0, x0_, x_, config_.diff);

        project(u0_, v0_, u_, v_);
        
        advect(1, u_, u0_, u0_, v0_);
        advect(2, v_, v0_, u0_, v0_);
        
        project(u_, v_, u0_, v0_);
        
        advect(0, x_, x0_, u_, v_);

        cuda_check(cudaStreamSynchronize(streams_[0]), "cudaStreamSynchronize(streams_[0])");
    }

    hasInput_ = false;

    clear_sources();
    if (reference_)
    {
        reference_->clear_sources();
    }
}

void cuda_fluid_solver::clear_sources() const
{
    const auto size = get_pixel_count() * sizeof(float);
    cuda_check(cudaMemset(x0_, 0, size), "cudaMemset(x0)");
    cuda_check(cudaMemset(u0_, 0, size), "cudaMemset(u0)");
    cuda_check(cudaMemset(v0_, 0, size), "cudaMemset(v0)");
}

bool cuda_fluid_solver::compare_state(const char* label) const
{
    bool error = false;
    error |= compare_state(label, "x", reference_->x(), x_, streams_[0]);
    error |= compare_state(label, "x0", reference_->x0(), x0_, streams_[0]);
    error |= compare_state(label, "u", reference_->u(), u_, streams_[1]);
    error |= compare_state(label, "u0", reference_->u0(), u0_, streams_[1]);
    error |= compare_state(label, "v", reference_->v(), v_, streams_[2]);
    error |= compare_state(label, "v0", reference_->v0(), v0_, streams_[2]);
    return error;
}

bool cuda_fluid_solver::compare_state(const char* step, const char* component, const float* expected,
                                      const float* actual,
                                      const cudaStream_t stream) const
{
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto count = get_pixel_count();
    cuda_check(cudaMemcpy(solver_state_, actual, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const auto gs = config_.n + 2;
    for (int i = 0; i < count; ++i)
    {
        const auto error = std::abs(expected[i] - solver_state_[i]);
        if (error > 0.0001f)
        {
            const auto x = i % gs;
            const auto y = i / gs;
            printf("error in %s(%s) at index %d ([%d,%d]]), expected %f, actual %f error %f\n", step, component, i, x,
                   y,
                   expected[i], solver_state_[i], error);
            return true;
        }
    }
    return false;
}

void cuda_fluid_solver::add_source(float* x, const float* s, cudaStream_t stream) const
{
    add_source_kernel<<< add_source_grid_size_, add_source_block_size_, 0, stream >>>(x, s, config_.dt);
}

void cuda_fluid_solver::set_bnd(const int b, float* x, cudaStream_t stream) const
{
    set_bnd_kernel<<< setbnd_grid_size_, setbnd_block_size_, 0, stream>>>(b, x);
}

void cuda_fluid_solver::diffuse(const int b, float* x, const float* x0, const float diff) const
{
    const auto a = config_.dt * diff * static_cast<float>(config_.n * config_.n);
    lin_solve(b, x, x0, a, 1 + 4 * a, streams_[b]);
}

void cuda_fluid_solver::lin_solve(const int b, float* x, const float* x0, const float a, const float c,
                                  cudaStream_t stream) const
{
    for (int k = 0; k < config_.k * 2; ++k)
    {
        lin_solve_kernel<<< view_grid_size_, view_block_size_, 0, stream >>>(x, x0, a, c, 0);
        lin_solve_kernel<<< view_grid_size_, view_block_size_, 0, stream >>>(x, x0, a, c, 1);
        set_bnd(b, x, stream);
    }
}

void cuda_fluid_solver::project(float* u, float* v, float* p, float* div) const
{
    cudaMemsetAsync(p, 0, 0, streams_[2]);
    send_event(2);
    // init_p_kernel<<< view_grid_size_, view_block_size_, 0, streams_[1] >>>(p);
    // set_bnd(0, p, streams_[1]);

    init_div_kernel<<< view_grid_size_, view_block_size_, 0, streams_[1] >>>(div, u, v);
    set_bnd(0, div, streams_[1]);

    wait_event(1, 2);
    lin_solve(0, p, div, 1, 4, streams_[1]);
    send_event(1);

    project_kernel<<< view_grid_size_, view_block_size_, 0 , streams_[1]>>>(u, p, 1, 0);
    send_event(1);

    wait_event(2, 1); // wait lin_solve on stream 1
    project_kernel<<< view_grid_size_, view_block_size_, 0 , streams_[2]>>>(v, p, 0, 1);
    send_event(2);

    wait_event(1, 2); // wait project_kernel on stream 2 (v)
    set_bnd(1, u, streams_[1]);

    wait_event(2, 1); // wait project_kernel on stream 1 (u)
    set_bnd(2, v, streams_[2]);

    send_event(1);
    send_event(2);
}

void cuda_fluid_solver::advect(const int b, float* d, const float* d0, const float* u, const float* v) const
{
    if (b == 0)
    {
        wait_event(0, 1);
        wait_event(0, 2);
    }
    else if (b == 1)
        wait_event(1, 2);
    else
        wait_event(2, 1);

    const auto dt0 = config_.dt * static_cast<float>(config_.n);
    advect_kernel<<< view_grid_size_, view_block_size_, 0 , streams_[b] >>>(d, d0, u, v, dt0);

    set_bnd(b, d, streams_[b]);
    send_event(b);
}

void cuda_fluid_solver::test_advect(float* d, const float* d0, const float* u, const float* v) const
{
    const auto dt0 = config_.dt * static_cast<float>(config_.n);
    advect_kernel<<< view_grid_size_, view_block_size_, 0 >>>(d, d0, u, v, dt0);
}

void cuda_fluid_solver::update_density_texture(const GLuint texture)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    update_cuda_texture(cuda_dens_texture_, x_);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void cuda_fluid_solver::update_velocity_textures(const GLuint u_texture, const GLuint v_texture)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    update_cuda_texture(cuda_u_texture_, u_);
    glBindTexture(GL_TEXTURE_2D, u_texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    update_cuda_texture(cuda_v_texture_, v_);
    glBindTexture(GL_TEXTURE_2D, v_texture);
}

void cuda_fluid_solver::update_cuda_texture(cudaGraphicsResource_t cuda_texture, const float* src) const
{
    cuda_check(cudaGraphicsMapResources(1, &cuda_texture), "cudaGraphicsMapResources");
    cudaArray_t array;
    cuda_check(cudaGraphicsSubResourceGetMappedArray(&array, cuda_texture, 0, 0),
               "cudaGraphicsSubResourceGetMappedArray");
    cudaResourceDesc desc{.resType = cudaResourceTypeArray, .res = {{array}}, .flags = 0};
    cudaSurfaceObject_t surface;
    cuda_check(cudaCreateSurfaceObject(&surface, &desc), "cudaCreateSurfaceObject");
    update_texture_kernel<<< update_texture_grid_size_, update_texture_block_size_ >>>(surface, src);
    cuda_check(cudaGetLastError(), "launch update_texture_kernel");
    cuda_check(cudaDeviceSynchronize(), "update_texture_kernel");
    cuda_check(cudaDestroySurfaceObject(surface), "cudaDestroySurfaceObject");
    cuda_check(cudaGraphicsUnmapResources(1, &cuda_texture), "cudaGraphicsUnmapResources");
}

void cuda_fluid_solver::wait_event(const int stream, const int waitForStream) const
{
    cuda_check(cudaStreamWaitEvent(streams_[stream], uv_events_[waitForStream]), "project::cudaStreamWaitEvent()");
}

void cuda_fluid_solver::send_event(const int stream) const
{
    cuda_check(cudaEventRecord(uv_events_[stream], streams_[stream]), "cudaEventRecord()");
}


#pragma endregion

////////////////////////////////// CUDA Kernels ////////////////////////////////
#pragma region CUDA Kernels

static __device__ int grid_x()
{
    return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
}

static __device__ int grid_y()
{
    return static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
}

// 1 cell kernel
static __global__ void add_input_kernel(float* dst, const int index, const float input)
{
    dst[index] += input;
}

// 1D grid length
static __global__ void add_source_kernel(float* x, const float* x0, const float dt)
{
    const int i = grid_x();
    if (i < (N + 2) * (N + 2))
        x[i] += x0[i] * dt;
}

// 2D ceil(N,2)+1 x N
static __global__ void lin_solve_kernel(float* x, const float* x0, const float a, const float c, const int color)
{
    const auto j = grid_y() + 1;
    const auto i = grid_x() * 2 + 1 + j % 2 - color;
    if (i > 0 && i <= N && j <= N)
    {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)]
            + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

enum side_t // NOLINT(performance-enum-size)
{
    left = 0,
    top = 1,
    right = 2,
    bottom = 3
};

// 2D N x 4 : N threads per side (0 left 1 top 2 right 3 bottom) 
static __global__ void set_bnd_kernel(const int b, float* x)
{
    const auto i = grid_x() + 1;
    if (i > N)
        return;

    const auto side = static_cast<side_t>(grid_y());

    if (side == left)
    {
        float v = x[IX(1, i)];
        x[IX(0, i)] = b == 1 ? -v : v;
        if (i == 1)
        {
            // bottom left corner
            float v2 = b == 2 ? -v : v;
            x[IX(0, 0)] = 0.5f * (v + v2);
        }
    }
    else if (side == right)
    {
        float v = x[IX(N, i)];
        x[IX(N+1, i)] = b == 1 ? -v : v;
        if (i == N)
        {
            // top right corner
            float v2 = b == 2 ? -v : v;
            x[IX(N+1, N+1)] = 0.5f * (v + v2);
        }
    }
    else if (side == bottom)
    {
        float v = x[IX(i, 1)];
        x[IX(i, 0)] = b == 2 ? -v : v;
        if (i == N)
        {
            // bottom right corner
            float v2 = b == 1 ? -v : v;
            x[IX(N+1, 0)] = 0.5f * (v + v2);
        }
    }
    else if (side == top)
    {
        float v = x[IX(i, N)];
        x[IX(i, N+1)] = b == 2 ? -v : v;
        if (i == 1)
        {
            // top left corner
            float v2 = b == 1 ? -v : v;
            x[IX(0, N+1)] = 0.5f * (v + v2);
        }
    }
}

// view kernel
static __global__ void init_div_kernel(float* div, const float* u, const float* v)
{
    const auto i = grid_x() + 1;
    const auto j = grid_y() + 1;
    if (i <= N && j <= N)
    {
        div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / static_cast<
            float>(N);
    }
}


// view kernel
static __global__ void project_kernel(float* x, const float* p, int dx, int dy)
{
    const auto i = grid_x() + 1;
    const auto j = grid_y() + 1;
    if (i <= N && j <= N)
    {
        x[IX(i, j)] -= 0.5f * static_cast<float>(N) * (p[IX(i + dx, j + dy)] - p[IX(i - dx, j - dy)]);
    }
}

template <typename T>
static __device__ T clamp(const T v, const T min, const T max)
{
    return v < min ? min : v > max ? max : v;
}

// view kernel
static __global__ void advect_kernel(float* d, const float* d0, const float* u, const float* v, const float dt0)
{
    const auto i = grid_x() + 1;
    const auto j = grid_y() + 1;
    if (i <= N && j <= N)
    {
        float x = static_cast<float>(i) - dt0 * u[i];
        float y = static_cast<float>(j) - dt0 * v[j];
        x = clamp(x, 0.5f, static_cast<float>(N) + 0.5f);
        y = clamp(y, 0.5f, static_cast<float>(N) + 0.5f);
        const int i0 = static_cast<int>(x), i1 = i0 + 1;
        const int j0 = static_cast<int>(y), j1 = j0 + 1;
        const float s1 = x - static_cast<float>(i0), s0 = 1 - s1;
        const float t1 = y - static_cast<float>(j0), t0 = 1 - t1;
        d[i] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

// full grid kernel
static __global__ void update_texture_kernel(cudaSurfaceObject_t dst, const float* src)
{
    const auto x = grid_x();
    const auto y = grid_y();
    if (x < N + 2 && y < N + 2)
    {
        surf2Dwrite(src[IX(x, y)], dst, x * static_cast<int>(sizeof(float)), y, cudaBoundaryModeClamp);
    }
}

#pragma endregion

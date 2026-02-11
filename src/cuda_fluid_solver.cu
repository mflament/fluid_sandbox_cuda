// ReSharper disable CppClangTidyMiscUseAnonymousNamespace
#include "cuda_fluid_solver.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "cuda_support.h"

#pragma region CUDA Kernels declarations

static __device__ __constant__ int N = 0;

static __global__ void update_texture_kernel(cudaSurfaceObject_t dst, const float* src);

static __global__ void add_input_kernel(float* dst, int index, float input);

static __global__ void add_source_kernel(float* x, const float* x0, float dt);

static __global__ void lin_solve_kernel(float* x, const float* x0, float a, float c, int color);
static __global__ void lin_solve_kernel(float* x, const float* x0, float a, float c);

static __global__ void set_bnd_kernel(int b, float* x);

static __global__ void init_div_kernel(float* div, const float* u, const float* v);
static __global__ void init_p_kernel(float* p);
static __global__ void project_kernel(float* u, float* v, const float* p);
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

    full_block_size_ = {8, 8, 1};
    const auto n = config.n;
    full_grid_size_ = {ceil_div(n + 2, full_block_size_.x), ceil_div(n + 2, full_block_size_.y), 1};

    view_block_size_ = full_block_size_;
    view_grid_size_ = {ceil_div(n, view_block_size_.x), ceil_div(n, view_block_size_.y), 1};

    setbnd_block_size_ = {128, 4};
    setbnd_grid_size_ = {ceil_div(n, setbnd_block_size_.x), 1};

    cuda_check(cudaMemcpyToSymbol(N, &config.n, sizeof(int)), "initialize::cudaMemcpyToSymbol(N)");
    for (int i = 0; i < 3; ++i)
    {
        cuda_check(cudaStreamCreate(&streams_[i]), "initialize::cudaStreamCreate");
        cuda_check(cudaEventCreateWithFlags(&uv_events_[i], cudaEventDisableTiming), "cudaEventCreate");
    }

    if (reference)
        solver_state_ = new float[get_pixel_count()];
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
}

void cuda_fluid_solver::add_density(const int2 grid_pos, const float density)
{
    add_input_kernel<<<1 , 1, 0, streams_[0]>>>(x0_, idx(grid_pos), density);
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

        reference_->diffuse(0, reference_->x0(), reference_->x(), config.diff);
        reference_->diffuse(1, reference_->u0(), reference_->u(), config.visc);
        reference_->diffuse(2, reference_->v0(), reference_->v(), config.visc);

        diffuse(0, x0_, x_, config.diff);
        diffuse(1, u0_, u_, config.visc);
        diffuse(2, v0_, v_, config.visc);

        stopped_ |= compare_state("diffuse");

        // reference_->project(reference_->u0(), reference_->v0(), reference_->u(), reference_->v());
        // project(u0_, v0_, u_, v_);
        // stopped_ |= compare_state("project");

        if (hasInput_ && !stopped_)
        {
            printf("frame %d had input and no errors\n", render_state.frame);
        }
    }
    else
    {
        add_source(u_, u0_, streams_[1]);
        diffuse(1, u0_, u_, config.visc);

        add_source(v_, v0_, streams_[2]);
        diffuse(2, v0_, v_, config.visc);

        add_source(x_, x0_, streams_[0]);
        diffuse(0, x0_, x_, config.diff);

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
    error |= compare_state(label, "dens", reference_->x(), x_, streams_[0]);
    error |= compare_state(label, "dens0", reference_->x0(), x0_, streams_[0]);
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
    const auto gs = config.n + 2;
    for (int i = 0; i < count; ++i)
    {
        if (std::abs(expected[i] - solver_state_[i]) > 0.0001f)
        {
            const auto x = i % gs;
            const auto y = i / gs;
            printf("error in %s(%s) at index %d ([%d,%d]]), expected %f, actual %f\n", step, component, i, x, y,
                   expected[i], solver_state_[i]);
            return true;
        }
    }
    return false;
}

void cuda_fluid_solver::add_source(float* x, const float* s, cudaStream_t stream) const
{
    add_source_kernel<<< full_grid_size_, full_block_size_, 0, stream >>>(x, s, config.dt);
}

void cuda_fluid_solver::set_bnd(const int b, float* x, cudaStream_t stream) const
{
    set_bnd_kernel<<< setbnd_grid_size_, setbnd_block_size_, 0, stream>>>(b, x);
}

void cuda_fluid_solver::lin_solve(const int b, float* x, const float* x0, const float a, const float c,
                                  cudaStream_t stream) const
{
    for (int k = 0; k < config.k * 2; ++k)
    {
        // lin_solve_kernel<<< view_grid_size_, view_block_size_, 0, stream >>>(x, x0, a, c);
        lin_solve_kernel<<< view_grid_size_, view_block_size_, 0, stream >>>(x, x0, a, c, 0);
        lin_solve_kernel<<< view_grid_size_, view_block_size_, 0, stream >>>(x, x0, a, c, 1);
        set_bnd(b, x, stream);
    }
}

void cuda_fluid_solver::diffuse(const int b, float* x, const float* x0, const float diff) const
{
    const auto a = config.dt * diff * static_cast<float>(config.n * config.n);
    lin_solve(b, x, x0, a, 1 + 4 * a, streams_[b]);
}

void cuda_fluid_solver::project(float* u, float* v, float* p, float* div) const
{
    init_p_kernel<<< view_grid_size_, view_block_size_, 0, streams_[1] >>>(p);
    set_bnd(0, p, streams_[1]);
    wait_event(1, 2);

    init_div_kernel<<< view_grid_size_, view_block_size_, 0, streams_[2] >>>(div, u, v);
    set_bnd(0, div, streams_[2]);
    send_event(2);

    lin_solve(0, p, div, 1, 4, streams_[1]);

    project_kernel<<< view_grid_size_, view_block_size_, 0 , streams_[1]>>>(u, v, p);
    send_event(1);

    set_bnd(1, u, streams_[1]);

    wait_event(2, 1); // wait for project_kernel running on stream 1
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

    const auto dt0 = config.dt * static_cast<float>(config.n);
    advect_kernel<<< view_grid_size_, view_block_size_, 0 , streams_[b] >>>(d, d0, u, v, dt0);

    set_bnd(b, d, streams_[b]);
    send_event(b);
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
    update_texture_kernel<<< full_grid_size_, full_block_size_ >>>(surface, src);
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

static __device__ int2 get_grid_pos(const int offset = 0)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    return {x + offset, y + offset};
}

static __device__ int2 get_view_pos()
{
    return get_grid_pos(1);
}

static __device__ bool is_in_grid(const int2 pos)
{
    return pos.x < N + 2 && pos.y < N + 2;
}

static __device__ bool is_in_view(const int2 pos)
{
    return pos.x > 0 && pos.x <= N && pos.y > 0 && pos.y <= N;
}

static __device__ int idx(const int2 pos)
{
    return pos.y * (N + 2) + pos.x;
}

static __device__ int idx(const int x, const int y)
{
    return y * (N + 2) + x;
}

// 1 cell kernel
static __global__ void add_input_kernel(float* dst, const int index, const float input)
{
    dst[index] += input;
}

// full grid size
static __global__ void add_source_kernel(float* x, const float* x0, const float dt)
{
    const auto gp = get_grid_pos();
    if (is_in_grid(gp))
    {
        const auto i = idx(gp);
        x[i] += x0[i] * dt;
    }
}

// view kernel
static __global__ void lin_solve_kernel(float* x, const float* x0, const float a, const float c, const int color)
{
    const auto gp = get_view_pos();
    if (is_in_view(gp) && (gp.x + gp.y) % 2 == color)
    {
        const auto i = idx(gp);
        x[i] = (x0[i] + a * (x[idx(gp.x - 1, gp.y)] + x[idx(gp.x + 1, gp.y)]
            + x[idx(gp.x, gp.y - 1)] + x[idx(gp.x, gp.y + 1)])) / c;
    }
}

static __global__ void lin_solve_kernel(float* x, const float* x0, const float a, const float c)
{
    const auto gp = get_grid_pos(1);
    if (is_in_grid(gp))
    {
        const auto i = idx(gp);
        if ((gp.x + gp.y) % 2 == 0)
        {
            x[i] = (x0[i] + a * (x[idx(gp.x - 1, gp.y)] + x[idx(gp.x + 1, gp.y)]
                + x[idx(gp.x, gp.y - 1)] + x[idx(gp.x, gp.y + 1)])) / c;
        }
        __syncthreads();
        if ((gp.x + gp.y) % 2 != 0)
        {
            x[i] = (x0[i] + a * (x[idx(gp.x - 1, gp.y)] + x[idx(gp.x + 1, gp.y)]
                + x[idx(gp.x, gp.y - 1)] + x[idx(gp.x, gp.y + 1)])) / c;
        }
    }
}


enum side_t // NOLINT(performance-enum-size)
{
    left = 0,
    top = 1,
    right = 2,
    bottom = 3
};


static __device__ int get_bnd_dst_idx(const side_t side, const int n)
{
    switch (side)
    {
    case left: return idx(0, n);
    case top: return idx(n, 0);
    case right: return idx(N + 1, n);
    case bottom: return idx(n, N + 1);
    }
    return -1;
}

static __device__ float get_bnd_src_value(const int b, const float* x, const side_t side, const int n)
{
    switch (side)
    {
    case left: return b == 1 ? -x[idx(1, n)] : x[idx(1, n)];
    case top: return b == 2 ? -x[idx(n, 1)] : x[idx(n, 1)];
    case right: return b == 1 ? -x[idx(N, n)] : x[idx(N, n)];
    case bottom: return b == 2 ? -x[idx(n, N)] : x[idx(n, N)];
    }
    return 0;
}

// grid size = N x 4 : N threads per side (0 left 1 top 2 right 3 bottom) 
static __global__ void set_bnd_kernel(const int b, float* x)
{
    const auto gp = get_grid_pos();
    if (gp.x >= N)
        return;

    const auto side = static_cast<side_t>(gp.y);
    const auto dst_idx = get_bnd_dst_idx(side, 1 + gp.x);
    const float n = get_bnd_src_value(b, x, side, 1 + gp.x);
    x[dst_idx] = n;

    // corners
    if (gp.x == 0 && side == left) // top left, n is (0, 1) , fetch (1, 0)
    {
        x[idx(0, 0)] = 0.5f * (n + get_bnd_src_value(b, x, top, 1));
    }
    else if (gp.x == N - 1 && side == top) // top right, n is (N, 0) , fetch (N+1, 1)
    {
        x[idx(N + 1, 0)] = 0.5f * (n + get_bnd_src_value(b, x, right, 1));
    }
    else if (gp.x == N - 1 && side == right) // bottom right, n is (N+1, N) , fetch (N, N+1)
    {
        x[idx(N + 1, N + 1)] = 0.5f * (n + get_bnd_src_value(b, x, bottom, N));
    }
    else if (gp.x == 0 && side == bottom) // bottom left, n is (1, N+1) , fetch (0, N)
    {
        x[idx(0, N + 1)] = 0.5f * (n + get_bnd_src_value(b, x, left, N));
    }
}

// view kernel
static __global__ void init_div_kernel(float* div, const float* u, const float* v)
{
    const auto vp = get_view_pos();
    if (is_in_view(vp))
    {
        div[idx(vp)] = -0.5f * (u[idx(vp.x + 1, vp.y)] - u[idx(vp.x - 1, vp.y)] +
            v[idx(vp.x, vp.y + 1)] - v[idx(vp.x, vp.y - 1)]) / static_cast<float>(N);
    }
}

// view kernel
static __global__ void init_p_kernel(float* p)
{
    const auto vp = get_view_pos();
    if (is_in_view(vp))
    {
        p[idx(vp)] = 0;
    }
}

// view kernel
static __global__ void project_kernel(float* u, float* v, const float* p)
{
    const auto vp = get_view_pos();
    if (is_in_view(vp))
    {
        const auto i = idx(vp);
        u[i] -= 0.5f * static_cast<float>(N) * (p[idx(vp.x + 1, vp.y)] - p[idx(vp.x - 1, vp.y)]);
        v[i] -= 0.5f * static_cast<float>(N) * (p[idx(vp.x, vp.y + 1)] - p[idx(vp.x, vp.y - 1)]);
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
    const auto vp = get_view_pos();

    if (is_in_view(vp))
    {
        const auto i = idx(vp);
        float x = static_cast<float>(vp.x) - dt0 * u[i];
        float y = static_cast<float>(vp.y) - dt0 * v[i];
        x = clamp(x, 0.5f, static_cast<float>(N) + 0.5f);
        const int i0 = static_cast<int>(x);
        const int i1 = i0 + 1;
        y = clamp(y, 0.5f, static_cast<float>(N) + 0.5f);
        const int j0 = static_cast<int>(y);
        const int j1 = j0 + 1;
        const float s1 = x - static_cast<float>(i0);
        const float s0 = 1 - s1;
        const float t1 = y - static_cast<float>(j0);
        const float t0 = 1 - t1;
        d[i] = s0 * (t0 * d0[idx(i0, j0)] + t1 * d0[idx(i0, j1)]) + s1 * (t0 * d0[idx(i1, j0)] + t1 * d0[idx(i1, j1)]);
    }
}

// full grid kernel
static __global__ void update_texture_kernel(cudaSurfaceObject_t dst, const float* src)
{
    const auto gp = get_grid_pos();
    if (is_in_grid(gp))
    {
        surf2Dwrite(src[idx(gp)], dst, gp.x * static_cast<int>(sizeof(float)), gp.y, cudaBoundaryModeClamp);
    }
}

#pragma endregion

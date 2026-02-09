#pragma once
#include "fluid_solver.h"
#include <cuda_gl_interop.h>

#include "original_fluid_solver.h"


class cuda_fluid_solver : public fluid_solver
{
    float *x_, *x0_;

    float *u_, *u0_;
    float *v_, *v0_;

    cudaGraphicsResource_t cuda_dens_texture_{};
    cudaGraphicsResource_t cuda_u_texture_{};
    cudaGraphicsResource_t cuda_v_texture_{};

    dim3 full_grid_size_{};
    dim3 full_block_size_{};

    dim3 view_grid_size_{};
    dim3 view_block_size_{};

    dim3 setbnd_grid_size_{};
    dim3 setbnd_block_size_{};

    cudaStream_t streams_[3]{};
    cudaEvent_t uv_events_[3]{};

    original_fluid_solver* reference_{};
    bool hasInput_ = false;
    bool stopped_ = false;

    void wait_event(int stream, int waitForStream) const;
    void send_event(int stream) const;

    void update_cuda_texture(cudaGraphicsResource_t cuda_texture, const float* src) const;

    void add_source(float* x, const float* s, cudaStream_t stream) const;

    void set_bnd(int b, float* x, cudaStream_t stream) const;
    void lin_solve(int b, float* x, const float* x0, float a, float c, cudaStream_t stream) const;
    void diffuse(int b, float* x, const float* x0, float diff) const;
    void project(float* u, float* v, float* p, float* div) const;
    void advect(int b, float* d, const float* d0, const float* u, const float* v) const;

    float* solver_state_;

    bool compare_state(const char* label) const;
    bool compare_state(const char* step, const char* component, const float* expected, const float* actual, cudaStream_t stream) const;

    void clear_sources() const;
    
public:
    explicit cuda_fluid_solver(const fluid_solver_config& cfg = {}, original_fluid_solver* reference = nullptr);
    ~cuda_fluid_solver() override;

    void initialize(GLuint den_texture, GLuint u_texture, GLuint v_texture) override;

    void add_density(int2 grid_pos, float density) override;

    void add_velocity(int2 grid_pos, float2 velocity) override;

    void solve(const render_state& render_state) override;

    void update_velocity_textures(GLuint u_texture, GLuint v_texture) override;

    void update_density_texture(GLuint texture) override;
};

#pragma once

#include "fluid_solver.h"

class cpp_fluid_solver final : public fluid_solver
{
    float *x_;
    float *x0_;
    float2 *uv_;
    float2 *uv0_;

    
    template <class T>
    void add_source(T* v, const T* v0, float dt) const;
    
    void set_bnd(float2* v) const;
    void set_bnd(float* v) const;
    
    template <class T>
    void lin_solve(T* v, const T* v0, float a, float c) const;
    
    template <class T>
    void diffuse(T* v, const T* v0, float diff, float dt) const;
    float2 get_dest(int2 pos, const float2* uv, float2 dt0) const;

    template <class T>
    void advect(T* x, const T* x0, const float2* uv) const;

    void project(float2* uv, float2* p_div) const;
    
public:
    explicit cpp_fluid_solver(const fluid_sandbox_config& cfg = {});

    ~cpp_fluid_solver() override;

    void add_density(int2 grid_pos, float density) override;

    void add_velocity(int2 grid_pos, float2 velocity) override;

    void solve() override;

    void update_velocity_texture(GLuint texture) override;

    void update_density_texture(GLuint texture) override;
};

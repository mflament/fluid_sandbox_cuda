#pragma once

#include "fluid_solver.h"

class original_fluid_solver final : public fluid_solver
{
    float *x_, *x0_;

    float *u_, *u0_;
    float *v_, *v0_;

public:
    explicit original_fluid_solver(const fluid_solver_config& cfg = {});

    ~original_fluid_solver() override;

    void clear() const override;
    
    void add_density(int2 grid_pos, float density) override;

    void add_velocity(int2 grid_pos, float2 velocity) override;

    void solve(const render_state& render_state) override;

    void update_velocity_textures(GLuint u_texture, GLuint v_texture) override;

    void update_density_texture(GLuint texture) override;

    [[nodiscard]] float* x() const { return x_; }
    [[nodiscard]] float* x0() const { return x0_; }
    [[nodiscard]] float* u() const { return u_; }
    [[nodiscard]] float* u0() const { return u0_; }
    [[nodiscard]] float* v() const { return v_; }
    [[nodiscard]] float* v0() const { return v0_; }


    void add_source(float* x, const float* s) const;
    void set_bnd(int b, float* x) const;
    void lin_solve(int b, float* x, const float* x0, float a, float c) const;
    void diffuse(int b, float* x, const float* x0, float diff) const;
    void advect(int b, float* d, const float* d0, const float* u, const float* v) const;
    void project(float* u, float* v, float* p, float* div) const;

    void clear_sources() const;
    
};

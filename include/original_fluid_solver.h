#pragma once

#include "fluid_solver.h"

class original_fluid_solver final : public fluid_solver
{
    float *x_, *x0_;
    float *u_, *u0_;
    float *v_, *v0_;
    float2 *uv_;

public:
    explicit original_fluid_solver(const fluid_sandbox_config& cfg = {});

    ~original_fluid_solver() override;

    void add_density(int2 grid_pos, float density) override;

    void add_velocity(int2 grid_pos, float2 velocity) override;

    void solve() override;

    void update_velocity_texture(GLuint texture) override;

    void update_density_texture(GLuint texture) override;
};

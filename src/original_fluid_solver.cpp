#include "original_fluid_solver.h"

#include <algorithm>
#include <stdexcept>

// Real-Time Fluid Dynamics for Games
// Jos Stam https://www.dgp.toronto.edu/public_user/stam/reality/index.html


#define IX(i,j) ((i)+(N+2)*(j))
#define FOR_EACH_CELL for ( i=1 ; i<=N ; i++ ) { for ( j=1 ; j<=N ; j++ ) {
#define END_FOR }}

original_fluid_solver::original_fluid_solver(const fluid_solver_config& cfg) : fluid_solver(cfg)
{
    const auto count = static_cast<size_t>(get_pixel_count());
    x_ = new float[count]{};
    x0_ = new float[count]{};
    u_ = new float[count]{};
    u0_ = new float[count]{};
    v_ = new float[count]{};
    v0_ = new float[count]{};
}

original_fluid_solver::~original_fluid_solver()
{
    delete x_;
    delete x0_;
    delete u_;
    delete u0_;
    delete v_;
    delete v0_;
}

void original_fluid_solver::clear() const
{
    clear_sources();
}

void original_fluid_solver::add_density(int2 grid_pos, float density)
{
    x0_[idx(grid_pos)] += density;
}

void original_fluid_solver::add_velocity(int2 grid_pos, float2 velocity)
{
    int i = idx(grid_pos);
    u0_[i] += velocity.x;
    v0_[i] += velocity.y;
}

void original_fluid_solver::solve(const render_state& render_state)
{
    add_source(u_, u0_);
    add_source(v_, v0_);
    diffuse(1, u0_, u_, config.visc);
    diffuse(2, v0_, v_, config.visc);
    project(u0_, v0_, u_, v_);
    advect(1, u_, u0_, u0_, v0_);
    advect(2, v_, v0_, u0_, v0_);
    project(u_,v_, u0_, v0_);

    add_source(x_, x0_);
    diffuse(0, x0_, x_, config.diff);
    advect(0, x_, x0_, u_, v_);
    
    clear_sources();
}

void original_fluid_solver::clear_sources() const
{
    const auto size = get_pixel_count() * sizeof(float);
    memset(x0_, 0, size);
    memset(u0_, 0, size);
    memset(v0_, 0, size);
}

void original_fluid_solver::update_velocity_textures(const GLuint u_texture, const GLuint v_texture)
{
    const auto n = config.n + 2;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, u_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, u_);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, v_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, v_);
}

void original_fluid_solver::update_density_texture(GLuint texture)
{
    const auto n = config.n + 2;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, x_);
}

void original_fluid_solver::add_source(float* x, const float* s) const
{
    const int size = get_pixel_count();
    const auto dt = config.dt;
    for (int i = 0; i < size; i++) x[i] += dt * s[i];
}

void original_fluid_solver::set_bnd(int b, float* x) const
{
    const auto N = config.n;
    for (int i = 1; i <= N; i++)
    {
        x[idx(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N+1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N+1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N+1)] = 0.5f * (x[IX(1, N+1)] + x[IX(0, N)]);
    x[IX(N+1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N+1, 1)]);
    x[IX(N+1, N+1)] = 0.5f * (x[IX(N, N+1)] + x[IX(N+1, N)]);
}

void original_fluid_solver::lin_solve(int b, float* x, const float* x0, float a, float c) const
{
    const auto N = config.n;
    int i, j;
    for (int k = 0; k < config.k; k++)
    {
        FOR_EACH_CELL
            x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i-1, j)] + x[IX(i+1, j)] + x[IX(i, j-1)] + x[IX(i, j+1)])) / c;
        END_FOR
        set_bnd(b, x);
    }
}

void original_fluid_solver::diffuse(int b, float* x, const float* x0, float diff) const
{
    float a = config.dt * diff * static_cast<float>(config.n * config.n);
    lin_solve(b, x, x0, a, 1 + 4 * a);
}

void original_fluid_solver::project(float* u, float* v, float* p, float* div) const
{
    int i, j;
    const auto N = config.n;
    FOR_EACH_CELL
        div[IX(i, j)] = -0.5f * (u[IX(i+1, j)] - u[IX(i-1, j)] + v[IX(i, j+1)] - v[IX(i, j-1)]) / static_cast<float>
            (N);
        p[IX(i, j)] = 0;
    END_FOR
    set_bnd(0, div);
    set_bnd(0, p);

    lin_solve(0, p, div, 1, 4);

    FOR_EACH_CELL
        u[IX(i, j)] -= 0.5f * static_cast<float>(N) * (p[IX(i+1, j)] - p[IX(i-1, j)]);
        v[IX(i, j)] -= 0.5f * static_cast<float>(N) * (p[IX(i, j+1)] - p[IX(i, j-1)]);
    END_FOR
    set_bnd(1, u);
    set_bnd(2, v);
}

void original_fluid_solver::advect(int b, float* d, const float* d0, const float* u, const float* v) const
{
    int i, j;
    const auto N = config.n;
    float dt0 = config.dt * static_cast<float>(N);
    FOR_EACH_CELL
        float x = static_cast<float>(i) - dt0 * u[IX(i, j)];
        float y = static_cast<float>(j) - dt0 * v[IX(i, j)];
        x = std::max(x, 0.5f);
        x = std::min(x, static_cast<float>(N) + 0.5f);
        const int i0 = static_cast<int>(x);
        const int i1 = i0 + 1;
        y = std::max(0.5f, y);
        y = std::min(static_cast<float>(N) + 0.5f, y);
        const int j0 = static_cast<int>(y);
        const int j1 = j0 + 1;
        const float s1 = x - static_cast<float>(i0);
        const float s0 = 1 - s1;
        const float t1 = y - static_cast<float>(j0);
        const float t0 = 1 - t1;
        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    END_FOR
    set_bnd(b, d);
}


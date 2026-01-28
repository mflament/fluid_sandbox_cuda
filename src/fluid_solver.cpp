#include "fluid_solver.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "vector_operators.h"

namespace
{
    float2 flip_y(const float2 v) { return {v.x, -v.y}; }
    float2 flip_x(const float2 v) { return {-v.x, v.y}; }

    void print_uv(const char* prefix, const float2* uv, const int2 grid_size)
    {
        printf("%s :\n", prefix);
        for (int j = grid_size.y-1; j >= 0; --j)
        {
            for (int i = 0; i < grid_size.x; ++i)
            {
                printf("(%.3f,%.3f)", uv[i].x, uv[i].y);
                if (i < grid_size.x - 1) printf("  ");    
            }
            printf("\n");
        }
        printf("\n");
    }
    
}

void fluid_solver::set_renderer(const fluid_sandbox_renderer* renderer)
{
    renderer_ = renderer;
}

int2 fluid_solver::get_grid_size() const
{
    return {config.n.x + 2, config.n.y + 2};
}

int fluid_solver::get_pixel_count() const
{
    return (config.n.x + 2) * (config.n.y + 2);
}

cpp_fluid_solver::cpp_fluid_solver(const fluid_sandbox_config& cfg) : fluid_solver(cfg)
{
    const auto count = static_cast<size_t>(get_pixel_count());
    x_ = new float[count]{};
    x0_ = new float[count]{};

    uv_ = new float2[count]{};
    uv0_ = new float2[count]{};
}

cpp_fluid_solver::~cpp_fluid_solver()
{
    delete[] x_;
    delete[] x0_;
    delete[] uv_;
    delete[] uv0_;
}

int cpp_fluid_solver::idx(const int2 grid_pos) const
{
    return idx(grid_pos.x, grid_pos.y);
}

int cpp_fluid_solver::idx(const int x, const int y) const
{
    return x + (config.n.x + 2) * y;
}

void cpp_fluid_solver::add_density(const int2 grid_pos, const float density)
{
    x0_[idx(grid_pos)] += density;
}

void cpp_fluid_solver::add_velocity(const int2 grid_pos, const float2 velocity)
{
    const int i = idx(grid_pos);
    uv0_[i] += velocity;
    // if (!is_zero(uv0_[i]))
    //     printf("[%d,%d] = (%.3f, %.3f)\n", grid_pos.x, grid_pos.y, uv0_[i].x, uv0_[i].y);
}


void cpp_fluid_solver::solve()
{
    add_source(uv_, uv0_, config.dt);
    print_uv("add_source", uv_, get_grid_size());

    diffuse(uv0_, uv_, config.visc, config.dt);
    print_uv("diffuse", uv0_, get_grid_size());

    project(uv0_, uv_);
    print_uv("project", uv0_, get_grid_size());

    advect(uv_, uv0_, uv0_);
    print_uv("advect", uv_, get_grid_size());

    project(uv_, uv0_);
    print_uv("project2", uv_, get_grid_size());

    printf("\n");

    add_source(x_, x0_, config.dt);
    diffuse(x0_, x_, config.diff, config.dt);
    advect(x_, x0_, uv_);

    const auto pixel_count = get_pixel_count();
    memset(x0_, 0, pixel_count * sizeof(float));
    memset(uv0_, 0, pixel_count * sizeof(float2));
}

void cpp_fluid_solver::update_density_texture(GLuint texture)
{
    auto s = get_grid_size();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s.x, s.y, GL_RED, GL_FLOAT, x_);
}

void cpp_fluid_solver::update_velocity_texture(GLuint texture)
{
    const auto s = get_grid_size();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s.x, s.y, GL_RG, GL_FLOAT, uv_);
}

template <typename T>
void cpp_fluid_solver::add_source(T* v, const T* v0, const float dt) const
{
    const auto count = get_pixel_count();
    for (int i = 0; i < count; ++i)
    {
        v[i] = v[i] + v0[i] * dt;
    }
}

void cpp_fluid_solver::set_bnd(float2* v) const
{
    const auto n = config.n;
    for (int i = 1; i <= n.x; ++i)
    {
        v[idx(i, 0)] = flip_y(v[idx(i, 1)]);
        v[idx(i, n.y + 1)] = flip_y(v[idx(i, n.y)]);
    }

    for (int i = 1; i <= n.y; ++i)
    {
        v[idx(0, i)] = flip_x(v[idx(1, i)]);
        v[idx(n.x + 1, i)] = flip_x(v[idx(n.x, i)]);
    }

    v[idx(0, 0)] = float2{0, 0};
    v[idx(0, n.y + 1)] = float2{0, 0};
    v[idx(n.x + 1, 0)] = float2{0, 0};
    v[idx(n.x + 1, n.y + 1)] = float2{0, 0};
}

void cpp_fluid_solver::set_bnd(float* v) const
{
    const auto n = config.n;
    for (int i = 1; i <= n.x; ++i)
    {
        v[idx(i, 0)] = v[idx(i, 1)];
        v[idx(i, n.y + 1)] = v[idx(i, n.y)];
    }

    for (int i = 1; i <= n.y; ++i)
    {
        v[idx(0, i)] = v[idx(1, i)];
        v[idx(n.x + 1, i)] = v[idx(n.x, i)];
    }

    v[idx(0, 0)] = 0.5f * (v[idx(1, 0)] + v[idx(0, 1)]);
    v[idx(0, n.y + 1)] = 0.5f * (v[idx(1, n.y + 1)] + v[idx(0, n.y)]);
    v[idx(n.x + 1, 0)] = 0.5f * (v[idx(n.x, 0)] + v[idx(n.x + 1, 1)]);
    v[idx(n.x + 1, n.y + 1)] = 0.5f * (v[idx(n.x, n.y + 1)] + v[idx(n.x + 1, n.y)]);
}

template <typename T>
void cpp_fluid_solver::lin_solve(T* v, const T* v0, float a, float c) const
{
    const auto n = config.n;
    for (int k = 0; k < 20; k++)
    {
        for (int y = 0; y < n.y; y++)
        {
            for (int x = 0; x < n.x; ++x)
            {
                v[idx(x, y)] = (v0[idx(x, y)] + a * (v[idx(x - 1, y)] + v[idx(x + 1, y)]
                    + v[idx(x, y - 1)] + v[idx(x, y + 1)])) / c;
            }
        }
        set_bnd(v);
    }
}

template <typename T>
void cpp_fluid_solver::diffuse(T* v, const T* v0, const float diff, const float dt) const
{
    const auto n = config.n;
    float a = dt * diff * static_cast<float>(n.x * n.y);
    lin_solve(v, v0, a, 1 + 4 * a);
}

template <typename T>
void cpp_fluid_solver::advect(T* x, const T* x0, const float2* uv) const
{
    const auto n = config.n;
    const auto dt0 = to_float2(n) * config.dt;
    for (int2 pos{}; pos.y < n.y; ++pos.y)
    {
        for (; pos.x < n.x; ++pos.x)
        {
            const auto dst = get_dest(pos, uv, dt0);
            const auto ij0 = to_int2(dst);
            const auto ij1 = int2{pos.x + 1, pos.y + 1};
            const auto st1 = dst - to_float2(ij0);
            const auto st0 = float2{1, 1} - st1;
            x[idx(pos.x, pos.y)] = st0.x * (st0.y * x0[idx(ij0.x, ij0.y)] + st1.y * x0[idx(ij0.x, ij1.y)]) +
                st1.x * (st0.y * x0[idx(ij1.x, ij0.y)] + st1.y * x0[idx(ij1.x, ij1.y)]);
        }
    }

    set_bnd(x);
}

float2 cpp_fluid_solver::get_dest(const int2 pos, const float2* uv, const float2 dt0) const
{
    const auto n = to_float2(config.n);
    const float x = static_cast<float>(pos.x) - dt0.x * uv[idx(pos.x, pos.y)].x;
    const float y = static_cast<float>(pos.y) - dt0.y * uv[idx(pos.x, pos.y)].y;
    return {std::clamp(x, 0.5f, n.x + 0.5f), std::clamp(y, 0.5f, n.y + 0.5f)};
}

void cpp_fluid_solver::project(float2* uv, float2* p_div) const
{
    const auto n = config.n;
    const auto p = reinterpret_cast<float*>(p_div);
    const auto div = p + get_pixel_count() / 2;

    for (int2 pos{}; pos.y < n.y; ++pos.y)
    {
        for (; pos.x < n.x; ++pos.x)
        {
            // @formatter:off
            div[idx(pos)] = -0.5f * (uv[idx(pos.x + 1, pos.y)].x - uv[idx(pos.x - 1, pos.y)].x) / static_cast<float>(n.x)
                          + -0.5f * (uv[idx(pos.x, pos.y + 1)].y - uv[idx(pos.x, pos.y - 1)].y) / static_cast<float>(n.y);
            // @formatter:on
            p[idx(pos)] = 0;
        }
    }
    set_bnd(div);
    set_bnd(p);

    lin_solve(p, div, 1, 4);
    for (int2 pos{}; pos.y < n.y; ++pos.y)
    {
        for (; pos.x < n.x; ++pos.x)
        {
            uv[idx(pos)] = 0.5f * to_float2(n) * float2{
                p[idx(pos.x + 1, pos.y)] - p[idx(pos.x - 1, pos.y)],
                p[idx(pos.x, pos.y + 1)] - p[idx(pos.x, pos.y - 1)]
            };
        }
    }
    set_bnd(uv);
}

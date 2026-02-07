#pragma once
#include <vector_types.h>

#include "glad/glad.h"

struct fluid_sandbox_config final
{
    int2 n{128, 128};
    int k = 20;
    float force = 5;
    float source = 100;
    float diff = 0;
    float visc = 0;
    float dt = 0.1f;
};

class fluid_renderer;

class fluid_solver
{
protected:
    const fluid_renderer* renderer_{};

public:
    fluid_sandbox_config config{};

    explicit fluid_solver(const fluid_sandbox_config& cfg) : config(cfg)
    {
    }

    virtual void set_renderer(const fluid_renderer* renderer);

    virtual ~fluid_solver() = default;

    virtual void add_density(int2 grid_pos, float density) = 0;

    virtual void add_velocity(int2 grid_pos, float2 velocity) = 0;

    virtual void solve() = 0;

    virtual void update_velocity_texture(GLuint texture) = 0;

    virtual void update_density_texture(GLuint texture) = 0;

    [[nodiscard]] int2 get_grid_size() const;
    
    [[nodiscard]] int get_pixel_count() const;
    
    [[nodiscard]] int idx(const int x, const int y) const {
        return x + (config.n.x + 2) * y;
    }

    [[nodiscard]] int idx(const int2 grid_pos) const
    {
        return idx(grid_pos.x, grid_pos.y);
    }

};
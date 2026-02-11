#pragma once
#include <vector_types.h>

#include "glad/glad.h"
#include "renderer.h"

struct fluid_solver_config final
{
    int n = 128;
    int k = 20;
    float force = 5;
    float source = 100;
    float diff = 0;
    float visc = 0;
    float dt = 0.1f;

    static fluid_solver_config load();
};

class fluid_renderer;

class fluid_solver
{
public:
    fluid_solver_config config{};

    explicit fluid_solver(const fluid_solver_config& cfg) : config(cfg)
    {
    }

    virtual void initialize(GLuint den_texture, GLuint u_texture, GLuint v_texture);
    
    virtual ~fluid_solver() = default;

    virtual void add_density(int2 grid_pos, float density) = 0;

    virtual void add_velocity(int2 grid_pos, float2 velocity) = 0;

    virtual void solve(const render_state& render_state) = 0;

    virtual void update_velocity_textures(GLuint uTexture, GLuint vTexture);
    
    virtual void update_density_texture(GLuint texture) = 0;

    [[nodiscard]] int get_pixel_count() const;
    
    [[nodiscard]] int idx(const int x, const int y) const {
        return x + (config.n + 2) * y;
    }

    [[nodiscard]] int idx(const int2 grid_pos) const
    {
        return idx(grid_pos.x, grid_pos.y);
    }

    virtual void clear() const = 0;
    
};
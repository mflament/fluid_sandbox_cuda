#include "fluid_solver.h"

void fluid_solver::initialize(GLuint den_texture, GLuint u_texture, GLuint v_texture)
{
}

void fluid_solver::update_velocity_textures(GLuint uTexture, GLuint vTexture)
{
}

int fluid_solver::get_pixel_count() const
{
    const int g = config.n + 2;
    return g * g;
}

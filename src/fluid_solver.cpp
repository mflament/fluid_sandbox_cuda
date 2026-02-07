#include "fluid_solver.h"

#include "vector_operators.h"

void fluid_solver::set_renderer(const fluid_renderer* renderer)
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

#include <exception>
#include <iostream>
#include <ostream>

#include "original_fluid_solver.h"
#include "cuda_fluid_solver.h"
#include "fluid_renderer.h"
#include "render_loop.h"

int main()
{
    const fluid_solver_config cfg = fluid_solver_config::load();
    const auto original_solver = new original_fluid_solver(cfg);
    // const auto solver = new cuda_fluid_solver(cfg, original_solver);
    const auto solver = new cuda_fluid_solver(cfg);
    const auto fsr = new fluid_renderer(original_solver);
    try
    {
        render_loop::start(fsr);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw;
    }
    delete fsr;
    delete original_solver;
    delete solver;
    return 0;
}

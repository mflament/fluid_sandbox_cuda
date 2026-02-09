#include <exception>
#include <iostream>
#include <ostream>

#include "original_fluid_solver.h"
#include "cuda_fluid_solver.h"
#include "fluid_renderer.h"
#include "render_loop.h"

int main()
{
    constexpr fluid_solver_config cfg{.n = 128};
    const auto original_solver = new original_fluid_solver(cfg);
    const auto solver = new cuda_fluid_solver(cfg, original_solver);
    const auto fsr = new fluid_renderer(solver);
    try
    {
        render_loop::start(fsr);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw e;
    }
    delete fsr;
    delete original_solver;
    delete solver;
    return 0;
}

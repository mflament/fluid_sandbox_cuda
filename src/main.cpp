#include <exception>
#include <iostream>
#include <ostream>

#include "fluid_renderer.h"
#include "render_loop.h"

int main()
{
    const auto solver = new cpp_fluid_solver({.n = {1, 1}});
    const auto fsr = new fluid_sandbox_renderer(solver);
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
    delete solver;
    return 0;
}

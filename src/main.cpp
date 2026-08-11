#include <exception>
#include <iostream>
#include <ostream>
#include <random>

#include "original_fluid_solver.h"
#include "cuda_fluid_solver.h"
#include "cuda_support.h"
#include "fluid_renderer.h"
#include "render_loop.h"

namespace
{
    std::random_device rnd_dev;
    std::mt19937 rng(rnd_dev());
    std::uniform_real_distribution rngDist(-1.0f, 1.0f);

    void randomize(const int length, float* host, float* device)
    {
        for (int i = 0; i < length; ++i)
        {
            const auto v = rngDist(rng);
            host[i] = v;
        }
        cuda_check(cudaMemcpy(device, host, length * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    }

    int IX(int i, int j, int N)
    {
        return j * (N + 2) + i;
    }

    void advect(int N, float dt0, float* d, const float* d0, const float* u, const float* v)
    {
        for (int j = 1; j <= N; ++j)
        {
            for (int i = 1; i <= N; ++i)
            {
                float x = static_cast<float>(i) - dt0 * u[IX(i, j, N)];
                float y = static_cast<float>(j) - dt0 * v[IX(i, j, N)];
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
                // @formatter:off
                d[IX(i, j, N)] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                                 s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
                // @formatter:on
            }
        }
    }

    void test_advect(const fluid_solver_config& cfg, const original_fluid_solver* original_solver, const cuda_fluid_solver* solver)
    {
        const auto length = (cfg.n + 2) * (cfg.n + 2);
        randomize(length, original_solver->x(), solver->x());
        randomize(length, original_solver->x0(), solver->x0());
        randomize(length, original_solver->u(), solver->u());
        randomize(length, original_solver->u0(), solver->u0());
        randomize(length, original_solver->v(), solver->v());
        randomize(length, original_solver->v0(), solver->v0());

        advect(cfg.n, cfg.dt * static_cast<float>(cfg.n), original_solver->x(), original_solver->x0(),
               original_solver->u0(), original_solver->v0());
        solver->test_advect(solver->x(), solver->x0(), solver->u0(), solver->v0());

        float total = 0, max = 0;
        int count = 0, maxIndex = -1;
        auto expected = original_solver->x();
        auto actual = new float[length];
        cuda_check(cudaMemcpy(actual, solver->x(), length * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (int i = 0; i < length; ++i)
        {
            const auto absError = std::abs(expected[i] - actual[i]);
            if (absError > 0)
            {
                total += absError;
                count++;
                if (absError > max)
                {
                    max = absError;
                    maxIndex = i;
                }
            }
        }
        if (count > 0)
        {
            const auto mx = maxIndex % (cfg.n + 2);
            const auto my = maxIndex / (cfg.n + 2);
            printf("%d errors. avg=%f max=%f [%d,%d] (expected=%f actual=%f)\n", count, total / static_cast<float>(count),
                   max, mx, my, expected[maxIndex], actual[maxIndex]);
        }
        else
        {
            printf("success\n");
        }
        delete []actual;
    }
}

int main()
{
    const fluid_solver_config cfg = fluid_solver_config::load();
    const auto original_solver = new original_fluid_solver(cfg);
    const auto solver = new cuda_fluid_solver(cfg, original_solver);
    // const auto solver = new cuda_fluid_solver(cfg);
    const auto fsr = new fluid_renderer(solver);
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

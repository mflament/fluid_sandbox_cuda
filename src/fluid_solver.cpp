#include "fluid_solver.h"

#include <fstream>
#include <sstream>

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


// Helper: trim whitespace from both ends
static std::string trim(const std::string& s)
{
    const char* whitespace = " \t\n\r";
    size_t start = s.find_first_not_of(whitespace);
    size_t end = s.find_last_not_of(whitespace);

    if (start == std::string::npos)
        return "";

    return s.substr(start, end - start + 1);
}

fluid_solver_config fluid_solver_config::load()
{
    fluid_solver_config config;
    std::ifstream is = std::ifstream("config.cfg");
    if (!is)
    {
        return config;
    }
    std::string line;
    while (std::getline(is, line))
    {
        // Skip empty lines or comments
        if (line.empty() || line[0] == '#')
            continue;

        line = trim(line);
        if (line.empty())
            continue;

        // Split on '='
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, eqPos));
        std::string value = trim(line.substr(eqPos + 1));

        if (key.empty() || value.empty())
            continue;

        if (key == "n") config.n = static_cast<int>(stol(value));
        else if (key == "k") config.k = static_cast<int>(stol(value));
        else if (key == "force") config.force = stof(value);
        else if (key == "source") config.source = stof(value);
        else if (key == "diff") config.diff = stof(value);
        else if (key == "visc") config.visc = stof(value);
        else if (key == "dt") config.dt = stof(value);
    }
    is.close();
    return config;
}

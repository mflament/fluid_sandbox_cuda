#include "fluid_renderer.h"

#include <cstdio>

#include "gl_support.h"
#include "shaders.h"
#include "vector_operators.h"

fluid_renderer::fluid_renderer(fluid_solver* solver) : solver_(solver)
{
    render_velocity_ = true;
}

fluid_renderer::~fluid_renderer()
{
    glDeleteProgram(render_density_program_.program);
    glDeleteProgram(render_velocity_program_.program);

    glDeleteSamplers(1, &linear_sampler_);

    const GLuint textures[]{dens_texture_, u_texture_, v_texture_};
    glDeleteTextures(3, textures);
}

void fluid_renderer::initialize(GLFWwindow* window)
{
    glClearColor(0, 0, 0, 1);

    create_data_textures();

    render_density_program_ = create_render_density_program();
    render_velocity_program_ = create_render_velocity_program();

    create_vao();

    cursor_hover_ = glfwGetWindowAttrib(window, GLFW_HOVERED);
    update_title(window);

    solver_->initialize(dens_texture_, u_texture_, v_texture_);
}

int2 fluid_renderer::grid_position(const double2& mouse_position) const
{
    const auto fbs = framebuffer_size();
    const auto n = get_config().n;
    return int2{
        .x = 1 + static_cast<int>(mouse_position.x / fbs.x * n),
        .y = 1 + static_cast<int>((fbs.y - mouse_position.y - 1) / fbs.y * n),
    };
}

fluid_solver_config fluid_renderer::get_config() const
{
    return solver_->config;
}

void fluid_renderer::render(GLFWwindow* window, const render_state& render_state)
{
    solver_->solve(render_state);

    glViewport(0, 0, framebuffer_size().x, framebuffer_size().y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(vao_);
    if (render_velocity_)
    {
        solver_->update_velocity_textures(u_texture_, v_texture_);
        glUseProgram(render_velocity_program_.program);
        const auto n = get_config().n;
        const auto vertex_count = n * n * 2;
        glDrawArrays(GL_LINES, 0, vertex_count);
    }
    else
    {
        solver_->update_density_texture(dens_texture_);
        glBindSampler(0, linear_sampler_);
        glUseProgram(render_density_program_.program);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindSampler(0, 0);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
}

void fluid_renderer::handle_key_event(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        render_velocity_ = !render_velocity_;
        update_title(window);
    }
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
    {
        solver_->clear();
        if (render_velocity_)
        {
            solver_->update_velocity_textures(u_texture_, v_texture_);
        }
        else
        {
            solver_->update_density_texture(dens_texture_);
        }
    }
    base_renderer::handle_key_event(window, key, scancode, action, mods);
}

void fluid_renderer::handle_cursor_enter_event(GLFWwindow* window, int entered)
{
    cursor_hover_ = entered;
    base_renderer::handle_cursor_enter_event(window, entered);
}

bool fluid_renderer::is_mouse_button_pressed(int button) const
{
    return mouse_pressed_buttons_ & 1 << button;
}

bool fluid_renderer::is_in_client_area(const double2 pos) const
{
    const auto fbs = framebuffer_size();
    return pos.x >= 0 && pos.x < fbs.x && pos.y >= 0 && pos.y < fbs.y;
}

void fluid_renderer::handle_mouse_button_event(GLFWwindow* window, const int button, const int action, const int mods)
{
    if (action == GLFW_PRESS) mouse_pressed_buttons_ |= 1 << button;
    else mouse_pressed_buttons_ &= ~(1 << button);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        dragging_ = action == GLFW_PRESS;
        if (dragging_) last_mouse_pos_ = mouse_pos_;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS && is_in_client_area(mouse_pos_))
    {
        solver_->add_density(grid_position(mouse_pos_), get_config().source);
    }
    base_renderer::handle_mouse_button_event(window, button, action, mods);
}

void fluid_renderer::handle_cursor_pos_event(GLFWwindow* window, double xpos, double ypos)
{
    mouse_pos_ = double2{xpos, ypos};
    if (!is_in_client_area(mouse_pos_))
        return;
    if (dragging_)
    {
        const auto force = get_config().force;
        const auto movement = float2{
            static_cast<float>(mouse_pos_.x - last_mouse_pos_.x),
            static_cast<float>(last_mouse_pos_.y - mouse_pos_.y)
        };
        if (!is_zero(movement))
        {
            const auto grid_pos = grid_position(mouse_pos_);
            const auto velocity = movement * force;
            solver_->add_velocity(grid_pos, velocity);
        }
        last_mouse_pos_ = mouse_pos_;
    }

    if (is_mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
        solver_->add_density(grid_position(mouse_pos_), get_config().source);
    }

    base_renderer::handle_cursor_pos_event(window, xpos, ypos);
}

render_density_program fluid_renderer::create_render_density_program() const
{
    GLuint program = create_program(quad_vs, render_density_fs);
    const render_density_program rp = {
        .program = program,
        .size = glGetUniformLocation(program, "size"),
        .u_texture = glGetUniformLocation(program, "u_texture")
    };
    glUseProgram(rp.program);
    glUniform1i(rp.u_texture, 0);
    glUniform1i(rp.size, solver_->config.n);
    glUseProgram(0);
    return rp;
}

render_velocity_program fluid_renderer::create_render_velocity_program() const
{
    GLuint program = create_program(render_velocity_vs, render_velocity_fs);
    const render_velocity_program rp = {
        .program = program,
        .size = glGetUniformLocation(program, "size"),
        .u_texture = glGetUniformLocation(program, "u_texture"),
        .v_texture = glGetUniformLocation(program, "v_texture")
    };
    glUseProgram(rp.program);
    glUniform1i(rp.u_texture, 0);
    glUniform1i(rp.v_texture, 1);
    glUniform1i(rp.size, solver_->config.n);
    glUseProgram(0);
    return rp;
}

void fluid_renderer::update_title(GLFWwindow* window) const
{
    char title[64];
    (void)sprintf_s(title, 64, "fluid sandbox (%s)", render_velocity_ ? "velocity" : "density");
    glfwSetWindowTitle(window, title);
}

void fluid_renderer::create_vao()
{
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

GLuint fluid_renderer::setup_data_texture(GLuint texture) const
{
    glBindTexture(GL_TEXTURE_2D, texture);
    set_texture_sampler_params();
    const auto n = solver_->config.n + 2;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, n, n, 0, GL_RED, GL_FLOAT, nullptr);
    return texture;
}

void fluid_renderer::create_data_textures()
{
    GLuint textures[3];
    glCreateTextures(GL_TEXTURE_2D, 3, textures);
    u_texture_ = setup_data_texture(textures[0]);
    v_texture_ = setup_data_texture(textures[1]);
    dens_texture_ = setup_data_texture(textures[2]);
    glBindTexture(GL_TEXTURE_2D, 0);
    linear_sampler_ = create_sampler(GL_LINEAR, GL_LINEAR);
}

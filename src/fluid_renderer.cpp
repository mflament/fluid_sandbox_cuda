#include "fluid_renderer.h"

#include <cstdio>

#include "gl_support.h"
#include "cuda_support.h"
#include "cuda_kernels.cuh"
#include "shaders.h"
#include "vector_operators.h"

fluid_renderer::fluid_renderer(fluid_solver* solver) : solver_(solver)
{
    solver_->set_renderer(this);
    render_velocity_ = true;
}

fluid_renderer::~fluid_renderer()
{
    glDeleteProgram(render_density_program_.program);
    glDeleteProgram(render_velocity_program_.program);

    glDeleteSamplers(1, &linear_sampler_);

    const GLuint textures[]{dens_texture_, vel_texture_};
    glDeleteTextures(2, textures);
}

void fluid_renderer::initialize(GLFWwindow* window)
{
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    glClearColor(0, 0, 0, 1);

    create_data_textures();

    render_density_program_ = create_render_program(quad_vs, render_density_fs);
    render_velocity_program_ = create_render_program(render_velocity_vs, render_velocity_fs);

    create_vao();

    cursor_hover_ = glfwGetWindowAttrib(window, GLFW_HOVERED);
    update_title(window);
}

int2 fluid_renderer::grid_position(const double2& mouse_position) const
{
    const auto fbs = framebuffer_size();
    const auto gs = get_config().n;
    return int2{
        .x = 1 + static_cast<int>(mouse_position.x / fbs.x * gs.x),
        .y = 1 + static_cast<int>((fbs.y - mouse_position.y - 1) / fbs.y * gs.y),
    };
}

fluid_sandbox_config fluid_renderer::get_config() const
{
    return solver_->config;
}

void fluid_renderer::render(GLFWwindow* window, const render_state& render_state)
{
    solver_->solve();
    
    glViewport(0, 0, framebuffer_size().x, framebuffer_size().y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(vao_);
    glActiveTexture(GL_TEXTURE0);
    if (render_velocity_)
    {
        glBindTexture(GL_TEXTURE_2D, vel_texture_);
        solver_->update_velocity_texture(vel_texture_);
        glUseProgram(render_velocity_program_.program);
        const auto [x, y] = get_config().n;
        const auto vertex_count = x * y * 2;
        glDrawArrays(GL_LINES, 0, vertex_count);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, dens_texture_);
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
    base_renderer::handle_key_event(window, key, scancode, action, mods);
}

void fluid_renderer::handle_mouse_button_event(GLFWwindow* window, const int button, const int action,
                                                       const int mods)
{
    if (action == GLFW_PRESS) mouse_pressed_buttons_ |= 1 << button;
    else mouse_pressed_buttons_ &= ~(1 << button);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        dragging_ = action == GLFW_PRESS;
        if (dragging_) last_mouse_pos_ = mouse_pos_;
    }
    base_renderer::handle_mouse_button_event(window, button, action, mods);
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

void fluid_renderer::handle_cursor_pos_event(GLFWwindow* window, double xpos, double ypos)
{
    if (!cursor_hover_)
    {
        base_renderer::handle_cursor_pos_event(window, xpos, ypos);
        return;
    }

    mouse_pos_ = double2{xpos, ypos};
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

render_program fluid_renderer::create_render_program(const char* vs, const char* fs) const
{
    GLuint program = create_program(vs, fs);
    const render_program rp = {
        .program = program,
        .size = glGetUniformLocation(program, "size"),
        .u_texture = glGetUniformLocation(program, "u_texture")
    };
    set_program_uniforms(rp);
    return rp;
}

void fluid_renderer::set_program_uniforms(const render_program& rp) const
{
    glUseProgram(rp.program);
    glUniform1i(rp.u_texture, 0);
    auto [x, y] = solver_->config.n;
    glUniform2i(rp.size, x, y);
    glUseProgram(0);
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

void fluid_renderer::create_data_textures()
{
    const auto s = solver_->get_grid_size();
    GLuint textures[2];
    glCreateTextures(GL_TEXTURE_2D, 2, textures);
    vel_texture_ = textures[0];
    glBindTexture(GL_TEXTURE_2D, vel_texture_);
    set_texture_sampler_params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, s.x, s.y, 0, GL_RG, GL_FLOAT, nullptr);

    dens_texture_ = textures[1];
    glBindTexture(GL_TEXTURE_2D, dens_texture_);
    set_texture_sampler_params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, s.x, s.y, 0, GL_RED, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);
    linear_sampler_ = create_sampler(GL_LINEAR, GL_LINEAR);
}

// const size_t pixelCount = static_cast<size_t>(size) * size;
// const auto pixels = new float[pixelCount * 4];
// for (size_t i = 0; i < pixelCount; ++i)
// {
//     pixels[i * 4] = 1.0;
//     pixels[i * 4 + 1] = 1.0;
//     pixels[i * 4 + 2] = 0.0;
//     pixels[i * 4 + 3] = 1.0;
// }
//
// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size, size, 0, GL_RGBA, GL_FLOAT, nullptr);
// glBindTexture(GL_TEXTURE_2D, 0);

// cuda_check(cudaGraphicsGLRegisterImage(&cuda_texture_, gl_texture_, GL_TEXTURE_2D,
//                                        cudaGraphicsRegisterFlagsSurfaceLoadStore), "cudaGraphicsGLRegisterImage");
//
// cuda_check(cudaMalloc(&cuda_pixels_, static_cast<size_t>(size) * size * sizeof(float4)), "cudaMalloc");
// cuda_check(cudaMemcpy(cuda_pixels_, pixels, pixelCount * sizeof(float4), cudaMemcpyHostToDevice),
//            "cudaMemcpyHostToDevice");
// void fluid_sandbox_renderer::cuda_update_texture() const
// {
//     const auto size = config_.N + 2;
//     update_texture(cuda_texture_, cuda_pixels_, size, size);
//     // cuda_test(cuda_texture_, size, size);
// }

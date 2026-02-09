#pragma once

#include "render_loop.h"
#include "fluid_solver.h"

struct render_density_program
{
    GLuint program;
    GLint size;
    GLint u_texture;
};

struct render_velocity_program 
{
    GLuint program;
    GLint size;
    GLint u_texture;
    GLint v_texture;
};

class fluid_renderer final : public base_renderer // NOLINT(cppcoreguidelines-special-member-functions)
{
    GLuint vao_{};
    GLuint u_texture_{};
    GLuint v_texture_{};
    GLuint dens_texture_{};

    GLuint linear_sampler_{};

    render_density_program render_density_program_{};
    render_velocity_program render_velocity_program_{};

    double2 mouse_pos_{};
    double2 last_mouse_pos_{};
    int mouse_pressed_buttons_{};
    bool cursor_hover_{}, dragging_{};

    fluid_solver* solver_{};

    bool render_velocity_{};

    render_density_program create_render_density_program() const;
    render_velocity_program create_render_velocity_program() const;

    void create_data_textures();

    GLuint setup_data_texture(GLuint texture) const;

    void create_vao();
    
    void update_title(GLFWwindow* window) const;

    [[nodiscard]] int2 grid_position(const double2& mouse_position) const;

    [[nodiscard]] fluid_solver_config get_config() const;

    [[nodiscard]] bool is_mouse_button_pressed(int button) const;

    [[nodiscard]] bool is_in_client_area(double2 pos) const;

public:
    explicit fluid_renderer(fluid_solver* solver);

    ~fluid_renderer() override;

    void initialize(GLFWwindow* window) override;

    void render(GLFWwindow* window, const render_state& render_state) override;

    int window_events() override
    {
        return size_event | key_event | cursor_pos_event | mouse_buttons_event | cursor_enter_event;
    }

    void handle_key_event(GLFWwindow* window, int key, int scancode, int action, int mods) override;
    void handle_mouse_button_event(GLFWwindow* window, int button, int action, int mods) override;
    void handle_cursor_enter_event(GLFWwindow* window, int entered) override;
    void handle_cursor_pos_event(GLFWwindow* window, double xpos, double ypos) override;
};

/*
float4* cuda_pixels_{};
cudaGraphicsResource_t cuda_texture_{};
void cuda_update_texture() const;
*/

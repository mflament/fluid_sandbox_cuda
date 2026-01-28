#pragma once

#include "render_loop.h"
#include "fluid_solver.h"

struct render_program
{
    GLuint program;
    GLint size;
    GLint u_texture;
};

class fluid_sandbox_renderer final : public base_renderer // NOLINT(cppcoreguidelines-special-member-functions)
{
    GLuint vao_{};
    GLuint vel_texture_{};
    GLuint dens_texture_{};

    GLuint linear_sampler_{};

    render_program render_density_program_{};
    render_program render_velocity_program_{};

    double2 mouse_pos_{};
    double2 last_mouse_pos_{};
    int mouse_pressed_buttons_{};
    bool cursor_hover_{}, dragging_{};

    fluid_solver* solver_{};

    bool render_velocity_{};

    static render_program create_render_program(const char* vs, const char* fs);

    void set_program_uniforms(const render_program& rp, bool velocity) const;

    void create_data_textures();

    void create_vao();
    
    void update_title(GLFWwindow* window) const;

    [[nodiscard]] int2 grid_position(const double2& mouse_position) const;

    [[nodiscard]] fluid_sandbox_config get_config() const;

    [[nodiscard]] bool is_mouse_button_pressed(int button) const;
    
public:
    explicit fluid_sandbox_renderer(fluid_solver* solver);

    ~fluid_sandbox_renderer() override;

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

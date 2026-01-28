#pragma once

#include "GLFW/glfw3.h"

#include "vector_types.h"

struct render_state
{
    int frame;
    double time; // in secs since start
};

enum window_event // NOLINT(performance-enum-size)
{
    size_event = 1 << 0,
    key_event = 1 << 1,
    cursor_pos_event = 1 << 2,
    mouse_buttons_event = 1 << 3,
    cursor_enter_event = 1 << 4
};

class base_renderer // NOLINT(cppcoreguidelines-special-member-functions)
{
    int2 framebuffer_size_{};

public:
    [[nodiscard]] int2 framebuffer_size() const
    {
        return framebuffer_size_;
    }

    virtual ~base_renderer() = default;

    virtual void initialize(GLFWwindow* window) = 0;

    virtual void render(GLFWwindow* window, const render_state& render_state) = 0;

    [[nodiscard]] virtual int window_events()
    {
        return size_event | key_event | cursor_pos_event | mouse_buttons_event;
    }

    virtual void handle_size_event(GLFWwindow* window, int width, int height);
    virtual void handle_key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
    virtual void handle_mouse_button_event(GLFWwindow* window, int button, int action, int mods);
    virtual void handle_cursor_enter_event(GLFWwindow* window, int entered);
    virtual void handle_cursor_pos_event(GLFWwindow* window, double xpos, double ypos);

    bool accept_window_event(window_event event);
};

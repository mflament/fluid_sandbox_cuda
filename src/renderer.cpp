#include "renderer.h"

void base_renderer::handle_size_event(GLFWwindow* window, const int width, const int height)
{
    framebuffer_size_ = {width, height};
    glViewport(0, 0, framebuffer_size_.x, framebuffer_size_.y);
}

void base_renderer::handle_key_event(GLFWwindow* window, const int key, const int scancode, const int action,
                                     const int mods)
{
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, true);
}

void base_renderer::handle_mouse_button_event(GLFWwindow* window, const int button, const int action, const int mods)
{
}

void base_renderer::handle_cursor_enter_event(GLFWwindow* window, const int entered)
{
}

void base_renderer::handle_cursor_pos_event(GLFWwindow* window, const double xpos, const double ypos)
{
}

bool base_renderer::accept_window_event(const window_event event)
{
    return window_events() & event;
}

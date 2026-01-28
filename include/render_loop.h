#pragma once

#include "gl_support.h"
#include "renderer.h"
#include "vector_types.h"

class render_loop final
{
    static base_renderer* renderer_;

    static GLFWwindow* create_window(bool debug);

    static void framebuffer_size_callback(GLFWwindow* window, int w, int h);

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    static void cursor_enter_callback(GLFWwindow* window, int entered);

    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);

    static void debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
                               GLsizei length, const GLchar* message, const void* userParam);

public:
    [[nodiscard]] static base_renderer* get_renderer() { return renderer_; };

    static void start(base_renderer* renderer, bool debug = false);
};

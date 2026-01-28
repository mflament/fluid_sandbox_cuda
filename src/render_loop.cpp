#include "render_loop.h"
#include <iostream>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

base_renderer* render_loop::renderer_ = nullptr;

void render_loop::start(base_renderer* renderer, const bool debug)
{
    renderer_ = renderer;
    const auto window = create_window(debug);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    renderer->handle_size_event(window, width, height);

    renderer->initialize(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetCursorEnterCallback(window, cursor_enter_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    render_state state{};
    glfwSetTime(0.0);
    while (!glfwWindowShouldClose(window))
    {
        state.time = glfwGetTime();
        renderer->render(window, state);

        glfwSwapBuffers(window);
        glfwPollEvents();
        state.frame++;
    }

    glfwTerminate();
}

GLFWwindow* render_loop::create_window(const bool debug)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (debug)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

    int width = 800, height = 600;
    GLFWwindow* window = glfwCreateWindow(width, height, "fluid", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);

    // ReSharper disable once CppCStyleCast
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) // NOLINT(clang-diagnostic-cast-function-type-strict)
    {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    if (debug)
    {
        int flags;
        glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
        if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
            glDebugMessageCallback(debug_callback, nullptr);
    }

    return window;
}

void render_loop::framebuffer_size_callback(GLFWwindow* window, const int w, const int h)
{
    if (const auto renderer = get_renderer(); renderer && renderer->accept_window_event(size_event))
        renderer->handle_size_event(window, w, h);
}

void render_loop::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (const auto renderer = get_renderer(); renderer && renderer->accept_window_event(key_event))
        renderer->handle_key_event(window, key, scancode, action, mods);
}

void render_loop::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (const auto renderer = get_renderer(); renderer && renderer->accept_window_event(mouse_buttons_event))
        renderer->handle_mouse_button_event(window, button, action, mods);
}

void render_loop::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (const auto renderer = get_renderer(); renderer && renderer->accept_window_event(cursor_pos_event))
        renderer->handle_cursor_pos_event(window, xpos, ypos);
}

void render_loop::cursor_enter_callback(GLFWwindow* window, int entered)
{
    if (const auto renderer = get_renderer(); renderer && renderer->accept_window_event(cursor_enter_event))
        renderer->handle_cursor_enter_event(window, entered);
}

void render_loop::debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                 GLsizei length, const GLchar* message, const void* userParam)
{
    printf("Debug message is: %s\n", message);
}

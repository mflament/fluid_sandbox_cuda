#include "gl_support.h"

#include "glad/glad.h"
#include <iostream>

enum constants : std::uint16_t
{
    log_size = 1024
};

namespace
{
    GLuint create_shader(GLenum shaderType, const char* src, char* infoLog)
    {
        const auto shader = glCreateShader(shaderType);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        int success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, log_size, nullptr, infoLog);
            const auto st = shaderType == GL_VERTEX_SHADER ? "vertex" : "fragment";
            char message[log_size];
            (void)sprintf_s(message, log_size, "error compiling %s shader\n%s", st, infoLog);
            throw std::invalid_argument(message);
        }
        return shader;
    }
}

GLuint create_program(const char* vsSrc, const char* fsSrc)
{
    char info_log[log_size]{};
    int status;

    const GLuint vs = create_shader(GL_VERTEX_SHADER, vsSrc, info_log);
    const GLuint fs = create_shader(GL_FRAGMENT_SHADER, fsSrc, info_log);

    const auto program = glCreateProgram();

    glAttachShader(program, vs);
    glAttachShader(program, fs);

    glLinkProgram(program);

    glValidateProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramInfoLog(program, log_size, nullptr, info_log);
        char message[log_size];
        (void)sprintf_s(message, log_size, "error linking program\n\n%s", info_log);
        throw std::invalid_argument(message);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

GLuint create_texture(const GLint min_filter, const GLint mag_filter, const GLint wrap_s, const GLint wrap_t)
{
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    set_texture_sampler_params(texture, min_filter, mag_filter, wrap_s, wrap_t);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

void set_texture_sampler_params(const GLuint texture, const GLint min_filter, const GLint mag_filter,
                                const GLint wrap_s, const GLint wrap_t)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    set_texture_sampler_params(texture, min_filter, mag_filter, wrap_s, wrap_t);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void set_texture_sampler_params(const GLint min_filter, const GLint mag_filter,
                                const GLint wrap_s, const GLint wrap_t)
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
}

GLuint create_sampler(const GLint min_filter, const GLint mag_filter, const GLint wrap_s, const GLint wrap_t)
{
    GLuint sampler;
    glCreateSamplers(1, &sampler);
    set_sampler_params(sampler, min_filter, mag_filter, wrap_s, wrap_t);
    return sampler;
}

void set_sampler_params(const GLuint sampler, const GLint min_filter, const GLint mag_filter, const GLint wrap_s,
                        const GLint wrap_t)
{
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, min_filter);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, mag_filter);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, wrap_s);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, wrap_t);
}

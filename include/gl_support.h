#pragma once
#include "glad/glad.h"

GLuint create_program(const char* vsSrc, const char* fsSrc);

GLuint create_texture(GLint min_filter = GL_NEAREST, GLint mag_filter = GL_NEAREST,
                      GLint wrap_s = GL_CLAMP_TO_EDGE, GLint wrap_t = GL_CLAMP_TO_EDGE);

void set_texture_sampler_params(GLuint texture, GLint min_filter, GLint mag_filter, GLint wrap_s = GL_CLAMP_TO_EDGE, GLint wrap_t = GL_CLAMP_TO_EDGE);

void set_texture_sampler_params(GLint min_filter = GL_NEAREST, GLint mag_filter = GL_NEAREST, GLint wrap_s = GL_CLAMP_TO_EDGE, GLint wrap_t = GL_CLAMP_TO_EDGE);

GLuint create_sampler(GLint min_filter = GL_NEAREST, GLint mag_filter = GL_NEAREST,
                      GLint wrap_s = GL_CLAMP_TO_EDGE, GLint wrap_t = GL_CLAMP_TO_EDGE);

void set_sampler_params(GLuint sampler, GLint min_filter = GL_NEAREST, GLint mag_filter = GL_NEAREST,
                        GLint wrap_s = GL_CLAMP_TO_EDGE, GLint wrap_t = GL_CLAMP_TO_EDGE);

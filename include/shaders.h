#pragma once

// language=glsl
static auto quad_vs = R"(
#version 330 core

vec2 QUAD_VERTICES[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
int QUAD_INDICES[6] = int[](0, 1, 2, 2, 3, 0);

out vec2 uv;

void main() {
    vec2 position = QUAD_VERTICES[QUAD_INDICES[gl_VertexID]];
    gl_Position = vec4(position, 0.0, 1.0);
    uv = position * 0.5 + 0.5;
}
)";


// language=glsl
static auto test_fs = R"(
#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform float u_time;

void main()
{
    vec3 col = 0.5 + 0.5*cos(u_time+uv.xyx+vec3(0,2,4));
    fragColor = vec4(col, 1.0);
}
)";

// language=glsl
static auto texture_fs = R"(
#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_texture;

void main()
{
    vec3 col = texture(u_texture, uv).rgb;
    fragColor = vec4(col, 1.0);
}
)";


// language=glsl
static auto mouse_fs = R"(
#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform vec2 u_cursor_pos;
uniform int u_buttons;

void main()
{
    if (u_buttons == 3) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else if (u_buttons == 1) {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
    } else if (u_buttons == 2) {
        fragColor = vec4(0.0, 1.0, 1.0, 1.0);
    } else {
        fragColor = vec4(u_cursor_pos, 0.0, 1.0);
    }
}
)";


// language=glsl
static auto render_density_fs = R"(
#version 330 core

uniform sampler2D u_texture;
uniform vec2 size;

in vec2 uv;
out vec4 fragColor;

void main()
{
    vec2 grid_pos = 1.0 + uv * size; 
    vec2 coord =  grid_pos / (size + 2.0);
    float d = texture(u_texture, coord).r;
    fragColor = vec4(vec3(d), 1.0);
}
)";

// language=glsl
static auto render_velocity_vs = R"(
#version 330 core

uniform sampler2D u_texture;
uniform ivec2 size;

void main() {
    int vertexId = gl_VertexID;
    int cellId = vertexId / 2;
    ivec2 gridPos = ivec2(cellId % size.x, cellId / size.x);
    vec2 pos = (vec2(gridPos) + 0.5) / vec2(size);
    if (vertexId % 2 == 1) {
        vec2 uv = texelFetch(u_texture, gridPos + 1, 0).rg;
        uv = normalize(uv) * clamp(length(uv), 0.0, 2.0 / min(size.x, size.y));
        pos += uv;
    }
    gl_Position = vec4((pos * 2.0 - 1.0), 0.0, 1.0);
}
)";

// language=glsl
static auto render_velocity_fs = R"(
#version 330 core

in vec2 uv;
out vec4 fragColor;

void main() {
    fragColor = vec4(1.0);
}
)";

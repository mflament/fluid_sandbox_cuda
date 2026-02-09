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
static auto render_density_fs = R"(
#version 330 core

uniform sampler2D u_texture;
uniform int size;

in vec2 uv;
out vec4 fragColor;

void main()
{
    vec2 grid_pos = 1.0 + uv * float(size); 
    vec2 coord = grid_pos / float(size + 2);
    float d = texture(u_texture, coord).r;
    fragColor = vec4(vec3(d), 1.0);
}
)";

// language=glsl
static auto render_velocity_vs = R"(
#version 330 core

uniform sampler2D u_texture;
uniform sampler2D v_texture;
uniform int size;

void main() {
    int vertexId = gl_VertexID;
    int cellId = vertexId / 2;
    ivec2 gridPos = ivec2(cellId % size, cellId / size);
    vec2 pos = (vec2(gridPos) + 0.5) / float(size);
    if (vertexId % 2 == 1) {
        float u = texelFetch(u_texture, gridPos + 1, 0).r;
        float v = texelFetch(v_texture, gridPos + 1, 0).r;
        vec2 uv = vec2(u, v);
        uv = normalize(uv) * clamp(length(uv), 0.0, 2 * size);
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

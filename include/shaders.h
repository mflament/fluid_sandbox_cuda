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
uniform ivec2 size;

in vec2 uv;
out vec4 fragColor;

void main()
{
    vec2 grid_pos = 1.0 + uv * vec2(size); 
    vec2 coord = grid_pos / vec2(size + 2);
    //float d = texture(u_texture, coord).r;
    //fragColor = vec4(vec3(d), 1.0);
    float d = texture(u_texture, uv).r;
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
        uv = normalize(uv) * clamp(length(uv), 0.0, 2 * min(size.x, size.y));
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

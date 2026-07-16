#version 430

uniform mat4 u_modelviewMatrix;
uniform mat4 u_projMatrix;
uniform mat4 u_transformMatrix;
uniform int u_use_gpu_grid_normal;
uniform uint u_gpu_grid_dim0;
uniform uint u_gpu_grid_dim1;

in vec3 v_position;
in vec3 v_color;
in vec3 v_normal;
in vec2 v_texcoord;

layout(std430, binding = 19) readonly buffer GpuPositionBuffer {
    float gpu_position[];
};

out vec4 f_color;
out vec4 f_normal;
out vec4 f_light1;
out vec4 f_light2;
out vec4 f_light3;
out vec2 f_texcoord;

vec3 load_gpu_position(uint vertex_index)
{
    uint base = vertex_index * 3u;
    return vec3(gpu_position[base], gpu_position[base + 1u], gpu_position[base + 2u]);
}

uint grid_index(uint row, uint col)
{
    return row * u_gpu_grid_dim1 + col;
}

uint clamp_uint(int value, uint upper_exclusive)
{
    if (value < 0) {
        return 0u;
    }
    return min(uint(value), upper_exclusive - 1u);
}

vec3 gpu_grid_normal()
{
    uint vertex_index = uint(gl_VertexID);
    if (u_gpu_grid_dim0 == 0u || u_gpu_grid_dim1 == 0u) {
        return v_normal;
    }

    uint row = vertex_index / u_gpu_grid_dim1;
    uint col = vertex_index - row * u_gpu_grid_dim1;
    if (row >= u_gpu_grid_dim0) {
        return vec3(0.0, 1.0, 0.0);
    }

    uint left_col = clamp_uint(int(col) - 1, u_gpu_grid_dim1);
    uint right_col = clamp_uint(int(col) + 1, u_gpu_grid_dim1);
    uint up_row = clamp_uint(int(row) - 1, u_gpu_grid_dim0);
    uint down_row = clamp_uint(int(row) + 1, u_gpu_grid_dim0);

    vec3 left_pos = load_gpu_position(grid_index(row, left_col));
    vec3 right_pos = load_gpu_position(grid_index(row, right_col));
    vec3 up_pos = load_gpu_position(grid_index(up_row, col));
    vec3 down_pos = load_gpu_position(grid_index(down_row, col));

    vec3 value = cross(right_pos - left_pos, down_pos - up_pos);
    float len_sq = dot(value, value);
    if (len_sq > 1e-20) {
        return value * inversesqrt(len_sq);
    }
    return vec3(0.0, 1.0, 0.0);
}

void main()
{
    vec3 normal = (u_use_gpu_grid_normal != 0) ? gpu_grid_normal() : v_normal;

    f_color = vec4(v_color, 1.0);
    f_normal = transpose(inverse(u_modelviewMatrix)) * vec4(normal, 0.0);
    f_light1 = normalize(u_modelviewMatrix * vec4(0.0, 0.707, -0.707, 0.0));
    f_light2 = normalize(u_modelviewMatrix * vec4(-0.707, -0.0, 0.707, 0.0));
    f_light3 = normalize(u_modelviewMatrix * vec4(0.707, -0.0, 0.707, 0.0));
    f_texcoord = v_texcoord;

    gl_Position = u_projMatrix * u_modelviewMatrix * u_transformMatrix * vec4(v_position, 1.0);
}

#version 430

layout( local_size_x = 10, local_size_y = 10 ) in;

uniform vec3 m_external_force;
uniform float stiffness;
uniform float h_square;
uniform float m_rest_length;
uniform float m_mass_matrix; //  m_mesh->m_mass_matrix

uniform uint edge_size;

struct Edge
{
    uint m_v1, m_v2; // indices of endpoint vertices
    uint m_tri1, m_tri2; // indices of adjacent faces
};

layout(std430, binding = 0) buffer EdgeBuffer {
    Edge edges[];
};

layout(std430, binding = 1) buffer gradient {
    float gradient[];
};

layout(std430, binding = 2) buffer x_pos {
    float x_pos[];
};

//layout(std430, binding = 3) buffer m_y {
//    float m_y[];
//};

void main() {
//    uint idx = gl_GlobalInvocationID.x;


    uvec3 nParticles = gl_NumWorkGroups * gl_WorkGroupSize;
    uint idx = gl_GlobalInvocationID.y * nParticles.x + gl_GlobalInvocationID.x;

    if(idx < edge_size){
        Edge e = edges[idx];

        // 获取顶点索引
        uint i = e.m_v1 * 3;
        uint j = e.m_v2 * 3;

        vec3 x_ij = vec3(x_pos[i],x_pos[i+1],x_pos[i+2]) - vec3(x_pos[j],x_pos[j+1],x_pos[j+2]);
        vec3 g_ij = stiffness * (length(x_ij) - m_rest_length) * normalize(x_ij);

        gradient[i] += g_ij.x;
        gradient[i+1] += g_ij.y;
        gradient[i+2] += g_ij.z;

        gradient[j] -= g_ij.x;
        gradient[j+1] -= g_ij.y;
        gradient[j+2] -= g_ij.z;


//        gradient[i] = m_mass_matrix * (x_pos[i] - m_y[i]) + h_square * gradient[i];
//        gradient[i+1] = m_mass_matrix * (x_pos[i+1] - m_y[i+1]) + h_square * gradient[i+1];
//        gradient[i+2] = m_mass_matrix * (x_pos[i+2] - m_y[i+2]) + h_square * gradient[i+2];
//
//        gradient[j] = m_mass_matrix * (x_pos[j] - m_y[j]) + h_square * gradient[j];
//        gradient[j+1] = m_mass_matrix * (x_pos[j+1] - m_y[j+1]) + h_square * gradient[j+1];
//        gradient[j+2] = m_mass_matrix * (x_pos[j+2] - m_y[j+2]) + h_square * gradient[j+2];
    }

}

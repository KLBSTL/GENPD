// ---------------------------------------------------------------------------------//
// Copyright (c) 2015, Regents of the University of Pennsylvania                    //
// All rights reserved.                                                             //
//                                                                                  //
// Redistribution and use in source and binary forms, with or without               //
// modification, are permitted provided that the following conditions are met:      //
//     * Redistributions of source code must retain the above copyright             //
//       notice, this list of conditions and the following disclaimer.              //
//     * Redistributions in binary form must reproduce the above copyright          //
//       notice, this list of conditions and the following disclaimer in the        //
//       documentation and/or other materials provided with the distribution.       //
//     * Neither the name of the <organization> nor the                             //
//       names of its contributors may be used to endorse or promote products       //
//       derived from this software without specific prior written permission.      //
//                                                                                  //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND  //
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED    //
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE           //
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY               //
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       //
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     //
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND      //
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT       //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS    //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                     //
//                                                                                  //
// Contact Tiantian Liu (ltt1598@gmail.com) if you have any questions.              //
//----------------------------------------------------------------------------------//

#ifndef _SIMULATION_H_
#define _SIMULATION_H_

//#define PARDISO_SUPPORT

#include <vector>
#include <deque>
#include <string>

#include "global_headers.h"
#include "anttweakbar_wrapper.h"
#include "mesh.h"
#include "constraint.h"
#include "scene.h"
#include "handle.h"

#ifdef PARDISO_SUPPORT
#include <Eigen/PardisoSupport>
#endif

//#include "Eigen/SVD"

class Mesh;
class AntTweakBarWrapper;

typedef enum
{
	INTEGRATION_IMPLICIT_EULER,
	INTEGRATION_IMPLICIT_BDF2,
	INTEGRATION_IMPLICIT_MIDPOINT,
	INTEGRATION_IMPLICIT_NEWMARK_BETA,
	INTEGRATION_QUASI_STATICS,
	INTEGRATION_TOTAL_NUM

} IntegrationMethod;
typedef enum
{
	PREFACTOR_M_PLUS_H2L,
	PREFACTOR_TOTAL_NUM

} PrefactorType;
typedef enum
{
	OPTIMIZATION_METHOD_GRADIENT_DESCENT,
	OPTIMIZATION_METHOD_NEWTON,
	OPTIMIZATION_METHOD_LBFGS,
	OPTIMIZATION_METHOD_NCG,
	OPTIMIZATION_METHOD_PNCG,
	OPTIMIZATION_METHOD_LOCALGLOBAL,
	OPTIMIZATION_METHOD_TOTAL_NUM

} OptimizationMethod;

typedef enum
{
	SOLVER_TYPE_DIRECT_LLT,
	SOLVER_TYPE_CG,
	SOLVER_TYPE_TOTAL_NUM
} SolverType;

typedef enum
{
	LBFGS_H0_IDENTITY,
	LBFGS_H0_LAPLACIAN,
	LBFGS_H0_TOTAL_NUM
} LBFGSH0Type;

typedef enum
{
	LS_TYPE_ARMIJO,
	LS_TYPE_WOLFE,
	LS_TYPE_TOTAL_NUM,
} LinesearchType;

typedef enum
{
	NCG_RESTART_NONE = 0,
	NCG_RESTART_PERIODIC = 1,
	NCG_RESTART_NON_DESCENT = 2
} NCGRestartMode;

typedef enum
{
	ADAPTIVE_LS_HISTORY_NONE,
	ADAPTIVE_LS_HISTORY_ITERATION,
	ADAPTIVE_LS_HISTORY_FRAME
} AdaptiveLineSearchHistoryMode;

struct AdaptiveLineSearchTraceRecord
{
	unsigned int iteration;
	unsigned int history_valid_before;
	unsigned int batch_id;
	ScalarType base_step;
	unsigned int candidate_count;
	ScalarType beta;
	int accepted;
	int chosen_i;
	ScalarType accepted_step;
	ScalarType accepted_energy;
	unsigned int candidate_evaluations;
	unsigned int fallback;
};

struct alignas(16) ParamsUBO {
	float t0;  // 0
	float beta; // 4
	int K; // 8
	int stiffness; // 12
	int edge_size; // 16
	float alpha; // 20
	float m_h; // 24
	int gradient_size; // 28
	int vertex_size; // 32
	int attachment_size; // 36
	int _pad0; // 40
	int _pad1; // 44
};

struct alignas(16) AttachmentGPU {
	unsigned int vertex_index;
	float stiffness;
	float _pad0;
	float _pad1;
	float fixed_point[4];
};

struct alignas(16) CollisionPrimitiveGPU {
	int type;
	int _pad0;
	int _pad1;
	int _pad2;
	float data0[4];
	float data1[4];
};

class Simulation
{
	friend class AntTweakBarWrapper;

public:
	Simulation();
	virtual ~Simulation();

	void Reset();
	void UpdateAnimation(const int fn);
	void Update();
	void Draw(const VBO& vbos);
	void LogFrameProfile(unsigned int frame, ScalarType fps_average, ScalarType fps_instant);
void ConfigureQualityMetrics(const std::string& reference_export_dir, const std::string& quality_reference_dir, unsigned int checkpoint_stride, bool enable_quality_metrics);
	bool PrepareCS2RenderBuffers();
	GLuint CS2RenderPositionBuffer() const;
	GLuint CS2RenderNormalBuffer() const;

	void GetOverlayChar(char* overlay, unsigned int overlay_length = 255);

	// select/unselect/move/save/load attachment constratins
	ScalarType TryToSelectAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir); // return ray_projection_plane_distance if hit; return -1 otherwise.
	bool TryToToggleAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir); // true if hit some vertex/constraint
	void SelectAtttachmentConstraint(AttachmentConstraint* ac);
	void UnselectAttachmentConstraint();
	AttachmentConstraint* AddAttachmentConstraint(unsigned int vertex_index); // add one attachment constraint at vertex_index
	AttachmentConstraint* AddAttachmentConstraint(unsigned int vertex_index, const EigenVector3& target); // add one attachment constraint at vertex_index
	void MoveSelectedAttachmentConstraintTo(const EigenVector3& target); // move selected attachement constraint to target
	void SaveAttachmentConstraint(const char* filename);
	void LoadAttachmentConstraint(const char* filename);


	// handles
	void NewHandle(const std::vector<unsigned int>& indices, const glm::vec3 color);
	void DeleteHandle();
	bool SelectHandle(std::vector<glm::vec3> ray);
	void MoveHandleTemporary(const glm::vec3& trans);
	void MoveHandleFinalize();
	void RotateHandleToValue();
	void RotateHandleSetStepSize();
	void RotateHandleTemporary(const glm::vec3& axis, const float& theta);
	void RotateHandleFinalize();
	void UpdateHandleInfoToConstraints(Handle& selected_handle);
	//void CombineVertexClassificationRegions(std::vector<RegionClassification>& rc);
	glm::vec3 SelectedHandleLocalCoM();
	glm::vec3 SelectedHandleCoM();
	// handle animations
	void SetHandleTranslationAnimation();
	void SetHandleTranslation();
	void SetHandleRotationAnimation();
	void SetHandleRotation();
	void AnimateHandle(const int current_frame);
	void SaveHandleAnimation(const char* filename);
	void LoadHandleAnimation(const char* filename);
	// save load reset
	void SaveHandles(const char* filename);
	void LoadHandles(const char* filename);
	void ResetHandles();

	// save laplacian matrix
	void SaveSparseMatrix(const SparseMatrix& A, const char* filename);
	void SaveLaplacianMatrix(const char* filename);

	// matlab debugger related
	void SetConvergedEnergy();

	// randomize points
	void RandomizePoints();

	// set material property for selected elements
	void SetMaterialProperty(std::vector<Constraint*>& constraints);
	void SetMaterialProperty(std::vector<Constraint*>& constraints, MaterialType type, ScalarType stretch, ScalarType bending, ScalarType kappa, ScalarType laplacian_coeff);
	// set material property for all elements
	void SetMaterialProperty();

	// select constraints and change material properties
	void GetPartialMaterialProperty();
	void SetPartialMaterialProperty();
	void SavePerConstraintMaterialProperties(const char* filename);
	void LoadPerConstraintMaterialProperties(const char* filename);
	void SelectTetConstraints(const std::vector<unsigned int>& indices);

	// eigen value visualization mesh
	void NewVisualizationMesh();
	void DeleteVisualizationMesh();
	void ResetVisualizationMesh();
	void SetVisualizationMesh();
	inline Mesh* GetEigenVectorVisMesh() { return m_eigenvector_vis_mesh; }
	void ResetVisualizationMeshHeight();

	// inline functions
	inline void SetReprefactorFlag()
	{
		m_precomputing_flag = false;
		m_prefactorization_flag = false;
		m_prefactorization_flag_newton = false;
	}
	inline void SetMesh(Mesh* mesh) { m_mesh = mesh; }
	inline void SetScene(Scene* scene) { m_scene = scene; }
	inline void SetStepMode(bool step_mode) { m_step_mode = step_mode; }
	inline ScalarType Timestep() { return m_h; }
	void SetTimestep(ScalarType timestep);
	void SetExperimentMaterialStiffness(ScalarType stretch, ScalarType bending);
inline void SetIterationsPerFrame(unsigned int iteration_count) { m_iterations_per_frame = iteration_count > 0u ? iteration_count : 1u; }
inline unsigned int IterationsPerFrame() const { return m_iterations_per_frame; }
	bool VerifyCSGradient(const std::string& output_dir);
	bool VerifyAdaptiveLineSearchHistoryInvalidation(const std::string& output_dir);
	void SetBatchedLineSearchK(unsigned int candidate_count);
	void SetArmijoBeta(ScalarType beta);
	void SetAdaptiveLineSearchHistoryMode(AdaptiveLineSearchHistoryMode mode);
	void SetNCGRestart(NCGRestartMode mode, unsigned int period);
	void SetProfileLineSearchDecisions(bool enabled);
	void SetForceCS2CpuStateRoundtrip(bool enabled);
protected:

	// simulation constants
	ScalarType m_h; // time_step
	unsigned int m_sub_stepping; // 
	bool m_step_mode;

	// simulation constants
	ScalarType m_gravity_constant;
	ScalarType m_wind_x;
	ScalarType m_wind_y;
	ScalarType m_wind_z;
	MaterialType m_material_type;
	ScalarType m_stiffness_attachment;
	ScalarType m_stiffness_stretch;
	ScalarType m_stiffness_high;
	ScalarType m_stiffness_bending;
	ScalarType m_stiffness_kappa;
	bool m_stiffness_auto_laplacian_stiffness;
	ScalarType m_stiffness_laplacian;
	ScalarType m_damping_coefficient;
	ScalarType m_restitution_coefficient;
	ScalarType m_friction_coefficient;

	// integration and optimization method
	IntegrationMethod m_integration_method;
	OptimizationMethod m_optimization_method;

	// key simulation components: mesh and scene
	Mesh* m_mesh;
	Scene* m_scene;
	// for other visualizations
	Mesh* m_eigenvector_vis_mesh = NULL;
	// key simulation components: constraints
	std::vector<Constraint*> m_constraints;
	//std::vector<Constraint*>::iterator m_attachment_constraint_start_point;
	AttachmentConstraint* m_selected_attachment_constraint;
	// collision constraints
	std::vector<CollisionSpringConstraint> m_collision_constraints;

	// partial material control
	std::vector<Constraint*> m_selected_constraints;
	MaterialType m_partial_material_type;
	ScalarType m_partial_stiffness_stretch;
	ScalarType m_partial_stiffness_bending;
	ScalarType m_partial_stiffness_kappa;

	// handle control
	std::vector<int> m_handle_id; // the id of the correspond handle of a vertex, -1 if no handle
	int m_selected_handle_id;
	std::vector<Handle> m_handles;
	// handle animation
	int m_keyframe_handle_id_translation;
	int m_keyframe_handle_id_rotation;
	int m_keyframe_handle_unit_translation_total_segments;
	int m_keyframe_handle_unit_rotation_total_segments;
	// translation	
	std::vector<int> m_keyframe_handle_unit_translation_end_frames;
	std::vector<EigenVector3> m_keyframe_handle_unit_translation_axis;
	std::vector<ScalarType> m_keyframe_handle_unit_translation_amount;
	std::vector<int> m_keyframe_handle_unit_rotation_end_frames;
	std::vector<EigenVector3> m_keyframe_handle_unit_rotation_axis;
	std::vector<ScalarType> m_keyframe_handle_unit_rotation_degree;

	// key simulation states
	VectorX m_qn;
	VectorX m_vn;
	VectorX m_qn_minus_one;
	VectorX m_vn_minus_one;
	VectorX m_qn_minus_two;
	VectorX m_vn_minus_two;

	// constant term in optimization:
	// 0.5(x-y)^2 M (x-y) + (c) * h^2 * E(x) - h^2 * x^T * z;
	VectorX m_y;
	VectorX m_z;

	// external force (gravity, wind, etc...)
	VectorX m_external_force;

	// for optimization method, number of iterations
	unsigned int m_iterations_per_frame;

	// for optimization method
	unsigned int m_current_iteration;

	// line search 
	bool m_enable_line_search;
	bool m_enable_exact_search;
	LinesearchType m_ls_type;
	ScalarType m_ls_alpha;
	ScalarType m_ls_beta;
	ScalarType m_ls_step_size;
	unsigned int m_batched_ls_k;
	AdaptiveLineSearchHistoryMode m_adaptive_ls_history_mode;
	bool m_adaptive_ls_was_active;
	std::vector<AdaptiveLineSearchTraceRecord> m_adaptive_ls_trace_records;
	NCGRestartMode m_ncg_restart_mode;
	unsigned int m_ncg_restart_period;
	// prefetched instructions in linesearch
	bool m_ls_is_first_iteration;
	VectorX m_ls_prefetched_gradient;
	VectorX m_ls_x_plus_tdx;
	ScalarType m_ls_prefetched_energy;

	// local global method
	bool m_enable_openmp;
	VectorX m_last_descent_dir;

	// for prefactorization
	SparseMatrix m_weighted_laplacian_1D;
	SparseMatrix m_weighted_laplacian;
	SparseMatrix m_J_matrix;
	Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver_1[PREFACTOR_TOTAL_NUM];

	bool m_precomputing_flag;
	bool m_prefactorization_flag;
	bool m_prefactorization_flag_newton;
#ifdef PARDISO_SUPPORT
	Eigen::PardisoLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver;
	Eigen::PardisoLLT<SparseMatrix, Eigen::Upper> m_newton_solver;
	Eigen::PardisoLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver_restpose_hessian;
	Eigen::PardisoLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver_dual;
#else
	Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver_1D;
	Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> m_prefactored_solver;
	Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> m_newton_solver;
#endif
	Eigen::ConjugateGradient<SparseMatrix> m_preloaded_cg_solver_1D;
	Eigen::ConjugateGradient<SparseMatrix> m_preloaded_cg_solver;

	// for Newton's method
	bool m_definiteness_fix;

	// solver type
	SolverType m_solver_type;
	int m_iterative_solver_max_iteration;

	// LBFGS
	bool m_lbfgs_restart_every_frame;
	LBFGSH0Type m_lbfgs_H0_type;
	int m_lbfgs_m; // back-track history length
	int m_lbfgs_max_m;
	bool m_lbfgs_need_update_H0;
	SparseMatrix m_lbfgs_B0;
	VectorX m_lbfgs_H0_diagonal;
	Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> m_lbfgs_H0_solver;
	VectorX m_lbfgs_last_x;
	VectorX m_lbfgs_last_gradient;
	VectorX m_ncg_lbfgs_pk;
	std::deque<VectorX> m_lbfgs_y_queue;
	std::deque<VectorX> m_lbfgs_s_queue;
	QueueLBFGS* m_lbfgs_queue;
	QueueLBFGS* ncg_lbfgs_queue;

	// volume
	ScalarType m_restshape_volume;
	ScalarType m_current_volume;

	// verbose
	bool m_verbose_show_converge;
	bool m_verbose_show_optimization_time;
	bool m_verbose_show_energy;
	bool m_verbose_show_factorization_warning;
	bool m_profile_logging_enabled;
bool m_quality_metrics_enabled;
bool m_profile_line_search_decisions;
std::string m_reference_export_dir;
std::string m_quality_reference_dir;
unsigned int m_quality_checkpoint_stride;
	bool m_last_profile_used_cs_ncg;
	bool m_last_profile_converged;
	bool m_last_profile_exploded;
	std::string m_last_profile_termination_reason;
	unsigned int m_last_profile_ncg_restarts;
	unsigned int m_last_profile_iterations;
	ScalarType m_last_profile_front_ms;
	ScalarType m_last_profile_transfer_ms;
	ScalarType m_last_profile_cs_y_upload_ms;
	ScalarType m_last_profile_cs_y_to_x_copy_ms;
	ScalarType m_last_profile_cs_x_readback_ms;
	ScalarType m_last_profile_cs_x_readback_wait_ms;
	ScalarType m_last_profile_cs_x_readback_copy_ms;
	ScalarType m_last_profile_iteration_ms;
	ScalarType m_last_profile_optimization_ms;
	ScalarType m_last_profile_back_ms;
	ScalarType m_last_profile_update_posvel_ms;
	ScalarType m_last_profile_position_stats_ms;
	ScalarType m_last_profile_collision_ms;
	ScalarType m_last_profile_total_ms;
	ScalarType m_last_profile_step_size;
	ScalarType m_last_profile_objective_energy;
	ScalarType m_last_profile_gradient_norm;
	bool m_last_profile_gradient_norm_sampled;
	ScalarType m_last_profile_max_displacement;
	ScalarType m_last_profile_max_position;

	// animation for demo
	bool m_animation_enable_swinging;
	int m_animation_swing_num;
	int m_animation_swing_half_period;
	ScalarType m_animation_swing_amp;
	ScalarType m_animation_swing_dir[3];

	// hard coded collision plane for demo
	bool m_processing_collision;

	// use compute shader
	bool use_cs = true;
	GLuint gradient_shader, gradient_scatter_shader, gradient_finalize_shader, energy_shader, descent_shader, iner_shader,
		energy_for_linesearch_shader, colliEnergy_shader, objective_shader,
		choose_valid_shader, choose_final_shader, compute_shader, computeX_shader, collision_resolve_shader, normal_from_triangles_shader, cs2_state_shader, xpbd_constraints_shader, xpbd_apply_shader, adaptive_ls_reset_shader;

	GLuint gradient_program, gradient_scatter_program, gradient_finalize_program, energy_program, computeX_program, descent_program, iner_program,
		energy_for_linesearch_program, colliEnergy_program, objective_program, choose_valid_program, choose_final_program, compute_program, collision_resolve_program, normal_from_triangles_program, cs2_state_program, xpbd_constraints_program, xpbd_apply_program, adaptive_ls_reset_program;

	GLuint edgeID, gradientID, xID, energyID, fixededgesID, FlagID, ResultID, DescentID, m_yID, inerID, testID;
	GLuint vertexEdgeOffsetID, vertexEdgeIndexID, attachmentID, collisionVelocityID, collisionPrimitiveID, csNormalID;
	GLuint csPositionID, csStateStatsID, xpbdDeltaID, xpbdLambdaID, adaptiveLineSearchStateID;
	bool m_cs_render_position_valid;
	bool m_cs_gpu_state_valid;
	bool m_cs_cpu_state_stale;
	bool m_cs_skip_cpu_damping_once;
	bool m_force_cs2_cpu_state_roundtrip;


	std::vector<float> test_;
	std::vector<unsigned int> m_cs_vertex_edge_offsets;
	std::vector<unsigned int> m_cs_vertex_edge_indices;
	std::vector<CollisionPrimitiveGPU> m_cs_collision_primitives;
	std::vector<ScalarType> m_cs_mass_diagonal;
	ScalarType m_cs_gradient_norm_sq;
	ScalarType m_cs_gradient_dot_descent;
	unsigned int m_cs_ncg_restart_count;
	bool m_cs_edge_buffer_dirty;
	unsigned int m_cs_spring_constraint_count;
	unsigned int m_cs_attachment_constraint_count;
	std::size_t m_cs_adaptive_ls_state_buffer_bytes;
	unsigned int m_cs_adaptive_ls_history_generation;

	bool li_test = true;
	

	const char* gradient_source = R"(#version 430
#extension GL_NV_shader_atomic_float : require
layout( local_size_x = 256 ) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};

struct Edge
{
    uint m_v1, m_v2;
    uint m_tri1, m_tri2;
    float rest_length;
    int stiffness;
    int fixed_point;
	float _pad0;
    vec4 fixed_;
};

layout(std430, binding = 0) buffer EdgeBuffer {
    Edge edges[];
};

layout(std430, binding = 1) buffer gradientBuffer {
    float gradient[];
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430, binding = 10) buffer testBuffer {
    float test[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;


    if(idx < edge_size){
		Edge e = edges[idx];
		if(idx == 0) {test[0] += 100;}
        if(e.fixed_point == 1){

           
            uint i = e.m_v1 * 3;

            vec3 x_i = vec3(x_pos[i],x_pos[i+1],x_pos[i+2]);

            vec3 m_g = e.stiffness * (x_i - vec3(e.fixed_));

            //gradient.block_vector(m_p0) += m_g;

            atomicAdd(gradient[i], m_g.x);
            atomicAdd(gradient[i+1], m_g.y);
            atomicAdd(gradient[i+2], m_g.z);


			test[0] += 1;
        }
        else {
            
            uint i = e.m_v1 * 3;
            uint j = e.m_v2 * 3;
            float res = e.rest_length;

            vec3 x_ij = vec3(x_pos[i], x_pos[i+1], x_pos[i+2]) - vec3(x_pos[j], x_pos[j+1], x_pos[j+2]);

            vec3 g_ij = e.stiffness * (length(x_ij) - res) * normalize(x_ij);

            atomicAdd(gradient[i], g_ij.x);
            atomicAdd(gradient[i+1], g_ij.y);
            atomicAdd(gradient[i+2], g_ij.z);

            atomicAdd(gradient[j], -g_ij.x);
            atomicAdd(gradient[j+1], -g_ij.y);
            atomicAdd(gradient[j+2], -g_ij.z);

        }
    }

}
)",
* energy_source = R"(#version 430

layout( local_size_x = 256 ) in;

// uniform float rest_length;

layout(std140,binding = 9) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};

struct Edge
{
    uint m_v1, m_v2;
    uint m_tri1, m_tri2;
    float rest_length;
    int stiffness;
    int fixed_point;
	float _pad0;
    vec4 fixed_;
};

layout(std430, binding = 0) buffer EdgeBuffer {
    Edge edges[];
};

layout(std430, binding = 1) buffer gradientBuffer {
    float gradient[];
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430, binding = 3) buffer Data {
    float energy[];
};
layout(std430, binding = 8) buffer y {
    float m_y[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if(idx < edge_size){
        Edge e = edges[idx];

        uint i = e.m_v1 * 3;
        uint j = e.m_v2 * 3;
		float res = e.rest_length;

        vec3 p_i = vec3(x_pos[i], x_pos[i+1], x_pos[i+2]);
        vec3 p_j = vec3(x_pos[j], x_pos[j+1], x_pos[j+2]);

        vec3 x_ij = p_i - p_j;

        float dist = length(x_ij);
        float ld_ij = dist - res;
        

        float e_ij = 0.5 * float(e.stiffness) * ld_ij * ld_ij;

		energy[0] += e_ij;
        // atomicAdd(energy, e_ij);
    }

}
)",
* objective_source = R"(#version 430

layout(local_size_x = 256) in;

layout(std140,binding = 0) uniform params {
    float t0;
    float beta;
    int K;
    int stiffness;
    int edge_size;
    float alpha;
    float m_h;
    int gradient_size;
};

struct Edge
{
    uint m_v1, m_v2;
    uint m_tri1, m_tri2;
    float rest_length;
    int stiffness;
    int fixed_point;
	float _pad0;
    vec4 fixed_;
};

layout(std430, binding = 0) buffer EdgeBuffer {
    Edge edges[];
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430, binding = 3) buffer Data {
    float energy[];
};

shared float energy_shared[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    float local_sum = 0.0;
    for (uint idx = tid; idx < uint(edge_size); idx += gl_WorkGroupSize.x) {
        Edge e = edges[idx];
        uint i = e.m_v1 * 3u;
        uint j = e.m_v2 * 3u;

        vec3 p_i = vec3(x_pos[i], x_pos[i + 1u], x_pos[i + 2u]);
        vec3 p_j = vec3(x_pos[j], x_pos[j + 1u], x_pos[j + 2u]);
        float dist = length(p_i - p_j) - e.rest_length;
        local_sum += 0.5 * float(e.stiffness) * dist * dist;
    }

    energy_shared[tid] = local_sum;
    barrier();

    for (uint stride = gl_WorkGroupSize.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            energy_shared[tid] += energy_shared[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) {
        energy[0] = energy_shared[0];
    }
}
)",
* energy_for_linesearch_source = R"(#version 430

layout( local_size_x = 256 ) in;

struct Edge
{
    uint m_v1, m_v2;
    uint m_tri1, m_tri2;
    float rest_length;
    int stiffness;
    int fixed_point;
	float _pad0;
    vec4 fixed_;
};


layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};

layout(std430, binding = 0) buffer EdgeBuffer {
    Edge edges[];
};
layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};
layout(std430, binding = 3) buffer Data {
    float energy[];
};
layout(std430,binding = 7) buffer Descent {
    float d[];
};
layout(std430,binding = 9) buffer inertia {
    float iner[];
};

uniform float alpha_k;

shared float energy_shared[256];


float compute_energy_contribution(uint cidx,float t) {
    uint i = edges[cidx].m_v1;
    uint j = edges[cidx].m_v2;
float res = edges[cidx].rest_length;

    vec3 xi = vec3(
        x_pos[i*3 + 0] + t*d[i*3 + 0],
        x_pos[i*3 + 1] + t*d[i*3 + 1],
        x_pos[i*3 + 2] + t*d[i*3 + 2]
    );
    vec3 xj = vec3(
        x_pos[j*3 + 0] + t*d[j*3 + 0],
        x_pos[j*3 + 1] + t*d[j*3 + 1],
        x_pos[j*3 + 2] + t*d[j*3 + 2]
    );

    float l = length(xi - xj) - res;

    return 0.5 * float(edges[cidx].stiffness) * l * l;
}

void main() {
    uint i = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    float t = t0 * pow(beta, float(i));

    float local_sum = 0.0;
    for (uint idx = tid; idx < edge_size ; idx += gl_WorkGroupSize.x) {
        local_sum += compute_energy_contribution(idx,t);
    }

    energy_shared[tid] = local_sum;
    barrier();

    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            energy_shared[tid] += energy_shared[tid + stride];
        }
        barrier();
    }

    if (tid == 0) {
        energy[i] = (energy_shared[0] * m_h * m_h + iner[i]);
    }
}
)"
, * colliEnergy_source = R"(#version 430

layout( local_size_x = 256 ) in;

// uniform float rest_length;

uniform uint edge_size;
uniform uint fixed_edge_size;

struct Edge
{
    uint m_v1, m_v2;
    uint m_tri1, m_tri2;
    float rest_length;
    int stiffness;
    int fixed_point;
	float _pad0;
    vec4 fixed_;
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430, binding = 3) buffer Data {
    float energy;
};

layout(std430, binding = 4) buffer FixedEdgeBuffer {
    Edge fixededges[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if(idx < fixed_edge_size){
        Edge e = fixededges[idx];

        uint i = e.m_v1 * 3;
        uint j = e.m_v2 * 3;

        vec3 x_ij = vec3(x_pos[i],x_pos[i+1],x_pos[i+2]) - vec3(x_pos[j],x_pos[j+1],x_pos[j+2]);
        float dist = length(x_ij);
        float e_i = 0.5 * 3000.0 * dist * dist;

		energy += e_i;
        //atomicAdd(energy, e_i);
    }

}


)",
* choose_valid_source = R"(#version 430
layout(local_size_x = 32) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};

uniform float currentEnergy;
uniform float grad_dot_d;


layout(std430, binding = 3) buffer Data {
    float energy[];
};

layout(std430,binding = 5) buffer Flags {
    int valid[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;

    if (i >= uint(K)) {
        return;
    }

    float t = t0 * pow(beta, float(i));
    float rhs = currentEnergy + alpha * t * grad_dot_d;

    valid[i] = (energy[i] <= rhs) ? 1 : 0;
}
)",
* choose_final_source = R"(#version 430
layout(local_size_x = 1) in;


layout(std430,binding = 5) buffer Flags {
    int valid[];
};

layout(std430,binding = 6) buffer Result {
    int chosen_i;
};

void main() {

    if (gl_GlobalInvocationID.x != 0) return;

    for (int i = 0; i < 8; ++i) {
        if (valid[i] == 1) {
            chosen_i = i;
            return;
        }
    }
    chosen_i = -1;
}
)",
* compute_source = R"(#version 430

layout( local_size_x = 256 ) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};


layout(std430, binding = 1) buffer gradientBuffer {
    float gradient[];
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430, binding = 8) buffer y {
    float m_y[];
};

layout(std430,binding = 7) buffer Descent {
    float d[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if(idx < gradient_size){
        gradient[idx] = 1.0 * (x_pos[idx] - m_y[idx]) + m_h * m_h * gradient[idx];
	
    }

}
)",
* computeX_source = R"(#version 430

layout( local_size_x = 256 ) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
};

uniform float alpha_k;

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430,binding = 7) buffer Descent {
    float d[];
};
uniform uint size;

void main() {
    uint idx = gl_GlobalInvocationID.x;

	

    if(idx < size){
       x_pos[idx] = x_pos[idx] + d[idx] * alpha_k;
//x_pos[idx] = 999.0;
    }

	//if(idx == 0){ x_pos[0] = 12345.0;}	

}
)", * descent_source = R"(#version 460

layout( local_size_x = 256 ) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
    vec4 fixed_point;
};


layout(std430, binding = 1) buffer gradientBuffer {
    float gradient[];
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};

layout(std430,binding = 7) buffer Descent {
    float d[];
};

uniform float beta_k;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if(idx < gradient_size){
        d[idx] = -gradient[idx] + beta_k * d[idx];
    }

}
)", * iner_source = R"(#version 460

layout( local_size_x = 256 ) in;

layout(std140,binding = 0) uniform params {
    float t0;  // 0
    float beta; // 4
    int K; // 8
    int stiffness; // 12
    int edge_size; // 16
    float alpha; // 20
    float m_h; // 24
    int gradient_size; // 28
    vec4 fixed_point;
};

layout(std430, binding = 2) buffer x_posBuffer {
    float x_pos[];
};
layout(std430, binding = 8) buffer y {
    float m_y[];
};
layout(std430,binding = 7) buffer Descent {
    float d[];
};
layout(std430,binding = 9) buffer inertia {
    float iner[];
};
shared float iner_shared[256];

void main() {
    uint i = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    float t = t0 * pow(beta, float(i));

    // ²¢ÐÐ¼ÆËã E(x + t d)
    float local_sum = 0.0;
    float temp = 0.0;
    for (uint idx = tid; idx < gradient_size ; idx += gl_WorkGroupSize.x) {
        temp = x_pos[idx] + t * d[idx];
        local_sum += 0.5 * (temp - m_y[idx]) * (temp - m_y[idx]);
    }

	iner_shared[tid] = local_sum;
    barrier();

	for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            iner_shared[tid] += iner_shared[tid + stride];
        }
        barrier();
    }

    if(tid == 0){
        iner[i] = iner_shared[0];
    }
}

)";


	ScalarType energy[8];
	ScalarType valid[8];
	ScalarType iner[8];
	int* Result;
	int* fixedsize;
	GLuint pUBO;


	ParamsUBO comp_params;


	std::vector<glm::vec3> fixed_points;
	std::string m_gradient_shader_file;
	std::string m_gradient_scatter_shader_file;
	std::string m_gradient_finalize_shader_file;
	std::string m_energy_shader_file;
	std::string m_objective_shader_file;
	std::string m_energy_for_linesearch_shader_file;
	std::string m_computeX_shader_file;
	std::string m_descent_shader_file;
	std::string m_iner_shader_file;
	std::string m_stats_shader_file;
	std::string m_collision_resolve_shader_file;
	std::string m_normal_from_triangles_shader_file;
	std::string m_cs2_state_shader_file;
std::string m_xpbd_constraints_shader_file;
std::string m_xpbd_apply_shader_file;
std::string m_adaptive_ls_reset_shader_file;


	VectorX gradient_dir;

	



private:

	// main update sub-routines
	void clearConstraints(); // cleanup all constraints
	void setupConstraints(); // initialize constraints
	void dampVelocity(); // damp velocity at the end of each iteration.
	void calculateExternalForce(); // wind force is propotional to the area of triangles projected on the tangential plane
	VectorX collisionDetectionPostProcessing(const VectorX& x); // detect collision and return a vector of penetration
	void collisionDetection(const VectorX& x);
	void collisionResolution(const VectorX& penetration, VectorX& x, VectorX& v);

	void integrateImplicitMethod();

	// all those "OneIteration" functions will be called in a loop
	// x is initially passed as the initial guess of the next postion (i.e. inertia term): x = y = current_pos + current_vel*h
	// x will be changed during these subroutines in EVERY iteration
	// the final value of x will be the next_pos that we used to update all vertices.
	bool performGradientDescentOneIteration(VectorX& x);
	bool performNewtonsMethodOneIteration(VectorX& x);
	bool performLBFGSOneIteration(VectorX& x);// our method
	bool performncg(VectorX& x);
	bool performNCG_CS(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir);
	bool performNCG_CS2(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir);
	void set_source();
	void set_shader();
	void Create_SSBO();
	void syncCSParams();
	void rebuildCSAdjacency();
	bool validateCSAdjacency(std::string& reason) const;
	void uploadCSResourcesIfNeeded();
	void uploadCSCollisionPrimitives();
	bool shouldUseCS2GpuState();
	void initializeCS2GpuStateIfNeeded();
	bool predictCS2GpuStateY();
	bool finalizeCS2GpuState(ScalarType& max_position, ScalarType& max_displacement, bool& x_is_finite);
	void syncCS2GpuStateToCPU();
	void invalidateCS2GpuState();
	void ensureAdaptiveLineSearchState();
	void resetAdaptiveLineSearchState();
	void dispatchCSGradient(bool pure_constraint_only, bool profile_gradient = true);
	void updateCSStats(bool readback = true);
	ScalarType readCSStatsFromGPU(bool profile_readback = true);
	void collisionPostProcessCS(VectorX& x, VectorX& v);
bool performGPUXPBD(VectorX& x);
	ScalarType evaluatePotentialEnergyCS(const VectorX& x);
	bool performNCG(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir);
	bool performNCG_LBFGS(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir);
	void LBFGSKernelLinearSolve(VectorX& r, VectorX gf_k, ScalarType scaled_identity_constant);
	bool integrateLocalGlobalOneIteration(VectorX& x);
	// key initializations and constants computations
	void computeConstantVectorsYandZ();
	void updatePosAndVel(const VectorX& new_pos);

	void Evaluatespringlength(const VectorX& x);

	// evaluate energy
	ScalarType evaluateEnergy(const VectorX& x);
	// evaluate gradient
	void evaluateGradient(const VectorX& x, VectorX& gradient, bool enable_omp = false);
	// evaluate gradient and energy
	ScalarType evaluateEnergyAndGradient(const VectorX& x, VectorX& gradient);
	// evaluate Hessian Matrix
	void evaluateHessian(const VectorX& x, SparseMatrix& hessian_matrix);
	void evaluateHessianSmart(const VectorX& x, SparseMatrix& hessian_matrix);
	// evaluate hessian
	void evaluateHessianForCG(const VectorX& x);
	// apply hessian
	void applyHessianForCG(const VectorX& x, VectorX& b);
	// evaluate Weighted Laplacian Matrix
	void evaluateLaplacian(SparseMatrix& laplacian_matrix);
	// evaluate Weighted Laplacian Matrix nxn
	void evaluateLaplacian1D(SparseMatrix& laplacian_matrix_1d);

	// accesser 
	// volume
	ScalarType getVolume(const VectorX& x);

	// testing
	// print volume of the elements
	void printVolumeTesting(const VectorX& x);

	// energy conservation
	ScalarType evaluatePotentialEnergy(const VectorX& x);
	ScalarType evaluateKineticEnergy(const VectorX& v);
	ScalarType evaluateTotalEnergy(const VectorX& x, const VectorX& v);

	// basic building blocks
	ScalarType evaluateEnergyPureConstraint(const VectorX& x, const VectorX& f_ext);
	void evaluateGradientPureConstraint(const VectorX& x, const VectorX& f_ext, VectorX& gradient);
	ScalarType evaluateEnergyAndGradientPureConstraint(const VectorX& x, const VectorX& f_ext, VectorX& gradient);
	void evaluateHessianPureConstraint(const VectorX& x, SparseMatrix& hessian_matrix);
	void evaluateHessianPureConstraintSmart(const VectorX& x, SparseMatrix& hessian_matrix);
	void evaluateLaplacianPureConstraint(SparseMatrix& laplacian_matrix);
	void evaluateLaplacianPureConstraint1D(SparseMatrix& laplacian_matrix_1d);
	void applyHessianForCGPureConstraint(const VectorX& x, VectorX& b); // b = H*x

	// collision
	ScalarType evaluateEnergyCollision(const VectorX& x);
	void evaluateGradientCollision(const VectorX& x, VectorX& gradient);
	ScalarType evaluateEnergyAndGradientCollision(const VectorX& x, VectorX& gradient);
	void evaluateHessianCollision(const VectorX& x, SparseMatrix& hessian_matrix);

	// line search
	ScalarType lineSearch(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir);
	ScalarType Simulation::lineSearch_CS(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir);
	ScalarType lineSearchCSSerial(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir);
	ScalarType linesearchWithPrefetchedEnergyAndGradientComputing(const VectorX& x, const ScalarType current_energy, const VectorX& gradient_dir, const VectorX& descent_dir, ScalarType& next_energy, VectorX& next_gradient_dir);
	ScalarType lineSearch_ncg(const VectorX& x, const  VectorX& gradient_dir, VectorX& descent_dir, const SparseMatrix& Hessian);

	// matrices and prefactorizations
	void precomputeLaplacianWeights();
	void precomputeLaplacian();
	void setWeightedLaplacianMatrix();
	void setWeightedLaplacianMatrix1D();
	void prefactorize();

	void setJMatrix();
	void setWeightedLaplacianMatrix_1();
	void prefactorize(PrefactorType type);
	void evaluateDVector(const VectorX& x, VectorX& d);
	void evaluateJMatrix(SparseMatrix& J);
	// newton solver
	void analyzeNewtonSolverPattern(const SparseMatrix& A);
	void factorizeNewtonSolver(const SparseMatrix& A, char* warning_msg = "");


	// utility functions
	// linear solver
	ScalarType linearSolve(VectorX& x, const SparseMatrix& A, const VectorX& b, char* msg = "");
	// conjugate gradient solver
	ScalarType conjugateGradientWithInitialGuess(VectorX& x, const SparseMatrix& A, const VectorX& b, const unsigned int max_it = 200, const ScalarType tol = 1e-5);
	void factorizeDirectSolverLLT(const SparseMatrix& A, Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>& lltSolver, char* warning_msg = ""); // factorize matrix A using LLT decomposition

	void generateRandomVector(const unsigned int size, VectorX& x); // generate random vector varing from [-1 1].

};

#endif

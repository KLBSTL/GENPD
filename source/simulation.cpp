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

#pragma warning( disable : 4244 4267 4305 4996)

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>

#include <Eigen/Eigenvalues>

#include "simulation.h"
#include "timer_wrapper.h"
#include "runtime_paths.h"
#include "experiment_variant.h"
#include "quality_metrics.h"

#ifdef ENABLE_MATLAB_DEBUGGING
#include "matlab_debugger.h"
extern MatlabDebugger* g_debugger;
#endif

//#define USE_STL_QUEUE_IMPLEMENTATION
//#define OUTPUT_LS_ITERATIONS
#ifdef OUTPUT_LS_ITERATIONS
#define OUTPUT_LS_ITERATIONS_EVERY_N_FRAMES 100
ScalarType g_total_ls_iterations;
int g_total_iterations;
#endif

TimerWrapper g_integration_timer;
TimerWrapper g_lbfgs_timer;

ScalarType rest_length_adjust = 1; // 1  = normal spring, 0 = zero length spring

namespace
{
	const GLuint kCSReductionLocalSize = 256u;
	const GLuint kCSGradientLocalSize = 256u;
	const GLuint kCSScratchBinding = 10u;
	const GLuint kCSReductionOutputBinding = 15u;
	const unsigned int kCSUnitStepShortcutBudget = 64u;
	const unsigned int kCSLargeClothUnitStepShortcutBudget = 256u;
	const unsigned int kCSLargeClothVertexThreshold = 512u * 512u;
	const unsigned int kCSLargeClothIterationCap = 3u;

	ScalarType g_cs_profile_linesearch_ms = 0.0;
	ScalarType g_cs_profile_gradstats_ms = 0.0;
	ScalarType g_cs_profile_gradient_gpu_ms = 0.0;
	ScalarType g_cs_profile_reduction_gpu_ms = 0.0;
	ScalarType g_cs_profile_stats_readback_ms = 0.0;
	GLuint g_cs_profile_gradient_query = 0;
	GLuint g_cs_profile_reduction_query = 0;
	GLuint g_cs_profile_xupdate_query = 0;
	GLuint g_cs_profile_descent_query = 0;
	bool g_cs_profile_gradient_query_pending = false;
	bool g_cs_profile_reduction_query_pending = false;
	bool g_cs_profile_xupdate_query_pending = false;
	bool g_cs_profile_descent_query_pending = false;
	ScalarType g_cs_profile_xupdate_ms = 0.0;
	ScalarType g_cs_profile_xupdate_gpu_ms = 0.0;
	ScalarType g_cs_profile_descent_ms = 0.0;
	ScalarType g_cs_profile_descent_gpu_ms = 0.0;
	unsigned int g_cs_profile_full_linesearch_calls = 0;
	unsigned int g_cs_profile_skipped_linesearch_calls = 0;
	unsigned int g_cs_profile_unit_step_accepts = 0;
	unsigned int g_cs_profile_gradient_dispatches = 0;
	unsigned int g_cs_profile_stats_dispatches = 0;
	unsigned int g_cs_profile_reduction_dispatches = 0;
	unsigned int g_cs_profile_xupdate_dispatches = 0;
	unsigned int g_cs_profile_descent_dispatches = 0;
	unsigned int g_cs_profile_host_readbacks = 0;
	unsigned int g_cs_profile_solver_finish_calls = 0;
	unsigned int g_cs_unit_step_shortcut_budget = 0;
	bool g_cs_prefetched_energy_valid = false;
	std::size_t g_cs_gradient_buffer_bytes = 0;
	std::size_t g_cs_descent_buffer_bytes = 0;
	std::size_t g_cs_x_buffer_bytes = 0;
	std::size_t g_cs_y_buffer_bytes = 0;
	std::size_t g_cs_energy_buffer_bytes = 0;
	std::size_t g_cs_inertia_buffer_bytes = 0;
	std::size_t g_cs_collision_velocity_buffer_bytes = 0;
	std::size_t g_cs_collision_primitive_buffer_bytes = 0;
	std::size_t g_cs_render_normal_buffer_bytes = 0;
	std::size_t g_cs_state_position_buffer_bytes = 0;
	std::size_t g_cs_state_stats_buffer_bytes = 0;
	unsigned int g_cs_active_iteration_budget = 0;
	bool g_cs_gpu_state_active_frame = false;

	struct ConstraintQualityMetrics
{
    ScalarType constraint_energy;
    ScalarType mean_stretch_strain;
    ScalarType max_stretch_strain;
};

struct CSLineSearchResultGPU
	{
		int chosen_i;
		int accepted;
		float step;
		float accepted_energy;
	};

	bool VectorIsFinite(const VectorX& x)
	{
		for (int i = 0; i < x.size(); ++i)
		{
			if (!std::isfinite(x[i]))
			{
				return false;
			}

		}
		return true;
	}

	ConstraintQualityMetrics ComputeConstraintQualityMetrics(const Mesh* mesh, const VectorX& positions)
{
    ConstraintQualityMetrics metrics = {};
    if (!mesh || positions.size() != mesh->m_system_dimension)
    {
        return metrics;
    }

    unsigned int stretch_count = 0;
    for (std::vector<Edge>::const_iterator edge_it = mesh->my_edge.begin(); edge_it != mesh->my_edge.end(); ++edge_it)
    {
        const Edge& edge = *edge_it;
        if (edge.m_v1 >= mesh->m_vertices_number || edge.m_v2 >= mesh->m_vertices_number)
        {
            continue;
        }

        if (edge.fixed_point == 1)
        {
            const EigenVector3 fixed_point(edge.fixed_.x, edge.fixed_.y, edge.fixed_.z);
            const EigenVector3 delta = positions.block_vector(edge.m_v1) - fixed_point;
            metrics.constraint_energy += static_cast<ScalarType>(0.5f * edge.stiffness) * delta.squaredNorm();
            continue;
        }

        const ScalarType rest_length = std::max(static_cast<ScalarType>(edge.rest_length), static_cast<ScalarType>(EPSILON));
        const ScalarType current_length = (positions.block_vector(edge.m_v1) - positions.block_vector(edge.m_v2)).norm();
        const ScalarType stretch = current_length - static_cast<ScalarType>(edge.rest_length);
        const ScalarType strain = std::abs(stretch) / rest_length;
        metrics.constraint_energy += static_cast<ScalarType>(0.5f * edge.stiffness) * stretch * stretch;
        metrics.mean_stretch_strain += strain;
        metrics.max_stretch_strain = std::max(metrics.max_stretch_strain, strain);
        ++stretch_count;
    }

    if (stretch_count > 0)
    {
        metrics.mean_stretch_strain /= static_cast<ScalarType>(stretch_count);
    }
    return metrics;
}

ScalarType ComputeMaxPenetrationDepth(Scene* scene, bool processing_collision, const VectorX& positions, unsigned int vertex_count)
{
    if (!processing_collision || !scene || scene->IsEmpty())
    {
        return 0.0;
    }

    ScalarType max_penetration = 0.0;
    EigenVector3 normal;
    ScalarType distance = 0.0;
    for (unsigned int vertex = 0; vertex < vertex_count; ++vertex)
    {
        if (scene->StaticIntersectionTest(positions.block_vector(vertex), normal, distance) && distance < 0.0)
        {
            max_penetration = std::max(max_penetration, -distance);
        }
    }
    return max_penetration;
}

ScalarType VectorInfinityNorm(const VectorX& x)
	{
		ScalarType max_abs_value = 0.0;
		for (int i = 0; i < x.size(); ++i)
		{
			ScalarType abs_value = std::abs(x[i]);
			if (abs_value > max_abs_value)
			{
				max_abs_value = abs_value;
			}
		}
		return max_abs_value;
	}

	bool ComputePositionStats(const VectorX& x, const VectorX& previous_position, ScalarType& max_position, ScalarType& max_displacement)
	{
		bool finite = true;
		max_position = 0.0;
		max_displacement = 0.0;
		for (int i = 0; i < x.size(); ++i)
		{
			const ScalarType value = x[i];
			if (!std::isfinite(value))
			{
				finite = false;
			}
			const ScalarType abs_value = std::abs(value);
			if (abs_value > max_position)
			{
				max_position = abs_value;
			}
			const ScalarType displacement = std::abs(value - previous_position[i]);
			if (displacement > max_displacement)
			{
				max_displacement = displacement;
			}
		}
		return finite;
	}

	GLuint ComputeCSPartialGroupCount(std::size_t item_count)
	{
		if (item_count == 0)
		{
			return 1u;
		}

		return static_cast<GLuint>((item_count + kCSReductionLocalSize - 1u) / kCSReductionLocalSize);
	}

	GLuint ComputeCSGradientGroupCount(std::size_t vertex_count)
	{
		if (vertex_count == 0)
		{
			return 1u;
		}

		return static_cast<GLuint>((vertex_count + kCSGradientLocalSize - 1u) / kCSGradientLocalSize);
	}

	void BeginCSGpuTimer(GLuint& query)
	{
		if (query == 0)
		{
			glGenQueries(1, &query);
		}
		glBeginQuery(GL_TIME_ELAPSED, query);
	}

	void EndCSGpuTimer(bool& pending)
	{
		glEndQuery(GL_TIME_ELAPSED);
		pending = true;
	}

	typedef void (GLAPIENTRY * CSDebugGroupPushProc)(GLenum source, GLuint id, GLsizei length, const GLchar* message);
	typedef void (GLAPIENTRY * CSDebugGroupPopProc)(void);

	bool IsInvalidOpenGLProcAddress(PROC proc)
	{
		return proc == NULL || proc == reinterpret_cast<PROC>(1) || proc == reinterpret_cast<PROC>(2)
			|| proc == reinterpret_cast<PROC>(3) || proc == reinterpret_cast<PROC>(-1);
	}

	CSDebugGroupPushProc GetCSDebugGroupPushProc()
	{
		static CSDebugGroupPushProc proc = NULL;
		static bool resolved = false;
		if (!resolved)
		{
			const PROC raw_proc = wglGetProcAddress("glPushDebugGroup");
			proc = IsInvalidOpenGLProcAddress(raw_proc) ? NULL : reinterpret_cast<CSDebugGroupPushProc>(raw_proc);
			resolved = true;
		}
		return proc;
	}

	CSDebugGroupPopProc GetCSDebugGroupPopProc()
	{
		static CSDebugGroupPopProc proc = NULL;
		static bool resolved = false;
		if (!resolved)
		{
			const PROC raw_proc = wglGetProcAddress("glPopDebugGroup");
			proc = IsInvalidOpenGLProcAddress(raw_proc) ? NULL : reinterpret_cast<CSDebugGroupPopProc>(raw_proc);
			resolved = true;
		}
		return proc;
	}

	struct ScopedCSDebugGroup
	{
		CSDebugGroupPopProc pop_proc;
		bool active;

		ScopedCSDebugGroup(const char* label)
			: pop_proc(NULL), active(false)
		{
			const CSDebugGroupPushProc push_proc = GetCSDebugGroupPushProc();
			pop_proc = GetCSDebugGroupPopProc();
			if (GenPDProfileGpuQueriesEnabled() && push_proc != NULL && pop_proc != NULL)
			{
				push_proc(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label);
				active = true;
			}
		}

		~ScopedCSDebugGroup()
		{
			if (active)
			{
				pop_proc();
			}
		}
	};
	ScalarType ConsumeCSGpuTimerMs(GLuint query, bool& pending)
	{
		if (!pending || query == 0)
		{
			return 0.0;
		}

		GLuint64 elapsed_ns = 0;
		glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsed_ns);
		pending = false;
		return static_cast<ScalarType>(elapsed_ns) * static_cast<ScalarType>(1.0e-6);
	}

	void EnsureFloatScratchBuffer(GLuint buffer, std::vector<float>& scratch_tracker, std::size_t required_float_count)
	{
		if (required_float_count == 0)
		{
			required_float_count = 1;
		}

		if (scratch_tracker.size() < required_float_count)
		{
			scratch_tracker.resize(required_float_count);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
			glBufferData(GL_SHADER_STORAGE_BUFFER, scratch_tracker.size() * sizeof(float), NULL, GL_DYNAMIC_DRAW);
		}

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kCSScratchBinding, buffer);
	}

	void DispatchCSReduction(GLuint reduction_program, GLuint scratch_buffer, GLuint output_buffer, GLuint partial_count, GLuint output_count, bool profile_stats_reduction = false, bool require_cpu_visibility = true, GLuint output_offset = 0u)
	{
		++g_cs_profile_reduction_dispatches;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kCSScratchBinding, scratch_buffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kCSReductionOutputBinding, output_buffer);

		glUseProgram(reduction_program);
		static GLint partial_count_location = -1;
		static GLint output_count_location = -1;
		static GLint output_offset_location = -1;
		if (partial_count_location < 0)
		{
			partial_count_location = glGetUniformLocation(reduction_program, "partial_count");
			output_count_location = glGetUniformLocation(reduction_program, "output_count");
			output_offset_location = glGetUniformLocation(reduction_program, "output_offset");
		}
		glUniform1ui(partial_count_location, partial_count);
		glUniform1ui(output_count_location, output_count);
		glUniform1ui(output_offset_location, output_offset);

		if (profile_stats_reduction)
		{
			BeginCSGpuTimer(g_cs_profile_reduction_query);
		}
		glDispatchCompute(output_count, 1, 1);
		if (profile_stats_reduction)
		{
			EndCSGpuTimer(g_cs_profile_reduction_query_pending);
		}
		GLbitfield barrier_bits = GL_SHADER_STORAGE_BARRIER_BIT;
		if (require_cpu_visibility)
		{
			barrier_bits |= GL_BUFFER_UPDATE_BARRIER_BIT;
		}
		glMemoryBarrier(barrier_bits);
	}

	void DispatchCSReductionIndirect(GLuint reduction_program, GLuint scratch_buffer, GLuint output_buffer, GLuint partial_count, GLuint output_count, GLuint output_offset, GLuint indirect_buffer, GLintptr indirect_offset)
	{
		++g_cs_profile_reduction_dispatches;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kCSScratchBinding, scratch_buffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kCSReductionOutputBinding, output_buffer);

		glUseProgram(reduction_program);
		static GLint partial_count_location = -1;
		static GLint output_count_location = -1;
		static GLint output_offset_location = -1;
		if (partial_count_location < 0)
		{
			partial_count_location = glGetUniformLocation(reduction_program, "partial_count");
			output_count_location = glGetUniformLocation(reduction_program, "output_count");
			output_offset_location = glGetUniformLocation(reduction_program, "output_offset");
		}
		glUniform1ui(partial_count_location, partial_count);
		glUniform1ui(output_count_location, output_count);
		glUniform1ui(output_offset_location, output_offset);

		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, indirect_buffer);
		glDispatchComputeIndirect(indirect_offset);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
	void EnsureCSBufferStorage(GLuint buffer, std::size_t required_bytes, std::size_t& tracked_bytes)
	{
		if (tracked_bytes != required_bytes)
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
			glBufferData(GL_SHADER_STORAGE_BUFFER, required_bytes, NULL, GL_DYNAMIC_DRAW);
			tracked_bytes = required_bytes;
		}
	}

	void UploadCSLineSearchResult(GLuint buffer, ScalarType step, int chosen_i, int accepted, ScalarType energy)
	{
		CSLineSearchResultGPU result = {};
		result.chosen_i = chosen_i;
		result.accepted = accepted;
		result.step = static_cast<float>(step);
		result.accepted_energy = static_cast<float>(energy);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(result), &result, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, buffer);
	}
	void ResetCSNCGProfileMetrics()
	{
		g_cs_profile_linesearch_ms = 0.0;
		g_cs_profile_gradstats_ms = 0.0;
		g_cs_profile_gradient_gpu_ms = 0.0;
		g_cs_profile_reduction_gpu_ms = 0.0;
		g_cs_profile_stats_readback_ms = 0.0;
		g_cs_profile_xupdate_ms = 0.0;
		g_cs_profile_xupdate_gpu_ms = 0.0;
		g_cs_profile_descent_ms = 0.0;
		g_cs_profile_descent_gpu_ms = 0.0;
		g_cs_profile_full_linesearch_calls = 0;
		g_cs_profile_skipped_linesearch_calls = 0;
		g_cs_profile_unit_step_accepts = 0;
		g_cs_profile_gradient_dispatches = 0;
		g_cs_profile_stats_dispatches = 0;
		g_cs_profile_reduction_dispatches = 0;
		g_cs_profile_xupdate_dispatches = 0;
		g_cs_profile_descent_dispatches = 0;
		g_cs_profile_host_readbacks = 0;
		g_cs_profile_solver_finish_calls = 0;
		g_cs_prefetched_energy_valid = false;
	}

	void ResetCSBufferStorageTrackers()
	{
		g_cs_gradient_buffer_bytes = 0;
		g_cs_descent_buffer_bytes = 0;
		g_cs_x_buffer_bytes = 0;
		g_cs_y_buffer_bytes = 0;
		g_cs_energy_buffer_bytes = 0;
		g_cs_inertia_buffer_bytes = 0;
		g_cs_collision_velocity_buffer_bytes = 0;
		g_cs_collision_primitive_buffer_bytes = 0;
		g_cs_render_normal_buffer_bytes = 0;
		g_cs_state_position_buffer_bytes = 0;
		g_cs_state_stats_buffer_bytes = 0;
	}

	std::size_t CurrentCSProfileBufferBytes(const std::vector<float>& scratch)
	{
		return g_cs_gradient_buffer_bytes + g_cs_descent_buffer_bytes + g_cs_x_buffer_bytes + g_cs_y_buffer_bytes
			+ g_cs_energy_buffer_bytes + g_cs_inertia_buffer_bytes + g_cs_collision_velocity_buffer_bytes
			+ g_cs_collision_primitive_buffer_bytes + g_cs_render_normal_buffer_bytes
			+ g_cs_state_position_buffer_bytes + g_cs_state_stats_buffer_bytes + scratch.size() * sizeof(float);
	}
}

// comparison function for sort
bool compareTriplet(SparseMatrixTriplet i, SparseMatrixTriplet j)
{
	return (abs(i.value()) < abs(j.value()));
}

void Vector3mx1ToMatrixmx3(const VectorX& x, EigenMatrixx3& m)
{
	for (unsigned int i = 0; i < m.rows(); i++)
	{
		m.block<1, 3>(i, 0) = x.block_vector(i).transpose();
	}
}
void Vector3mx1ToMatrixmx3(const VectorX& x, Matrix& m)
{
	for (unsigned int i = 0; i < m.rows(); i++)
	{
		m.block<1, 3>(i, 0) = x.block_vector(i).transpose();
	}
}

void Matrixmx3ToVector3mx1(const EigenMatrixx3& m, VectorX& x)
{
	for (unsigned int i = 0; i < m.rows(); i++)
	{
		x.block_vector(i) = m.block<1, 3>(i, 0).transpose();
	}
}
void Matrixmx3ToVector3mx1(const Matrix& m, VectorX& x)
{
	assert(m.cols() == 3);

	for (unsigned int i = 0; i < m.rows(); i++)
	{
		x.block_vector(i) = m.block<1, 3>(i, 0).transpose();
	}
}

void EigenSparseDiagonalToVector(VectorX& dst, const SparseMatrix& src)
{
	assert(src.rows() == src.cols());
	dst.resize(src.rows());

	for (unsigned int i = 0; i != src.rows(); ++i)
	{
		dst(i) = src.coeff(i, i);
	}
}



Simulation::Simulation()
{


	m_lbfgs_queue = NULL;
	ncg_lbfgs_queue = NULL;
	pUBO = 0;
	Result = NULL;
	fixedsize = NULL;

	m_processing_collision = true;

	m_verbose_show_converge = false;
	m_verbose_show_optimization_time = false;
	m_verbose_show_energy = false;
	m_verbose_show_factorization_warning = true;
	m_profile_logging_enabled = true;
m_quality_metrics_enabled = false;
m_quality_checkpoint_stride = 1;
	m_last_profile_used_cs_ncg = false;
	m_last_profile_converged = false;
	m_last_profile_exploded = false;
	m_last_profile_iterations = 0;
	m_last_profile_front_ms = 0.0;
	m_last_profile_transfer_ms = 0.0;
	m_last_profile_cs_y_upload_ms = 0.0;
	m_last_profile_cs_y_to_x_copy_ms = 0.0;
	m_last_profile_cs_x_readback_ms = 0.0;
	m_last_profile_cs_x_readback_wait_ms = 0.0;
	m_last_profile_cs_x_readback_copy_ms = 0.0;
	m_last_profile_iteration_ms = 0.0;
	m_last_profile_optimization_ms = 0.0;
	m_last_profile_back_ms = 0.0;
	m_last_profile_update_posvel_ms = 0.0;
	m_last_profile_position_stats_ms = 0.0;
	m_last_profile_collision_ms = 0.0;
	m_last_profile_total_ms = 0.0;
	m_last_profile_step_size = 0.0;
	m_last_profile_objective_energy = 0.0;
	m_last_profile_gradient_norm = 0.0;
	m_last_profile_max_displacement = 0.0;
	m_last_profile_max_position = 0.0;

	m_cs_gradient_norm_sq = 0.0;
	m_cs_gradient_dot_descent = 0.0;
	m_cs_edge_buffer_dirty = true;
	m_cs_render_position_valid = false;
	m_cs_gpu_state_valid = false;
	m_cs_cpu_state_stale = false;
	m_cs_skip_cpu_damping_once = false;

	m_gradient_shader_file = "./shaders/gradient.comp";
	m_gradient_scatter_shader_file = "./shaders/gradient_scatter.comp";
	m_gradient_finalize_shader_file = "./shaders/gradient_finalize.comp";
	m_energy_shader_file = "./shaders/energy.comp";
	m_objective_shader_file = "./shaders/objective.comp";
	m_energy_for_linesearch_shader_file = "./shaders/energy_for_linesearch.comp";
	m_computeX_shader_file = "./shaders/computeX.comp";
	m_descent_shader_file = "./shaders/descent.comp";
	m_iner_shader_file = "./shaders/iner.comp";
	m_stats_shader_file = "./shaders/stats.comp";
	m_collision_resolve_shader_file = "./shaders/collisionResolve.comp";
	m_normal_from_triangles_shader_file = "./shaders/normalFromTriangles.comp";
	m_cs2_state_shader_file = "./shaders/cs2State.comp";


	use_cs = true;

	if (use_cs)
	{

		set_shader();

		glGenBuffers(1, &edgeID);
		glGenBuffers(1, &gradientID);
		glGenBuffers(1, &xID);
		glGenBuffers(1, &energyID);
		glGenBuffers(1, &fixededgesID);
		glGenBuffers(1, &FlagID);
		glGenBuffers(1, &ResultID);
		glGenBuffers(1, &DescentID);
		glGenBuffers(1, &m_yID);
		glGenBuffers(1, &inerID);
		glGenBuffers(1, &testID);
		glGenBuffers(1, &vertexEdgeOffsetID);
		glGenBuffers(1, &vertexEdgeIndexID);
		glGenBuffers(1, &attachmentID);
		glGenBuffers(1, &collisionVelocityID);
		glGenBuffers(1, &collisionPrimitiveID);
		glGenBuffers(1, &csNormalID);
		glGenBuffers(1, &csPositionID);
		glGenBuffers(1, &csStateStatsID);

	}

	ResetCSBufferStorageTrackers();
}

void Simulation::set_source()
{/*
	gradient_source = {}, energy_source = {}, energy_for_linesearch_source = {}
	, colliEnergy_source = {}, choose_valid_source = {}, choose_final_source = {};*/

}

void Simulation::set_shader()
{
	auto load_shader_source = [](const std::string& path, const char* fallback_source)
	{
		std::vector<std::string> candidate_paths;
		candidate_paths.push_back(path);

		std::string normalized_path = path;
		if (normalized_path.size() > 2 && normalized_path[0] == '.' &&
			(normalized_path[1] == '/' || normalized_path[1] == '\\'))
		{
			normalized_path = normalized_path.substr(2);
		}
		candidate_paths.push_back(normalized_path);
		candidate_paths.push_back("../" + normalized_path);
		candidate_paths.push_back("../../" + normalized_path);

		for (std::vector<std::string>::const_iterator candidate = candidate_paths.begin(); candidate != candidate_paths.end(); ++candidate)
		{
			std::ifstream input(candidate->c_str(), std::ios::binary);
			if (input)
			{
				return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
			}
		}

		return std::string(fallback_source);
	};

	auto compile_compute_program = [](GLuint& shader_handle, GLuint& program_handle, const std::string& source, const char* label)
	{
		shader_handle = glCreateShader(GL_COMPUTE_SHADER);
		const char* source_ptr = source.c_str();
		glShaderSource(shader_handle, 1, &source_ptr, NULL);
		glCompileShader(shader_handle);

		GLint success = 0;
		glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			char infoLog[2048];
			glGetShaderInfoLog(shader_handle, 2048, NULL, infoLog);
			std::cerr << label << " compilation failed:\n" << infoLog << std::endl;
		}

		program_handle = glCreateProgram();
		glAttachShader(program_handle, shader_handle);
		glLinkProgram(program_handle);
		glGetProgramiv(program_handle, GL_LINK_STATUS, &success);
		if (!success)
		{
			char infoLog[2048];
			glGetProgramInfoLog(program_handle, 2048, NULL, infoLog);
			std::cerr << label << " linking failed:\n" << infoLog << std::endl;
		}
	};

	compile_compute_program(gradient_shader, gradient_program, load_shader_source(m_gradient_shader_file, gradient_source), "gradient compute shader");
	compile_compute_program(gradient_scatter_shader, gradient_scatter_program, load_shader_source(m_gradient_scatter_shader_file, ""), "gradient scatter compute shader");
	compile_compute_program(gradient_finalize_shader, gradient_finalize_program, load_shader_source(m_gradient_finalize_shader_file, ""), "gradient finalize compute shader");
	compile_compute_program(iner_shader, iner_program, load_shader_source(m_iner_shader_file, iner_source), "inertia compute shader");
	compile_compute_program(descent_shader, descent_program, load_shader_source(m_descent_shader_file, descent_source), "descent compute shader");
	compile_compute_program(energy_shader, energy_program, load_shader_source(m_energy_shader_file, energy_source), "energy compute shader");
	compile_compute_program(energy_for_linesearch_shader, energy_for_linesearch_program, load_shader_source(m_energy_for_linesearch_shader_file, energy_for_linesearch_source), "line-search energy compute shader");
	compile_compute_program(objective_shader, objective_program, load_shader_source(m_objective_shader_file, objective_source), "objective compute shader");
	compile_compute_program(colliEnergy_shader, colliEnergy_program, load_shader_source(m_stats_shader_file, colliEnergy_source), "stats compute shader");
	compile_compute_program(choose_valid_shader, choose_valid_program, load_shader_source("./shaders/choose_valid.comp", choose_valid_source), "choose-valid compute shader");
	compile_compute_program(choose_final_shader, choose_final_program, load_shader_source("./shaders/choose_final.comp", choose_final_source), "choose-final compute shader");
	compile_compute_program(compute_shader, compute_program, load_shader_source("./shaders/reduce.comp", compute_source), "reduction compute shader");
	compile_compute_program(computeX_shader, computeX_program, load_shader_source(m_computeX_shader_file, computeX_source), "computeX compute shader");
	compile_compute_program(collision_resolve_shader, collision_resolve_program, load_shader_source(m_collision_resolve_shader_file, ""), "collision-resolve compute shader");
	compile_compute_program(normal_from_triangles_shader, normal_from_triangles_program, load_shader_source(m_normal_from_triangles_shader_file, ""), "GPU normal compute shader");
	compile_compute_program(cs2_state_shader, cs2_state_program, load_shader_source(m_cs2_state_shader_file, ""), "CS2 GPU state compute shader");
}

void Simulation::Create_SSBO()
{

}

void Simulation::syncCSParams()
{
	if (!use_cs || !m_mesh)
	{
		return;
	}

	comp_params.t0 = 1.0f;
	comp_params.beta = static_cast<float>(m_ls_beta);
	comp_params.K = 8;
	comp_params.stiffness = static_cast<int>(m_stiffness_stretch);
	comp_params.edge_size = static_cast<int>(m_mesh->my_edge.size());
	comp_params.alpha = static_cast<float>(m_ls_alpha);
	comp_params.m_h = static_cast<float>(m_h);
	comp_params.gradient_size = static_cast<int>(m_mesh->m_system_dimension);
	comp_params.vertex_size = static_cast<int>(m_mesh->m_vertices_number);
	comp_params.attachment_size = 0;
	for (std::vector<Edge>::const_iterator edge_it = m_mesh->my_edge.begin(); edge_it != m_mesh->my_edge.end(); ++edge_it)
	{
		if (edge_it->fixed_point == 1)
		{
			++comp_params.attachment_size;
		}
	}
	comp_params._pad0 = 0;
	comp_params._pad1 = 0;

	if (pUBO == 0)
	{
		glGenBuffers(1, &pUBO);
	}

	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(comp_params), &comp_params, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);
}

void Simulation::rebuildCSAdjacency()
{
	if (!m_mesh)
	{
		return;
	}

	std::vector<Edge> spring_edges;
	spring_edges.reserve(m_mesh->my_edge.size());
	for (std::vector<Edge>::const_iterator edge_it = m_mesh->my_edge.begin(); edge_it != m_mesh->my_edge.end(); ++edge_it)
	{
		if (edge_it->fixed_point != 1)
		{
			spring_edges.push_back(*edge_it);
		}
	}

	unsigned int spring_index = 0;
	unsigned int attachment_count = 0;
	for (std::vector<Constraint*>::iterator constraint_it = m_constraints.begin(); constraint_it != m_constraints.end(); ++constraint_it)
	{
		Constraint* constraint = *constraint_it;
		if (constraint->Type() == CONSTRAINT_TYPE_SPRING || constraint->Type() == CONSTRAINT_TYPE_SPRING_BENDING)
		{
			if (spring_index < spring_edges.size())
			{
				ScalarType stiffness = 0.0;
				constraint->GetMaterialProperty(stiffness);
				spring_edges[spring_index].stiffness = static_cast<float>(stiffness);
				spring_edges[spring_index].fixed_point = 0;
				spring_edges[spring_index]._pad0 = 0.0f;
				spring_edges[spring_index].fixed_ = glm::vec4(0.0f);
			}
			++spring_index;
		}
		else if (constraint->Type() == CONSTRAINT_TYPE_ATTACHMENT)
		{
			++attachment_count;
		}
	}

	m_mesh->my_edge.clear();
	m_mesh->my_edge.reserve(spring_edges.size() + attachment_count);
	for (std::vector<Edge>::const_iterator edge_it = spring_edges.begin(); edge_it != spring_edges.end(); ++edge_it)
	{
		m_mesh->my_edge.push_back(*edge_it);
	}

	for (std::vector<Constraint*>::iterator constraint_it = m_constraints.begin(); constraint_it != m_constraints.end(); ++constraint_it)
	{
		AttachmentConstraint* attachment_constraint = dynamic_cast<AttachmentConstraint*>(*constraint_it);
		if (!attachment_constraint)
		{
			continue;
		}

		ScalarType stiffness = 0.0;
		attachment_constraint->GetMaterialProperty(stiffness);
		EigenVector3 fixed_point = attachment_constraint->GetFixedPoint();

		Edge attachment_edge = {};
		attachment_edge.m_v1 = attachment_constraint->GetConstrainedVertexIndex();
		attachment_edge.m_v2 = attachment_edge.m_v1;
		attachment_edge.rest_length = 0.0f;
		attachment_edge.stiffness = static_cast<float>(stiffness);
		attachment_edge.fixed_point = 1;
		attachment_edge._pad0 = 0.0f;
		attachment_edge.fixed_ = glm::vec4(
			static_cast<float>(fixed_point[0]),
			static_cast<float>(fixed_point[1]),
			static_cast<float>(fixed_point[2]),
			0.0f);
		m_mesh->my_edge.push_back(attachment_edge);
	}

	const unsigned int vertex_count = static_cast<unsigned int>(m_mesh->m_vertices_number);
	m_cs_vertex_edge_offsets.assign(vertex_count + 1, 0u);

	unsigned int adjacency_count = 0;
	for (unsigned int edge_index = 0; edge_index < m_mesh->my_edge.size(); ++edge_index)
	{
		const Edge& edge = m_mesh->my_edge[edge_index];
		++m_cs_vertex_edge_offsets[edge.m_v1 + 1];
		++adjacency_count;
		if (edge.fixed_point != 1 && edge.m_v2 != edge.m_v1)
		{
			++m_cs_vertex_edge_offsets[edge.m_v2 + 1];
			++adjacency_count;
		}
	}

	for (unsigned int vertex_index = 1; vertex_index < m_cs_vertex_edge_offsets.size(); ++vertex_index)
	{
		m_cs_vertex_edge_offsets[vertex_index] += m_cs_vertex_edge_offsets[vertex_index - 1];
	}

	m_cs_vertex_edge_indices.assign(adjacency_count, 0u);
	std::vector<unsigned int> cursor = m_cs_vertex_edge_offsets;
	for (unsigned int edge_index = 0; edge_index < m_mesh->my_edge.size(); ++edge_index)
	{
		const Edge& edge = m_mesh->my_edge[edge_index];
		m_cs_vertex_edge_indices[cursor[edge.m_v1]++] = edge_index;
		if (edge.fixed_point != 1 && edge.m_v2 != edge.m_v1)
		{
			m_cs_vertex_edge_indices[cursor[edge.m_v2]++] = edge_index;
		}
	}

	m_cs_mass_diagonal.resize(m_mesh->m_system_dimension);
	for (unsigned int i = 0; i < m_cs_mass_diagonal.size(); ++i)
	{
		m_cs_mass_diagonal[i] = m_mesh->m_mass_matrix.coeff(i, i);
	}
}

void Simulation::uploadCSResourcesIfNeeded()
{
	if (!use_cs || !m_mesh)
	{
		return;
	}

	if (m_cs_edge_buffer_dirty)
	{
		rebuildCSAdjacency();

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeID);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			m_mesh->my_edge.size() * sizeof(Edge),
			m_mesh->my_edge.empty() ? NULL : m_mesh->my_edge.data(),
			GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, edgeID);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexEdgeOffsetID);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			m_cs_vertex_edge_offsets.size() * sizeof(unsigned int),
			m_cs_vertex_edge_offsets.empty() ? NULL : m_cs_vertex_edge_offsets.data(),
			GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, vertexEdgeOffsetID);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexEdgeIndexID);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			m_cs_vertex_edge_indices.size() * sizeof(unsigned int),
			m_cs_vertex_edge_indices.empty() ? NULL : m_cs_vertex_edge_indices.data(),
			GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, vertexEdgeIndexID);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, attachmentID);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			m_cs_mass_diagonal.size() * sizeof(ScalarType),
			m_cs_mass_diagonal.empty() ? NULL : m_cs_mass_diagonal.data(),
			GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, attachmentID);

		syncCSParams();
		m_cs_edge_buffer_dirty = false;
	}
	else
	{
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, edgeID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, vertexEdgeOffsetID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, vertexEdgeIndexID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, attachmentID);
	}
}

void Simulation::uploadCSCollisionPrimitives()
{
	m_cs_collision_primitives.clear();
	if (!m_scene)
	{
		return;
	}

	const std::vector<Primitive*>& primitives = m_scene->GetPrimitives();
	for (std::vector<Primitive*>::const_iterator primitive_it = primitives.begin(); primitive_it != primitives.end(); ++primitive_it)
	{
		Primitive* primitive = *primitive_it;
		if (!primitive)
		{
			continue;
		}

		CollisionPrimitiveGPU primitive_gpu = {};
		switch (primitive->type())
		{
		case PLANE:
		{
			Plane* plane = dynamic_cast<Plane*>(primitive);
			if (!plane)
			{
				continue;
			}

			primitive_gpu.type = static_cast<int>(PLANE);
			const glm::vec3& normal = plane->Normal();
			primitive_gpu.data0[0] = normal.x;
			primitive_gpu.data0[1] = normal.y;
			primitive_gpu.data0[2] = normal.z;
			primitive_gpu.data0[3] = primitive->m_pos.y;
			break;
		}
		case SPHERE:
		{
			Sphere* sphere = dynamic_cast<Sphere*>(primitive);
			if (!sphere)
			{
				continue;
			}

			primitive_gpu.type = static_cast<int>(SPHERE);
			primitive_gpu.data0[0] = primitive->m_pos.x;
			primitive_gpu.data0[1] = primitive->m_pos.y;
			primitive_gpu.data0[2] = primitive->m_pos.z;
			primitive_gpu.data0[3] = sphere->Radius();
			break;
		}
		case TORUS:
		{
			Torus* torus = dynamic_cast<Torus*>(primitive);
			if (!torus)
			{
				continue;
			}

			primitive_gpu.type = static_cast<int>(TORUS);
			primitive_gpu.data0[0] = primitive->m_pos.x;
			primitive_gpu.data0[1] = primitive->m_pos.y;
			primitive_gpu.data0[2] = primitive->m_pos.z;
			primitive_gpu.data0[3] = torus->MajorRadius();
			primitive_gpu.data1[0] = torus->MinorRadius();
			break;
		}
		default:
			continue;
		}

		m_cs_collision_primitives.push_back(primitive_gpu);
	}

	if (m_cs_collision_primitives.empty())
	{
		return;
	}

	const std::size_t primitive_buffer_bytes = m_cs_collision_primitives.size() * sizeof(CollisionPrimitiveGPU);
	EnsureCSBufferStorage(collisionPrimitiveID, primitive_buffer_bytes, g_cs_collision_primitive_buffer_bytes);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionPrimitiveID);
	glBufferSubData(
		GL_SHADER_STORAGE_BUFFER,
		0,
		primitive_buffer_bytes,
		m_cs_collision_primitives.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, collisionPrimitiveID);
}

void Simulation::dispatchCSGradient(bool pure_constraint_only, bool profile_gradient_query)
{
	const bool use_edge_scatter = GenPDExperimentUsesEdgeScatter();
	const bool fuse_gradient_stats = !use_edge_scatter && GenPDExperimentUsesFusedGradientStats();
	ScopedCSDebugGroup debug_group(
		use_edge_scatter ? "GenPD gradient edge scatter" :
		(fuse_gradient_stats ? "GenPD gradient gather fusion" : "GenPD gradient gather"));
	if (!use_cs || !m_mesh)
	{
		return;
	}

	uploadCSResourcesIfNeeded();

	const GLuint vertex_group_count = ComputeCSGradientGroupCount(static_cast<std::size_t>(m_mesh->m_vertices_number));
	if (!pure_constraint_only)
	{
		EnsureFloatScratchBuffer(testID, test_, static_cast<std::size_t>(2u) * vertex_group_count);
	}

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);
	const bool profile_gradient = (profile_gradient_query || GenPDProfileGpuQueriesEnabled()) && !pure_constraint_only;
	if (profile_gradient)
	{
		BeginCSGpuTimer(g_cs_profile_gradient_query);
	}

	if (use_edge_scatter)
	{
		const float zero_gradient = 0.0f;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
		glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zero_gradient);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, edgeID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradientID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
		glUseProgram(gradient_scatter_program);
		const GLuint constraint_group_count = ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.edge_size));
		++g_cs_profile_gradient_dispatches;
		glDispatchCompute(constraint_group_count, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		if (!pure_constraint_only)
		{
			ScopedCSDebugGroup finalize_group("GenPD gradient scatter finalize");
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_yID);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, attachmentID);
			glUseProgram(gradient_finalize_program);
			++g_cs_profile_gradient_dispatches;
			glDispatchCompute(vertex_group_count, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}
	else
	{
		glUseProgram(gradient_program);
		static GLint pure_constraint_location = -1;
		static GLint write_stats_location = -1;
		if (pure_constraint_location < 0)
		{
			pure_constraint_location = glGetUniformLocation(gradient_program, "pure_constraint_only");
			write_stats_location = glGetUniformLocation(gradient_program, "write_stats");
		}
		glUniform1i(pure_constraint_location, pure_constraint_only ? 1 : 0);
		glUniform1i(write_stats_location, (!pure_constraint_only && fuse_gradient_stats) ? 1 : 0);
		++g_cs_profile_gradient_dispatches;
		glDispatchCompute(vertex_group_count, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	if (profile_gradient)
	{
		EndCSGpuTimer(g_cs_profile_gradient_query_pending);
	}

	if (!pure_constraint_only && !fuse_gradient_stats)
	{
		ScopedCSDebugGroup stats_group("GenPD gradient stats pass");
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradientID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, DescentID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, testID);
		glUseProgram(colliEnergy_program);
		++g_cs_profile_stats_dispatches;
		glDispatchCompute(vertex_group_count, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}
ScalarType Simulation::readCSStatsFromGPU(bool profile_readback)
{
	ScalarType stats[2] = { 0, 0 };
	++g_cs_profile_host_readbacks;
	TimerWrapper t_stats_readback;
	if (profile_readback)
	{
		t_stats_readback.Tic();
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, fixededgesID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(stats), stats);
	ScalarType stats_readback_ms = 0.0;
	if (profile_readback)
	{
		t_stats_readback.Toc();
		stats_readback_ms = t_stats_readback.DurationInSeconds() * 1000.0;
		g_cs_profile_stats_readback_ms += stats_readback_ms;
	}
	m_cs_gradient_norm_sq = stats[0];
	m_cs_gradient_dot_descent = stats[1];
	return stats_readback_ms;
}

void Simulation::updateCSStats(bool readback)
{
	ScopedCSDebugGroup debug_group("GenPD stats reduction");
	if (!use_cs || !m_mesh)
	{
		return;
	}

	const GLuint partial_group_count = ComputeCSGradientGroupCount(static_cast<std::size_t>(m_mesh->m_vertices_number));

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);

	const bool profile_stats = readback || GenPDProfileGpuQueriesEnabled();

	DispatchCSReduction(compute_program, testID, fixededgesID, partial_group_count, 2u, profile_stats, profile_stats);
	if (!profile_stats)
	{
		return;
	}

	const ScalarType stats_readback_ms = readCSStatsFromGPU(true);
	const ScalarType gradient_gpu_ms = ConsumeCSGpuTimerMs(g_cs_profile_gradient_query, g_cs_profile_gradient_query_pending);
	const ScalarType reduction_gpu_ms = ConsumeCSGpuTimerMs(g_cs_profile_reduction_query, g_cs_profile_reduction_query_pending);
	g_cs_profile_gradient_gpu_ms += gradient_gpu_ms;
	g_cs_profile_reduction_gpu_ms += reduction_gpu_ms;
	if (m_verbose_show_optimization_time)
	{
		printf("t_gradient_gpu, time elapse: %.3f milliseconds.\n", static_cast<double>(gradient_gpu_ms));
		printf("t_reduction_gpu, time elapse: %.3f milliseconds.\n", static_cast<double>(reduction_gpu_ms));
		printf("t_stats_readback, time elapse: %.3f milliseconds.\n", static_cast<double>(stats_readback_ms));
	}
}

void Simulation::collisionPostProcessCS(VectorX& x, VectorX& v)
{
	if (!m_processing_collision || !m_scene || m_scene->IsEmpty())
	{
		return;
	}

	if (!use_cs)
	{
		VectorX penetration = collisionDetectionPostProcessing(x);
		collisionResolution(penetration, x, v);
		return;
	}

	uploadCSCollisionPrimitives();
	if (m_cs_collision_primitives.empty())
	{
		return;
	}

	const std::size_t vector_buffer_bytes = x.size() * sizeof(ScalarType);
	EnsureCSBufferStorage(xID, vector_buffer_bytes, g_cs_x_buffer_bytes);
	EnsureCSBufferStorage(collisionVelocityID, vector_buffer_bytes, g_cs_collision_velocity_buffer_bytes);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, x.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionVelocityID);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, v.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, collisionVelocityID);

	glUseProgram(collision_resolve_program);
	static GLint vertex_count_location = -1;
	static GLint primitive_count_location = -1;
	static GLint restitution_location = -1;
	static GLint friction_location = -1;
	if (vertex_count_location < 0)
	{
		vertex_count_location = glGetUniformLocation(collision_resolve_program, "vertex_count");
		primitive_count_location = glGetUniformLocation(collision_resolve_program, "primitive_count");
		restitution_location = glGetUniformLocation(collision_resolve_program, "restitution_coefficient");
		friction_location = glGetUniformLocation(collision_resolve_program, "friction_coefficient");
	}

	glUniform1ui(vertex_count_location, static_cast<GLuint>(m_mesh->m_vertices_number));
	glUniform1ui(primitive_count_location, static_cast<GLuint>(m_cs_collision_primitives.size()));
	glUniform1f(restitution_location, static_cast<float>(m_restitution_coefficient));
	glUniform1f(friction_location, static_cast<float>(m_friction_coefficient));
	glDispatchCompute((m_mesh->m_vertices_number + 255) / 256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, x.data());

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionVelocityID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, v.data());

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

GLuint Simulation::CS2RenderPositionBuffer() const
{
	return xID;
}

GLuint Simulation::CS2RenderNormalBuffer() const
{
	return csNormalID;
}

bool Simulation::PrepareCS2RenderBuffers()
{
	if (!use_cs || !m_cs_render_position_valid || !m_mesh)
	{
		return false;
	}

	if (m_mesh->m_mesh_type != MESH_TYPE_CLOTH || m_mesh->m_vertices_number == 0)
	{
		return false;
	}

	const unsigned int vertex_count = m_mesh->m_vertices_number;
	if (m_mesh->m_dim[0] == 0 || m_mesh->m_dim[1] == 0 ||
		m_mesh->m_dim[0] * m_mesh->m_dim[1] != vertex_count)
	{
		return false;
	}

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 19, xID);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	return true;
}
bool Simulation::shouldUseCS2GpuState()
{
	if (!GenPDExperimentUsesPersistentBuffers() || !use_cs || !m_mesh || cs2_state_program == 0)
	{
		return false;
	}

	if (m_optimization_method != OPTIMIZATION_METHOD_NCG || m_integration_method != INTEGRATION_IMPLICIT_EULER)
	{
		return false;
	}

	if (m_sub_stepping != 1 || sizeof(ScalarType) != sizeof(float))
	{
		return false;
	}

	if (m_mesh->m_mesh_type != MESH_TYPE_CLOTH || m_mesh->m_vertices_number < kCSLargeClothVertexThreshold)
	{
		return false;
	}

	if (m_step_mode || m_animation_enable_swinging || m_selected_handle_id >= 0 || m_selected_attachment_constraint != NULL)
	{
		return false;
	}

	if (m_processing_collision && m_scene && !m_scene->IsEmpty())
	{
		return false;
	}

	return true;
}

void Simulation::initializeCS2GpuStateIfNeeded()
{
	if (!shouldUseCS2GpuState() || m_cs_gpu_state_valid)
	{
		return;
	}

	const std::size_t vector_buffer_bytes = static_cast<std::size_t>(m_mesh->m_system_dimension) * sizeof(ScalarType);
	EnsureCSBufferStorage(csPositionID, vector_buffer_bytes, g_cs_state_position_buffer_bytes);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, csPositionID);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, m_mesh->m_current_positions.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, csPositionID);

	EnsureCSBufferStorage(collisionVelocityID, vector_buffer_bytes, g_cs_collision_velocity_buffer_bytes);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionVelocityID);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, m_mesh->m_current_velocities.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, collisionVelocityID);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	m_cs_gpu_state_valid = true;
	m_cs_cpu_state_stale = false;
	m_cs_skip_cpu_damping_once = false;
}

bool Simulation::predictCS2GpuStateY()
{
	if (!shouldUseCS2GpuState())
	{
		return false;
	}

	initializeCS2GpuStateIfNeeded();
	if (!m_cs_gpu_state_valid)
	{
		return false;
	}

	const std::size_t vector_buffer_bytes = static_cast<std::size_t>(m_mesh->m_system_dimension) * sizeof(ScalarType);
	EnsureCSBufferStorage(xID, vector_buffer_bytes, g_cs_x_buffer_bytes);
	EnsureCSBufferStorage(m_yID, vector_buffer_bytes, g_cs_y_buffer_bytes);
	EnsureCSBufferStorage(csPositionID, vector_buffer_bytes, g_cs_state_position_buffer_bytes);
	EnsureCSBufferStorage(collisionVelocityID, vector_buffer_bytes, g_cs_collision_velocity_buffer_bytes);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_yID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, collisionVelocityID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, csPositionID);

	glUseProgram(cs2_state_program);
	static GLint mode_location = -1;
	static GLint vertex_count_location = -1;
	static GLint timestep_location = -1;
	static GLint damping_location = -1;
	static GLint acceleration_location = -1;
	if (mode_location < 0)
	{
		mode_location = glGetUniformLocation(cs2_state_program, "update_mode");
		vertex_count_location = glGetUniformLocation(cs2_state_program, "vertex_count");
		timestep_location = glGetUniformLocation(cs2_state_program, "timestep");
		damping_location = glGetUniformLocation(cs2_state_program, "damping_coefficient");
		acceleration_location = glGetUniformLocation(cs2_state_program, "acceleration");
	}
	glUniform1i(mode_location, 1);
	glUniform1ui(vertex_count_location, static_cast<GLuint>(m_mesh->m_vertices_number));
	glUniform1f(timestep_location, static_cast<float>(m_h));
	glUniform1f(damping_location, static_cast<float>(m_damping_coefficient));
	glUniform3f(
		acceleration_location,
		static_cast<float>(m_wind_x),
		static_cast<float>(-m_gravity_constant + m_wind_y),
		static_cast<float>(m_wind_z));
	glDispatchCompute((m_mesh->m_vertices_number + 255) / 256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	return true;
}

bool Simulation::finalizeCS2GpuState(ScalarType& max_position, ScalarType& max_displacement, bool& x_is_finite)
{
	max_position = 0.0;
	max_displacement = 0.0;
	x_is_finite = false;

	if (!shouldUseCS2GpuState() || !m_cs_gpu_state_valid)
	{
		return false;
	}

	const GLuint group_count = (m_mesh->m_vertices_number + 255) / 256;

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, collisionVelocityID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, csPositionID);

	glUseProgram(cs2_state_program);
	static GLint mode_location = -1;
	static GLint vertex_count_location = -1;
	static GLint timestep_location = -1;
	static GLint damping_location = -1;
	static GLint acceleration_location = -1;
	if (mode_location < 0)
	{
		mode_location = glGetUniformLocation(cs2_state_program, "update_mode");
		vertex_count_location = glGetUniformLocation(cs2_state_program, "vertex_count");
		timestep_location = glGetUniformLocation(cs2_state_program, "timestep");
		damping_location = glGetUniformLocation(cs2_state_program, "damping_coefficient");
		acceleration_location = glGetUniformLocation(cs2_state_program, "acceleration");
	}
	glUniform1i(mode_location, 3);
	glUniform1ui(vertex_count_location, static_cast<GLuint>(m_mesh->m_vertices_number));
	glUniform1f(timestep_location, static_cast<float>(m_h));
	glUniform1f(damping_location, static_cast<float>(m_damping_coefficient));
	glUniform3f(acceleration_location, 0.0f, 0.0f, 0.0f);
	glDispatchCompute(group_count, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

	x_is_finite = true;

	m_cs_cpu_state_stale = true;
	m_cs_skip_cpu_damping_once = true;
	m_cs_render_position_valid = x_is_finite;
	return true;
}

void Simulation::syncCS2GpuStateToCPU()
{
	if (!m_mesh)
	{
		return;
	}

	if (!m_cs_gpu_state_valid || !m_cs_cpu_state_stale)
	{
		m_cs_cpu_state_stale = false;
		m_cs_skip_cpu_damping_once = false;
		return;
	}

	const std::size_t vector_buffer_bytes = static_cast<std::size_t>(m_mesh->m_system_dimension) * sizeof(ScalarType);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, csPositionID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, m_mesh->m_current_positions.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionVelocityID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vector_buffer_bytes, m_mesh->m_current_velocities.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	m_mesh->m_previous_positions = m_mesh->m_current_positions;
	m_mesh->m_previous_velocities = m_mesh->m_current_velocities;
	m_cs_cpu_state_stale = false;
	m_cs_skip_cpu_damping_once = false;
	m_cs_render_position_valid = false;
}

void Simulation::invalidateCS2GpuState()
{
	m_cs_gpu_state_valid = false;
	m_cs_cpu_state_stale = false;
	m_cs_skip_cpu_damping_once = false;
	m_cs_render_position_valid = false;
}
Simulation::~Simulation()
{
	clearConstraints();
	m_handles.clear();
	m_handle_id.clear();


	glDeleteShader(gradient_shader);
	glDeleteProgram(gradient_program);

	glDeleteShader(gradient_scatter_shader);
	glDeleteProgram(gradient_scatter_program);

	glDeleteShader(gradient_finalize_shader);
	glDeleteProgram(gradient_finalize_program);

	glDeleteShader(energy_shader);
	glDeleteProgram(energy_program);

	glDeleteShader(energy_for_linesearch_shader);
	glDeleteProgram(energy_for_linesearch_program);

	glDeleteShader(objective_shader);
	glDeleteProgram(objective_program);

	glDeleteShader(colliEnergy_shader);
	glDeleteProgram(colliEnergy_program);

	glDeleteShader(choose_valid_shader);
	glDeleteProgram(choose_valid_program);

	glDeleteShader(choose_final_shader);
	glDeleteProgram(choose_final_program);

	glDeleteShader(compute_shader);
	glDeleteProgram(compute_program);

	glDeleteShader(computeX_shader);
	glDeleteProgram(computeX_program);

	glDeleteShader(collision_resolve_shader);
	glDeleteProgram(collision_resolve_program);

	glDeleteShader(normal_from_triangles_shader);
	glDeleteProgram(normal_from_triangles_program);

	glDeleteShader(cs2_state_shader);
	glDeleteProgram(cs2_state_program);

	glDeleteShader(descent_shader);
	glDeleteProgram(descent_program);

	glDeleteShader(iner_shader);
	glDeleteProgram(iner_program);

	glDeleteBuffers(1, &edgeID);
	glDeleteBuffers(1, &gradientID);
	glDeleteBuffers(1, &xID);
	glDeleteBuffers(1, &energyID);
	glDeleteBuffers(1, &fixededgesID);
	glDeleteBuffers(1, &FlagID);
	glDeleteBuffers(1, &ResultID);
	glDeleteBuffers(1, &DescentID);
	glDeleteBuffers(1, &m_yID);
	glDeleteBuffers(1, &inerID);
	glDeleteBuffers(1, &testID);
	glDeleteBuffers(1, &vertexEdgeOffsetID);
	glDeleteBuffers(1, &vertexEdgeIndexID);
	glDeleteBuffers(1, &attachmentID);
	glDeleteBuffers(1, &collisionVelocityID);
	glDeleteBuffers(1, &collisionPrimitiveID);
	glDeleteBuffers(1, &csNormalID);
	glDeleteBuffers(1, &csPositionID);
	glDeleteBuffers(1, &csStateStatsID);
	if (pUBO != 0)
	{
		glDeleteBuffers(1, &pUBO);
	}

	DeleteVisualizationMesh();
}

void Simulation::Reset()
{


	m_y.resize(m_mesh->m_system_dimension);
	m_external_force.resize(m_mesh->m_system_dimension);

	// handles
	if (!m_handles.empty())
		m_handles.clear();
	m_handle_id.resize(m_mesh->m_vertices_number);
	for (unsigned int i = 0; i != m_handle_id.size(); ++i)
	{
		m_handle_id[i] = -1;
	}
	m_selected_handle_id = -1;

	m_mesh->m_expanded_system_dimension = 0;
	m_mesh->m_expanded_system_dimension_1d = 0;

	setupConstraints();
	SetMaterialProperty();

	// compute shader
	if (use_cs)
	{
		ResetCSBufferStorageTrackers();

		ScalarType zero_values[8] = { 0 };
		int zero_flags[8] = { 0 };
		ScalarType zero_stats[8] = { 0 };

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, energyID);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_values), zero_values, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, energyID);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, inerID);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_values), zero_values, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, FlagID);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_flags), zero_flags, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, FlagID);

		UploadCSLineSearchResult(ResultID, m_ls_step_size, 0, 1, 0.0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, fixededgesID);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_stats), zero_stats, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);

		gradient_dir.resize(m_mesh->m_system_dimension);
		gradient_dir.setZero();

		m_cs_gradient_norm_sq = 0.0;
		m_cs_gradient_dot_descent = 0.0;
		m_cs_edge_buffer_dirty = true;
		invalidateCS2GpuState();
		uploadCSResourcesIfNeeded();
	}

	m_selected_attachment_constraint = NULL;
	m_step_mode = false;

	// lbfgs
	m_lbfgs_restart_every_frame = true;
	m_lbfgs_need_update_H0 = true;

	// solver type
	m_solver_type = SOLVER_TYPE_DIRECT_LLT;
	m_iterative_solver_max_iteration = 10;

	// volume
	m_restshape_volume = getVolume(m_mesh->m_current_positions);
	m_current_volume = m_restshape_volume;

	// animation
	m_keyframe_handle_unit_translation_total_segments = 0;
	m_keyframe_handle_unit_rotation_total_segments = 0;

	// partial material property editting
	m_selected_constraints.clear();

	// collision
	m_collision_constraints.clear();

#ifdef OUTPUT_LS_ITERATIONS
	g_total_ls_iterations = 0;
	g_total_iterations = 0;
#endif
}

void Simulation::UpdateAnimation(const int fn)
{
	if (m_animation_enable_swinging)
	{
		int swing_num = 0;
		ScalarType swing_step = m_animation_swing_amp / m_animation_swing_half_period;
		EigenVector3 swing_dir(m_animation_swing_dir[0], m_animation_swing_dir[1], m_animation_swing_dir[2]);
		int positive_direction = ((fn / m_animation_swing_half_period) % 2) ? -1 : 1;
		for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
		{
			AttachmentConstraint* ac;
			if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
			{
				EigenVector3 new_fixed_point = ac->GetFixedPoint() + swing_dir * swing_step * positive_direction;
				ac->SetFixedPoint(new_fixed_point);
				m_cs_edge_buffer_dirty = true;
				if (++swing_num >= m_animation_swing_num)
				{
					break;
				}
			}
		}
	}
}

void Simulation::Update()
{

	/*TimerWrapper my_ti;
	my_ti.Tic();*/



	const bool use_cs_gpu_state_for_update = shouldUseCS2GpuState();

	// update external force
	if (!use_cs_gpu_state_for_update)
	{
		calculateExternalForce();
	}

	ScalarType old_h = m_h;
	m_h = m_h / m_sub_stepping;

	m_last_descent_dir.resize(m_mesh->m_system_dimension);
	m_last_descent_dir.setZero();

	for (unsigned int substepping_i = 0; substepping_i != m_sub_stepping; substepping_i++)
	{
		const bool use_cs_gpu_state = use_cs_gpu_state_for_update;
		if (m_cs_cpu_state_stale && !use_cs_gpu_state)
		{
			syncCS2GpuStateToCPU();
		}

		// update inertia term
		if (!use_cs_gpu_state)
		{
			computeConstantVectorsYandZ();
		}

		// update
		switch (m_integration_method)
		{
		case INTEGRATION_QUASI_STATICS:
		case INTEGRATION_IMPLICIT_EULER:
		case INTEGRATION_IMPLICIT_BDF2:
		case INTEGRATION_IMPLICIT_MIDPOINT:
		case INTEGRATION_IMPLICIT_NEWMARK_BETA:
			integrateImplicitMethod();
			break;
		}

		// damping
		if (m_cs_skip_cpu_damping_once)
		{
			m_cs_skip_cpu_damping_once = false;
		}
		else
		{
			dampVelocity();
		}
	}
	//???????α???
	//Evaluatespringlength(m_mesh->m_current_positions);

	//// volume
	//m_current_volume = getVolume(m_mesh->m_current_positions);

	if (m_verbose_show_energy)
	{
		if (m_integration_method == INTEGRATION_QUASI_STATICS)
		{
			ScalarType W = evaluatePotentialEnergy(m_mesh->m_current_positions);
			std::cout << "Potential Energy = " << W << std::endl;


		}
		else
		{
			std::string filePath = GenPDResolveOutputPath("energy\\data.txt");
			//std::string filePath_0 = GenPDResolveOutputPath("energy\\G.txt");
			std::string filePath_1 = GenPDResolveOutputPath("energy\\K.txt");
			std::string filePath_2 = GenPDResolveOutputPath("energy\\P.txt");

			GenPDEnsureDirectoryForFile(filePath);
			std::ofstream outFile(filePath, std::ios::out | std::ios::app);
			//std::ofstream outFile_0(filePath_0, std::ios::out | std::ios::app);
			GenPDEnsureDirectoryForFile(filePath_1);
			std::ofstream outFile_1(filePath_1, std::ios::out | std::ios::app);
			GenPDEnsureDirectoryForFile(filePath_2);
			std::ofstream outFile_2(filePath_2, std::ios::out | std::ios::app);

			// output total energy;
			ScalarType K = evaluateKineticEnergy(m_mesh->m_current_velocities);
			ScalarType W = evaluatePotentialEnergy(m_mesh->m_current_positions);
			ScalarType K_plus_W = K + W;
			//ScalarType sum=evaluateTotalEnergy(m_mesh->m_current_positions, m_mesh->m_current_velocities);
			std::cout << "Kinetic Energy = " << K << std::endl;
			std::cout << "Potential Energy = " << W << std::endl;
			std::cout << "Total Energy = " << K_plus_W << std::endl;

			if (!outFile.is_open() || !outFile_1.is_open() || outFile_2.is_open()) {
				std::cerr << "Failed to open file for writing." << std::endl;
			}

			// д??????
			outFile << K_plus_W << std::endl;
			outFile_1 << K << std::endl;
			outFile_2 << W << std::endl;
			// ??????
			outFile.close();
			outFile_1.close();
			outFile_2.close();
		}
	}
	m_h = old_h;


	/*my_ti.Toc();
	my_ti.Report("time :");*/
}

void Simulation::Draw(const VBO& vbos)
{
	//// draw attachment constraints
	//for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	//{
	//	(*it)->Draw(vbos);
	//}
	// draw handles
	for (std::vector<Handle>::iterator it = m_handles.begin(); it != m_handles.end(); ++it)
	{
		it->Draw(vbos, (it->ID() == m_selected_handle_id));
	}
}

void Simulation::ConfigureQualityMetrics(const std::string& reference_export_dir, const std::string& quality_reference_dir, unsigned int checkpoint_stride)
{
    m_reference_export_dir = reference_export_dir;
    m_quality_reference_dir = quality_reference_dir;
    m_quality_checkpoint_stride = std::max(1u, checkpoint_stride);
    m_quality_metrics_enabled = !m_reference_export_dir.empty() || !m_quality_reference_dir.empty();
    if (!m_reference_export_dir.empty() && !GenPDEnsureDirectory(m_reference_export_dir))
    {
        std::cerr << "Warning: cannot create reference checkpoint directory: " << m_reference_export_dir << std::endl;
        m_reference_export_dir.clear();
        m_quality_metrics_enabled = !m_quality_reference_dir.empty();
    }
}

void Simulation::LogFrameProfile(unsigned int frame, ScalarType fps_average, ScalarType fps_instant)
{
	if (!m_profile_logging_enabled)
	{
		return;
	}

	static bool initialized = false;
	static std::ofstream profile_file;
	static std::ofstream experiment_profile_file;
static std::ofstream quality_profile_file;
	if (!initialized)
	{
		const std::string profile_path = GenPDResolveOutputPath("frame_profile.csv");
		GenPDEnsureDirectoryForFile(profile_path);
		profile_file.open(profile_path.c_str(), std::ios::out | std::ios::trunc);

		if (profile_file.is_open())
		{
			profile_file << "frame,fps_avg,fps_inst,use_cs_ncg,converged,exploded,iterations,total_ms,front_ms,transfer_ms,cs_y_upload_ms,cs_y_to_x_copy_ms,cs_x_readback_ms,cs_x_readback_wait_ms,cs_x_readback_copy_ms,iteration_ms,optimization_ms,back_ms,update_posvel_ms,position_stats_ms,collision_ms,step_size,objective_energy,gradient_norm,max_displacement,max_position,cs_linesearch_ms,cs_gradstats_ms,cs_gradient_gpu_ms,cs_reduction_gpu_ms,cs_stats_readback_ms,cs_xupdate_ms,cs_xupdate_gpu_ms,cs_descent_ms,cs_descent_gpu_ms,cs_full_ls,cs_skip_ls,cs_unit_accepts\n";
			profile_file.flush();
		}
		const std::string experiment_profile_path = GenPDResolveOutputPath("frame_profile_experiment.csv");
		GenPDEnsureDirectoryForFile(experiment_profile_path);
		experiment_profile_file.open(experiment_profile_path.c_str(), std::ios::out | std::ios::trunc);
		if (experiment_profile_file.is_open())
		{
			experiment_profile_file << "frame,solver_variant,persistent_buffers_active,gradient_dispatches,stats_dispatches,reduction_dispatches,xupdate_dispatches,descent_dispatches,full_linesearch_calls,skipped_linesearch_calls,host_readbacks,solver_gl_finish_calls,tracked_buffer_bytes,gradient_buffer_bytes,descent_buffer_bytes,x_buffer_bytes,y_buffer_bytes,scratch_buffer_bytes,state_position_buffer_bytes\n";
			experiment_profile_file.flush();
		}
		if (m_quality_metrics_enabled)
{
    const std::string quality_profile_path = GenPDResolveOutputPath("quality_metrics.csv");
    GenPDEnsureDirectoryForFile(quality_profile_path);
    quality_profile_file.open(quality_profile_path.c_str(), std::ios::out | std::ios::trunc);
    if (quality_profile_file.is_open())
    {
        quality_profile_file << "frame,has_reference,finite,exploded,position_rel_l2,velocity_rel_l2,constraint_energy,reference_constraint_energy,constraint_energy_rel_error,mean_stretch_strain,reference_mean_stretch_strain,max_stretch_strain,reference_max_stretch_strain,max_penetration_depth,reference_max_penetration_depth\n";
        quality_profile_file.flush();
    }
}
initialized = true;
	}

	if (!profile_file.is_open())
	{
		return;
	}

	profile_file << frame << ","
		<< fps_average << ","
		<< fps_instant << ","
		<< (m_last_profile_used_cs_ncg ? 1 : 0) << ","
		<< (m_last_profile_converged ? 1 : 0) << ","
		<< (m_last_profile_exploded ? 1 : 0) << ","
		<< m_last_profile_iterations << ","
		<< m_last_profile_total_ms << ","
		<< m_last_profile_front_ms << ","
		<< m_last_profile_transfer_ms << ","
		<< m_last_profile_cs_y_upload_ms << ","
		<< m_last_profile_cs_y_to_x_copy_ms << ","
		<< m_last_profile_cs_x_readback_ms << ","
		<< m_last_profile_cs_x_readback_wait_ms << ","
		<< m_last_profile_cs_x_readback_copy_ms << ","
		<< m_last_profile_iteration_ms << ","
		<< m_last_profile_optimization_ms << ","
		<< m_last_profile_back_ms << ","
		<< m_last_profile_update_posvel_ms << ","
		<< m_last_profile_position_stats_ms << ","
		<< m_last_profile_collision_ms << ","
		<< m_last_profile_step_size << ","
		<< m_last_profile_objective_energy << ","
		<< m_last_profile_gradient_norm << ","
		<< m_last_profile_max_displacement << ","
		<< m_last_profile_max_position << ","
		<< g_cs_profile_linesearch_ms << ","
		<< g_cs_profile_gradstats_ms << ","
		<< g_cs_profile_gradient_gpu_ms << ","
		<< g_cs_profile_reduction_gpu_ms << ","
		<< g_cs_profile_stats_readback_ms << ","
		<< g_cs_profile_xupdate_ms << ","
		<< g_cs_profile_xupdate_gpu_ms << ","
		<< g_cs_profile_descent_ms << ","
		<< g_cs_profile_descent_gpu_ms << ","
		<< g_cs_profile_full_linesearch_calls << ","
		<< g_cs_profile_skipped_linesearch_calls << ","
		<< g_cs_profile_unit_step_accepts << "\n";
	profile_file.flush();

	if (experiment_profile_file.is_open())
	{
		experiment_profile_file << frame << ","
			<< GenPDExperimentVariantName() << ","
			<< ((GenPDExperimentUsesPersistentBuffers() && m_cs_gpu_state_valid) ? 1 : 0) << ","
			<< g_cs_profile_gradient_dispatches << ","
			<< g_cs_profile_stats_dispatches << ","
			<< g_cs_profile_reduction_dispatches << ","
			<< g_cs_profile_xupdate_dispatches << ","
			<< g_cs_profile_descent_dispatches << ","
			<< g_cs_profile_full_linesearch_calls << ","
			<< g_cs_profile_skipped_linesearch_calls << ","
			<< g_cs_profile_host_readbacks << ","
			<< g_cs_profile_solver_finish_calls << ","
			<< CurrentCSProfileBufferBytes(test_) << ","
			<< g_cs_gradient_buffer_bytes << ","
			<< g_cs_descent_buffer_bytes << ","
			<< g_cs_x_buffer_bytes << ","
			<< g_cs_y_buffer_bytes << ","
			<< test_.size() * sizeof(float) << ","
			<< g_cs_state_position_buffer_bytes << "\n";
		experiment_profile_file.flush();

}

if (m_quality_metrics_enabled && quality_profile_file.is_open() && m_mesh)
{
    if (m_cs_cpu_state_stale)
    {
        syncCS2GpuStateToCPU();
    }
    const bool checkpoint_frame = (frame % m_quality_checkpoint_stride) == 0u;
    if (checkpoint_frame && !m_reference_export_dir.empty()
        && !GenPDWriteReferenceCheckpoint(m_reference_export_dir, frame, m_mesh->m_current_positions, m_mesh->m_current_velocities))
    {
        std::cerr << "Warning: cannot write reference checkpoint for frame " << frame << std::endl;
    }

    VectorX reference_positions;
    VectorX reference_velocities;
    bool has_reference = checkpoint_frame && !m_quality_reference_dir.empty()
        && GenPDReadReferenceCheckpoint(m_quality_reference_dir, frame, reference_positions, reference_velocities)
        && reference_positions.size() == m_mesh->m_system_dimension
        && reference_velocities.size() == m_mesh->m_system_dimension
        && GenPDVectorIsFinite(reference_positions) && GenPDVectorIsFinite(reference_velocities);

    const bool finite = GenPDVectorIsFinite(m_mesh->m_current_positions) && GenPDVectorIsFinite(m_mesh->m_current_velocities);
    const ConstraintQualityMetrics current_metrics = ComputeConstraintQualityMetrics(m_mesh, m_mesh->m_current_positions);
    ConstraintQualityMetrics reference_metrics = {};
    ScalarType position_relative_l2 = 0.0;
    ScalarType velocity_relative_l2 = 0.0;
    ScalarType constraint_energy_relative_error = 0.0;
    ScalarType reference_penetration = 0.0;
    if (has_reference)
    {
        position_relative_l2 = GenPDRelativeL2(m_mesh->m_current_positions, reference_positions);
        velocity_relative_l2 = GenPDRelativeL2(m_mesh->m_current_velocities, reference_velocities);
        reference_metrics = ComputeConstraintQualityMetrics(m_mesh, reference_positions);
        constraint_energy_relative_error = std::abs(current_metrics.constraint_energy - reference_metrics.constraint_energy)
            / std::max(std::abs(reference_metrics.constraint_energy), static_cast<ScalarType>(EPSILON));
        reference_penetration = ComputeMaxPenetrationDepth(m_scene, m_processing_collision, reference_positions, m_mesh->m_vertices_number);
    }

    const ScalarType current_penetration = ComputeMaxPenetrationDepth(m_scene, m_processing_collision, m_mesh->m_current_positions, m_mesh->m_vertices_number);
    quality_profile_file << frame << ","
        << (has_reference ? 1 : 0) << ","
        << (finite ? 1 : 0) << ","
        << (m_last_profile_exploded ? 1 : 0) << ","
        << position_relative_l2 << ","
        << velocity_relative_l2 << ","
        << current_metrics.constraint_energy << ","
        << reference_metrics.constraint_energy << ","
        << constraint_energy_relative_error << ","
        << current_metrics.mean_stretch_strain << ","
        << reference_metrics.mean_stretch_strain << ","
        << current_metrics.max_stretch_strain << ","
        << reference_metrics.max_stretch_strain << ","
        << current_penetration << ","
        << reference_penetration << "\n";
    quality_profile_file.flush();
}
}

void Simulation::GetOverlayChar(char* overlay, unsigned int size)
{
	if (m_mesh->m_mesh_type == MESH_TYPE_TET)
	{
		sprintf_s(
			overlay,
			size,
			"| #vertices: %u, #elements: %u | iter: %u | total: %.2f ms | step: %.4g | |g|: %.4g | max|x|: %.4g | Restshape Volume: %.1lf, Current Volume: %.1lf (%.1lf%%)%s",
			m_mesh->m_vertices_number,
			static_cast<unsigned int>(m_constraints.size()),
			m_last_profile_iterations,
			m_last_profile_total_ms,
			m_last_profile_step_size,
			m_last_profile_gradient_norm,
			m_last_profile_max_position,
			m_restshape_volume,
			m_current_volume,
			m_current_volume / m_restshape_volume * 100,
			m_last_profile_exploded ? " | INVALID STEP" : "");
	}
	else
	{
		sprintf_s(
			overlay,
			size,
			"| #vertices: %u, #elements: %u | iter: %u | total: %.2f ms | step: %.4g | |g|: %.4g | max|x|: %.4g%s",
			m_mesh->m_vertices_number,
			static_cast<unsigned int>(m_constraints.size()),
			m_last_profile_iterations,
			m_last_profile_total_ms,
			m_last_profile_step_size,
			m_last_profile_gradient_norm,
			m_last_profile_max_position,
			m_last_profile_exploded ? " | INVALID STEP" : "");
	}
}

ScalarType Simulation::TryToSelectAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir)
{
	ScalarType ray_point_dist;
	ScalarType min_dist = 100.0;
	AttachmentConstraint* best_candidate = NULL;

	bool current_state_on = false;
	for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
	{
		AttachmentConstraint* ac;
		if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
		{
			ray_point_dist = ((ac->GetFixedPoint() - p0).cross(dir)).norm();
			if (ray_point_dist < min_dist)
			{
				min_dist = ray_point_dist;
				best_candidate = ac;
			}
		}
	}
	// exit if no one fits
	if (min_dist > DEFAULT_SELECTION_RADIUS)
	{
		UnselectAttachmentConstraint();

		return -1;
	}
	else
	{
		SelectAtttachmentConstraint(best_candidate);
		EigenVector3 fixed_point_temp = m_mesh->m_current_positions.block_vector(m_selected_attachment_constraint->GetConstrainedVertexIndex());

		return (fixed_point_temp - p0).dot(dir); // this is m_cached_projection_plane_distance
	}
}

bool Simulation::TryToToggleAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir)
{
	EigenVector3 p1;

	ScalarType ray_point_dist;
	ScalarType min_dist = 100.0;
	unsigned int best_candidate = 0;
	// first pass: choose nearest point
	for (unsigned int i = 0; i != m_mesh->m_vertices_number; i++)
	{
		p1 = m_mesh->m_current_positions.block_vector(i);

		ray_point_dist = ((p1 - p0).cross(dir)).norm();
		if (ray_point_dist < min_dist)
		{
			min_dist = ray_point_dist;
			best_candidate = i;
		}
	}
	for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
	{
		AttachmentConstraint* ac;
		if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
		{
			ray_point_dist = ((ac->GetFixedPoint() - p0).cross(dir)).norm();
			if (ray_point_dist < min_dist)
			{
				min_dist = ray_point_dist;
				best_candidate = ac->GetConstrainedVertexIndex();
			}
		}
	}
	// exit if no one fits
	if (min_dist > DEFAULT_SELECTION_RADIUS)
	{
		return false;
	}
	// second pass: toggle that point's fixed position constraint
	bool current_state_on = false;
	for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
	{
		AttachmentConstraint* ac;
		if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
		{
			if (ac->GetConstrainedVertexIndex() == best_candidate)
			{
				current_state_on = true;
				m_constraints.erase(c);
				delete ac;
				m_mesh->m_expanded_system_dimension -= 3;
				m_mesh->m_expanded_system_dimension_1d -= 1;
				m_cs_edge_buffer_dirty = true;
				break;
			}
		}
	}
	if (!current_state_on)
	{
		AddAttachmentConstraint(best_candidate);
	}

	return true;
}

void Simulation::SelectAtttachmentConstraint(AttachmentConstraint* ac)
{
	//m_selected_attachment_constraint = ac;
	//m_selected_attachment_constraint->Select();
}

void Simulation::UnselectAttachmentConstraint()
{
	//if (m_selected_attachment_constraint)
	//{
	//	m_selected_attachment_constraint->UnSelect();
	//}
	//m_selected_attachment_constraint = NULL;
}

AttachmentConstraint* Simulation::AddAttachmentConstraint(unsigned int vertex_index)
{
	AttachmentConstraint* ac = new AttachmentConstraint(vertex_index, m_mesh->m_current_positions.block_vector(vertex_index));
	ac->SetMaterialProperty(m_stiffness_attachment);
	m_constraints.push_back(ac);
	m_mesh->m_expanded_system_dimension += 3;
	m_mesh->m_expanded_system_dimension_1d += 1;
	m_cs_edge_buffer_dirty = true;

	return ac;
}

AttachmentConstraint* Simulation::AddAttachmentConstraint(unsigned int vertex_index, const EigenVector3& target)
{
	AttachmentConstraint* ac = new AttachmentConstraint(vertex_index, target);
	ac->SetMaterialProperty(m_stiffness_attachment);
	m_constraints.push_back(ac);
	m_mesh->m_expanded_system_dimension += 3;
	m_mesh->m_expanded_system_dimension_1d += 1;
	m_cs_edge_buffer_dirty = true;

	return ac;
}

void Simulation::MoveSelectedAttachmentConstraintTo(const EigenVector3& target)
{
	if (m_selected_attachment_constraint)
	{
		m_selected_attachment_constraint->SetFixedPoint(target);
		m_cs_edge_buffer_dirty = true;
	}
}

void Simulation::SaveAttachmentConstraint(const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename, std::ifstream::out);
	if (outfile.is_open())
	{
		int existing_vertices = 0;
		for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
		{
			(*c)->WriteToFileOBJHead(outfile);
		}
		outfile << std::endl;
		for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
		{
			(*c)->WriteToFileOBJ(outfile, existing_vertices);
		}

		outfile.close();
	}
}

void Simulation::LoadAttachmentConstraint(const char* filename)
{
	// clear current attachement constraints
	for (std::vector<Constraint*>::iterator& c = m_constraints.begin(); c != m_constraints.end(); )
	{
		AttachmentConstraint* ac;
		if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
		{
			c = m_constraints.erase(c);
			delete ac;
			m_mesh->m_expanded_system_dimension -= 3;
			m_mesh->m_expanded_system_dimension_1d -= 1;
		}
		else
		{
			c++;
		}
	}
	m_cs_edge_buffer_dirty = true;

	// read from file
	std::ifstream infile;
	infile.open(filename, std::ifstream::in);
	char ignore[256];
	if (infile.is_open())
	{
		while (!infile.eof())
		{
			int id;
			EigenVector3 p;
			if (infile >> ignore >> id >> p[0] >> p[1] >> p[2])
			{
				if (strcmp(ignore, "v") == 0)
					break;
				AttachmentConstraint* ac = new AttachmentConstraint(id, p);
				ac->SetMaterialProperty(m_stiffness_attachment);
				m_constraints.push_back(ac);
				m_mesh->m_expanded_system_dimension += 3;
				m_mesh->m_expanded_system_dimension_1d += 1;
			}
			else
				break;
		}

		infile.close();
	}

}

void Simulation::NewHandle(const std::vector<unsigned int>& indices, const glm::vec3 color)
{
	// check if any points has already been assigned as handle
	for (unsigned int i = 0; i != indices.size(); ++i)
	{
		if (m_handle_id[indices[i]] >= 0)
		{
			std::cerr << "Some of the vertices in the selection has already been assigned as handle. Please select again." << std::endl;
			return;
		}
	}

	// new handle
	VectorX vertices(3 * indices.size());
	for (unsigned int i = 0; i != indices.size(); ++i)
	{
		vertices.block<3, 1>(i * 3, 0) = m_mesh->m_current_positions.block<3, 1>(indices[i] * 3, 0);
	}

	int id = m_handles.size();
	Handle handle = Handle(indices, vertices, color, id);
	// update the handle id list
	for (unsigned int i = 0; i != indices.size(); ++i)
	{
		m_handle_id[indices[i]] = id;
	}

	handle.attachment_constraints.clear();
	// new attachment constraints according to this handle
	for (unsigned int i = 0; i != handle.Indices().size(); ++i)
	{
		handle.attachment_constraints.push_back(AddAttachmentConstraint(handle.Indices()[i]));
	}
	m_handles.push_back(handle);
}

void Simulation::DeleteHandle()
{
	if (m_selected_handle_id >= 0)
	{
		// remove attachment constraints
		Handle& handle = m_handles[m_selected_handle_id];

		// TODO: remove attachment constraits
		std::vector<Constraint*>::iterator it;
		bool find_ac = false;
		for (it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			if ((*it) == handle.attachment_constraints[0])
			{
				find_ac = true;
				break;
			}
		}
		if (find_ac == true)
		{
			m_constraints.erase(it, it + handle.attachment_constraints.size());
		}
		else
		{
			assert(find_ac == true);
		}
		//AddAttachmentConstraint(handle.Indices()[i]);

		// delete handle
		m_handles.erase(m_handles.begin() + m_selected_handle_id);

		// reconstruct the id of the handles
		for (unsigned int i = m_selected_handle_id; i < m_handles.size(); ++i)
		{
			m_handles[i].ID()--;
		}

		// reconstruct the handle id list
		for (unsigned int i = 0; i != m_handle_id.size(); ++i)
		{
			if (m_handle_id[i] == m_selected_handle_id)
			{
				m_handle_id[i] = -1; // unselected
			}
			else if (m_handle_id[i] > m_selected_handle_id)
			{
				m_handle_id[i]--;
			}
		}
		m_selected_handle_id = -1;
		m_cs_edge_buffer_dirty = true;
	}
}

bool Simulation::SelectHandle(std::vector<glm::vec3> ray)
{
	unsigned int nearest_handle_id = -1;
	float dist_min = 1e10;
	float dist;

	for (unsigned int i = 0; i != m_handles.size(); ++i)
	{
		if (m_handles[i].Select(ray, dist))
		{
			if (dist_min > dist)
			{
				dist_min = dist;
				nearest_handle_id = i;
			}
		}
	}
	m_selected_handle_id = nearest_handle_id;
	return (m_selected_handle_id == -1) ? false : true;
}

void Simulation::MoveHandleTemporary(const glm::vec3& trans)
{
	if (m_selected_handle_id >= 0)
	{
		m_handles[m_selected_handle_id].MoveTemporary(trans);
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::MoveHandleFinalize()
{
	if (m_selected_handle_id >= 0)
	{
		m_handles[m_selected_handle_id].MoveFinalize();
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::RotateHandleToValue()
{
	if (m_selected_handle_id >= 0)
	{
		ScalarType theta;
		std::cout << "Assign rotation(degree) for handle [" << m_selected_handle_id << "]." << std::endl;
		std::cin >> theta;
		theta = theta / 180.0 * 3.1415926;
		m_handles[m_selected_handle_id].RotateToValue(theta);
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}//handle?????????????????????
void Simulation::RotateHandleSetStepSize()
{
	if (m_selected_handle_id >= 0)
	{
		//m_keyframe_handle_id = m_selected_handle_id;
		//std::cout << "Assign rotation(degree) for handle [" << m_selected_handle_id << "]." << std::endl;
		//std::cin >> m_keyframe_handle_unit_rotation;
		//m_keyframe_handle_unit_rotation = m_keyframe_handle_unit_rotation / 180.0 * 3.1415926;
		//UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::RotateHandleTemporary(const glm::vec3& axis, const float& theta)
{
	if (m_selected_handle_id >= 0)
	{
		m_handles[m_selected_handle_id].RotateTemporary(axis, theta);
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::RotateHandleFinalize()
{
	if (m_selected_handle_id >= 0)
	{
		m_handles[m_selected_handle_id].RotateFinalize();
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}

void Simulation::UpdateHandleInfoToConstraints(Handle& selected_handle)
{
	for (unsigned int i = 0; i != selected_handle.attachment_constraints.size(); ++i)
	{
		selected_handle.attachment_constraints[i]->SetFixedPoint(selected_handle[i]);
	}
	if (!selected_handle.attachment_constraints.empty())
	{
		m_cs_edge_buffer_dirty = true;
	}
}

glm::vec3 Simulation::SelectedHandleLocalCoM()
{
	assert(m_selected_handle_id >= 0);
	assert(m_selected_handle_id < m_handles.size());

	return m_handles[m_selected_handle_id].GetLocalCoM();
}

glm::vec3 Simulation::SelectedHandleCoM()
{
	assert(m_selected_handle_id >= 0);
	assert(m_selected_handle_id < m_handles.size());

	return m_handles[m_selected_handle_id].GetCoM();
}

void Simulation::SetHandleTranslationAnimation()
{
	if (m_selected_handle_id >= 0)
	{
		m_keyframe_handle_id_translation = m_selected_handle_id;
		m_keyframe_handle_unit_translation_axis.clear();
		m_keyframe_handle_unit_translation_amount.clear();
		m_keyframe_handle_unit_translation_end_frames.clear();

		// report handle
		std::cout << "Assignment Translation Animation for handle [" << m_selected_handle_id << "]." << std::endl;

		// set number of segments
		std::cout << "How many segments of translation do you want?" << std::endl;
		std::cin >> m_keyframe_handle_unit_translation_total_segments;

		int last_end_frame = 0;
		for (int i = 0; i != m_keyframe_handle_unit_translation_total_segments; i++)
		{
			EigenVector3 axis;
			ScalarType t;
			int end_frame;
			std::cout << "Input Translation Segment[" << i + 1 << "]." << std::endl;
			std::cout << "Assign translation axis: (will be normalized afterwards)" << std::endl;
			std::cin >> axis[0] >> axis[1] >> axis[2];
			std::cout << "Assign translation amount:" << std::endl;
			std::cin >> t;
			std::cout << "Assign this segment duration: (#frames)" << std::endl;
			std::cin >> end_frame;
			end_frame += last_end_frame;
			last_end_frame = end_frame;
			m_keyframe_handle_unit_translation_axis.push_back(axis);
			m_keyframe_handle_unit_translation_amount.push_back(t);
			m_keyframe_handle_unit_translation_end_frames.push_back(end_frame);
		}
	}
}
void Simulation::SetHandleTranslation()
{
	if (m_selected_handle_id >= 0)
	{
		EigenVector3 axis;
		ScalarType t;
		std::cout << "Assignment Translation for handle [" << m_selected_handle_id << "]." << std::endl;
		std::cout << "Assign translation axis: (will be normalized afterwards)" << std::endl;
		std::cin >> axis[0] >> axis[1] >> axis[2];
		std::cout << "Assign translation amount:" << std::endl;
		std::cin >> t;

		m_handles[m_selected_handle_id].ChangeTranslation(axis, t);
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::SetHandleRotationAnimation()
{
	if (m_selected_handle_id >= 0)
	{
		m_keyframe_handle_id_rotation = m_selected_handle_id;
		m_keyframe_handle_unit_rotation_axis.clear();
		m_keyframe_handle_unit_rotation_degree.clear();
		m_keyframe_handle_unit_rotation_end_frames.clear();

		// report handle
		std::cout << "Assignment Rotation Animation for handle [" << m_selected_handle_id << "]." << std::endl;

		// set number of segments
		std::cout << "How many segments of rotation do you want?" << std::endl;
		std::cin >> m_keyframe_handle_unit_rotation_total_segments;

		int last_end_frame = 0;
		for (int i = 0; i != m_keyframe_handle_unit_rotation_total_segments; i++)
		{
			EigenVector3 axis;
			ScalarType t;
			int end_frame;
			std::cout << "Input Rotation Segment[" << i + 1 << "]." << std::endl;
			std::cout << "Assign rotation axis: (will be normalized afterwards)" << std::endl;
			std::cin >> axis[0] >> axis[1] >> axis[2];
			std::cout << "Assign rotation degree:" << std::endl;
			std::cin >> t;
			std::cout << "Assign this segment duration: (#frames)" << std::endl;
			std::cin >> end_frame;
			end_frame += last_end_frame;
			last_end_frame = end_frame;
			m_keyframe_handle_unit_rotation_axis.push_back(axis);
			m_keyframe_handle_unit_rotation_degree.push_back(t);
			m_keyframe_handle_unit_rotation_end_frames.push_back(end_frame);
		}
	}
}
void Simulation::SetHandleRotation()
{
	if (m_selected_handle_id >= 0)
	{
		EigenVector3 axis;
		ScalarType t;
		std::cout << "Assignment Rotation for handle [" << m_selected_handle_id << "]." << std::endl;
		std::cout << "Assign rotation axis: (will be normalized afterwards)" << std::endl;
		std::cin >> axis[0] >> axis[1] >> axis[2];
		std::cout << "Assign rotation degree:" << std::endl;
		std::cin >> t;

		m_handles[m_selected_handle_id].ChangeRotation(axis, t);
		UpdateHandleInfoToConstraints(m_handles[m_selected_handle_id]);
	}
}
void Simulation::AnimateHandle(const int current_frame)
{
	int segment = 0;
	// translation
	// find first segment;
	for (segment = 0; segment < m_keyframe_handle_unit_translation_total_segments /*This end condition should never be hit*/; segment++)
	{
		if (m_keyframe_handle_unit_translation_end_frames[segment] > current_frame)
		{
			break;
		}
	}
	if (segment == m_keyframe_handle_unit_translation_total_segments)
	{
		// do nothing
	}
	else
	{
		if (m_keyframe_handle_unit_translation_amount[segment] > EPSILON)
		{
			m_handles[m_keyframe_handle_id_translation].ChangeTranslation(m_keyframe_handle_unit_translation_axis[segment], m_keyframe_handle_unit_translation_amount[segment]);
			UpdateHandleInfoToConstraints(m_handles[m_keyframe_handle_id_translation]);
		}
	}

	// rotation
	// find first segment;
	for (segment = 0; segment < m_keyframe_handle_unit_rotation_total_segments /*This end condition should never be hit*/; segment++)
	{
		if (m_keyframe_handle_unit_rotation_end_frames[segment] > current_frame)
		{
			break;
		}
	}
	if (segment == m_keyframe_handle_unit_rotation_total_segments)
	{
		// do nothing
	}
	else
	{
		if (m_keyframe_handle_unit_rotation_degree[segment] > EPSILON)
		{

			m_handles[m_keyframe_handle_id_rotation].ChangeRotation(m_keyframe_handle_unit_rotation_axis[segment], m_keyframe_handle_unit_rotation_degree[segment]);
			UpdateHandleInfoToConstraints(m_handles[m_keyframe_handle_id_rotation]);
		}
	}

}

void Simulation::SaveHandleAnimation(const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename, std::ifstream::out);
	if (outfile.is_open())
	{
		// TODO: change it to memory dump.
		outfile << "KeyframedHandleIDT   " << m_keyframe_handle_id_translation << std::endl << std::endl;
		outfile << "AnimationSegmentsT   " << m_keyframe_handle_unit_translation_total_segments << std::endl << std::endl;

		for (int i = 0; i != m_keyframe_handle_unit_translation_total_segments; ++i)
		{
			outfile << m_keyframe_handle_unit_translation_axis[i].x() << " " \
				<< m_keyframe_handle_unit_translation_axis[i].y() << " " \
				<< m_keyframe_handle_unit_translation_axis[i].z() << " " \
				<< m_keyframe_handle_unit_translation_amount[i] << " " \
				<< m_keyframe_handle_unit_translation_end_frames[i] << std::endl << std::endl;
		}

		outfile << "KeyframedHandleIDR   " << m_keyframe_handle_id_rotation << std::endl << std::endl;
		outfile << "AnimationSegmentsR   " << m_keyframe_handle_unit_rotation_total_segments << std::endl << std::endl;

		for (int i = 0; i != m_keyframe_handle_unit_rotation_total_segments; ++i)
		{
			outfile << m_keyframe_handle_unit_rotation_axis[i].x() << " " \
				<< m_keyframe_handle_unit_rotation_axis[i].y() << " " \
				<< m_keyframe_handle_unit_rotation_axis[i].z() << " " \
				<< m_keyframe_handle_unit_rotation_degree[i] << " " \
				<< m_keyframe_handle_unit_rotation_end_frames[i] << std::endl << std::endl;
		}

		outfile.close();
	}
	else
	{
		std::cerr << "Warning: Can not write handle animation file. Settings not saved." << std::endl;
	}
}
void Simulation::LoadHandleAnimation(const char* filename)
{
	// clear current handle animation info
	m_keyframe_handle_unit_translation_axis.clear();
	m_keyframe_handle_unit_translation_amount.clear();
	m_keyframe_handle_unit_translation_end_frames.clear();

	bool successfulRead = false;
	// read file
	std::ifstream infile;
	infile.open(filename, std::ifstream::in);
	if (successfulRead = infile.is_open())
	{
		char ignoreToken[256];

		infile >> ignoreToken >> m_keyframe_handle_id_translation;
		infile >> ignoreToken >> m_keyframe_handle_unit_translation_total_segments;

		for (int i = 0; i != m_keyframe_handle_unit_translation_total_segments; ++i)
		{
			EigenVector3 translation_axis;
			ScalarType translation_amount;
			int end_frame;
			infile >> translation_axis[0] \
				>> translation_axis[1] \
				>> translation_axis[2] \
				>> translation_amount \
				>> end_frame;

			m_keyframe_handle_unit_translation_axis.push_back(translation_axis);
			m_keyframe_handle_unit_translation_amount.push_back(translation_amount);
			m_keyframe_handle_unit_translation_end_frames.push_back(end_frame);
		}

		infile >> ignoreToken >> m_keyframe_handle_id_rotation;
		infile >> ignoreToken >> m_keyframe_handle_unit_rotation_total_segments;

		for (int i = 0; i != m_keyframe_handle_unit_rotation_total_segments; ++i)
		{
			EigenVector3 rotation_axis;
			ScalarType rotation_amount;
			int end_frame;
			infile >> rotation_axis[0] \
				>> rotation_axis[1] \
				>> rotation_axis[2] \
				>> rotation_amount \
				>> end_frame;

			m_keyframe_handle_unit_rotation_axis.push_back(rotation_axis);//??????????????洢
			m_keyframe_handle_unit_rotation_degree.push_back(rotation_amount);
			m_keyframe_handle_unit_rotation_end_frames.push_back(end_frame);
		}
	}
	if (!successfulRead)
	{
		std::cerr << "Waning: failed loading handles animation." << std::endl;
	}
}
void Simulation::SaveHandles(const char* filename)
{
	unsigned int handle_num = m_handles.size();

	//const std::vector<unsigned int>& indices, const glm::vec3 color;
	std::ofstream outfile;
	outfile.open(filename, std::ifstream::out);
	if (outfile.is_open())
	{
		// TODO: change it to memory dump.
		outfile << "HandleNum   " << m_handles.size() << std::endl << std::endl;

		for (unsigned int i = 0; i != m_handles.size(); ++i)
		{
			outfile << "Color       "
				<< m_handles[i].Color().x << " "
				<< m_handles[i].Color().y << " "
				<< m_handles[i].Color().z << std::endl;
			//outfile << "Rotation2D  " << m_handles[i].RotationAngle2d() << std::endl;
			outfile << "RotationCenter "
				<< m_handles[i].CoM()[0] << " "
				<< m_handles[i].CoM()[1] << " "
				<< m_handles[i].CoM()[2] << std::endl;
			outfile << "Translation "
				<< m_handles[i].Translation()[0] << " "
				<< m_handles[i].Translation()[1] << " "
				<< m_handles[i].Translation()[2] << std::endl;
			EigenAngleAxis aa(m_handles[i].Rotation());
			outfile << "Rotation "
				<< aa.axis()[0] << " "
				<< aa.axis()[1] << " "
				<< aa.axis()[2] << " "
				<< aa.angle() << std::endl;
			outfile << "VerticesNum " << m_handles[i].Indices().size() << std::endl;
			for (unsigned int j = 0; j != m_handles[i].Indices().size(); ++j)
			{
				outfile << m_handles[i].Indices()[j] << " ";
			}
			outfile << std::endl << std::endl;
		}

		outfile.close();
	}
	else
	{
		std::cerr << "Warning: Can not write handle file. Settings not saved." << std::endl;
	}
}

void Simulation::LoadHandles(const char* filename)//????ж?????????
{
	// clear current handles and attachment constraints
	for (int i = m_handles.size() - 1; i >= 0; --i)
	{
		m_selected_handle_id = i;
		DeleteHandle();
	}
	m_handles.clear();
	for (unsigned int i = 0; i != m_handle_id.size(); ++i)
	{
		m_handle_id[i] = -1;
	}

	bool successfulRead = false;
	// read file
	std::ifstream infile;
	infile.open(filename, std::ifstream::in);
	if (successfulRead = infile.is_open())
	{
		char ignoreToken[256];

		unsigned int handle_num;
		infile >> ignoreToken >> handle_num;

		unsigned int handle_size;
		glm::vec3 color;
		std::vector<unsigned int> ids;

		EigenVector3 translation;
		EigenVector3 com;
		EigenVector3 rotation_axis;
		ScalarType rotation_angle;
		for (unsigned int hi = 0; hi != handle_num; ++hi)
		{
			infile >> ignoreToken >> color[0] >> color[1] >> color[2];
			//infile >> ignoreToken >> rotation2d;
			infile >> ignoreToken >> com[0] >> com[1] >> com[2];
			infile >> ignoreToken >> translation[0] >> translation[1] >> translation[2];
			infile >> ignoreToken >> rotation_axis[0] >> rotation_axis[1] >> rotation_axis[2] >> rotation_angle;
			infile >> ignoreToken >> handle_size;
			ids.resize(handle_size);
			for (unsigned int j = 0; j != handle_size; ++j)
			{
				infile >> ids[j];
			}
			VectorX vertices(3 * ids.size());
			for (unsigned int i = 0; i != ids.size(); ++i)
			{
				vertices.block<3, 1>(i * 3, 0) = m_mesh->m_restpose_positions.block<3, 1>(3 * ids[i], 0).transpose();
			}
			Handle h(ids, vertices, color, hi);
			//h.RotationAngle2d() = rotation2d;
			h.Translation() = translation;
			h.Rotation() = EigenAngleAxis(rotation_angle, rotation_axis).matrix();
			h.CoM() = com;
			h.Update();

			h.attachment_constraints.clear();
			// new attachment constraints according to this handle
			for (unsigned int i = 0; i != h.Indices().size(); ++i)
			{
				h.attachment_constraints.push_back(AddAttachmentConstraint(h.Indices()[i], h[i]));
			}

			m_handles.push_back(h);

			// update handle id
			for (unsigned int i = 0; i != h.Indices().size(); ++i)
			{
				m_handle_id[h.Indices()[i]] = h.ID();
			}
		}
	}
	if (!successfulRead)
	{
		std::cerr << "Waning: failed loading handles." << std::endl;
	}
}

void Simulation::ResetHandles()
{
	for (unsigned int i = 0; i != m_handles.size(); i++)
	{
		m_handles[i].Reset();
		UpdateHandleInfoToConstraints(m_handles[i]);
	}
}

void Simulation::SaveSparseMatrix(const SparseMatrix& A, const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename, std::ifstream::out);
	if (outfile.is_open())
	{
		std::vector<SparseMatrixTriplet> A_triplets;
		EigenSparseMatrixToTriplets(A, A_triplets);

		// my format
		outfile << A.rows() << " " << A.cols() << " " << A_triplets.size() << std::endl;
		for (unsigned int i = 0; i != A_triplets.size(); i++)
		{
			SparseMatrixTriplet& Ai = A_triplets[i];
			outfile << Ai.row() << " " << Ai.col() << " " << Ai.value() << std::endl;
		}

		//// cuSolver format
		//outfile << "%%MatrixMarket matrix coordinate real symmetric\n% Generated 20 - Nov - 2014" << std::endl;
		//outfile << m_mass_plus_h2_weighted_laplacian.rows() << " " << m_mass_plus_h2_weighted_laplacian.cols() << " " << (L_triplets.size()+m_mass_plus_h2_weighted_laplacian.rows())/2 << std::endl;
		//for (unsigned int i = 0; i != L_triplets.size(); i++)
		//{
		//	SparseMatrixTriplet& Li = L_triplets[i];
		//	if (Li.row() >= Li.col())
		//	{
		//		outfile << Li.row() + 1 << " " << Li.col() + 1 << " " << Li.value() << std::endl;
		//	}
		//}

		outfile.close();
	}
}

void Simulation::SaveLaplacianMatrix(const char* filename)
{
	SaveSparseMatrix(m_weighted_laplacian, filename);
	//std::ofstream outfile;
	//outfile.open(filename, std::ifstream::out);
	//if (outfile.is_open())
	//{
	//	std::vector<SparseMatrixTriplet> L_triplets;
	//	EigenSparseMatrixToTriplets(m_weighted_laplacian, L_triplets);

	//	// my format
	//	outfile << m_weighted_laplacian.rows() << " " << m_weighted_laplacian.cols() << " " << L_triplets.size() << std::endl;
	//	for (unsigned int i = 0; i != L_triplets.size(); i++)
	//	{
	//		SparseMatrixTriplet& Li = L_triplets[i];
	//		outfile << Li.row() << " " << Li.col() << " " << Li.value() << std::endl;
	//	}

	//	outfile.close();
	//}
}

void Simulation::SetConvergedEnergy()
{
#ifdef ENABLE_MATLAB_DEBUGGING
	ScalarType energy = evaluateEnergy(m_mesh->m_current_positions);
	g_debugger->SetConvergedEnergy(energy);
#endif // ENABLE_MATLAB_DEBUGGING
}

void Simulation::RandomizePoints()
{
	VectorX x;
	generateRandomVector(m_mesh->m_system_dimension, x);//random?????????????

	m_mesh->m_current_positions = x;// ??????????????????????????λ??
}

// set material property for selected elements
void Simulation::SetMaterialProperty(std::vector<Constraint*>& constraints)
{
	for (std::vector<Constraint*>::iterator it = constraints.begin(); it != constraints.end(); ++it)
	{
		switch ((*it)->Type())
		{
		case CONSTRAINT_TYPE_TET:
			(*it)->SetMaterialProperty(m_material_type, m_stiffness_stretch, m_stiffness_bending, m_stiffness_kappa, m_stiffness_laplacian);
			break;
		case CONSTRAINT_TYPE_SPRING:
			(*it)->SetMaterialProperty(m_stiffness_stretch);
			break;
		case CONSTRAINT_TYPE_SPRING_BENDING:
			(*it)->SetMaterialProperty(m_stiffness_bending);
			break;
		case CONSTRAINT_TYPE_ATTACHMENT:
			(*it)->SetMaterialProperty(m_stiffness_attachment);
			break;
		}
	}
	SetReprefactorFlag();
	m_cs_edge_buffer_dirty = true;
}
void Simulation::SetMaterialProperty(std::vector<Constraint*>& constraints, MaterialType type, ScalarType stretch, ScalarType bending, ScalarType kappa, ScalarType laplacian_coeff)
{
	for (std::vector<Constraint*>::iterator it = constraints.begin(); it != constraints.end(); ++it)
	{
		switch ((*it)->Type())
		{
		case CONSTRAINT_TYPE_TET:
			(*it)->SetMaterialProperty(type, stretch, bending, kappa, laplacian_coeff);
			break;
		case CONSTRAINT_TYPE_SPRING:
			(*it)->SetMaterialProperty(stretch);
			break;
		}
	}
	SetReprefactorFlag();
	m_cs_edge_buffer_dirty = true;
}

// set material property for all elements
void Simulation::SetMaterialProperty()
{
	SetMaterialProperty(m_constraints);
}

void Simulation::GetPartialMaterialProperty()
{
	if (!m_selected_constraints.empty())
	{
		Constraint* c = m_selected_constraints[0];
		if (c->Type() == CONSTRAINT_TYPE_SPRING)
		{
			c->GetMaterialProperty(m_partial_stiffness_stretch);
		}
		else if (c->Type() == CONSTRAINT_TYPE_TET)
		{
			c->GetMaterialProperty(m_partial_material_type, m_partial_stiffness_stretch, m_partial_stiffness_bending, m_partial_stiffness_kappa);
		}
	}
}
void Simulation::SetPartialMaterialProperty()
{
	if (!m_selected_constraints.empty())
	{
		SetMaterialProperty(m_selected_constraints, m_partial_material_type, m_partial_stiffness_stretch, m_partial_stiffness_bending, m_partial_stiffness_kappa, 2 * m_partial_stiffness_stretch + m_partial_stiffness_bending);
	}

}
void Simulation::SavePerConstraintMaterialProperties(const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename, std::ifstream::out);
	if (outfile.is_open())
	{
		MaterialType material_type;
		ScalarType stiffness, mu, lambda, kappa;
		for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
		{
			if ((*c)->Type() == CONSTRAINT_TYPE_TET)
			{
				(*c)->GetMaterialProperty(material_type, mu, lambda, kappa);
				outfile << material_type << " " << mu << " " << lambda << " " << kappa << std::endl;
			}
			else
			{
				(*c)->GetMaterialProperty(stiffness);
				outfile << stiffness << std::endl;
			}
		}

		outfile.close();
	}

}
void Simulation::LoadPerConstraintMaterialProperties(const char* filename)
{
	std::ifstream infile;
	infile.open(filename, std::ifstream::in);
	char ignore[256];
	if (infile.is_open())
	{
		MaterialType material_type;
		ScalarType stiffness, mu, lambda, kappa;
		int temp_enum;
		for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
		{
			if ((*c)->Type() == CONSTRAINT_TYPE_TET)
			{
				infile >> temp_enum; material_type = MaterialType(temp_enum);
				infile >> mu;
				infile >> lambda;
				infile >> kappa;//?????????????????
				(*c)->SetMaterialProperty(material_type, mu, lambda, kappa, 2 * mu + lambda);
			}
			else
			{
				infile >> stiffness;//????????????????
				(*c)->SetMaterialProperty(stiffness);
			}
		}

		infile.close();
	}
	m_cs_edge_buffer_dirty = true;
}
void Simulation::SelectTetConstraints(const std::vector<unsigned int>& indices)
{
	m_selected_constraints.clear();
	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); it++)
	{
		if ((*it)->Type() == CONSTRAINT_TYPE_SPRING || (*it)->Type() == CONSTRAINT_TYPE_TET)
		{
			for (unsigned int i = 0; i != indices.size(); i++)
			{
				if ((*it)->VertexIncluded(indices[i]))
				{
					m_selected_constraints.push_back(*it);
					break;
				}
			}
		}
	}
}

// eigen value visualization mesh
void Simulation::NewVisualizationMesh()
{
	if (m_mesh->GetMeshType() == MESH_TYPE_CLOTH)
	{
		m_eigenvector_vis_mesh = new ClothMesh();
	}
	else
	{
		m_eigenvector_vis_mesh = new TetMesh();
	}
}
void Simulation::DeleteVisualizationMesh()
{
	if (m_eigenvector_vis_mesh != NULL)
	{
		delete m_eigenvector_vis_mesh;
	}
}
void Simulation::ResetVisualizationMesh()
{
	DeleteVisualizationMesh();
	NewVisualizationMesh();
}
void Simulation::SetVisualizationMesh()
{
	m_eigenvector_vis_mesh->CopyFromClothMesh(m_mesh);
}
void Simulation::ResetVisualizationMeshHeight()
{
	if (m_eigenvector_vis_mesh->GetMeshType() == MESH_TYPE_CLOTH)
	{
		VectorX& x = m_eigenvector_vis_mesh->m_current_positions;

		for (unsigned int i = 0; 3 * i != x.size(); i++)
		{
			// set y component of x to 0
			x(3 * i + 1) = 0;
		}
		m_eigenvector_vis_mesh->Update();
	}
}

void Simulation::clearConstraints()
{
	for (unsigned int i = 0; i < m_constraints.size(); ++i)
	{
		delete m_constraints[i];
	}
	m_constraints.clear();
}

void Simulation::setupConstraints()
{
	clearConstraints();
	m_mesh->my_edge.clear();
	m_cs_edge_buffer_dirty = true;

	m_stiffness_high = 1e5;

	switch (m_mesh->m_mesh_type)
	{
	case MESH_TYPE_CLOTH:
		// procedurally generate constraints including to attachment constraints?????????????????????????
	{
		// generate stretch constraints. assign a stretch constraint for each edge.???????
		EigenVector3 p1, p2;
		for (std::vector<Edge>::iterator e = m_mesh->m_edge_list.begin(); e != m_mesh->m_edge_list.end(); ++e)
		{
			p1 = m_mesh->m_current_positions.block_vector(e->m_v1);
			p2 = m_mesh->m_current_positions.block_vector(e->m_v2);
			SpringConstraint* c;
			//if (e - m_mesh->m_edge_list.begin() < 100)
			//{
			//	c = new SpringConstraint(&m_stiffness_high, e->m_v1, e->m_v2, (p1 - p2).norm());
			//}
			//else
			{
				ScalarType rest_length = (p1 - p2).norm();
				rest_length *= rest_length_adjust;
				e->rest_length = rest_length;
				Edge temp;
				temp.m_v1 = e->m_v1, temp.m_v2 = e->m_v2;
				temp.rest_length = rest_length;
				temp.stiffness = m_stiffness_stretch;
				m_mesh->my_edge.push_back(temp);
				c = new SpringConstraint(e->m_v1, e->m_v2, rest_length);
			}
			m_constraints.push_back(c);
			m_mesh->m_expanded_system_dimension += 6;
			m_mesh->m_expanded_system_dimension_1d += 2;
		}

		// generate bending constraints. naive???????
		unsigned int i, k;
		for (i = 0; i < m_mesh->m_dim[0]; ++i)
		{
			for (k = 0; k < m_mesh->m_dim[1]; ++k)
			{
				unsigned int index_self = m_mesh->m_dim[1] * i + k;
				p1 = m_mesh->m_current_positions.block_vector(index_self);
				if (i + 2 < m_mesh->m_dim[0])
				{
					unsigned int index_row_1 = m_mesh->m_dim[1] * (i + 2) + k;
					p2 = m_mesh->m_current_positions.block_vector(index_row_1);
					ScalarType rest_length = (p1 - p2).norm();
					rest_length *= rest_length_adjust;
					// e->rest_length = rest_length;
					SpringConstraint* c = new SpringConstraint(CONSTRAINT_TYPE_SPRING_BENDING, index_self, index_row_1, rest_length);
					m_constraints.push_back(c);
					Edge temp;
					temp.m_v1 = index_self, temp.m_v2 = index_row_1;
					temp.rest_length = rest_length;
					temp.stiffness = m_stiffness_bending;
					m_mesh->my_edge.push_back(temp);
					m_mesh->m_expanded_system_dimension += 6;
					m_mesh->m_expanded_system_dimension_1d += 2;
				}
				if (k + 2 < m_mesh->m_dim[1])
				{
					unsigned int index_column_1 = m_mesh->m_dim[1] * i + k + 2;
					p2 = m_mesh->m_current_positions.block_vector(index_column_1);
					ScalarType rest_length = (p1 - p2).norm();
					rest_length *= rest_length_adjust;
					SpringConstraint* c = new SpringConstraint(CONSTRAINT_TYPE_SPRING_BENDING, index_self, index_column_1, rest_length);
					m_constraints.push_back(c);
					Edge temp;
					temp.m_v1 = index_self, temp.m_v2 = index_column_1;
					temp.rest_length = rest_length;
					temp.stiffness = m_stiffness_bending;
					m_mesh->my_edge.push_back(temp);
					m_mesh->m_expanded_system_dimension += 6;
					m_mesh->m_expanded_system_dimension_1d += 2;
				}
			}
		}

		// generating attachment constraints.
		//std::vector<unsigned int> handle_1_indices; handle_1_indices.clear(); handle_1_indices.push_back(0);
		//std::vector<unsigned int> handle_2_indices; handle_2_indices.clear(); handle_2_indices.push_back(m_mesh->m_dim[1] * (m_mesh->m_dim[0] - 1));
		NewHandle({ 0 }, glm::vec3(1.0, 0.0, 0.0));
		NewHandle({ m_mesh->m_dim[1] * (m_mesh->m_dim[0] - 1) }, glm::vec3(1.0, 0.0, 0.0));
		//AddAttachmentConstraint(0);
		//AddAttachmentConstraint(m_mesh->m_dim[1]*(m_mesh->m_dim[0]-1));
	}
	break;
	case MESH_TYPE_TET:
	{
		//// generate stretch constraints. assign a stretch constraint for each edge.
		//EigenVector3 p1, p2;
		//for(std::vector<Edge>::iterator e = m_mesh->m_edge_list.begin(); e != m_mesh->m_edge_list.end(); ++e)
		//{
		//	p1 = m_mesh->m_current_positions.block_vector(e->m_v1);
		//	p2 = m_mesh->m_current_positions.block_vector(e->m_v2);
		//	SpringConstraint *c = new SpringConstraint(&m_stiffness_stretch, e->m_v1, e->m_v2, (p1-p2).norm());
		//	m_constraints.push_back(c);
		//}

		// reset mass matrix for tet simulation:
		ScalarType total_volume = 0;
		std::vector<SparseMatrixTriplet> mass_triplets;
		std::vector<SparseMatrixTriplet> mass_1d_triplets;
		mass_triplets.clear();
		mass_1d_triplets.clear();

		VectorX& x = m_mesh->m_current_positions;
		TetMesh* tet_mesh = dynamic_cast<TetMesh*>(m_mesh);
		for (unsigned int i = 0; i < tet_mesh->m_loaded_mesh->m_tets.size(); ++i)
		{
			MeshLoader::Tet& tet = tet_mesh->m_loaded_mesh->m_tets[i];
			TetConstraint* c = new TetConstraint(tet.id1, tet.id2, tet.id3, tet.id4, x);
			m_constraints.push_back(c);

			total_volume += c->SetMassMatrix(mass_triplets, mass_1d_triplets);

			m_mesh->m_expanded_system_dimension += 9;
			m_mesh->m_expanded_system_dimension_1d += 3;
		}

		m_mesh->m_mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
		m_mesh->m_mass_matrix_1d.setFromTriplets(mass_1d_triplets.begin(), mass_1d_triplets.end());

		m_mesh->m_mass_matrix = m_mesh->m_mass_matrix * (m_mesh->m_total_mass / total_volume);
		m_mesh->m_mass_matrix_1d = m_mesh->m_mass_matrix_1d * (m_mesh->m_total_mass / total_volume);

		std::vector<SparseMatrixTriplet> mass_inv_triplets;
		mass_inv_triplets.clear();
		std::vector<SparseMatrixTriplet> mass_inv_1d_triplets;
		mass_inv_1d_triplets.clear();
		for (unsigned int i = 0; i != m_mesh->m_mass_matrix.rows(); i++)
		{
			ScalarType mi = m_mesh->m_mass_matrix.coeff(i, i);
			ScalarType mi_inv;
			if (std::abs(mi) > 1e-12)
			{
				mi_inv = 1.0 / mi;
			}
			else
			{
				// ugly ugly!
				m_mesh->m_mass_matrix.coeffRef(i, i) = 1e-12;
				mi_inv = 1e12;
			}
			mass_inv_triplets.push_back(SparseMatrixTriplet(i, i, mi_inv));
		}
		for (unsigned int i = 0; i != m_mesh->m_mass_matrix_1d.rows(); i++)
		{
			ScalarType mi = m_mesh->m_mass_matrix_1d.coeff(i, i);
			ScalarType mi_inv;
			if (std::abs(mi) > 1e-12)
			{
				mi_inv = 1.0 / mi;
			}
			else
			{
				// ugly ugly!
				m_mesh->m_mass_matrix_1d.coeffRef(i, i) = 1e-12;
				mi_inv = 1e12;
			}
			mass_inv_1d_triplets.push_back(SparseMatrixTriplet(i, i, mi_inv));
		}

		m_mesh->m_inv_mass_matrix.setFromTriplets(mass_inv_triplets.begin(), mass_inv_triplets.end());
		m_mesh->m_inv_mass_matrix_1d.setFromTriplets(mass_inv_1d_triplets.begin(), mass_inv_1d_triplets.end());

#ifdef ENABLE_MATLAB_DEBUGGING
		g_debugger->SendSparseMatrix(m_mesh->m_mass_matrix, "M");
		g_debugger->SendSparseMatrix(m_mesh->m_inv_mass_matrix, "M_inv");
		g_debugger->SendSparseMatrix(m_mesh->m_mass_matrix_1d, "M_1d");
#endif
	}
	break;
	}
}

void Simulation::dampVelocity()//???????????е????????????Ч??
{
	if (std::abs(m_damping_coefficient) < EPSILON)
		return;

	m_mesh->m_current_velocities *= 1 - m_damping_coefficient;//??????????????0

	//// post-processing damping
	//EigenVector3 pos_mc(0.0, 0.0, 0.0), vel_mc(0.0, 0.0, 0.0);
	//unsigned int i, size;
	//ScalarType denominator(0.0), mass(0.0);
	//size = m_mesh->m_vertices_number;
	//for(i = 0; i < size; ++i)
	//{
	//	mass = m_mesh->m_mass_matrix.coeff(i*3, i*3);

	//	pos_mc += mass * m_mesh->m_current_positions.block_vector(i);
	//	vel_mc += mass * m_mesh->m_current_velocities.block_vector(i);
	//	denominator += mass;
	//}
	//assert(denominator != 0.0);
	//pos_mc /= denominator;
	//vel_mc /= denominator;

	//EigenVector3 angular_momentum(0.0, 0.0, 0.0), r(0.0, 0.0, 0.0);
	//EigenMatrix3 inertia, r_mat;
	//inertia.setZero(); r_mat.setZero();

	//for(i = 0; i < size; ++i)
	//{
	//	mass = m_mesh->m_mass_matrix.coeff(i*3, i*3);

	//	r = m_mesh->m_current_positions.block_vector(i) - pos_mc;
	//	angular_momentum += r.cross(mass * m_mesh->m_current_velocities.block_vector(i));

	//	//r_mat = EigenMatrix3(0.0,  r.z, -r.y,
	//	//					-r.z, 0.0,  r.x,
	//	//					r.y, -r.x, 0.0);

	//	r_mat.coeffRef(0, 1) = r[2];
	//	r_mat.coeffRef(0, 2) = -r[1];
	//	r_mat.coeffRef(1, 0) = -r[2];
	//	r_mat.coeffRef(1, 2) = r[0];
	//	r_mat.coeffRef(2, 0) = r[1];
	//	r_mat.coeffRef(2, 1) = -r[0];

	//	inertia += r_mat * r_mat.transpose() * mass;
	//}
	//EigenVector3 angular_vel = inertia.inverse() * angular_momentum;

	//EigenVector3 delta_v(0.0, 0.0, 0.0);
	//for(i = 0; i < size; ++i)
	//{
	//	r = m_mesh->m_current_positions.block_vector(i) - pos_mc;
	//	delta_v = vel_mc + angular_vel.cross(r) - m_mesh->m_current_velocities.block_vector(i);
	//	m_mesh->m_current_velocities.block_vector(i) += m_damping_coefficient * delta_v;
	//}
}

void Simulation::calculateExternalForce()
{
	m_external_force.resize(m_mesh->m_system_dimension);
	m_external_force.setZero();

	// gravity
	for (unsigned int i = 0; i < m_mesh->m_vertices_number; ++i)
	{
		m_external_force[3 * i + 1] += -m_gravity_constant;
		//wind
		m_external_force[3 * i + 0] += m_wind_x;
		m_external_force[3 * i + 1] += m_wind_y;
		m_external_force[3 * i + 2] += m_wind_z;
	}



#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendSparseMatrix(m_mesh->m_mass_matrix, "M");
#endif
	m_external_force = m_mesh->m_mass_matrix * m_external_force;
}

VectorX Simulation::collisionDetectionPostProcessing(const VectorX& x)
{//?????????????????????о???????????????
	// Naive implementation of collision detection
	VectorX penetration(m_mesh->m_system_dimension);
	penetration.setZero();
	EigenVector3 normal;
	ScalarType dist;

	for (unsigned int i = 0; i != m_mesh->m_vertices_number; ++i)
	{
		EigenVector3 xi = x.block_vector(i);

		if (m_scene->StaticIntersectionTest(xi, normal, dist))
		{
			penetration.block_vector(i) += (dist)*normal;//??????
		}
	}

	return penetration;
}

void Simulation::collisionDetection(const VectorX& x)
{
	if (!m_scene->IsEmpty())
	{
		m_collision_constraints.clear();

		EigenVector3 surface_point;
		EigenVector3 normal;
		ScalarType dist;

		for (unsigned int i = 0; i != m_mesh->m_vertices_number; ++i)
		{
			EigenVector3 xi = x.block_vector(i);

			if (m_scene->StaticIntersectionTest(xi, normal, dist))
			{
				surface_point = xi - normal * dist; // dist is negative...
				m_collision_constraints.push_back(CollisionSpringConstraint(1e3, i, surface_point, normal));
			}//?????????? ??????????
		}

	}
}

void Simulation::collisionResolution(const VectorX& penetration, VectorX& x, VectorX& v)
//?????????λ??????
{
	EigenVector3 xi, vi, pi, ni;
	EigenVector3 vin, vit;
	for (unsigned int i = 0; i != m_mesh->m_vertices_number; ++i)
	{
		xi = x.block_vector(i);
		vi = v.block_vector(i);
		pi = penetration.block_vector(i);

		ScalarType dist = pi.norm();//????????
		if (dist > EPSILON) // there is collision
		{
			ni = -pi / dist; // normalize
			xi -= pi;
			vin = vi.dot(ni) * ni;//??????
			vit = vi - vin;//??????
			vi = -(m_restitution_coefficient)*vin + (1 - m_friction_coefficient) * vit;
			//?????????????????????????????????????????????????С????????????????
			x.block_vector(i) = xi;
			v.block_vector(i) = vi;
		}
	}
}

void Simulation::integrateImplicitMethod()
{

	TimerWrapper myti, inteti, backtime, transtime;
	inteti.Tic();
	myti.Tic();
	const bool use_cs_ncg = use_cs && GenPDExperimentUsesCSNCG() && (m_optimization_method == OPTIMIZATION_METHOD_NCG);
	bool use_cs_gpu_state = use_cs_ncg && shouldUseCS2GpuState();
	if (m_cs_cpu_state_stale && !use_cs_gpu_state)
	{
		syncCS2GpuStateToCPU();
	}
	m_cs_render_position_valid = false;
	// take a initial guess
	VectorX x = m_y;
	//VectorX x = m_mesh->m_current_positions;

	// init method specific constants
	// for l-bfgs only
	if (m_lbfgs_restart_every_frame == true)
	{
		m_lbfgs_need_update_H0 = true;
	}
	ScalarType total_time = ScalarType(1e-5);


	VectorX gradient_dir;
	gradient_dir.resize(m_mesh->m_system_dimension);

	VectorX descent_dir;
	descent_dir.resize(m_mesh->m_system_dimension);

	if (!use_cs_ncg)
	{
		gradient_dir.setZero();
		descent_dir.setZero();
	}

	if (!use_cs_ncg)
	{
		if (m_optimization_method == OPTIMIZATION_METHOD_PNCG && m_enable_line_search)
		{
			m_ls_prefetched_energy = evaluateEnergyAndGradient(x, gradient_dir);
			m_ls_prefetched_gradient = gradient_dir;
		}
		else
		{
			evaluateGradient(x, gradient_dir);
		}
		descent_dir = -gradient_dir;
	}

	TimerWrapper t_optimization;
	t_optimization.Tic();
	g_lbfgs_timer.Tic();
	g_lbfgs_timer.Pause();
	// while loop until converge or exceeds maximum iterations
	bool converge = false;
	ScalarType beta = 0;
	m_ls_is_first_iteration = true;

	//for (m_current_iteration = 0; !converge && m_current_iteration < 20; ++m_current_iteration)
	//{
	//	if (m_processing_collision)
	//	{
	//		// Collision Detection every iteration
	//		collisionDetection(x_n);
	//	}
	//	g_integration_timer.Tic();
	//
	//	m_solver_type = SOLVER_TYPE_DIRECT_LLT;
	//	converge = performNewtonsMethodOneIteration(x_n);
	//
	//	m_ls_is_first_iteration = false;
	//	g_integration_timer.Toc();
	//}

	ScalarType m_double1x1_time[200];
	ScalarType m_double1x1_energy[200];

	if (m_step_mode)
	{
#ifdef ENABLE_MATLAB_DEBUGGING
		ScalarType energy = evaluateEnergy(x);
		//ScalarType energy = evaluatePotentialEnergy(x);
		VectorX gradient;
		evaluateGradient(x, gradient);

		ScalarType gradient_norm = gradient.norm();
		g_debugger->SendData(x, energy, gradient_norm, 0, total_time);


		m_double1x1_time[0] = total_time;
		m_double1x1_energy[0] = energy;

		/*	GenPDEnsureDirectoryForFile(filePath_e);
	std::ofstream outFile_e(filePath_e, std::ios::out | std::ios::app);

			if (!outFile_e.is_open() ) {
				std::cerr << "Failed to open file for writing." << std::endl;
			}*/

			//ScalarType error = (x - x_n).lpNorm<Eigen::Infinity>();;

			/*outFile_e << energy << std::endl;
			outFile_e.close();*/

#endif // ENABLE_MATLAB_DEBUGGING
	}


	m_ls_is_first_iteration = true;
	ResetCSNCGProfileMetrics();

	myti.Toc();

	myti.Report("front time:", m_verbose_show_optimization_time);

	transtime.Tic();
	ScalarType cs_y_upload_ms = 0.0;
	ScalarType cs_y_to_x_copy_ms = 0.0;

	if (use_cs_ncg)
	{
		uploadCSResourcesIfNeeded();

		const std::size_t vector_buffer_bytes = static_cast<std::size_t>(m_mesh->m_system_dimension) * sizeof(ScalarType);
		const std::size_t position_buffer_bytes = vector_buffer_bytes;
		EnsureCSBufferStorage(gradientID, vector_buffer_bytes, g_cs_gradient_buffer_bytes);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradientID);
		EnsureCSBufferStorage(DescentID, vector_buffer_bytes, g_cs_descent_buffer_bytes);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, DescentID);
		EnsureCSBufferStorage(xID, position_buffer_bytes, g_cs_x_buffer_bytes);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
		EnsureCSBufferStorage(m_yID, position_buffer_bytes, g_cs_y_buffer_bytes);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_yID);

		if (use_cs_gpu_state)
		{
			use_cs_gpu_state = predictCS2GpuStateY();
		}

		if (!use_cs_gpu_state)
		{
			if (m_cs_cpu_state_stale)
			{
				syncCS2GpuStateToCPU();
				computeConstantVectorsYandZ();
				x = m_y;
			}

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_yID);
			TimerWrapper y_upload_timer;
			y_upload_timer.Tic();
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, position_buffer_bytes, m_y.data());
			y_upload_timer.Toc();
			cs_y_upload_ms = y_upload_timer.DurationInSeconds() * 1000.0;
			glBindBuffer(GL_COPY_READ_BUFFER, m_yID);
			glBindBuffer(GL_COPY_WRITE_BUFFER, xID);
			TimerWrapper y_to_x_copy_timer;
			y_to_x_copy_timer.Tic();
			glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, position_buffer_bytes);
			y_to_x_copy_timer.Toc();
			cs_y_to_x_copy_ms = y_to_x_copy_timer.DurationInSeconds() * 1000.0;
			glBindBuffer(GL_COPY_READ_BUFFER, 0);
			glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		dispatchCSGradient(false);
		if (use_cs_gpu_state)
		{
			updateCSStats(false);
			if (m_cs_gradient_norm_sq < EPSILON * EPSILON)
			{
				m_cs_gradient_norm_sq = static_cast<ScalarType>(4.0) * EPSILON * EPSILON;
				m_cs_gradient_dot_descent = -m_cs_gradient_norm_sq;
			}
		}
		else
		{
			updateCSStats(true);
		}

		ScopedCSDebugGroup debug_group_descent("GenPD descent update");
		glUseProgram(descent_program);
		static GLint beta_uniform = -1;
		static GLint descent_mode_uniform = -1;
		if (beta_uniform < 0)
		{
			beta_uniform = glGetUniformLocation(descent_program, "beta_k");
			descent_mode_uniform = glGetUniformLocation(descent_program, "update_mode");
		}
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);
		glUniform1f(beta_uniform, 0.0f);
		glUniform1i(descent_mode_uniform, 0);
		++g_cs_profile_descent_dispatches;
		glDispatchCompute((gradient_dir.size() + 255) / 256, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

		m_cs_gradient_dot_descent = -m_cs_gradient_norm_sq;
	}

	transtime.Toc();
	transtime.Report("trans time:", m_verbose_show_optimization_time);




	TimerWrapper ti;
	ti.Tic();

	unsigned int iteration_budget = m_iterations_per_frame;
	if (use_cs_ncg && m_mesh && m_mesh->m_vertices_number >= kCSLargeClothVertexThreshold)
	{
		iteration_budget = 10;//std::min<unsigned int>(iteration_budget, kCSLargeClothIterationCap);
	}
	g_cs_active_iteration_budget = iteration_budget;
	g_cs_gpu_state_active_frame = use_cs_gpu_state;

	for (m_current_iteration = 0; !converge && m_current_iteration < iteration_budget; ++m_current_iteration)
	{
		//if (m_processing_collision)
		//{
		//	// Collision Detection every iteration
		//	collisionDetection(x);
		//}

		g_integration_timer.Tic();

		switch (m_optimization_method)
		{
		case OPTIMIZATION_METHOD_GRADIENT_DESCENT:
			converge = performGradientDescentOneIteration(x);
			break;
		case OPTIMIZATION_METHOD_NEWTON:
			//m_solver_type = SOLVER_TYPE_CG;
			converge = performNewtonsMethodOneIteration(x);
			break;
		case OPTIMIZATION_METHOD_NCG:


			if (use_cs_ncg)
			{
				converge = performNCG_CS2(x, beta, gradient_dir, descent_dir);
				//std::cout << "use_cs" << std::endl;
			}
			else
			{
				converge = performNCG(x, beta, gradient_dir, descent_dir);
			}

			break;
		case OPTIMIZATION_METHOD_PNCG:
			converge = performNCG_LBFGS(x, beta, gradient_dir, descent_dir);
			break;
		case OPTIMIZATION_METHOD_LBFGS:
			//m_solver_type = SOLVER_TYPE_CG;
			converge = performLBFGSOneIteration(x);
			break;
		case OPTIMIZATION_METHOD_LOCALGLOBAL:
			converge = integrateLocalGlobalOneIteration(x);
			break;
		default:
			break;
		}
		m_ls_is_first_iteration = false;
		g_integration_timer.Toc();

		if (m_verbose_show_converge)
		{
			if (converge && m_current_iteration != 0)
			{
				std::cout << "Optimization Converged in iteration #" << m_current_iteration << std::endl;
			}
		}

	}

	TimerWrapper getD;
	getD.Tic();
	ScalarType cs_x_readback_ms = 0.0;
	ScalarType cs_x_readback_wait_ms = 0.0;
	ScalarType cs_x_readback_copy_ms = 0.0;
	if (use_cs_ncg && !use_cs_gpu_state)
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
		TimerWrapper x_readback_wait_timer;
		x_readback_wait_timer.Tic();
		++g_cs_profile_solver_finish_calls;
		glFinish();
		x_readback_wait_timer.Toc();
		cs_x_readback_wait_ms = x_readback_wait_timer.DurationInSeconds() * 1000.0;

		TimerWrapper x_readback_copy_timer;
		x_readback_copy_timer.Tic();
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
			x.size() * sizeof(ScalarType),
			x.data());
		x_readback_copy_timer.Toc();
		cs_x_readback_copy_ms = x_readback_copy_timer.DurationInSeconds() * 1000.0;
		cs_x_readback_ms = cs_x_readback_wait_ms + cs_x_readback_copy_ms;
		g_cs_profile_xupdate_gpu_ms += ConsumeCSGpuTimerMs(g_cs_profile_xupdate_query, g_cs_profile_xupdate_query_pending);
		g_cs_profile_descent_gpu_ms += ConsumeCSGpuTimerMs(g_cs_profile_descent_query, g_cs_profile_descent_query_pending);
		if (!m_mesh || m_mesh->m_vertices_number < kCSLargeClothVertexThreshold)
		{
			readCSStatsFromGPU(true);
		}
		else
		{
			// The next frame refreshes CS stats before solver decisions; avoid this
			// large-cloth end-of-frame sync that only feeds reporting checks.
		}
	}

	getD.Toc();
	getD.Report("get data:", m_verbose_show_optimization_time);

	ScalarType max_displacement = 0.0;
	ScalarType max_position = 0.0;
	ScalarType gpu_state_update_ms = 0.0;
	ScalarType position_stats_ms = 0.0;
	bool x_is_finite = true;
	if (use_cs_gpu_state)
	{
		TimerWrapper gpu_state_timer;
		gpu_state_timer.Tic();
		const bool finalized_gpu_state = finalizeCS2GpuState(max_position, max_displacement, x_is_finite);
		gpu_state_timer.Toc();
		gpu_state_update_ms = gpu_state_timer.DurationInSeconds() * 1000.0;
		position_stats_ms = gpu_state_update_ms;
		if (!finalized_gpu_state)
		{
			x_is_finite = false;
			max_position = 1e30f;
		}
	}
	else
	{
		TimerWrapper position_stats_timer;
		position_stats_timer.Tic();
		x_is_finite = ComputePositionStats(x, m_mesh->m_current_positions, max_position, max_displacement);
		position_stats_timer.Toc();
		position_stats_ms = position_stats_timer.DurationInSeconds() * 1000.0;
	}

	m_last_profile_exploded = false;
	if (!x_is_finite || !std::isfinite(m_cs_gradient_norm_sq) || !std::isfinite(m_cs_gradient_dot_descent) || max_position > 1e3f)
	{
		if (use_cs_gpu_state)
		{
			invalidateCS2GpuState();
			use_cs_gpu_state = false;
		}
		m_last_profile_exploded = true;
		g_cs_unit_step_shortcut_budget = 0;
		g_cs_prefetched_energy_valid = false;
		x = m_mesh->m_current_positions;
		max_displacement = 0.0;
		max_position = VectorInfinityNorm(x);
		converge = true;
	}


	ti.Toc();
	ti.Report("iter time:", m_verbose_show_optimization_time);


	backtime.Tic();
	t_optimization.Toc();
	t_optimization.Report("Optimization", m_verbose_show_optimization_time);
	g_lbfgs_timer.Resume();
	g_lbfgs_timer.TocAndReport("L-BFGS overhead", m_verbose_show_converge, TIMER_OUTPUT_MILLISECONDS);

	m_last_profile_used_cs_ncg = use_cs_ncg;
	m_last_profile_converged = converge;
	m_last_profile_iterations = m_current_iteration;
	m_last_profile_front_ms = myti.DurationInSeconds() * 1000.0;
	m_last_profile_transfer_ms = transtime.DurationInSeconds() * 1000.0;
	m_last_profile_cs_y_upload_ms = cs_y_upload_ms;
	m_last_profile_cs_y_to_x_copy_ms = cs_y_to_x_copy_ms;
	m_last_profile_cs_x_readback_ms = cs_x_readback_ms;
	m_last_profile_cs_x_readback_wait_ms = cs_x_readback_wait_ms;
	m_last_profile_cs_x_readback_copy_ms = cs_x_readback_copy_ms;
	m_last_profile_iteration_ms = ti.DurationInSeconds() * 1000.0;
	m_last_profile_optimization_ms = t_optimization.DurationInSeconds() * 1000.0;
	m_last_profile_position_stats_ms = position_stats_ms;
	m_last_profile_step_size = m_ls_step_size;
	if (use_cs_ncg)
	{
		m_last_profile_objective_energy = g_cs_prefetched_energy_valid ? m_ls_prefetched_energy : 0.0;
	}
	else if (m_optimization_method == OPTIMIZATION_METHOD_PNCG && m_enable_line_search)
	{
		m_last_profile_objective_energy = m_ls_prefetched_energy;
	}
	else
	{
		try
		{
			m_last_profile_objective_energy = evaluateEnergy(x);
		}
		catch (const std::exception&)
		{
			m_last_profile_objective_energy = 0.0;
		}
	}
	m_last_profile_gradient_norm = use_cs_ncg ? std::sqrt(std::max<ScalarType>(0.0, m_cs_gradient_norm_sq)) : gradient_dir.norm();
	m_last_profile_max_displacement = max_displacement;
	m_last_profile_max_position = max_position;

	const bool used_gpu_state_update = use_cs_gpu_state && !m_last_profile_exploded;
	TimerWrapper update;
	update.Tic();

	// update constants
	if (!used_gpu_state_update)
	{
		updatePosAndVel(x);
	}

	update.Toc();
	update.Report("update", m_verbose_show_optimization_time);
	m_last_profile_update_posvel_ms = used_gpu_state_update ? gpu_state_update_ms : update.DurationInSeconds() * 1000.0;
	TimerWrapper colli;
	colli.Tic();

	if (!used_gpu_state_update)
	{
		collisionPostProcessCS(m_mesh->m_current_positions, m_mesh->m_current_velocities);
	}
	m_cs_render_position_valid = use_cs_ncg && !m_last_profile_exploded;

	colli.Toc();
	colli.Report("colli", m_verbose_show_optimization_time);
	m_last_profile_collision_ms = colli.DurationInSeconds() * 1000.0;
	backtime.Toc();
	backtime.Report("back time:", m_verbose_show_optimization_time);
	m_last_profile_back_ms = backtime.DurationInSeconds() * 1000.0;

	inteti.Toc();
	inteti.Report("integrate time", m_verbose_show_optimization_time);
	m_last_profile_total_ms = inteti.DurationInSeconds() * 1000.0;

}



bool Simulation::performGradientDescentOneIteration(VectorX& x)
{
	// evaluate gradient direction
	VectorX gradient;
	evaluateGradient(x, gradient);

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient, "g");
#endif

	if (gradient.norm() < EPSILON)
		return true;

	// assign descent direction
	//VectorX descent_dir = -m_mesh->m_inv_mass_matrix*gradient;
	VectorX descent_dir = -gradient;

	// line search
	ScalarType step_size = lineSearch(x, gradient, descent_dir);

	// update x
	x = x + descent_dir * step_size;

	// report convergence
	if (step_size < EPSILON)
		return true;
	else
		return false;
}
bool Simulation::performncg(VectorX& x)
{
	VectorX gradient_dir;
	evaluateGradient(x, gradient_dir);
	VectorX descent_dir = -gradient_dir;

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient_dir, "g");
#endif
	// evaluate hessian matrix
	SparseMatrix hessian_1;
	evaluateHessian(x, hessian_1);
	//SparseMatrix hessian_2;
	//evaluateHessianSmart(x, hessian_2);
	ScalarType beta;
	SparseMatrix& hessian = hessian_1;
	if (gradient_dir.norm() < EPSILON)
		return true;

	// assign descent direction
	//VectorX descent_dir = -m_mesh->m_inv_mass_matrix*gradient;


	// line search
	ScalarType step_size = lineSearch_ncg(x, gradient_dir, descent_dir, hessian);


	// update x
	x = x + descent_dir * step_size;

	/*VectorX gradient_dir_tmp = gradient_dir;

	evaluateGradient(x, gradient_dir);
	beta = gradient_dir.norm() * gradient_dir.norm() / (gradient_dir_tmp.norm() * gradient_dir_tmp.norm());

	descent_dir = -gradient_dir + beta * descent_dir;*/

	// report convergence
	if (-descent_dir.dot(gradient_dir) < EPSILON)
		return true;
	else
		return false;
}


bool Simulation::performNCG_CS2(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir)
{
	if (m_cs_gradient_norm_sq < EPSILON * EPSILON)
	{
		return true;
	}

	TimerWrapper t_linesearch;
	t_linesearch.Tic();
	lineSearch_CS(x, gradient_dir, descent_dir);
	t_linesearch.Toc();
	t_linesearch.Report("linesearch", m_verbose_show_optimization_time);
	g_cs_profile_linesearch_ms += t_linesearch.DurationInSeconds() * 1000.0;

	TimerWrapper t_xupdate;
	t_xupdate.Tic();
	glUseProgram(computeX_program);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ResultID);
	static GLint size_location = -1;
	if (size_location < 0)
	{
		size_location = glGetUniformLocation(computeX_program, "size");
	}
	glUniform1ui(size_location, static_cast<GLuint>(gradient_dir.size()));
	++g_cs_profile_xupdate_dispatches;
	const bool profile_xupdate_gpu = !g_cs_gpu_state_active_frame;
	if (profile_xupdate_gpu)
	{
		BeginCSGpuTimer(g_cs_profile_xupdate_query);
	}
	glDispatchCompute((descent_dir.size() + 255) / 256, 1, 1);
	if (profile_xupdate_gpu)
	{
		EndCSGpuTimer(g_cs_profile_xupdate_query_pending);
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	t_xupdate.Toc();
	t_xupdate.Report("t_xupdate", m_verbose_show_optimization_time);
	g_cs_profile_xupdate_ms += t_xupdate.DurationInSeconds() * 1000.0;

	if (m_current_iteration + 1u >= g_cs_active_iteration_budget)
	{
		return false;
	}

	TimerWrapper t_gradstats;
	t_gradstats.Tic();
	dispatchCSGradient(false, false);
	updateCSStats(false);
	t_gradstats.Toc();
	t_gradstats.Report("t_gradstats", m_verbose_show_optimization_time);
	g_cs_profile_gradstats_ms += t_gradstats.DurationInSeconds() * 1000.0;

	TimerWrapper t_descent;
	t_descent.Tic();
	ScopedCSDebugGroup debug_group_descent("GenPD descent update");
	glUseProgram(descent_program);
	static GLint beta_location = -1;
	static GLint descent_mode_location = -1;
	if (beta_location < 0)
	{
		beta_location = glGetUniformLocation(descent_program, "beta_k");
		descent_mode_location = glGetUniformLocation(descent_program, "update_mode");
	}
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);
	glUniform1f(beta_location, 0.0f);
	glUniform1i(descent_mode_location, 1);
	g_cs_profile_descent_dispatches += 2u;
	const bool profile_descent_gpu = !g_cs_gpu_state_active_frame;
	if (profile_descent_gpu)
	{
		BeginCSGpuTimer(g_cs_profile_descent_query);
	}
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	glUniform1i(descent_mode_location, 2);
	glDispatchCompute((descent_dir.size() + 255) / 256, 1, 1);
	if (profile_descent_gpu)
	{
		EndCSGpuTimer(g_cs_profile_descent_query_pending);
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	t_descent.Toc();
	t_descent.Report("t_descent", m_verbose_show_optimization_time);
	g_cs_profile_descent_ms += t_descent.DurationInSeconds() * 1000.0;
	return false;

}
//bool Simulation::performNCG_CS2(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir)
//{
//	// test2
//	ScalarType current_energy;
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		gradient_dir.size() * sizeof(float),
//		gradient_dir.data());
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		x.size() * sizeof(float),
//		x.data());
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, DescentID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		descent_dir.size() * sizeof(float),
//		descent_dir.data());
//
//
//	ScalarType alpha_k = lineSearch(x, gradient_dir, descent_dir);
//
//
//	// x = x + descent_dir * alpha_k;
//	//evaluateGradient(x, gradient_dir);
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER,
//		gradient_dir.size() * sizeof(ScalarType), gradient_dir.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradientID);
//
//	//// ??x????SSBO
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER, x.size() * sizeof(ScalarType),
//		x.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, DescentID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER, descent_dir.size() * sizeof(ScalarType),
//		descent_dir.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, DescentID);
//
//	// 更新 params UBO 数据并确保绑定到着色器使用的 binding = 0
//	comp_params.gradient_size = static_cast<int>(m_y.size());
//	comp_params.edge_size = static_cast<int>(m_mesh->m_edge_list.size());
//	comp_params.m_h = m_h;
//	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
//	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
//	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);
//
//	/*GLint b = glGetUniformLocation(computeX_program, "beta_k");
//	glUniform1f(b, beta);
//
//	ScopedCSDebugGroup debug_group_descent("GenPD descent update");
	//	glUseProgram(descent_program);
//	glDispatchCompute((descent_dir.size() + 255) / 256, 1, 1);
//	glMemoryBarrier(GL_ALL_BARRIER_BITS);*/
//
//	//// descent_dir = -gradient_dir + beta * descent_dir;
//
//	//std::cout << x[10];
//
//
//	////x = x + descent_dir * alpha_k;
//	glUseProgram(computeX_program);
//	GLint ce = glGetUniformLocation(computeX_program, "alpha_k");
//	glUniform1f(ce, alpha_k);
//	GLint size = glGetUniformLocation(computeX_program, "size");
//	glUniform1ui(size, static_cast<GLuint>(gradient_dir.size()));
//	glDispatchCompute((descent_dir.size() + 255) / 256, 1, 1);
//	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//	//// ????compute shader????
//	glUseProgram(gradient_program);
//	glDispatchCompute((m_mesh->m_edge_list.size() + 255) / 256, 1, 1);
//	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//	glUseProgram(compute_program);
//	glDispatchCompute((gradient_dir.size() + 255) / 256, 1, 1);
//	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		gradient_dir.size() * sizeof(float),
//		gradient_dir.data());
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		x.size() * sizeof(float),
//		x.data());
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, DescentID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		descent_dir.size() * sizeof(float),
//		descent_dir.data());
//
//
//
//
//	return true;
//
//}

//bool Simulation::performNCG_CS(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir)
//{
//	//const char* computeShaderSource = {};
//
//	VectorX LBFGS_Pk;
//
//	LBFGS_Pk = -gradient_dir;
//
//	ScalarType current_energy;
//
//#ifdef ENABLE_MATLAB_DEBUGGING
//	g_debugger->SendVector(gradient_dir, "g");
//#endif
//
//
//	if (gradient_dir.norm() < EPSILON)
//		return true;
//
//	int my_m = 2;
//
//	m_lbfgs_last_x = x;
//	m_lbfgs_last_gradient = gradient_dir;
//
//
//
//	descent_dir = LBFGS_Pk + beta * descent_dir;
//
//	ScalarType alpha_k = lineSearch(x, gradient_dir, descent_dir);
//
//	x = x + descent_dir * alpha_k;
//
//	// ??edge_list????SSBO
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER, m_mesh->m_edge_list.size() * sizeof(Edge),
//		m_mesh->m_edge_list.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, edgeID);
//
//	// ??gradient_dir????SSBO
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER,
//		m_mesh->m_vertices_number*3 * sizeof(ScalarType), gradient_dir.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradientID);
//
//	// ??x????SSBO
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
//	glBufferData(GL_SHADER_STORAGE_BUFFER, x.size() * sizeof(ScalarType),
//		x.data(), GL_DYNAMIC_DRAW);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);
//
//	VectorX gradient_dir_tmp = gradient_dir;
//
//	// ????compute shader?е?uniform????
//	GLint loc_edge_size = glGetUniformLocation(computeProgram, "edge_size");
//	glUniform1ui(loc_edge_size, m_mesh->m_edge_list.size());
//
//
//	// ????compute shader????
//	glUseProgram(computeProgram);
//	glDispatchCompute((m_mesh->m_edge_list.size() + 9) / 10, 1, 1);
//	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//	// ??gradient???????SSBO?ж???cpu
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradientID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		gradient_dir.size() * sizeof(float),
//		gradient_dir.data());
//	// ??x?????????cpu
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
//		x.size() * sizeof(float),
//		x.data());
//
//	//gradient_dir = m_mesh->m_mass_matrix * (x - m_y) + m_h * m_h * gradient_dir;
//	// ??????compute shader?? remain_todo
//	for (size_t i = 0; i < gradient_dir.size(); ++i) {
//		gradient_dir[i] = 1.0 * (x[i] - m_y[i]) + m_h * m_h * gradient_dir[i];
//	}
//
//	// ?????
//	//evaluateGradient(x, gradient_dir);
//
//
//
//
//
//	/*VectorX g_yk = gradient_dir - gradient_dir_tmp;
//	VectorX x_sk = x - m_lbfgs_last_x;*/
//	beta = gradient_dir.norm() * gradient_dir.norm() / (gradient_dir_tmp.norm() * gradient_dir_tmp.norm());
//
//	if (-descent_dir.dot(gradient_dir) < EPSILON_SQUARE)
//		return true;
//	else
//		return false;
//
//}

bool Simulation::performNCG(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir)
{

	VectorX LBFGS_Pk = -gradient_dir;

	VectorX gradient_dir_temp;
	ScalarType current_energy;



#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient_dir, "g");
#endif
	// evaluate hessian matrix
	/*SparseMatrix hessian_1;
	evaluateHessian(x, hessian_1);

	ScalarType beta;
	SparseMatrix& hessian = hessian_1;*/


	if (gradient_dir.norm() < EPSILON)
		return true;


	int my_m = 2;

	//std::cout << " m_current_iteration  #" << m_current_iteration;;


	m_lbfgs_last_x = x;
	// m_lbfgs_last_gradient = gradient_dir;


	// assign descent direction
	//VectorX descent_dir = -m_mesh->m_inv_mass_matrix*gradient;

	descent_dir = LBFGS_Pk + beta * descent_dir;

	// line search
	TimerWrapper ti;
	ti.Tic();
	ScalarType alpha_k = lineSearch(x, gradient_dir, descent_dir);
	ti.Toc();
	ti.Report("linesearch TIme :");

	// update x
	x = x + descent_dir * alpha_k;

	VectorX gradient_dir_tmp = gradient_dir;

	evaluateGradient(x, gradient_dir);
	 VectorX g_yk = gradient_dir - gradient_dir_tmp;
	 VectorX x_sk = x - m_lbfgs_last_x;
	TimerWrapper test;
	test.Tic();

	// FR
	//beta = gradient_dir.norm() * gradient_dir.norm() / (gradient_dir_tmp.norm() * gradient_dir_tmp.norm());

	// PR
	//beta = gradient_dir.dot(g_yk) / (gradient_dir_tmp.norm() * gradient_dir_tmp.norm());
	//std::cout << "   beta  #" << beta<<std::endl;

	// HS
	//beta = gradient_dir.dot(g_yk) / descent_dir.dot(g_yk);

	// Dy
	//beta= gradient_dir.norm() * gradient_dir.norm() / descent_dir.dot(g_yk);

	// DK
	//beta = gradient_dir.dot(g_yk) / descent_dir.dot(g_yk) - (g_yk.norm() * g_yk.norm() / x_sk.dot(g_yk)) * (gradient_dir.dot(x_sk) / descent_dir.dot(g_yk));
	//std::cout << "-descent_dir.dot(gradient_dir) #" << -descent_dir.dot(gradient_dir) << std::endl;

	// N
	beta = g_yk.dot(gradient_dir) / descent_dir.dot(g_yk) - 2 * (g_yk.norm() * g_yk.norm()) *
		descent_dir.dot(gradient_dir) / (descent_dir.dot(g_yk) * descent_dir.dot(g_yk));

	test.Toc();
	test.Report("beta time");
	if (-descent_dir.dot(gradient_dir) < EPSILON_SQUARE)
		return true;
	else
		return false;

	//// report convergence
	/*if (gradient_dir.norm() < EPSILON)
		return true;
	else
		return false;*/
}
bool Simulation::performNCG_LBFGS(VectorX& x, ScalarType& beta, VectorX& gradient_dir, VectorX& descent_dir)
{

	m_ncg_lbfgs_pk.resize(gradient_dir.size());
	m_ncg_lbfgs_pk = -gradient_dir;
	VectorX& LBFGS_Pk = m_ncg_lbfgs_pk;

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient_dir, "g");
#endif


	if (gradient_dir.squaredNorm() < EPSILON_SQUARE)
		return true;


	int my_m = 2;

	//std::cout << " m_current_iteration  #" << m_current_iteration;

	if (m_current_iteration == 0) {

		const unsigned int queue_vector_size = static_cast<unsigned int>(x.size());
		if (ncg_lbfgs_queue == NULL || ncg_lbfgs_queue->vectorSize() != static_cast<int>(queue_vector_size) || ncg_lbfgs_queue->capacity() != my_m)
		{
			delete ncg_lbfgs_queue;
			ncg_lbfgs_queue = new QueueLBFGS(queue_vector_size, my_m);
		}
		else
		{
			ncg_lbfgs_queue->clear();
		}
		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gradient_dir;
	}
	else {
		VectorX s_k = x - m_lbfgs_last_x;
		VectorX y_k = gradient_dir - m_lbfgs_last_gradient;

		// my implementation
		if (ncg_lbfgs_queue->isFull())
		{
			ncg_lbfgs_queue->dequeue();
		}
		ncg_lbfgs_queue->enqueue(s_k, y_k);

		int m_queue_size = ncg_lbfgs_queue->size();

		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gradient_dir;

		ScalarType alpha[2];
		ScalarType y_dot_s[2];

		int m_queue_visit_upper_bound = (my_m < m_queue_size) ? my_m : m_queue_size;
		ScalarType* s_i = NULL;
		ScalarType* y_i = NULL;

		for (int i = 0; i != m_queue_visit_upper_bound; i++) {
			ncg_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType yi_dot_si = y_i_eigen.dot(s_i_eigen);
			if (yi_dot_si < EPSILON_SQUARE)
				return true;
			const ScalarType alpha_i = s_i_eigen.dot(LBFGS_Pk) / yi_dot_si;
			alpha[i] = alpha_i;
			y_dot_s[i] = yi_dot_si;
			LBFGS_Pk -= alpha_i * y_i_eigen;

		}

		if (alpha[0] < EPSILON) // should not be negative
		{
			alpha[0] = EPSILON;
		}
		LBFGS_Pk *= alpha[0];


		for (int i = m_queue_visit_upper_bound - 1; i >= 0; i--) {
			ncg_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType beta = y_i_eigen.dot(LBFGS_Pk) / y_dot_s[i];
			LBFGS_Pk += s_i_eigen * (alpha[i] - beta);

		}


	}


	// assign descent direction
	//VectorX descent_dir = -m_mesh->m_inv_mass_matrix*gradient;

	ScalarType* const descent_data = descent_dir.data();
	const ScalarType* const lbfgs_pk_data = LBFGS_Pk.data();
	const int descent_size = static_cast<int>(descent_dir.size());
	for (int i = 0; i != descent_size; ++i)
	{
		descent_data[i] = beta * descent_data[i] + lbfgs_pk_data[i];
	}

	/*if (-descent_dir.dot(gradient_dir) < 0)
		return false;*/

	ScalarType alpha_k;
	if (m_enable_line_search)
	{
		alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(
			x,
			m_ls_prefetched_energy,
			gradient_dir,
			descent_dir,
			m_ls_prefetched_energy,
			m_ls_prefetched_gradient);
		x += alpha_k * descent_dir;
		gradient_dir = m_ls_prefetched_gradient;
	}
	else
	{
		alpha_k = lineSearch(x, gradient_dir, descent_dir);
		x += alpha_k * descent_dir;
		evaluateGradient(x, gradient_dir);
	}
	ScalarType descent_dot_y = 0;
	ScalarType gradient_dot_y = 0;
	ScalarType y_norm_sq = 0;
	ScalarType x_dot_y = 0;
	ScalarType gradient_dot_x = 0;
	const ScalarType* const current_gradient = gradient_dir.data();
	const ScalarType* const previous_gradient = m_lbfgs_last_gradient.data();
	const ScalarType* const current_x = x.data();
	const ScalarType* const previous_x = m_lbfgs_last_x.data();
	const ScalarType* const current_descent = descent_dir.data();
	const int beta_update_size = static_cast<int>(gradient_dir.size());
	for (int i = 0; i != beta_update_size; ++i)
	{
		const ScalarType y_delta = current_gradient[i] - previous_gradient[i];
		const ScalarType x_delta = current_x[i] - previous_x[i];
		descent_dot_y += current_descent[i] * y_delta;
		gradient_dot_y += current_gradient[i] * y_delta;
		y_norm_sq += y_delta * y_delta;
		x_dot_y += x_delta * y_delta;
		gradient_dot_x += current_gradient[i] * x_delta;
	}
	beta = gradient_dot_y / descent_dot_y - (y_norm_sq / x_dot_y) * (gradient_dot_x / descent_dot_y);


	/*if (-descent_dir.dot(gradient_dir) < EPSILON_SQUARE)
		return true;
	else
		return false;*/

		// report convergence
		/*if (gradient_dir.norm() < EPSILON)
			return true;
		else
			return false;*/
	return false;

}

bool Simulation::integrateLocalGlobalOneIteration(VectorX& x)
{
	prefactorize(PREFACTOR_M_PLUS_H2L);

	VectorX d;
	evaluateDVector(x, d);

	VectorX b = m_mesh->m_mass_matrix * m_y + m_h * m_h * (m_J_matrix * d + m_external_force);

	x = m_prefactored_solver_1[PREFACTOR_M_PLUS_H2L].solve(b);

	return false;
}

bool Simulation::performNewtonsMethodOneIteration(VectorX& x)
{
	TimerWrapper timer; timer.Tic();
	// evaluate gradient direction
	VectorX gradient;
	evaluateGradient(x, gradient, true);
	//QSEvaluateGradient(x, gradient, m_ss->m_quasi_static);
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient, "g");
#endif

	timer.TocAndReport("evaluate gradient", m_verbose_show_converge);
	timer.Tic();

	// evaluate hessian matrix
	SparseMatrix hessian_1;
	evaluateHessian(x, hessian_1);
	//SparseMatrix hessian_2;
	//evaluateHessianSmart(x, hessian_2);

	SparseMatrix& hessian = hessian_1;

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendSparseMatrix(hessian_1, "H");
	//g_debugger->SendSparseMatrix(hessian_2, "H2");
#endif

	timer.TocAndReport("evaluate hessian", m_verbose_show_converge);
	timer.Tic();
	VectorX descent_dir;

	linearSolve(descent_dir, hessian, gradient);
	//????????????????????????CG???
	descent_dir = -descent_dir;

	timer.TocAndReport("solve time", m_verbose_show_converge);
	timer.Tic();

	// line search
	ScalarType step_size = lineSearch(x, gradient, descent_dir);
	//if (step_size < EPSILON)
	//{
	//	std::cout << "correct step size to 1" << std::endl;
	//	step_size = 1;
	//}
	// update x
	x = x + descent_dir * step_size;

	//if (step_size < EPSILON)
	//{
	//	printVolumeTesting(x);
	//}

	timer.TocAndReport("line search", m_verbose_show_converge);
	//timer.Toc();
	//std::cout << "newton: " << timer.Duration() << std::endl;

	if (-descent_dir.dot(gradient) < EPSILON_SQUARE)
		return true;
	else
		return false;
}

bool Simulation::performLBFGSOneIteration(VectorX& x)
{
	bool converged = false;
	ScalarType current_energy;
	VectorX gf_k;

	// set xk and gfk
	if (m_ls_is_first_iteration || !m_enable_line_search)
	{
		current_energy = evaluateEnergyAndGradient(x, gf_k);
	}
	else
	{
		current_energy = m_ls_prefetched_energy;
		gf_k = m_ls_prefetched_gradient;
	}
	//current_energy = evaluateEnergyAndGradient(x, gf_k);

	if (m_lbfgs_need_update_H0) // first iteration
	{
		// clear sk and yk and alpha_k
#ifdef USE_STL_QUEUE_IMPLEMENTATION
		// stl implementation
		m_lbfgs_y_queue.clear();
		m_lbfgs_s_queue.clear();
#else
		// my implementation
		delete m_lbfgs_queue;
		m_lbfgs_queue = new QueueLBFGS(x.size(), m_lbfgs_m);
#endif

		// decide H0 and it's factorization precomputation
		switch (m_lbfgs_H0_type)
		{
		case LBFGS_H0_LAPLACIAN:
			prefactorize();
			break;
		default:
			//prefactorize();
			break;
		}

		g_lbfgs_timer.Resume();
		// store them before wipeout
		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gf_k;

		g_lbfgs_timer.Pause();
		// first iteration
		VectorX r;
		LBFGSKernelLinearSolve(r, gf_k, 1);
		g_lbfgs_timer.Resume();

		// update
		VectorX p_k = -r;
		g_lbfgs_timer.Pause();

		if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.norm() / x.norm() < LARGER_EPSILON)
		{
			converged = true;
		}

		ScalarType alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(x, current_energy, gf_k, p_k, m_ls_prefetched_energy, m_ls_prefetched_gradient);
		x += alpha_k * p_k;

		// final touch
		m_lbfgs_need_update_H0 = false;
	}
	else // otherwise
	{
		TimerWrapper t_local;
		TimerWrapper t_global;
		TimerWrapper t_linesearch;
		TimerWrapper t_other;
		bool verbose = false;

		t_other.Tic();
		g_lbfgs_timer.Resume();
		// enqueue stuff
		VectorX s_k = x - m_lbfgs_last_x;
		VectorX y_k = gf_k - m_lbfgs_last_gradient;

#ifdef USE_STL_QUEUE_IMPLEMENTATION
		//stl implementation
		if (m_lbfgs_s_queue.size() > m_lbfgs_m)
		{
			m_lbfgs_s_queue.pop_back();
			m_lbfgs_y_queue.pop_back();
		}
		// enqueue stuff
		m_lbfgs_s_queue.push_front(s_k);
		m_lbfgs_y_queue.push_front(y_k);

		int m_queue_size = m_lbfgs_s_queue.size();
#else
		// my implementation
		if (m_lbfgs_queue->isFull())
		{
			m_lbfgs_queue->dequeue();
		}
		m_lbfgs_queue->enqueue(s_k, y_k);

		int m_queue_size = m_lbfgs_queue->size();
#endif

		// store them before wipeout
		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gf_k;
		VectorX q = gf_k;

		// loop 1 of l-BFGS
		std::vector<ScalarType> rho;
		rho.clear();
		std::vector<ScalarType> alpha;
		alpha.clear();
		int m_queue_visit_upper_bound = (m_lbfgs_m < m_queue_size) ? m_lbfgs_m : m_queue_size;
		ScalarType* s_i = NULL;
		ScalarType* y_i = NULL;
		for (int i = 0; i != m_queue_visit_upper_bound; i++)
		{
#ifdef USE_STL_QUEUE_IMPLEMENTATION
			// stl implementation
			ScalarType yi_dot_si = m_lbfgs_y_queue[i].dot(m_lbfgs_s_queue[i]);
			if (yi_dot_si < EPSILON_SQUARE)
			{
				return true;
			}
			ScalarType rho_i = 1.0 / yi_dot_si;
			rho.push_back(rho_i);
			alpha.push_back(rho[i] * m_lbfgs_s_queue[i].dot(q));
			q = q - alpha[i] * m_lbfgs_y_queue[i];
#else
			// my implementation
			m_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType yi_dot_si = (y_i_eigen.dot(s_i_eigen));
			if (yi_dot_si < EPSILON_SQUARE)
			{
				return true;
			}
			ScalarType rho_i = 1.0 / yi_dot_si;
			rho.push_back(rho_i);
			ScalarType alpha_i = rho_i * s_i_eigen.dot(q);
			alpha.push_back(alpha_i);
			q -= alpha_i * y_i_eigen;
#endif
		}
		// compute H0 * q
		g_lbfgs_timer.Pause();
		t_other.Pause();
		t_global.Tic();
		VectorX r;
		// compute the scaling parameter on the fly
		ScalarType scaling_parameter = (s_k.transpose() * y_k).trace() / (y_k.transpose() * y_k).trace();
		if (scaling_parameter < EPSILON) // should not be negative
		{
			scaling_parameter = EPSILON;
		}
		LBFGSKernelLinearSolve(r, q, scaling_parameter);
		t_global.Toc();
		t_other.Resume();
		g_lbfgs_timer.Resume();
		// loop 2 of l-BFGS
		for (int i = m_queue_visit_upper_bound - 1; i >= 0; i--)
		{
#ifdef USE_STL_QUEUE_IMPLEMENTATION
			// stl implementation
			ScalarType beta = rho[i] * m_lbfgs_y_queue[i].dot(r);
			r = r + m_lbfgs_s_queue[i] * (alpha[i] - beta);
#else
			// my implementation
			m_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType beta = rho[i] * y_i_eigen.dot(r);
			r += s_i_eigen * (alpha[i] - beta);
#endif
		}
		// update
		VectorX p_k = -r;
		if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.squaredNorm() < EPSILON_SQUARE)
		{
			converged = true;
		}
		g_lbfgs_timer.Pause();
		t_other.Toc();

		t_linesearch.Tic();
		//ScalarType alpha_k = lineSearch(x, gf_k, p_k);
		ScalarType alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(x, current_energy, gf_k, p_k, m_ls_prefetched_energy, m_ls_prefetched_gradient);
		t_linesearch.Toc();

		x += alpha_k * p_k;

		t_global.Report("Forward Backward Substitution", verbose, TIMER_OUTPUT_MICROSECONDS);
		t_other.Report("Two loop overhead", verbose, TIMER_OUTPUT_MICROSECONDS);
		t_linesearch.Report("Linesearch", verbose, TIMER_OUTPUT_MICROSECONDS);
	}

	return converged;
}

void Simulation::LBFGSKernelLinearSolve(VectorX& r, VectorX rhs, ScalarType scaled_identity_constant) // Ar = rhs
{
	r.resize(rhs.size());
	switch (m_lbfgs_H0_type)
	{
	case LBFGS_H0_IDENTITY:
		r = rhs / scaled_identity_constant;
		break;
	case LBFGS_H0_LAPLACIAN: // h^2*laplacian+mass
	{
		// solve the linear system in reduced dimension because of the pattern of the Laplacian matrix
		// convert to nx3 space
		EigenMatrixx3 rhs_n3(rhs.size() / 3, 3);
		Vector3mx1ToMatrixmx3(rhs, rhs_n3);
		// solve using the nxn laplacian
		EigenMatrixx3 r_n3;
		if (m_solver_type == SOLVER_TYPE_CG)
		{
			m_preloaded_cg_solver_1D.setMaxIterations(m_iterative_solver_max_iteration);
			r_n3 = m_preloaded_cg_solver_1D.solve(rhs_n3);
		}
		else
		{
			r_n3 = m_prefactored_solver_1D.solve(rhs_n3);
		}
		// convert the result back
		Matrixmx3ToVector3mx1(r_n3, r);


		////// conventional solve using 3nx3n system
		//if (m_solver_type == SOLVER_TYPE_CG)
		//{
		//	m_preloaded_cg_solver.setMaxIterations(m_iterative_solver_max_iteration);
		//	r = m_preloaded_cg_solver.solve(rhs);
		//}
		//else
		//{
		//	r = m_prefactored_solver.solve(rhs);
		//}
	}
	break;
	default:
		break;
	}
}

void Simulation::computeConstantVectorsYandZ()
{

	//Eigen::VectorXd increment(3);
	//increment << 1.0, 0.0, 0.0;
	//VectorX
	//m_mesh->m_current_velocities += increment;

	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		m_y = m_mesh->m_current_positions;
		break;
	case INTEGRATION_IMPLICIT_EULER:
		m_y = m_mesh->m_current_positions + m_mesh->m_current_velocities * m_h + m_h * m_h * m_mesh->m_inv_mass_matrix * m_external_force;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		m_y = (4 * m_mesh->m_current_positions - m_mesh->m_previous_positions) / 3 + (4 * m_mesh->m_current_velocities - m_mesh->m_previous_velocities + 2 * m_h * m_mesh->m_inv_mass_matrix * m_external_force) * m_h * 2.0 / 9.0;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		m_y = m_mesh->m_current_positions + m_mesh->m_current_velocities * m_h + 0.5 * m_h * m_h * m_mesh->m_inv_mass_matrix * m_external_force;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		m_y = m_mesh->m_current_positions + m_mesh->m_current_velocities * m_h + 0.5 * m_h * m_h * m_mesh->m_inv_mass_matrix * m_external_force;
		evaluateGradientPureConstraint(m_mesh->m_current_positions, m_external_force, m_z);
		break;
	default:
		break;
	}
}

void Simulation::updatePosAndVel(const VectorX& new_pos)
{
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		m_mesh->m_previous_positions = m_mesh->m_current_positions;
		m_mesh->m_current_positions = new_pos;
		break;
	case INTEGRATION_IMPLICIT_EULER:
		m_mesh->m_previous_velocities = m_mesh->m_current_velocities;
		m_mesh->m_previous_positions = m_mesh->m_current_positions;
		m_mesh->m_current_velocities = (new_pos - m_mesh->m_current_positions) / m_h;
		m_mesh->m_current_positions = new_pos;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
	{
		m_mesh->m_previous_velocities = m_mesh->m_current_velocities;
		m_mesh->m_current_velocities = 1.5 * (new_pos - (4 * m_mesh->m_current_positions - m_mesh->m_previous_positions) / 3) / m_h;;
		m_mesh->m_previous_positions = m_mesh->m_current_positions;
		m_mesh->m_current_positions = new_pos;
	}
	break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		m_mesh->m_previous_velocities = m_mesh->m_current_velocities;
		m_mesh->m_previous_positions = m_mesh->m_current_positions;
		m_mesh->m_current_velocities = 2 * (new_pos - m_mesh->m_current_positions) / m_h - m_mesh->m_current_velocities;
		m_mesh->m_current_positions = new_pos;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		m_mesh->m_previous_velocities = m_mesh->m_current_velocities;
		m_mesh->m_previous_positions = m_mesh->m_current_positions;
		m_mesh->m_current_velocities = 2 * (new_pos - m_mesh->m_current_positions) / m_h - m_mesh->m_current_velocities;
		m_mesh->m_current_positions = new_pos;
		break;
	default:
		break;
	}
}

ScalarType Simulation::evaluateEnergy(const VectorX& x)
{
	ScalarType energy_pure_constraints, energy;

	ScalarType inertia_term = 0.5 * (x - m_y).transpose() * m_mesh->m_mass_matrix * (x - m_y);
	ScalarType h_square = m_h * m_h;
	//std::cout << inertia_term << std::endl;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		energy = evaluateEnergyPureConstraint(x, m_external_force);
		energy -= m_external_force.dot(x);
		break;
	case INTEGRATION_IMPLICIT_EULER:
		energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
		energy = inertia_term + h_square * energy_pure_constraints;
		//std::cout << "IMPLICIT " << energy <<std::endl;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
		energy = inertia_term + h_square * 4.0 / 9.0 * energy_pure_constraints;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		energy_pure_constraints = evaluateEnergyPureConstraint((x + m_mesh->m_current_positions) / 2, m_external_force);
		energy = inertia_term + h_square * (energy_pure_constraints);
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
		energy = inertia_term + h_square / 4 * (energy_pure_constraints + m_z.dot(x));
		break;
	}

	//std::cout << "energy " << energy << std::endl;

	return energy;
}

void Simulation::evaluateGradient(const VectorX& x, VectorX& gradient, bool enable_omp)
{
	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient -= m_external_force;
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square * gradient;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + (h_square * 4.0 / 9.0) * gradient;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateGradientPureConstraint((x + m_mesh->m_current_positions) / 2, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 2 * (gradient);
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 4 * (gradient + m_z);
		break;
	}
}

ScalarType Simulation::evaluatePotentialEnergyCS(const VectorX& x)
{
	if (!use_cs)
	{
		ScalarType energy = evaluateEnergyPureConstraint(x, m_external_force);
		energy -= m_external_force.dot(x);
		return energy;
	}

	uploadCSResourcesIfNeeded();

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, xID);
	glBufferData(GL_SHADER_STORAGE_BUFFER, x.size() * sizeof(ScalarType), x.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xID);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_yID);
	glBufferData(GL_SHADER_STORAGE_BUFFER, m_y.size() * sizeof(ScalarType), m_y.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_yID);

	const GLuint inertia_partial_group_count = ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.gradient_size));
	const GLuint energy_partial_group_count = ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.edge_size));
	const GLuint objective_partial_group_count = std::max(inertia_partial_group_count, energy_partial_group_count);
	EnsureFloatScratchBuffer(
		testID,
		test_,
		static_cast<std::size_t>(objective_partial_group_count));

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, energyID);

	comp_params.t0 = 0.0f;
	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);
	glUseProgram(energy_for_linesearch_program);
	glDispatchCompute(1, objective_partial_group_count, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	DispatchCSReduction(compute_program, testID, energyID, objective_partial_group_count, 1u);

	ScalarType energy = 0.0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, energyID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(ScalarType), &energy);

	if (m_processing_collision)
	{
		energy += evaluateEnergyCollision(x);
	}

	energy -= m_external_force.dot(x);
	return energy;
}

ScalarType Simulation::evaluateEnergyAndGradient(const VectorX& x, VectorX& gradient)
{
	ScalarType h_square = m_h * m_h;
	ScalarType energy_pure_constraints, energy;
	if (m_integration_method == INTEGRATION_IMPLICIT_EULER &&
		m_cs_mass_diagonal.size() == static_cast<std::size_t>(x.size()))
	{
		ScalarType inertia_term = 0.5 * (x - m_y).transpose() * m_mesh->m_mass_matrix * (x - m_y);
		energy_pure_constraints = evaluateEnergyAndGradientPureConstraint(x, m_external_force, gradient);

		const ScalarType* const mass_diagonal = m_cs_mass_diagonal.data();
		const ScalarType* const x_data = x.data();
		const ScalarType* const y_data = m_y.data();
		const int system_dimension = static_cast<int>(x.size());
		for (int i = 0; i != system_dimension; ++i)
		{
			const ScalarType delta = x_data[i] - y_data[i];
			gradient[i] = mass_diagonal[i] * delta + h_square * gradient[i];
		}
		energy = inertia_term + h_square * energy_pure_constraints;
		return energy;
	}
	ScalarType inertia_term = 0.5 * (x - m_y).transpose() * m_mesh->m_mass_matrix * (x - m_y);

	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		energy = evaluateEnergyAndGradientPureConstraint(x, m_external_force, gradient);
		energy -= m_external_force.dot(x);
		gradient -= m_external_force;
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		energy_pure_constraints = evaluateEnergyAndGradientPureConstraint(x, m_external_force, gradient);
		energy = inertia_term + h_square * energy_pure_constraints;
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square * gradient;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		energy_pure_constraints = evaluateEnergyAndGradientPureConstraint(x, m_external_force, gradient);
		energy = inertia_term + h_square * 4.0 / 9.0 * energy_pure_constraints;
		gradient = m_mesh->m_mass_matrix * (x - m_y) + (h_square * 4.0 / 9.0) * gradient;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		energy_pure_constraints = evaluateEnergyAndGradientPureConstraint((x + m_mesh->m_current_positions) / 2, m_external_force, gradient);
		energy = inertia_term + h_square * (energy_pure_constraints);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 2 * (gradient);
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		energy_pure_constraints = evaluateEnergyAndGradientPureConstraint(x, m_external_force, gradient);
		energy = inertia_term + h_square / 4 * (energy_pure_constraints + m_z.dot(x));
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 4 * (gradient + m_z);
		break;
	}

	return energy;
}

void Simulation::evaluateHessian(const VectorX& x, SparseMatrix& hessian_matrix)
{
	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateHessianPureConstraint(x, hessian_matrix);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square * 4.0 / 9.0 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateHessianPureConstraint((x + m_mesh->m_current_positions) / 2, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	}
}

void Simulation::evaluateHessianSmart(const VectorX& x, SparseMatrix& hessian_matrix)
{
	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square * 4.0 / 9.0 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateHessianPureConstraintSmart((x + m_mesh->m_current_positions) / 2, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	}
}

// evaluate hessian
void Simulation::evaluateHessianForCG(const VectorX& x)
{
	VectorX x_evaluated_point;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
	case INTEGRATION_IMPLICIT_EULER:
	case INTEGRATION_IMPLICIT_BDF2:
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		x_evaluated_point = x;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		x_evaluated_point = (x + m_mesh->m_current_positions) / 2;
		break;
	}

	if (!m_enable_openmp)
	{
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			(*it)->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
		}

		if (m_processing_collision)
		{
			for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
			{
				it->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
		}
	}
	else
	{
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
#pragma omp for
			for (i = 0; i < m_collision_constraints.size(); i++)
			{
				m_collision_constraints[i].EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
		}
	}
}
// apply hessian
void Simulation::applyHessianForCG(const VectorX& x, VectorX& b)
{
	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		applyHessianForCGPureConstraint(x, b);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square * b;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square * 4.0 / 9.0 * b;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square / 4 * b;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square / 4 * b;
		break;
	}

}

void Simulation::evaluateLaplacian(SparseMatrix& laplacian_matrix)
{
	evaluateLaplacianPureConstraint(laplacian_matrix);

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendSparseMatrix(m_weighted_laplacian, "L");
#endif

	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		laplacian_matrix = m_mesh->m_mass_matrix + h_square * laplacian_matrix;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		laplacian_matrix = m_mesh->m_mass_matrix + h_square * 4.0 / 9.0 * laplacian_matrix;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		laplacian_matrix = m_mesh->m_mass_matrix + h_square / 4 * laplacian_matrix;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		laplacian_matrix = m_mesh->m_mass_matrix + h_square / 4 * laplacian_matrix;
		break;
	}

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendSparseMatrix(laplacian_matrix, "A");
#endif

}

void Simulation::evaluateLaplacian1D(SparseMatrix& laplacian_matrix_1d)
{
	evaluateLaplacianPureConstraint1D(laplacian_matrix_1d);

	ScalarType h_square = m_h * m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		laplacian_matrix_1d = m_mesh->m_mass_matrix_1d + h_square * laplacian_matrix_1d;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		laplacian_matrix_1d = m_mesh->m_mass_matrix_1d + h_square * 4.0 / 9.0 * laplacian_matrix_1d;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		laplacian_matrix_1d = m_mesh->m_mass_matrix_1d + h_square / 4 * laplacian_matrix_1d;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		laplacian_matrix_1d = m_mesh->m_mass_matrix_1d + h_square / 4 * laplacian_matrix_1d;
		break;
	}
}

ScalarType Simulation::getVolume(const VectorX& x)
{
	ScalarType volume = 0.0;
	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		volume += (*it)->GetVolume(x);
	}

	return volume;
}

void Simulation::printVolumeTesting(const VectorX& x)
{
	unsigned int index = 0;
	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		if ((*it)->Type() == CONSTRAINT_TYPE_TET)
		{
			TetConstraint* tc = dynamic_cast<TetConstraint*> (*it);
			ScalarType rest_vol = tc->GetVolume(m_mesh->m_restpose_positions);
			ScalarType vol = tc->GetVolume(x);

			if (vol / rest_vol < 0.1)
			{
				std::cout << "Volume of element: " << index << std::endl;
				std::cout << "rest pose vol: " << rest_vol << std::endl;
				std::cout << "current pose vol: " << vol << std::endl << std::endl;
			}

			index++;
		}
	}
}

ScalarType Simulation::evaluateEnergyPureConstraint(const VectorX& x, const VectorX& f_ext)
{
	ScalarType energy = 0.0;

	if (!m_enable_openmp)
	{
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			energy += (*it)->EvaluateEnergy(x);
		}
	}
	else
	{
		// openmp get all energy
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->EvaluateEnergy(x);
			}
		}

		// reduction
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			energy += (*it)->GetEnergy();
		}
	}

	if (m_processing_collision)
	{
		energy += evaluateEnergyCollision(x);
	}

	return energy;
}


void Simulation::Evaluatespringlength(const VectorX& x)
{
	ScalarType length = 0.0;

	if (!m_enable_openmp)
	{
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			length += (*it)->Evaluatelength(x);
		}
	}
	else
	{
		// openmp get all energy
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->Evaluatelength(x);
			}
		}

		// reduction
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			length += (*it)->Getlength();
		}
	}

	std::string filePath_e = GenPDResolveOutputPath("stress.txt");
	GenPDEnsureDirectoryForFile(filePath_e);
	std::ofstream outFile_e(filePath_e, std::ios::out | std::ios::app);

	if (!outFile_e.is_open()) {
		std::cerr << "Failed to open file for writing." << std::endl;
	}
	outFile_e << length << std::endl;
	outFile_e.close();


}

void Simulation::evaluateGradientPureConstraint(const VectorX& x, const VectorX& f_ext, VectorX& gradient)
{
	gradient.resize(m_mesh->m_system_dimension);
	gradient.setZero();

	if (!m_enable_openmp)
	{
		// constraints single thread
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			(*it)->EvaluateGradient(x, gradient);
		}
	}
	else
	{
		// constraints omp
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->EvaluateGradient(x);
			}
		}

		for (i = 0; i < m_constraints.size(); i++)
		{
			m_constraints[i]->GetGradient(gradient);
		}
	}

	// hardcoded collision plane
	if (m_processing_collision)
	{
		VectorX gc;

		evaluateGradientCollision(x, gc);

		gradient += gc;
	}
}
ScalarType Simulation::evaluateEnergyAndGradientPureConstraint(const VectorX& x, const VectorX& f_ext, VectorX& gradient)
{
	ScalarType energy = 0.0;
	gradient.resize(m_mesh->m_system_dimension);
	gradient.setZero();

	if (!m_enable_openmp)
	{
		// constraints single thread
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			energy += (*it)->EvaluateEnergyAndGradient(x, gradient);
		}
	}
	else
	{
		// constraints omp
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->EvaluateEnergyAndGradient(x);
			}
		}

		// collect the results in a sequential way
		for (i = 0; i < m_constraints.size(); i++)
		{
			energy += m_constraints[i]->GetEnergyAndGradient(gradient);
		}
	}

	// hardcoded collision plane
	if (m_processing_collision)
	{
		VectorX gc;

		energy += evaluateEnergyAndGradientCollision(x, gc);

		gradient += gc;
	}

	return energy;
}

void Simulation::evaluateHessianPureConstraint(const VectorX& x, SparseMatrix& hessian_matrix)
{
	hessian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> h_triplets;
	h_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateHessian(x, h_triplets, m_definiteness_fix);
	}

	hessian_matrix.setFromTriplets(h_triplets.begin(), h_triplets.end());

	if (m_processing_collision)
	{
		SparseMatrix HC;
		evaluateHessianCollision(x, HC);
		hessian_matrix += HC;
	}
}

void Simulation::evaluateHessianPureConstraintSmart(const VectorX& x, SparseMatrix& hessian_matrix)
{
	hessian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> h_triplets;
	h_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateHessian(x, h_triplets, m_definiteness_fix);
	}

	// sort triplets using ascent order of ||triplet.value()||
	std::sort(h_triplets.begin(), h_triplets.end(), compareTriplet);

	hessian_matrix.setFromTriplets(h_triplets.begin(), h_triplets.end());

	if (m_processing_collision)
	{
		SparseMatrix HC;
		evaluateHessianCollision(x, HC);
		hessian_matrix += HC;
	}
}

void Simulation::evaluateLaplacianPureConstraint(SparseMatrix& laplacian_matrix)
{
	laplacian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> l_triplets;
	l_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateWeightedLaplacian(l_triplets);
	}

	laplacian_matrix.setFromTriplets(l_triplets.begin(), l_triplets.end());
}

void Simulation::evaluateLaplacianPureConstraint1D(SparseMatrix& laplacian_matrix_1d)
{
	laplacian_matrix_1d.resize(m_mesh->m_vertices_number, m_mesh->m_vertices_number);
	std::vector<SparseMatrixTriplet> l_1d_triplets;
	l_1d_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateWeightedLaplacian1D(l_1d_triplets);
	}

	laplacian_matrix_1d.setFromTriplets(l_1d_triplets.begin(), l_1d_triplets.end());
}

void Simulation::applyHessianForCGPureConstraint(const VectorX& x, VectorX& b)
{
	b.resize(x.size());
	b.setZero();
	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->ApplyHessian(x, b);
	}
}

ScalarType Simulation::evaluateEnergyCollision(const VectorX& x)
{
	ScalarType energy = 0.0;

	if (!m_enable_openmp)
	{
		for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
		{
			energy += it->EvaluateEnergy(x);
		}
	}
	else
	{
		// openmp get all energy
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_collision_constraints.size(); i++)
			{
				m_collision_constraints[i].EvaluateEnergy(x);
			}
		}

		// reduction
		for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
		{
			energy += it->GetEnergy();
		}
	}

	return energy;
}
void Simulation::evaluateGradientCollision(const VectorX& x, VectorX& gradient)
{
	gradient.resize(m_mesh->m_system_dimension);
	gradient.setZero();

	if (!m_enable_openmp)
	{
		// constraints single thread
		for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
		{
			it->EvaluateGradient(x, gradient);
		}
	}
	else
	{
		// constraints omp
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_collision_constraints.size(); i++)
			{
				m_collision_constraints[i].EvaluateGradient(x);
			}
		}

		for (i = 0; i < m_collision_constraints.size(); i++)
		{
			m_collision_constraints[i].GetGradient(gradient);
		}
	}
}

ScalarType Simulation::evaluateEnergyAndGradientCollision(const VectorX& x, VectorX& gradient)
{
	ScalarType energy = 0.0;
	gradient.resize(m_mesh->m_system_dimension);
	gradient.setZero();

	if (!m_enable_openmp)
	{
		// constraints single thread
		for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
		{
			energy += it->EvaluateEnergyAndGradient(x, gradient);
		}
	}
	else
	{
		// constraints omp
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_collision_constraints.size(); i++)
			{
				m_collision_constraints[i].EvaluateEnergyAndGradient(x);
			}
		}

		// collect the results in a sequential way
		for (i = 0; i < m_collision_constraints.size(); i++)
		{
			energy += m_collision_constraints[i].GetEnergyAndGradient(gradient);
		}
	}

	return energy;
}
void Simulation::evaluateHessianCollision(const VectorX& x, SparseMatrix& hessian_matrix)
{
	hessian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> h_triplets;
	h_triplets.clear();

	for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
	{
		it->EvaluateHessian(x, h_triplets, m_definiteness_fix);
	}

	hessian_matrix.setFromTriplets(h_triplets.begin(), h_triplets.end());
}

ScalarType Simulation::lineSearchCSSerial(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir)
{
	(void)x;
	(void)gradient_dir;
	(void)descent_dir;

	if (!GenPDExperimentUsesBatchedLineSearch())
	{
		return lineSearchCSSerial(x, gradient_dir, descent_dir);
	}

	if (!m_enable_line_search)
	{
		UploadCSLineSearchResult(ResultID, m_ls_step_size, 0, 1, m_ls_prefetched_energy);
		return m_ls_step_size;
	}

	uploadCSResourcesIfNeeded();
	const int base_K = comp_params.K;
	const GLuint partial_group_count = std::max(
		ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.gradient_size)),
		ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.edge_size)));
	EnsureFloatScratchBuffer(testID, test_, partial_group_count);
	EnsureCSBufferStorage(energyID, sizeof(ScalarType), g_cs_energy_buffer_bytes);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, energyID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ResultID);

	static GLint include_current_candidate_location = -1;
	static GLint gpu_line_search_mode_location = -1;
	if (include_current_candidate_location < 0)
	{
		include_current_candidate_location = glGetUniformLocation(energy_for_linesearch_program, "include_current_candidate");
		gpu_line_search_mode_location = glGetUniformLocation(energy_for_linesearch_program, "gpu_line_search_mode");
	}

	auto evaluate_candidate_energy = [&](ScalarType step)
	{
		comp_params.t0 = static_cast<float>(step);
		comp_params.K = 1;
		glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);
		glUseProgram(energy_for_linesearch_program);
		glUniform1i(include_current_candidate_location, 0);
		glUniform1i(gpu_line_search_mode_location, 0);
		glDispatchCompute(1, partial_group_count, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		DispatchCSReduction(compute_program, testID, energyID, partial_group_count, 1u);
		ScalarType energy = 0.0;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, energyID);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(energy), &energy);
		return energy;
	};

	++g_cs_profile_full_linesearch_calls;
	const ScalarType current_energy = g_cs_prefetched_energy_valid ? m_ls_prefetched_energy : evaluate_candidate_energy(0.0);
	readCSStatsFromGPU(true);
	ScalarType step = 1.0;
	ScalarType accepted_energy = current_energy;
	bool accepted = false;
	while (step > 1e-5)
	{
		const ScalarType candidate_energy = evaluate_candidate_energy(step);
		const ScalarType armijo_rhs = current_energy + m_ls_alpha * step * m_cs_gradient_dot_descent;
		if (candidate_energy < armijo_rhs)
		{
			accepted = true;
			accepted_energy = candidate_energy;
			break;
		}
		step *= m_ls_beta;
	}
	if (!accepted)
	{
		step = 0.0;
	}

	m_ls_step_size = step;
	m_ls_prefetched_energy = accepted_energy;
	g_cs_prefetched_energy_valid = accepted;
	g_cs_unit_step_shortcut_budget = 0;
	UploadCSLineSearchResult(ResultID, step, 0, accepted ? 1 : 0, accepted_energy);

	comp_params.t0 = 1.0f;
	comp_params.K = base_K;
	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);
	return step;
}
ScalarType Simulation::lineSearch_CS(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir)
{
	ScopedCSDebugGroup debug_group("GenPD line search");
	(void)x;
	(void)gradient_dir;
	(void)descent_dir;

	if (!m_enable_line_search)
	{
		UploadCSLineSearchResult(ResultID, m_ls_step_size, 0, 1, m_ls_prefetched_energy);
		return m_ls_step_size;
	}

	/*if (li_test)
	{
		li_test = false;
		return m_ls_step_size;
	}
	else
	{
		li_test = true;
	}*/

	const bool gpu_resident_line_search = g_cs_gpu_state_active_frame;
	if (!gpu_resident_line_search && g_cs_unit_step_shortcut_budget > 0u)
	{
		--g_cs_unit_step_shortcut_budget;
		++g_cs_profile_skipped_linesearch_calls;
		g_cs_prefetched_energy_valid = false;
		m_ls_step_size = 1.0f;
		UploadCSLineSearchResult(ResultID, m_ls_step_size, 0, 1, m_ls_prefetched_energy);
		return m_ls_step_size;
	}

	uploadCSResourcesIfNeeded();

	const int base_K = comp_params.K;
	const int max_batches = 6;
	const int fallback_K = base_K * max_batches;
	const GLuint inertia_partial_group_count = ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.gradient_size));
	const GLuint energy_partial_group_count = ComputeCSPartialGroupCount(static_cast<std::size_t>(comp_params.edge_size));
	const GLuint max_partial_group_count = std::max(inertia_partial_group_count, energy_partial_group_count);
	static GLint include_current_candidate_location = -1;
	static GLint gpu_line_search_mode_location = -1;
	if (include_current_candidate_location < 0)
	{
		include_current_candidate_location = glGetUniformLocation(energy_for_linesearch_program, "include_current_candidate");
		gpu_line_search_mode_location = glGetUniformLocation(energy_for_linesearch_program, "gpu_line_search_mode");
	}

	if (gpu_resident_line_search)
	{
		const unsigned int accepted_shortcut_budget =
			(m_mesh && m_mesh->m_vertices_number >= kCSLargeClothVertexThreshold)
			? kCSLargeClothUnitStepShortcutBudget
			: kCSUnitStepShortcutBudget;
		const GLuint unit_output_count = 2u;
		const GLuint fallback_output_count = static_cast<GLuint>(fallback_K);
		const std::size_t scratch_candidate_count = std::max<std::size_t>(
			static_cast<std::size_t>(unit_output_count),
			static_cast<std::size_t>(fallback_output_count));
		EnsureFloatScratchBuffer(testID, test_, scratch_candidate_count * max_partial_group_count);
		EnsureCSBufferStorage(energyID, static_cast<std::size_t>(fallback_output_count + 1u) * sizeof(ScalarType), g_cs_energy_buffer_bytes);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, energyID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, FlagID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ResultID);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, fixededgesID);

		++g_cs_profile_full_linesearch_calls;

		static GLint gpu_final_current_energy_location = -1;
		static GLint gpu_final_gradient_dot_location = -1;
		static GLint gpu_final_state_location = -1;
		static GLint gpu_final_control_mode_location = -1;
		static GLint gpu_final_keep_accept_location = -1;
		static GLint gpu_final_write_fallback_location = -1;
		static GLint gpu_final_dispatch_group_location = -1;
		static GLint gpu_final_dispatch_partial_location = -1;
		static GLint gpu_final_shortcut_budget_location = -1;
		static GLint gpu_final_candidate_count_location = -1;
		static GLint gpu_final_first_step_location = -1;
		if (gpu_final_current_energy_location < 0)
		{
			gpu_final_current_energy_location = glGetUniformLocation(choose_final_program, "currentEnergy");
			gpu_final_gradient_dot_location = glGetUniformLocation(choose_final_program, "grad_dot_d");
			gpu_final_state_location = glGetUniformLocation(choose_final_program, "use_gpu_line_search_state");
			gpu_final_control_mode_location = glGetUniformLocation(choose_final_program, "control_mode");
			gpu_final_keep_accept_location = glGetUniformLocation(choose_final_program, "keep_existing_accept");
			gpu_final_write_fallback_location = glGetUniformLocation(choose_final_program, "write_fallback_dispatch");
			gpu_final_dispatch_group_location = glGetUniformLocation(choose_final_program, "dispatch_group_count");
			gpu_final_dispatch_partial_location = glGetUniformLocation(choose_final_program, "dispatch_partial_count");
			gpu_final_shortcut_budget_location = glGetUniformLocation(choose_final_program, "shortcut_budget_on_accept");
			gpu_final_candidate_count_location = glGetUniformLocation(choose_final_program, "gpu_candidate_count");
			gpu_final_first_step_location = glGetUniformLocation(choose_final_program, "gpu_first_step");
		}

		auto configure_choose_final = [&](int control_mode, int keep_existing_accept, int write_fallback_dispatch, GLuint dispatch_group_count, GLuint dispatch_partial_count, GLuint shortcut_budget_on_accept, int candidate_count, float first_step)
		{
			glUseProgram(choose_final_program);
			glUniform1f(gpu_final_current_energy_location, 0.0f);
			glUniform1f(gpu_final_gradient_dot_location, 0.0f);
			glUniform1i(gpu_final_state_location, 1);
			glUniform1i(gpu_final_control_mode_location, control_mode);
			glUniform1i(gpu_final_keep_accept_location, keep_existing_accept);
			glUniform1i(gpu_final_write_fallback_location, write_fallback_dispatch);
			glUniform1ui(gpu_final_dispatch_group_location, dispatch_group_count);
			glUniform1ui(gpu_final_dispatch_partial_location, dispatch_partial_count);
			glUniform1ui(gpu_final_shortcut_budget_location, shortcut_budget_on_accept);
			glUniform1i(gpu_final_candidate_count_location, candidate_count);
			glUniform1f(gpu_final_first_step_location, first_step);
		};

		configure_choose_final(1, 0, 0, unit_output_count, max_partial_group_count, 0u, 0, 1.0f);
		glDispatchCompute(1, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

		glUseProgram(energy_for_linesearch_program);
		glUniform1i(include_current_candidate_location, 0);
		glUniform1i(gpu_line_search_mode_location, 1);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, FlagID);
		glDispatchComputeIndirect(0);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		DispatchCSReductionIndirect(compute_program, testID, energyID, max_partial_group_count, unit_output_count, 0u, FlagID, static_cast<GLintptr>(4u * sizeof(GLuint)));

		configure_choose_final(0, 1, 1, fallback_output_count, max_partial_group_count, accepted_shortcut_budget, 1, 1.0f);
		glDispatchCompute(1, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

		glUseProgram(energy_for_linesearch_program);
		glUniform1i(include_current_candidate_location, 0);
		glUniform1i(gpu_line_search_mode_location, 2);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, FlagID);
		glDispatchComputeIndirect(0);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		DispatchCSReductionIndirect(compute_program, testID, energyID, max_partial_group_count, fallback_output_count, 1u, FlagID, static_cast<GLintptr>(4u * sizeof(GLuint)));

		configure_choose_final(0, 1, 0, 0u, 1u, 0u, fallback_K, static_cast<float>(m_ls_beta));
		glDispatchCompute(1, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		glUseProgram(energy_for_linesearch_program);
		glUniform1i(include_current_candidate_location, 0);
		glUniform1i(gpu_line_search_mode_location, 0);

		g_cs_unit_step_shortcut_budget = 0;
		g_cs_prefetched_energy_valid = false;
		m_ls_step_size = 1.0f;
		return m_ls_step_size;
	}
	EnsureFloatScratchBuffer(testID, test_, static_cast<std::size_t>(fallback_K) * max_partial_group_count);
	EnsureCSBufferStorage(energyID, static_cast<std::size_t>(fallback_K) * sizeof(ScalarType), g_cs_energy_buffer_bytes);
	ScalarType current_objective_value = 0;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, energyID);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ResultID);

	++g_cs_profile_full_linesearch_calls;

	if (!g_cs_prefetched_energy_valid)
	{
		comp_params.t0 = 0.0f;
		comp_params.K = base_K;
		glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);

		glUseProgram(energy_for_linesearch_program);
		glUniform1i(include_current_candidate_location, 0);
		glUniform1i(gpu_line_search_mode_location, 0);
		glDispatchCompute(1, max_partial_group_count, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		DispatchCSReduction(compute_program, testID, energyID, max_partial_group_count, 1u);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, energyID);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(current_objective_value), &current_objective_value);
	}
	else
	{
		current_objective_value = m_ls_prefetched_energy;
	}

	comp_params.t0 = 1.0f;
	comp_params.K = base_K;
	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);

	glUseProgram(energy_for_linesearch_program);
	glUniform1i(include_current_candidate_location, 0);
	glUniform1i(gpu_line_search_mode_location, 0);
	glDispatchCompute(1, max_partial_group_count, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	DispatchCSReduction(compute_program, testID, energyID, max_partial_group_count, 1u);

	ScalarType candidate_energy = 0.0f;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, energyID);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(candidate_energy), &candidate_energy);

	readCSStatsFromGPU(true);

	const ScalarType armijo_rhs = current_objective_value + m_ls_alpha * m_cs_gradient_dot_descent;
	if (candidate_energy <= armijo_rhs)
	{
		const unsigned int accepted_shortcut_budget =
			(m_mesh && m_mesh->m_vertices_number >= kCSLargeClothVertexThreshold)
			? kCSLargeClothUnitStepShortcutBudget
			: kCSUnitStepShortcutBudget;
		m_ls_prefetched_energy = candidate_energy;
		g_cs_prefetched_energy_valid = true;
		g_cs_unit_step_shortcut_budget = accepted_shortcut_budget;
		++g_cs_profile_unit_step_accepts;
		m_ls_step_size = 1.0f;
		UploadCSLineSearchResult(ResultID, m_ls_step_size, 0, 1, m_ls_prefetched_energy);
		return m_ls_step_size;
	}

	g_cs_unit_step_shortcut_budget = 0;
	g_cs_prefetched_energy_valid = false;

	comp_params.t0 = static_cast<float>(m_ls_beta);
	comp_params.K = fallback_K;
	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);

	glUseProgram(energy_for_linesearch_program);
	glUniform1i(include_current_candidate_location, 0);
	glUniform1i(gpu_line_search_mode_location, 0);
	glDispatchCompute(static_cast<GLuint>(fallback_K), max_partial_group_count, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	DispatchCSReduction(compute_program, testID, energyID, max_partial_group_count, static_cast<GLuint>(fallback_K));

	glUseProgram(choose_final_program);
	static GLint final_current_energy_location = -1;
	static GLint final_gradient_dot_location = -1;
	static GLint final_state_location = -1;
	static GLint final_control_mode_location = -1;
	static GLint final_keep_accept_location = -1;
	static GLint final_write_fallback_location = -1;
	static GLint final_dispatch_group_location = -1;
	static GLint final_dispatch_partial_location = -1;
	static GLint final_shortcut_budget_location = -1;
	if (final_current_energy_location < 0)
	{
		final_current_energy_location = glGetUniformLocation(choose_final_program, "currentEnergy");
		final_gradient_dot_location = glGetUniformLocation(choose_final_program, "grad_dot_d");
		final_state_location = glGetUniformLocation(choose_final_program, "use_gpu_line_search_state");
		final_control_mode_location = glGetUniformLocation(choose_final_program, "control_mode");
		final_keep_accept_location = glGetUniformLocation(choose_final_program, "keep_existing_accept");
		final_write_fallback_location = glGetUniformLocation(choose_final_program, "write_fallback_dispatch");
		final_dispatch_group_location = glGetUniformLocation(choose_final_program, "dispatch_group_count");
		final_dispatch_partial_location = glGetUniformLocation(choose_final_program, "dispatch_partial_count");
		final_shortcut_budget_location = glGetUniformLocation(choose_final_program, "shortcut_budget_on_accept");
	}
	glUniform1f(final_current_energy_location, static_cast<float>(current_objective_value));
	glUniform1f(final_gradient_dot_location, static_cast<float>(m_cs_gradient_dot_descent));
	glUniform1i(final_state_location, 0);
	glUniform1i(final_control_mode_location, 0);
	glUniform1i(final_keep_accept_location, 0);
	glUniform1i(final_write_fallback_location, 0);
	glUniform1ui(final_dispatch_group_location, 0u);
	glUniform1ui(final_dispatch_partial_location, 1u);
	glUniform1ui(final_shortcut_budget_location, 0u);
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	comp_params.t0 = 1.0f;
	comp_params.K = base_K;
	glBindBuffer(GL_UNIFORM_BUFFER, pUBO);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(comp_params), &comp_params);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pUBO);

	return m_ls_step_size;
}
ScalarType Simulation::lineSearch(const VectorX& x, const VectorX& gradient_dir, const VectorX& descent_dir)
{


	if (m_enable_line_search)
	{
		VectorX x_plus_tdx(m_mesh->m_system_dimension);
		ScalarType t = 1.0 / m_ls_beta;
		//ScalarType t = m_ls_step_size/m_ls_beta;
		//ScalarType m_ls = 0.85;
		ScalarType lhs, rhs, lhs_g, rhs_g;
		VectorX g_tmp;

		ScalarType currentObjectiveValue;
		try
		{
			currentObjectiveValue = evaluateEnergy(x);

		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
		do
		{
#ifdef OUTPUT_LS_ITERATIONS
			g_total_ls_iterations++;
#endif
			t *= m_ls_beta;
			x_plus_tdx = x + t * descent_dir;

			lhs = 1e15;
			rhs = 0;
			try
			{
				//evaluateGradient(x_plus_tdx, g_tmp);
				lhs = evaluateEnergy(x_plus_tdx);
				//lhs_g = g_tmp.dot(descent_dir);
				//if (lhs_g < 0)lhs_g = -lhs_g;
			}
			catch (const std::exception&)
			{
				continue;
			}

			//rhs_g = gradient_dir.dot(descent_dir);
			//if (rhs_g < 0)rhs_g = -rhs_g;
			rhs = currentObjectiveValue + m_ls_alpha * t * (gradient_dir.transpose() * descent_dir)(0);


			//|| lhs_g<=m_ls*rhs_g
			if (lhs >= rhs)
			{
				continue; // keep looping
			}

			break; // exit looping

		} while (t > 1e-5);

		if (t < 1e-5)
		{
			t = 0.0;
		}
		m_ls_step_size = t;

		// std::cout << t << " and " << std::endl;

		if (m_verbose_show_converge)
		{
			std::cout << "Linesearch Stepsize = " << t << std::endl;
			std::cout << "lhs (current energy) = " << lhs << std::endl;
			std::cout << "previous energy = " << currentObjectiveValue << std::endl;
			std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
		}

#ifdef OUTPUT_LS_ITERATIONS
		g_total_iterations++;
		if (g_total_iterations % OUTPUT_LS_ITERATIONS_EVERY_N_FRAMES == 0)
		{
			std::cout << "Avg LS Iterations = " << g_total_ls_iterations / g_total_iterations << std::endl;
			g_total_ls_iterations = 0;
			g_total_iterations = 0;
		}
#endif
		return t;
	}
	else
	{
		return m_ls_step_size;
	}
}


ScalarType Simulation::lineSearch_ncg(const VectorX& x, const  VectorX& gradient_dir, VectorX& descent_dir, const SparseMatrix& Hessian)
{
	if (m_enable_line_search)
	{
		VectorX x_plus_tdx(m_mesh->m_system_dimension);
		//ScalarType t = 1.0 / m_ls_beta;
		//ScalarType t = m_ls_step_size/m_ls_beta;
		ScalarType t;
		ScalarType lhs, rhs, beta;


		ScalarType currentObjectiveValue;
		try
		{
			currentObjectiveValue = evaluateEnergy(x);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
		do
		{
#ifdef OUTPUT_LS_ITERATIONS
			g_total_ls_iterations++;
#endif
			//t *= m_ls_beta;

			t = -(gradient_dir.transpose() * descent_dir)(0) / (descent_dir.transpose() * (Hessian * descent_dir))(0);

			x_plus_tdx = x + t * descent_dir;

			VectorX gradient_dir_tmp = gradient_dir;

			evaluateGradient(x_plus_tdx, gradient_dir_tmp);

			beta = gradient_dir_tmp.norm() * gradient_dir_tmp.norm() / gradient_dir.norm() * gradient_dir.norm();

			descent_dir = -gradient_dir_tmp + beta * descent_dir;

			lhs = 1e15;
			rhs = 0;
			try
			{
				lhs = evaluateEnergy(x_plus_tdx);
			}
			catch (const std::exception&)
			{
				continue;
			}
			rhs = currentObjectiveValue + t * (gradient_dir_tmp.transpose() * descent_dir)(0);

			if (lhs >= rhs)
			{
				continue; // keep looping
			}

			break; // exit looping

		} while (t > 1e-5);

		if (t < 1e-5)
		{
			t = 0.0;
		}
		m_ls_step_size = t;

		if (m_verbose_show_converge)
		{
			std::cout << "Linesearch Stepsize = " << t << std::endl;
			std::cout << "lhs (current energy) = " << lhs << std::endl;
			std::cout << "previous energy = " << currentObjectiveValue << std::endl;
			std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
		}

#ifdef OUTPUT_LS_ITERATIONS
		g_total_iterations++;
		if (g_total_iterations % OUTPUT_LS_ITERATIONS_EVERY_N_FRAMES == 0)
		{
			std::cout << "Avg LS Iterations = " << g_total_ls_iterations / g_total_iterations << std::endl;
			g_total_ls_iterations = 0;
			g_total_iterations = 0;
		}
#endif
		return t;
	}
	else
	{
		return m_ls_step_size;
	}
}


ScalarType Simulation::linesearchWithPrefetchedEnergyAndGradientComputing(const VectorX& x, const ScalarType current_energy, const VectorX& gradient_dir, const VectorX& descent_dir, ScalarType& next_energy, VectorX& next_gradient_dir)
{
	if (m_enable_line_search)
	{
		m_ls_x_plus_tdx.resize(m_mesh->m_system_dimension);
		VectorX& x_plus_tdx = m_ls_x_plus_tdx;
		ScalarType t = 1.0 / m_ls_beta;
		ScalarType lhs, rhs;

		ScalarType currentObjectiveValue = current_energy;
		const ScalarType gradient_dot_descent = gradient_dir.dot(descent_dir);

		do
		{
#ifdef OUTPUT_LS_ITERATIONS
			g_total_ls_iterations++;
#endif

			t *= m_ls_beta;
			x_plus_tdx.noalias() = x + t * descent_dir;

			lhs = 1e15;
			rhs = 0;
			try
			{
				lhs = evaluateEnergyAndGradient(x_plus_tdx, next_gradient_dir);
			}
			catch (const std::exception&)
			{
				continue;
			}
			rhs = currentObjectiveValue + m_ls_alpha * t * gradient_dot_descent;
			if (lhs >= rhs)
			{
				continue; // keep looping
			}

			next_energy = lhs;
			break; // exit looping

		} while (t > 1e-5);

		if (t < 1e-5)
		{
			t = 0.0;
			next_energy = current_energy;
			next_gradient_dir = gradient_dir;
		}
		m_ls_step_size = t;

		if (m_verbose_show_converge)
		{
			std::cout << "Linesearch Stepsize = " << t << std::endl;
			std::cout << "lhs (current energy) = " << lhs << std::endl;
			std::cout << "previous energy = " << currentObjectiveValue << std::endl;
			std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
		}

#ifdef OUTPUT_LS_ITERATIONS
		g_total_iterations++;
		if (g_total_iterations % OUTPUT_LS_ITERATIONS_EVERY_N_FRAMES == 0)
		{
			std::cout << "Avg LS Iterations = " << g_total_ls_iterations / g_total_iterations << std::endl;
			g_total_ls_iterations = 0;
			g_total_iterations = 0;
		}
#endif

		return t;
	}
	else
	{
		return m_ls_step_size;
	}
}

ScalarType Simulation::evaluatePotentialEnergy(const VectorX& x)
{
	/*if (use_cs)
	{
		return evaluatePotentialEnergyCS(x);
	}*/

	ScalarType energy = evaluateEnergyPureConstraint(x, m_external_force);
	energy -= m_external_force.dot(x);

	return energy;
}
ScalarType Simulation::evaluateKineticEnergy(const VectorX& v)
{
	return (0.5 * v.transpose() * m_mesh->m_mass_matrix * v);
}
ScalarType Simulation::evaluateTotalEnergy(const VectorX& x, const VectorX& v)
{
	return (evaluatePotentialEnergy(x) + evaluateKineticEnergy(v));
}

#pragma region matrices and prefactorization
void Simulation::setWeightedLaplacianMatrix()
{
	evaluateLaplacian(m_weighted_laplacian);
}

void Simulation::setWeightedLaplacianMatrix1D()
{
	evaluateLaplacian1D(m_weighted_laplacian_1D);
}

void Simulation::precomputeLaplacianWeights()
{
	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		if ((*it)->Type() == CONSTRAINT_TYPE_TET)
		{
			m_stiffness_laplacian = (*it)->ComputeLaplacianWeight();
		}
	}
	SetReprefactorFlag();
}

void Simulation::precomputeLaplacian()
{
	if (m_precomputing_flag == false)
	{
		setWeightedLaplacianMatrix();

		m_precomputing_flag = true;
	}
}

void Simulation::prefactorize()
{
	if (m_prefactorization_flag == false)
	{
		// update laplacian coefficients
		if (m_stiffness_auto_laplacian_stiffness)
		{
			precomputeLaplacianWeights();
		}
		else
		{
			SetMaterialProperty();
		}

		// full space laplacian 3n x 3n
		setWeightedLaplacianMatrix();
		factorizeDirectSolverLLT(m_weighted_laplacian, m_prefactored_solver, "Our Method");		// prefactorization of laplacian
		m_preloaded_cg_solver.compute(m_weighted_laplacian);		// load the cg solver

		// reduced dim space laplacian nxn
		setWeightedLaplacianMatrix1D();
		factorizeDirectSolverLLT(m_weighted_laplacian_1D, m_prefactored_solver_1D, "Our Method Reduced Space");
		m_preloaded_cg_solver_1D.compute(m_weighted_laplacian_1D);

#ifdef ENABLE_MATLAB_DEBUGGING
		g_debugger->SendSparseMatrix(m_weighted_laplacian, "L");
		g_debugger->SendSparseMatrix(m_weighted_laplacian_1D, "L1");
#endif
		m_prefactorization_flag = true;
	}
}
void Simulation::setWeightedLaplacianMatrix_1()
{
	m_weighted_laplacian.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> l_triplets;
	l_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateWeightedLaplacian(l_triplets);
	}

	m_weighted_laplacian.setFromTriplets(l_triplets.begin(), l_triplets.end());
}

void Simulation::setJMatrix()
{
	m_J_matrix;

	evaluateJMatrix(m_J_matrix);
}
// for local global method only
void Simulation::evaluateDVector(const VectorX& x, VectorX& d)
{
	d.resize(m_constraints.size() * 3);
	d.setZero();

	for (unsigned int index = 0; index < m_constraints.size(); ++index)
	{
		m_constraints[index]->EvaluateDVector(index, x, d);
	}
}
void Simulation::evaluateJMatrix(SparseMatrix& J)
{
	J.resize(m_mesh->m_system_dimension, m_constraints.size() * 3);
	std::vector<SparseMatrixTriplet> J_triplets;
	J_triplets.clear();

	for (unsigned int index = 0; index < m_constraints.size(); ++index)
	{
		m_constraints[index]->EvaluateJMatrix(index, J_triplets);
	}
	J.setFromTriplets(J_triplets.begin(), J_triplets.end());
}



void Simulation::prefactorize(PrefactorType type)
{
	if (m_prefactorization_flag == false)
	{
		SparseMatrix A;
		ScalarType h2 = m_h * m_h;

		// choose matrix
		switch (type)
		{
		case PREFACTOR_M_PLUS_H2L:
			setWeightedLaplacianMatrix_1();
			setJMatrix();
			A = m_mesh->m_mass_matrix + h2 * m_weighted_laplacian;
			break;
		}

		factorizeDirectSolverLLT(A, m_prefactored_solver_1[type]);
		m_prefactorization_flag = true;
	}
}

#pragma endregion

#pragma region newton_solver
void Simulation::analyzeNewtonSolverPattern(const SparseMatrix& A)
{
	if (!m_prefactorization_flag_newton)
	{
		m_newton_solver.analyzePattern(A);
		m_prefactorization_flag_newton = true;
	}
}

void Simulation::factorizeNewtonSolver(const SparseMatrix& A, char* warning_msg)
{
	SparseMatrix A_prime = A;
	m_newton_solver.factorize(A_prime);
	ScalarType Regularization = ScalarType(1e-10);
	bool success = true;
	SparseMatrix I;
	while (m_newton_solver.info() != Eigen::Success)
	{
		if (success == true) // first time here
		{
			EigenMakeSparseIdentityMatrix(A.rows(), A.cols(), I);
		}
		Regularization *= 10;
		A_prime = A_prime + Regularization * I;
		m_newton_solver.factorize(A_prime);
		success = false;
	}
	if (!success && m_verbose_show_factorization_warning)
		std::cout << "Warning: " << warning_msg << " adding " << Regularization << " identites.(llt solver)" << std::endl;
}
#pragma endregion

#pragma region utilities

ScalarType Simulation::linearSolve(VectorX& x, const SparseMatrix& A, const VectorX& b, char* msg)
{
	ScalarType residual = 0;

	switch (m_solver_type)
	{
	case SOLVER_TYPE_DIRECT_LLT:
	{
#ifdef PARDISO_SUPPORT
		Eigen::PardisoLLT<SparseMatrix, Eigen::Upper> A_solver;
#else
		Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> A_solver;
#endif
		factorizeDirectSolverLLT(A, A_solver, msg);
		x = A_solver.solve(b);
	}
	break;
	case SOLVER_TYPE_CG:
	{
		x.resize(b.size());
		x.setZero();
		residual = conjugateGradientWithInitialGuess(x, A, b, m_iterative_solver_max_iteration);
	}
	break;
	default:
		break;
	}

	return residual;
}

ScalarType Simulation::conjugateGradientWithInitialGuess(VectorX& x, const SparseMatrix& A, const VectorX& b, const unsigned int max_it /* = 200 */, const ScalarType tol /* = 1e-5 */)
{
	VectorX r = b - A * x;
	VectorX p = r;
	ScalarType rsold = r.dot(r);
	ScalarType rsnew;

	VectorX Ap;
	Ap.resize(x.size());
	ScalarType alpha;

	for (unsigned int i = 1; i != max_it; ++i)
	{
		Ap = A * p;
		alpha = rsold / p.dot(Ap);
		x = x + alpha * p;

		r = r - alpha * Ap;
		rsnew = r.dot(r);
		if (sqrt(rsnew) < tol)
		{
			break;
		}
		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
	}

	return sqrt(rsnew);
}
//Cholesky ???
void Simulation::factorizeDirectSolverLLT(const SparseMatrix& A, Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>& lltSolver, char* warning_msg)
{
	SparseMatrix A_prime = A;
	lltSolver.analyzePattern(A_prime);
	lltSolver.factorize(A_prime);
	ScalarType Regularization = ScalarType(1e-10);
	bool success = true;
	SparseMatrix I;
	while (lltSolver.info() != Eigen::Success)
	{
		if (success == true) // first time factorization failed
		{
			EigenMakeSparseIdentityMatrix(A.rows(), A.cols(), I);
		}
		Regularization *= 10;
		A_prime = A_prime + Regularization * I;
		lltSolver.factorize(A_prime);
		success = false;
	}
	if (!success && m_verbose_show_factorization_warning)
		std::cout << "Warning: " << warning_msg << " adding " << Regularization << " identites.(llt solver)" << std::endl;
}

#ifdef PARDISO_SUPPORT
void Simulation::factorizeDirectSolverLLT(const SparseMatrix& A, Eigen::PardisoLLT<SparseMatrix, Eigen::Upper>& lltSolver, char* warning_msg)
{
	SparseMatrix A_prime = A;
	lltSolver.analyzePattern(A_prime);
	lltSolver.factorize(A_prime);
	ScalarType Regularization = ScalarType(1e-10);
	bool success = true;
	SparseMatrix I;
	while (lltSolver.info() != Eigen::Success)
	{
		if (success == true) // first time factorization failed
		{
			EigenMakeSparseIdentityMatrix(A.rows(), A.cols(), I);
		}
		Regularization *= 10;
		A_prime = A_prime + Regularization * I;
		lltSolver.factorize(A_prime);
		success = false;
	}
	if (!success && m_verbose_show_factorization_warning)
		std::cout << "Warning: " << warning_msg << " adding " << Regularization << " identites.(llt solver)" << std::endl;
}
#endif

void Simulation::generateRandomVector(const unsigned int size, VectorX& x)
{
	x.resize(size);
	for (unsigned int i = 0; i < size; i++)
	{
		x(i) = ((ScalarType)(rand()) / (ScalarType)(RAND_MAX + 1) - 0.5) * 2;
	}

	//x.resize(size);
	//unsigned int dim = 0;
	//ScalarType scale[3];
	//scale[0] = scale[2] = 1;
	//scale[1] = 1e6;
	//for (unsigned int i = 0; i < size; i++)
	//{
	//	x(i) = ((ScalarType)(rand()) / (ScalarType)(RAND_MAX + 1) - 0.5) * 2 * scale[dim];
	//	dim = (dim + 1) % 3;
	//}
}

#pragma endregion




























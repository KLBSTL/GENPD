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

#pragma warning( disable : 4244)
#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <string>

//----------Headers--------------//
#include "global_headers.h"
#include <cstdlib>
#include <fstream>
#include "math_headers.h"
#include "openGL_headers.h"
//----------Framework--------------//
#include "fps.h"
#include "timer_wrapper.h"
#include "stb_image_write.h"
#include "glsl_wrapper.h"
#include "runtime_paths.h"
#include "experiment_variant.h"
#include "AntTweakBar.h"
#include "anttweakbar_wrapper.h"
#include "camera.h"
#include "scene.h"
#include "selection_tool.h"
//----------Core--------------//
#include "mesh.h"
#include "simulation.h"

#define max_t 100

//----------Project Key Globals--------------//
AntTweakBarWrapper* g_config_bar;
Camera* g_camera;
RenderWrapper* g_renderer;
Scene* g_scene;
Mesh* g_mesh;
Simulation * g_simulation;
SelectionTool* g_selection_tool;
#ifdef ENABLE_MATLAB_DEBUGGING
#include "matlab_debugger.h"
MatlabDebugger * g_debugger;
#endif // ENABLE_MATLAB_DEBUGGING
TimerWrapper g_global_timer;
TimerWrapper g_benchmark_frame_timer;
TimerWrapper g_benchmark_render_timer;
bool g_benchmark_frame_timer_active = false;
unsigned int g_benchmark_profile_frame = 0;
int g_interval = 100;
ScalarType total_time = 0;

//----------Global Parameters----------------//
int g_screen_width = DEFAULT_SCREEN_WIDTH;
int g_screen_height = DEFAULT_SCREEN_HEIGHT;
glm::vec3 g_handle_color;
bool g_random_handle_color;

//----------State Control--------------------//
bool g_only_show_sim = false;
bool g_record = false;
bool g_pause = true;
bool g_show_mesh = true;
bool g_show_wireframe = false;
int  g_wireframe_linewidth = 1;
bool g_show_texture = false;
bool g_texture_load_succeed = false;

//----------Mouse Control--------------------//
int g_mouse_old_x, g_mouse_old_y;
int g_mouse_wheel_pos;
unsigned char g_button_mask = 0x00;
bool g_mouse_down = false;

//----------Frame Rate/Frame Number----------//
mmc::FpsTracker g_fps_tracker;
int g_max_fps = 30;
int g_timestep = 1000 / g_max_fps;

//----------Recording Related----------------//
bool g_recording_limit = false;
int g_current_frame = 0;
int g_total_frame = 0;
bool g_export_obj = true;

//----------Benchmark Mode-------------------//
bool g_benchmark_mode = false;
bool g_benchmark_no_render = false;
bool g_benchmark_hide_window = false;
bool g_benchmark_swing_attachments = false;
bool g_benchmark_uncapped = false;
bool g_benchmark_sync_gpu = false;
bool g_benchmark_disable_vsync = false;
bool g_benchmark_finish_after_display = false;
int g_benchmark_frames = 300;
int g_benchmark_warmup_frames = 30;
std::string g_cli_project_root;
std::string g_cli_output_dir;
std::string g_cli_run_label;
std::string g_cli_solver_variant;
int g_cli_iterations_per_frame = 0;
std::string g_cli_reference_export_dir;
std::string g_cli_quality_reference_dir;
int g_cli_quality_checkpoint_stride = 1;
bool g_cli_quality_metrics = false;
ScalarType g_cli_timestep = static_cast<ScalarType>(-1);
ScalarType g_cli_stretch_stiffness = static_cast<ScalarType>(-1);
ScalarType g_cli_bending_stiffness = static_cast<ScalarType>(-1);
int g_cli_cloth_dimension = 0;
int g_cli_cloth_width = 0;
int g_cli_cloth_height = 0;
int g_cli_batched_ls_k = 0;
ScalarType g_cli_armijo_beta = static_cast<ScalarType>(-1);
std::string g_cli_adaptive_ls_history = "frame";
std::string g_cli_ncg_restart_mode;
int g_cli_ncg_restart_period = 0;
std::string g_cli_verify_cs_gradient_dir;
std::string g_cli_verify_adaptive_ls_history_reset_dir;
std::string g_cli_scene;
int g_cli_capture_frame = -1;
std::string g_cli_capture_output;
int g_cli_capture_width = 0;
int g_cli_capture_height = 0;
int g_cli_render_width = 0;
int g_cli_render_height = 0;
bool g_cli_capture_pending = false;
bool g_cli_restore_iterations_per_frame = false;
unsigned int g_cli_original_iterations_per_frame = 0;
bool g_cli_profile_gpu_queries = false;
bool g_cli_profile_line_search_decisions = false;
bool g_cli_force_cpu_state_roundtrip = false;
bool g_cli_energy_audit = false;
int g_cli_xpbd_fuse_apply_collision = -1;
int g_cli_xpbd_cached_pins = -1;
bool g_cli_print_paths = false;

//----------glut function handlers-----------//
void resize(int, int);
void timeout(int);
void display(void);
void key_press(unsigned char, int, int);
void mouse_click(int, int, int, int);
void mouse_motion(int, int);
void mouse_wheel(int, int, int, int);
void mouse_over(int, int);
void parse_command_line(int, char**);
int parse_positive_int(const char*, const int);
int parse_nonnegative_int(const char*, const int);
bool should_use_uncapped_benchmark(void);
void schedule_next_timeout(void);
void finish_display_frame(void);
void log_benchmark_presentation_frame(void);
void apply_benchmark_swap_interval(void);
void maybe_finish_benchmark(void);
void maybe_capture_benchmark_frame(void);

//----------anttweakbar handlers----------//
void TW_CALL set_handle(void*);
void TW_CALL save_handle(void*);
void TW_CALL load_handle(void*);
void TW_CALL reset_handle(void*);
void TW_CALL reset_simulation(void*);
void TW_CALL step_through(void*);
void TW_CALL reset_camera(void*);
void TW_CALL set_partial_material_property(void*);
void TW_CALL matlab_reset_current_data(void*);
void TW_CALL matlab_reset_all_data(void*);
void TW_CALL matlab_set_converged_energy(void*);
void TW_CALL matlab_new_data(void*);
void TW_CALL matlab_remove_last_data(void*);
void TW_CALL matlab_export_data(void*);
void TW_CALL matlab_import_data(void*);
void TW_CALL matlab_plot(void*);
void TW_CALL matlab_plot_all(void*);
void TW_CALL matlab_visualize_vector(void*);

int niuma = 0;
float cowhorse = 0;
//----------other utility functions----------//
void init(void);
void cleanup(void);
void draw_overlay(void);
void grab_screen(void);
void grab_screen(const char* filename);

//#include "src\plugins\BlockMethods.h"

inline const Eigen::Matrix3Xd::ConstColXpr getColumn(const Eigen::Matrix3Xd& A, unsigned int i)
{
	const Eigen::Matrix3Xd::ConstColXpr col = A.col(i);
	std::cout << "in function col address\n" << &col << std::endl;
	std::cout << "in function col\n" << col << std::endl;
	return col;
	//return A.col(i);
}

void test()
{
	Eigen::Matrix3Xd A(3, 5);
	A.setRandom();
	std::cout << "A address\n" << &A << std::endl;
	std::cout << "A data address\n" << A.data() << std::endl;
	std::cout << A << std::endl;

	const Eigen::Vector3d& a = A.col(0);
	std::cout << "a address\n" << &a << std::endl;
	std::cout << "a\n" << a << std::endl;

	const Eigen::Vector3d& b = getColumn(A, 0);
	std::cout << "b address\n" << &b << std::endl;
	std::cout << "b\n" << b << std::endl;

	return;
}

int main(int argc, char ** argv)
{
	//test();
	parse_command_line(argc, argv);
	if (g_cli_render_width > 0 && g_cli_render_height > 0)
	{
		g_screen_width = g_cli_render_width;
		g_screen_height = g_cli_render_height;
	}
	if (g_cli_capture_width > 0 && g_cli_capture_height > 0)
	{
		g_screen_width = g_cli_capture_width;
		g_screen_height = g_cli_capture_height;
	}
	GenPDInitializeRuntimePaths(
		argc > 0 ? argv[0] : NULL,
		g_cli_project_root,
		g_cli_output_dir,
		g_cli_run_label,
		g_cli_profile_gpu_queries,
		g_cli_print_paths);

    // gl init
    glutInit(&argc, argv);
#ifdef HIGH_PRECISION
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
#else
	glutInitDisplayMode(GLUT_RGBA);
#endif

    glutCreateWindow("Mass-Spring System Simulation T.L.");
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glutInitWindowSize(g_screen_width, g_screen_height);
    glViewport(0, 0, g_screen_width, g_screen_height);

    // user init
    init();
	if (!g_cli_solver_variant.empty())
	{
		GenPDExperimentVariant variant;
		if (!GenPDParseExperimentVariant(g_cli_solver_variant, variant))
		{
			std::cerr << "Unknown --solver-variant: " << g_cli_solver_variant << std::endl;
			exit(EXIT_FAILURE);
		}
		GenPDSetExperimentVariant(variant);
	}
	if (_putenv_s("GENPD_SOLVER_VARIANT", GenPDExperimentVariantName()) != 0)
	{
		std::cerr << "Warning: cannot set solver-variant metadata environment." << std::endl;
	}
std::cout << "GenPD solver variant: " << GenPDExperimentVariantName() << std::endl;
if (GenPDExperimentUsesAdaptiveLineSearch() && g_cli_batched_ls_k == 0)
{
    g_simulation->SetBatchedLineSearchK(4u);
}
if (g_cli_batched_ls_k > 0)
{
    g_simulation->SetBatchedLineSearchK(static_cast<unsigned int>(g_cli_batched_ls_k));
}
if (g_cli_armijo_beta > 0)
{
    if (g_cli_armijo_beta >= 1.0)
    {
        std::cerr << "--armijo-beta must be in (0, 1)." << std::endl;
        exit(EXIT_FAILURE);
    }
    g_simulation->SetArmijoBeta(g_cli_armijo_beta);
}
AdaptiveLineSearchHistoryMode adaptive_history_mode = ADAPTIVE_LS_HISTORY_FRAME;
if (g_cli_adaptive_ls_history == "none")
{
    adaptive_history_mode = ADAPTIVE_LS_HISTORY_NONE;
}
else if (g_cli_adaptive_ls_history == "iteration")
{
    adaptive_history_mode = ADAPTIVE_LS_HISTORY_ITERATION;
}
else if (g_cli_adaptive_ls_history != "frame")
{
    std::cerr << "Unknown --adaptive-ls-history: " << g_cli_adaptive_ls_history << std::endl;
    exit(EXIT_FAILURE);
}
g_simulation->SetAdaptiveLineSearchHistoryMode(adaptive_history_mode);
if (!g_cli_ncg_restart_mode.empty())
{
    NCGRestartMode restart_mode = NCG_RESTART_NONE;
    if (g_cli_ncg_restart_mode == "periodic")
    {
        restart_mode = NCG_RESTART_PERIODIC;
    }
    else if (g_cli_ncg_restart_mode == "non-descent")
    {
        restart_mode = NCG_RESTART_NON_DESCENT;
    }
    else if (g_cli_ncg_restart_mode != "none")
    {
        std::cerr << "Unknown --ncg-restart-mode: " << g_cli_ncg_restart_mode << std::endl;
        exit(EXIT_FAILURE);
    }
    g_simulation->SetNCGRestart(restart_mode, static_cast<unsigned int>(g_cli_ncg_restart_period));
}
else if (g_cli_ncg_restart_period > 0)
{
    std::cerr << "--ncg-restart-period requires --ncg-restart-mode periodic." << std::endl;
    exit(EXIT_FAILURE);
}
g_simulation->SetProfileLineSearchDecisions(g_cli_profile_line_search_decisions);
g_simulation->SetForceCS2CpuStateRoundtrip(g_cli_force_cpu_state_roundtrip);
g_simulation->SetEnergyAudit(g_cli_energy_audit);
if (g_cli_xpbd_fuse_apply_collision >= 0)
{
    g_simulation->SetXPBDFuseApplyCollision(g_cli_xpbd_fuse_apply_collision != 0);
}
if (g_cli_xpbd_cached_pins >= 0)
{
    g_simulation->SetXPBDCachedPins(g_cli_xpbd_cached_pins != 0);
}
_putenv_s("GENPD_PROFILE_LINE_SEARCH_DECISIONS", g_cli_profile_line_search_decisions ? "1" : "0");
_putenv_s("GENPD_FORCE_CPU_STATE_ROUNDTRIP", g_cli_force_cpu_state_roundtrip ? "1" : "0");
_putenv_s("GENPD_ENERGY_AUDIT", g_cli_energy_audit ? "1" : "0");
_putenv_s("GENPD_XPBD_FUSE_APPLY_COLLISION", g_cli_xpbd_fuse_apply_collision == 0 ? "0" : "1");
_putenv_s("GENPD_XPBD_CACHED_PINS", g_cli_xpbd_cached_pins == 0 ? "0" : "1");
if (g_cli_iterations_per_frame > 0)
{
    g_cli_original_iterations_per_frame = g_simulation->IterationsPerFrame();
g_cli_restore_iterations_per_frame = true;
g_simulation->SetIterationsPerFrame(static_cast<unsigned int>(g_cli_iterations_per_frame));
    std::cout << "GenPD iterations per frame: " << g_simulation->IterationsPerFrame() << std::endl;
}
const std::string reference_export_dir = g_cli_reference_export_dir.empty() ? std::string() : GenPDResolveProjectPath(g_cli_reference_export_dir);
const std::string quality_reference_dir = g_cli_quality_reference_dir.empty() ? std::string() : GenPDResolveProjectPath(g_cli_quality_reference_dir);
g_simulation->ConfigureQualityMetrics(
    reference_export_dir,
    quality_reference_dir,
    static_cast<unsigned int>(g_cli_quality_checkpoint_stride),
    g_cli_quality_metrics);
const std::string iterations_per_frame_metadata = std::to_string(g_simulation->IterationsPerFrame());
const std::string selected_scene = GenPDResolveProjectPath(g_cli_scene.empty() ? std::string(DEFAULT_SCENE_FILE) : g_cli_scene);
const std::string timestep_metadata = g_cli_timestep > 0 ? std::to_string(g_cli_timestep) : std::string();
const std::string stretch_metadata = g_cli_stretch_stiffness > 0 ? std::to_string(g_cli_stretch_stiffness) : std::string();
const std::string bending_metadata = g_cli_bending_stiffness > 0 ? std::to_string(g_cli_bending_stiffness) : std::string();
const std::string cloth_dimension_metadata = g_cli_cloth_dimension > 0 ? std::to_string(g_cli_cloth_dimension) : std::string();
const std::string cloth_width_metadata = g_cli_cloth_width > 0 ? std::to_string(g_cli_cloth_width) : std::string();
const std::string cloth_height_metadata = g_cli_cloth_height > 0 ? std::to_string(g_cli_cloth_height) : std::string();
const std::string batched_ls_k_metadata = g_cli_batched_ls_k > 0 ? std::to_string(g_cli_batched_ls_k) : std::string();
const std::string armijo_beta_metadata = g_cli_armijo_beta > 0 ? std::to_string(g_cli_armijo_beta) : std::string();
const std::string adaptive_ls_history_metadata = g_cli_adaptive_ls_history;
const std::string ncg_restart_period_metadata = g_cli_ncg_restart_period > 0 ? std::to_string(g_cli_ncg_restart_period) : std::string();
const bool quality_metrics_enabled = g_cli_quality_metrics || !reference_export_dir.empty() || !quality_reference_dir.empty();
_putenv_s("GENPD_ITERATIONS_PER_FRAME", iterations_per_frame_metadata.c_str());
_putenv_s("GENPD_REFERENCE_EXPORT_DIR", reference_export_dir.c_str());
_putenv_s("GENPD_QUALITY_REFERENCE_DIR", quality_reference_dir.c_str());
_putenv_s("GENPD_QUALITY_CHECKPOINT_STRIDE", std::to_string(g_cli_quality_checkpoint_stride).c_str());
_putenv_s("GENPD_QUALITY_METRICS", quality_metrics_enabled ? "1" : "0");
_putenv_s("GENPD_TIMESTEP_OVERRIDE", timestep_metadata.c_str());
_putenv_s("GENPD_STRETCH_STIFFNESS_OVERRIDE", stretch_metadata.c_str());
_putenv_s("GENPD_BENDING_STIFFNESS_OVERRIDE", bending_metadata.c_str());
_putenv_s("GENPD_CLOTH_DIMENSION_OVERRIDE", cloth_dimension_metadata.c_str());
_putenv_s("GENPD_CLOTH_WIDTH_OVERRIDE", cloth_width_metadata.c_str());
_putenv_s("GENPD_CLOTH_HEIGHT_OVERRIDE", cloth_height_metadata.c_str());
_putenv_s("GENPD_BATCHED_LS_K", batched_ls_k_metadata.c_str());
_putenv_s("GENPD_ARMIJO_BETA", armijo_beta_metadata.c_str());
_putenv_s("GENPD_ADAPTIVE_LS_HISTORY", adaptive_ls_history_metadata.c_str());
_putenv_s("GENPD_NCG_RESTART_MODE", g_cli_ncg_restart_mode.c_str());
_putenv_s("GENPD_NCG_RESTART_PERIOD", ncg_restart_period_metadata.c_str());
_putenv_s("GENPD_SCENE", selected_scene.c_str());
// Configuration loading may reset the window size; CLI rendering dimensions win.
if (g_cli_render_width > 0 && g_cli_render_height > 0)
{
    g_screen_width = g_cli_render_width;
    g_screen_height = g_cli_render_height;
}
if (g_cli_capture_width > 0 && g_cli_capture_height > 0)
{
    g_screen_width = g_cli_capture_width;
    g_screen_height = g_cli_capture_height;
}
glutReshapeWindow(g_screen_width, g_screen_height);
	GenPDWriteRunMetadata(
		argc,
		argv,
		g_benchmark_mode,
		g_benchmark_frames,
		g_benchmark_warmup_frames,
		g_benchmark_no_render,
		g_benchmark_uncapped,
		g_benchmark_sync_gpu,
		g_benchmark_disable_vsync,
		g_benchmark_hide_window,
		reinterpret_cast<const char*>(glGetString(GL_VENDOR)),
		reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
		reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	if (!g_cli_verify_cs_gradient_dir.empty())
	{
		const std::string verification_dir = GenPDResolveOutputPath(g_cli_verify_cs_gradient_dir);
		const bool verified = g_simulation->VerifyCSGradient(verification_dir);
		return verified ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	if (!g_cli_verify_adaptive_ls_history_reset_dir.empty())
	{
		const std::string verification_dir = GenPDResolveOutputPath(g_cli_verify_adaptive_ls_history_reset_dir);
		const bool verified = g_simulation->VerifyAdaptiveLineSearchHistoryInvalidation(verification_dir);
		return verified ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	if (g_benchmark_mode)
	{
		g_pause = false;
		g_record = false;
		g_recording_limit = false;
		g_only_show_sim = true;
		g_show_mesh = !g_benchmark_no_render;
		g_show_wireframe = false;
		g_show_texture = false;
		g_config_bar->Hide();
		if (g_benchmark_hide_window)
		{
			glutHideWindow();
		}
		std::cout << "Benchmark mode enabled: warmup=" << g_benchmark_warmup_frames
			<< " frames, measured=" << g_benchmark_frames
			<< " frames, no_render=" << (g_benchmark_no_render ? "true" : "false")
			<< ", uncapped=" << (g_benchmark_uncapped ? "true" : "false")
			<< ", sync_gpu=" << (g_benchmark_sync_gpu ? "true" : "false")
			<< ", disable_vsync=" << (g_benchmark_disable_vsync ? "true" : "false")
			<< std::endl;
	}

    // bind function callbacks
    glutDisplayFunc(display);
    schedule_next_timeout();
    glutReshapeFunc(resize);
    glutKeyboardFunc(key_press);
    glutMouseFunc(mouse_click);
    glutMotionFunc(mouse_motion);
    glutPassiveMotionFunc(mouse_over);
    glutMouseWheelFunc(mouse_wheel);
    glutCloseFunc(cleanup);
    glutIdleFunc(NULL);

	omp_set_num_threads(6);

    glutMainLoop();

    return 0;
}

bool should_use_uncapped_benchmark(void)
{
	return g_benchmark_mode && g_benchmark_uncapped;
}

void schedule_next_timeout(void)
{
	const int delay_ms = should_use_uncapped_benchmark() ? 0 : g_timestep;
	glutTimerFunc(delay_ms, timeout, delay_ms);
}

void finish_display_frame(void)
{
	if (g_benchmark_sync_gpu)
	{
		glFinish();
	}

	log_benchmark_presentation_frame();

	if (g_benchmark_finish_after_display)
	{
		std::cout << "Benchmark complete after " << g_current_frame
			<< " frames. Profile: " << GenPDResolveOutputPath("frame_profile.csv") << std::endl;
		cleanup();
		exit(EXIT_SUCCESS);
	}

	if (should_use_uncapped_benchmark() && !g_benchmark_no_render && g_current_frame > 0)
	{
		schedule_next_timeout();
	}
}

void log_benchmark_presentation_frame(void)
{
	if (!g_benchmark_frame_timer_active)
	{
		return;
	}

	g_benchmark_frame_timer.Toc();
	g_benchmark_render_timer.Toc();

	static bool initialized = false;
	static std::ofstream presentation_profile_file;
	if (!initialized)
	{
		const std::string profile_path = GenPDResolveOutputPath("frame_presentation.csv");
		GenPDEnsureDirectoryForFile(profile_path);
		presentation_profile_file.open(profile_path.c_str(), std::ios::out | std::ios::trunc);
		if (presentation_profile_file.is_open())
		{
			presentation_profile_file << "frame,rendered,frame_wall_ms,render_and_present_wall_ms,gpu_sync_enabled,screen_width,screen_height\n";
			presentation_profile_file.flush();
		}
		initialized = true;
	}

	if (presentation_profile_file.is_open())
	{
		presentation_profile_file << g_benchmark_profile_frame << ",1,"
			<< g_benchmark_frame_timer.DurationInSeconds() * 1000.0 << ","
			<< g_benchmark_render_timer.DurationInSeconds() * 1000.0 << ","
			<< (g_benchmark_sync_gpu ? 1 : 0) << ","
			<< g_screen_width << "," << g_screen_height << "\n";
		presentation_profile_file.flush();
	}

	g_benchmark_frame_timer_active = false;
}

void apply_benchmark_swap_interval(void)
{
	if (!g_benchmark_disable_vsync)
	{
		return;
	}

#ifdef _WIN32
	typedef int (__stdcall *SwapIntervalProc)(int);
	const auto raw_proc = wglGetProcAddress("wglSwapIntervalEXT");
	if (!raw_proc)
	{
		std::cout << "Benchmark disable vsync unavailable: wglSwapIntervalEXT missing." << std::endl;
		return;
	}

	const SwapIntervalProc swap_interval = reinterpret_cast<SwapIntervalProc>(raw_proc);
	const bool disabled = swap_interval(0) != 0;
	std::cout << "Benchmark disable vsync: " << (disabled ? "enabled" : "failed") << std::endl;
#else
	std::cout << "Benchmark disable vsync unavailable: WGL is only available on Windows." << std::endl;
#endif
}

void resize(int width, int height) {
	g_screen_width = width;
	g_screen_height = height;
    //set the viewport, more boilerplate
    glViewport(0, 0, width, height);
    g_camera->ResizeWindow(width, height);
	g_config_bar->ChangeTwBarWindowSize(g_screen_width, g_screen_height);

    glutPostRedisplay();
}

void timeout(int value)
{
	const bool uncapped_benchmark = should_use_uncapped_benchmark();
	if (!uncapped_benchmark)
	{
		schedule_next_timeout();
	}
    // keep track of time
    g_fps_tracker.timestamp();

    // ant tweak bar update
    int atb_feed_back = g_benchmark_mode ? 0 : g_config_bar->Update();
	if (atb_feed_back&ATB_RESHAPE_WINDOW)
	{
		glutReshapeWindow(g_screen_width, g_screen_height);
	}
	if (atb_feed_back&ATB_CHANGE_MATERIAL_PROPERTY)
	{
		g_simulation->SetMaterialProperty();
	}
	if (atb_feed_back&(ATB_CHANGE_TIME_STEP | ATB_CHANGE_INTEGRATION))
	{
		g_simulation->SetReprefactorFlag();
	}
	if (atb_feed_back&(ATB_INIT_MATLAB))
	{
#ifdef ENABLE_MATLAB_DEBUGGING
		g_debugger->Init();
#endif // ENABLE_MATLAB_DEBUGGING
	}

	if (g_recording_limit && g_current_frame > g_total_frame)
	{
		g_pause = true;
	}

	// simulation update
    if (!g_pause) 
    {
		// grab screen
		if (g_record)
		{
			char cap_name[64];
			sprintf_s(cap_name, 64, "screenshots\\ScreenCap%04d.png", g_current_frame);
			const std::string cap_filename = GenPDResolveOutputPath(cap_name);
			grab_screen(cap_filename.c_str());

			if (g_export_obj)
			{
				char mesh_name[64];
				sprintf_s(mesh_name, 64, "mesh\\Mesh%04d.obj", g_current_frame);
				const std::string mesh_filename = GenPDResolveOutputPath(mesh_name);
				GenPDEnsureDirectoryForFile(mesh_filename);
				g_mesh->ExportToOBJ(mesh_filename.c_str());

				char handle_name[64];
				sprintf_s(handle_name, 64, "handles\\Handle%04d.obj", g_current_frame);
				const std::string handle_filename = GenPDResolveOutputPath(handle_name);
				GenPDEnsureDirectoryForFile(handle_filename);
				g_simulation->SaveAttachmentConstraint(handle_filename.c_str());
			}
		}

		if (g_benchmark_mode && !g_benchmark_no_render)
		{
			g_benchmark_profile_frame = g_current_frame;
			g_benchmark_frame_timer.Tic();
			g_benchmark_frame_timer_active = true;
		}

		// update scene
		g_scene->Update(g_simulation->Timestep(), g_current_frame);

		// update animation
		g_simulation->AnimateHandle(g_current_frame);
		if (g_benchmark_swing_attachments)
		{
			g_simulation->UpdateAnimation(g_current_frame);
		}

		//g_global_timer.Tic();
		// update mesh
        g_simulation->Update();
        g_simulation->LogFrameProfile(g_current_frame, g_fps_tracker.fpsAverage(), g_fps_tracker.fpsInstant());
		//g_global_timer.Toc();
		//total_time += g_global_timer.Duration();
		//if ((g_current_frame+1)%g_interval == 0)
		//{
		//	std::cout << "Time per Frame = " << total_time/g_interval << " seconds" << std::endl;
		//	total_time = 0;
		//}
		g_current_frame ++;
		maybe_finish_benchmark();
    }

	if (!g_benchmark_no_render)
	{
		glutPostRedisplay();
	}
	else if (uncapped_benchmark && !g_benchmark_finish_after_display)
	{
		schedule_next_timeout();
	}
}

void display() {
	if (g_benchmark_no_render)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glutSwapBuffers();
		finish_display_frame();
		return;
	}

	if (g_benchmark_frame_timer_active)
	{
		g_benchmark_render_timer.Tic();
	}

    //Always and only do this at the start of a frame, it wipes the slate clean
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // aim camera
    g_renderer->SetCameraModelview(g_camera->GetViewMatrix());
    g_renderer->SetCameraProjection(g_camera->GetProjectionMatrix());

    // Draw world and cloth (using programmable shaders)
    g_renderer->ActivateShaderprog();
    g_scene->Draw(g_renderer->getVBO());

	bool use_gpu_mesh_draw = false;
	if ((g_show_mesh || g_show_wireframe) && g_simulation)
	{
		use_gpu_mesh_draw = g_simulation->PrepareCS2RenderBuffers();
	}

	if (g_show_mesh)
	{
		if (use_gpu_mesh_draw)
		{
			g_mesh->DrawGPUPositionNormal(
				g_renderer->getVBO(),
				g_simulation->CS2RenderPositionBuffer(),
				g_simulation->CS2RenderNormalBuffer(),
				g_show_texture & g_texture_load_succeed);
		}
		else
		{
			g_mesh->Draw(g_renderer->getVBO(), g_show_texture & g_texture_load_succeed);
		}
	}
	if (g_show_wireframe)
	{
		if (use_gpu_mesh_draw)
		{
			g_mesh->DrawWireFrameGPUPosition(g_renderer->getVBO(), g_simulation->CS2RenderPositionBuffer(), g_wireframe_linewidth);
		}
		else
		{
			g_mesh->DrawWireFrame(g_renderer->getVBO(), g_wireframe_linewidth);
		}
	}
	g_simulation->Draw(g_renderer->getVBO());

	// highlight selections
	g_selection_tool->HighlightSelectedVertices(g_renderer->getVBO());

	g_renderer->DeactivateShaderprog();

	g_selection_tool->Draw();
	
	if (!g_only_show_sim)
    {
        // Draw axis
        g_camera->DrawAxis();

        // Draw overlay
        draw_overlay();
    }

    // Draw tweak bar
    g_config_bar->Draw();

    maybe_capture_benchmark_frame();
    glutSwapBuffers();
	finish_display_frame();
}

void key_press(unsigned char key, int x, int y) {
    if (!TwEventKeyboardGLUT(key, x, y))
    {
        switch(key) {
        case 32:
            g_pause = !g_pause;
            break;
		case 'q':
		case 'Q':
			g_selection_tool->SetMode(GUI_MODE_SELECTION);
			break;
		case 'w':
		case 'W':
			g_selection_tool->SetMode(GUI_MODE_TRANSLATION);
			break;
		case 'e':
		case 'E':
			break;
		case 'r':
		case 'R':
			g_selection_tool->SetMode(GUI_MODE_ROTATION);
			break;
		case 'b':
		case 'B':
			g_show_mesh = !g_show_mesh;
			break;
		case 'n':
		case 'N':
			g_show_wireframe = !g_show_wireframe;
			break;
		case 't':
		case 'T':
			g_show_texture = !g_show_texture;
			break;
        case 'p':
        case 'P':
            step_through(NULL);
            break;
        case 27: // ascii code of esc key
            cleanup();
            exit(EXIT_SUCCESS);
            break;
        case '0':
            if (g_only_show_sim)
            {
                g_only_show_sim = false;
                g_config_bar->Show();
            }
            else
            {
                g_only_show_sim = true;
                g_config_bar->Hide();
            }
            break;
		case 's':
		case 'S':
			g_camera->SaveCamera();
			g_mesh->ExportToOBJ(DEFAULT_CONFIG_OBJ_FILE);
			g_simulation->SaveAttachmentConstraint(DEFAULT_CONFIG_TARGET_CONSTRAINT_FILE); // save for rendering
			g_simulation->SaveHandles(DEFAULT_CONFIG_HANDLE_FILE);
			g_simulation->SaveHandleAnimation(DEFAULT_CONFIG_ANIMATION_FILE);
			g_simulation->SavePerConstraintMaterialProperties(DEFAULT_CONFIG_MATERIAL_PROPERTIES);
			//g_simulation->SaveLaplacianMatrix(DEFAULT_LAPLACIAN_FILE);
			break;
		case 'l':
			if (g_mesh->ImportFromOBJ(DEFAULT_CONFIG_OBJ_FILE))
			{
				//g_simulation->LoadAttachmentConstraint(DEFAULT_CONFIG_TARGET_CONSTRAINT_FILE);
				g_simulation->LoadHandles(DEFAULT_CONFIG_HANDLE_FILE);
				g_simulation->LoadHandleAnimation(DEFAULT_CONFIG_ANIMATION_FILE);
			}
			g_camera->LoadCamera();
			g_mesh->Update();

			g_simulation->SetVisualizationMesh();
			g_simulation->ResetVisualizationMeshHeight();

			glutPostRedisplay();
			break;
		case 'L':
			if (g_mesh->ImportFromOBJ(DEFAULT_CONFIG_OBJ_FILE))
			{
				//g_simulation->LoadAttachmentConstraint(DEFAULT_CONFIG_TARGET_CONSTRAINT_FILE);
				g_simulation->LoadHandles(DEFAULT_CONFIG_HANDLE_FILE);
				g_simulation->LoadHandleAnimation(DEFAULT_CONFIG_ANIMATION_FILE);
				g_simulation->LoadPerConstraintMaterialProperties(DEFAULT_CONFIG_MATERIAL_PROPERTIES);
			}
			g_camera->LoadCamera();
			g_mesh->Update();

			g_simulation->SetVisualizationMesh();
			g_simulation->ResetVisualizationMeshHeight();

			glutPostRedisplay();
			break;
		case 'g':
		case 'G':
		{
			const std::string screenshot_path = GenPDResolveOutputPath("ScreenShot.png");
			const std::string mesh_path = GenPDResolveOutputPath("mesh.obj");
			const std::string handles_path = GenPDResolveOutputPath("handles.obj");
			grab_screen(screenshot_path.c_str());
			GenPDEnsureDirectoryForFile(mesh_path);
			g_mesh->ExportToOBJ(mesh_path.c_str());
			GenPDEnsureDirectoryForFile(handles_path);
			g_simulation->SaveAttachmentConstraint(handles_path.c_str());
			break;
		}
		case 'a':
		case 'A':
			g_mesh->Update();
			g_selection_tool->SelectVerticesHardCoded(g_mesh->m_positions);
			break;
        case 'f':
        case 'F':
            g_camera->Lookat(g_mesh);
            break;
		case '9':
			g_simulation->RandomizePoints();
			g_mesh->Update();
			glutPostRedisplay();
			break;
		case '.':
			//g_simulation->RotateHandleToValue();
			g_simulation->SetHandleTranslationAnimation();
			break;
		case '>':
			//g_simulation->RotateHandleToValue();
			g_simulation->SetHandleTranslation();
			break;
		case '/':
			g_simulation->SetHandleRotationAnimation();
			break;
		case '?':
			g_simulation->SetHandleRotation();
			break;
		case 8: // backspace
		case 127: // delete
			g_simulation->DeleteHandle();
			g_simulation->SetReprefactorFlag();
			break;
		}
    }

    glutPostRedisplay();
}

int parse_positive_int(const char* value, const int fallback)
{
	if (!value)
	{
		return fallback;
	}

	const int parsed_value = std::atoi(value);
	return parsed_value > 0 ? parsed_value : fallback;
}

int parse_nonnegative_int(const char* value, const int fallback)
{
	if (!value)
	{
		return fallback;
	}

	const int parsed_value = std::atoi(value);
	return parsed_value >= 0 ? parsed_value : fallback;
}

ScalarType parse_positive_scalar(const char* value, const ScalarType fallback)
{
	if (!value)
	{
		return fallback;
	}

	char* end = NULL;
	const double parsed_value = std::strtod(value, &end);
	return end != value && end && *end == '\0' && std::isfinite(parsed_value) && parsed_value > 0.0
		? static_cast<ScalarType>(parsed_value)
		: fallback;
}

void parse_command_line(int argc, char** argv)
{
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg(argv[i]);
		if (arg == "--benchmark")
		{
			g_benchmark_mode = true;
		}
		else if ((arg == "--frames" || arg == "--benchmark-frames") && i + 1 < argc)
		{
			g_benchmark_frames = parse_positive_int(argv[++i], g_benchmark_frames);
			g_benchmark_mode = true;
		}
		else if ((arg == "--warmup" || arg == "--benchmark-warmup") && i + 1 < argc)
		{
			g_benchmark_warmup_frames = parse_nonnegative_int(argv[++i], g_benchmark_warmup_frames);
			g_benchmark_mode = true;
		}
		else if (arg == "--no-render" || arg == "--benchmark-no-render")
		{
			g_benchmark_no_render = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--headless" || arg == "--benchmark-hide-window")
		{
			g_benchmark_hide_window = true;
			g_benchmark_no_render = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--benchmark-swing-attachments")
		{
			g_benchmark_swing_attachments = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--uncapped" || arg == "--benchmark-uncapped")
		{
			g_benchmark_uncapped = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--sync-gpu" || arg == "--benchmark-sync-gpu")
		{
			g_benchmark_sync_gpu = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--disable-vsync" || arg == "--benchmark-disable-vsync")
		{
			g_benchmark_disable_vsync = true;
			g_benchmark_mode = true;
		}
		else if (arg == "--project-root" && i + 1 < argc)
		{
			g_cli_project_root = argv[++i];
		}
		else if (arg == "--output-dir" && i + 1 < argc)
		{
			g_cli_output_dir = argv[++i];
		}
		else if (arg == "--run-label" && i + 1 < argc)
		{
			g_cli_run_label = argv[++i];
		}
		else if (arg == "--solver-variant" && i + 1 < argc)
		{
			g_cli_solver_variant = argv[++i];
		}
        else if (arg == "--iterations-per-frame" && i + 1 < argc)
        {
            g_cli_iterations_per_frame = parse_positive_int(argv[++i], g_cli_iterations_per_frame);
        }
        else if (arg == "--reference-export-dir" && i + 1 < argc)
        {
            g_cli_reference_export_dir = argv[++i];
        }
        else if (arg == "--quality-reference-dir" && i + 1 < argc)
        {
            g_cli_quality_reference_dir = argv[++i];
        }
        else if (arg == "--quality-checkpoint-stride" && i + 1 < argc)
        {
            g_cli_quality_checkpoint_stride = parse_positive_int(argv[++i], g_cli_quality_checkpoint_stride);
        }
        else if (arg == "--quality-metrics")
        {
            g_cli_quality_metrics = true;
        }
        else if (arg == "--timestep" && i + 1 < argc)
        {
            g_cli_timestep = parse_positive_scalar(argv[++i], g_cli_timestep);
        }
        else if (arg == "--stretch-stiffness" && i + 1 < argc)
        {
            g_cli_stretch_stiffness = parse_positive_scalar(argv[++i], g_cli_stretch_stiffness);
        }
        else if (arg == "--bending-stiffness" && i + 1 < argc)
        {
            g_cli_bending_stiffness = parse_positive_scalar(argv[++i], g_cli_bending_stiffness);
        }
        else if (arg == "--cloth-dimension" && i + 1 < argc)
        {
            g_cli_cloth_dimension = parse_positive_int(argv[++i], g_cli_cloth_dimension);
        }
        else if (arg == "--cloth-width" && i + 1 < argc)
        {
            g_cli_cloth_width = parse_positive_int(argv[++i], g_cli_cloth_width);
        }
        else if (arg == "--cloth-height" && i + 1 < argc)
        {
            g_cli_cloth_height = parse_positive_int(argv[++i], g_cli_cloth_height);
        }
        else if (arg == "--batched-ls-k" && i + 1 < argc)
        {
            g_cli_batched_ls_k = parse_positive_int(argv[++i], g_cli_batched_ls_k);
        }
        else if (arg == "--armijo-beta" && i + 1 < argc)
        {
            g_cli_armijo_beta = parse_positive_scalar(argv[++i], g_cli_armijo_beta);
        }
        else if (arg == "--adaptive-ls-history" && i + 1 < argc)
        {
            g_cli_adaptive_ls_history = argv[++i];
        }
        else if (arg == "--ncg-restart-mode" && i + 1 < argc)
        {
            g_cli_ncg_restart_mode = argv[++i];
        }
        else if (arg == "--ncg-restart-period" && i + 1 < argc)
        {
            g_cli_ncg_restart_period = parse_positive_int(argv[++i], g_cli_ncg_restart_period);
        }
        else if (arg == "--verify-cs-gradient" && i + 1 < argc)
        {
            g_cli_verify_cs_gradient_dir = argv[++i];
        }
		else if (arg == "--verify-adaptive-ls-history-resets" && i + 1 < argc)
		{
			g_cli_verify_adaptive_ls_history_reset_dir = argv[++i];
		}
        else if (arg == "--scene" && i + 1 < argc)
        {
            g_cli_scene = argv[++i];
        }
        else if (arg == "--capture-frame" && i + 1 < argc)
        {
            g_cli_capture_frame = parse_nonnegative_int(argv[++i], g_cli_capture_frame);
            g_cli_capture_pending = g_cli_capture_frame >= 0;
        }
        else if (arg == "--capture-output" && i + 1 < argc)
        {
            g_cli_capture_output = argv[++i];
        }
        else if (arg == "--capture-resolution" && i + 2 < argc)
        {
            const int width = parse_positive_int(argv[++i], 0);
            const int height = parse_positive_int(argv[++i], 0);
            if (width > 0 && height > 0)
            {
                g_cli_capture_width = width;
                g_cli_capture_height = height;
            }
        }
        else if (arg == "--render-resolution" && i + 2 < argc)
        {
            const int width = parse_positive_int(argv[++i], 0);
            const int height = parse_positive_int(argv[++i], 0);
            if (width > 0 && height > 0)
            {
                g_cli_render_width = width;
                g_cli_render_height = height;
            }
        }
        else if (arg == "--profile-gpu-queries")
		{
			g_cli_profile_gpu_queries = true;
        }
        else if (arg == "--profile-line-search-decisions")
        {
            g_cli_profile_line_search_decisions = true;
        }
        else if (arg == "--force-cpu-state-roundtrip")
        {
            g_cli_force_cpu_state_roundtrip = true;
        }
        else if (arg == "--energy-audit")
        {
            g_cli_energy_audit = true;
        }
        else if (arg == "--xpbd-fuse-apply-collision" && i + 1 < argc)
        {
            g_cli_xpbd_fuse_apply_collision = parse_nonnegative_int(argv[++i], g_cli_xpbd_fuse_apply_collision);
            if (g_cli_xpbd_fuse_apply_collision > 1)
            {
                std::cerr << "--xpbd-fuse-apply-collision must be 0 or 1." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--xpbd-cached-pins" && i + 1 < argc)
        {
            g_cli_xpbd_cached_pins = parse_nonnegative_int(argv[++i], g_cli_xpbd_cached_pins);
            if (g_cli_xpbd_cached_pins > 1)
            {
                std::cerr << "--xpbd-cached-pins must be 0 or 1." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--print-paths")
		{
			g_cli_print_paths = true;
		}
		else if (arg == "--help" || arg == "-h")
		{
			std::cout << "GenPD options:\n"
				<< "  --benchmark                 Run simulation automatically and exit.\n"
				<< "  --frames N                  Measured benchmark frames, default 300.\n"
				<< "  --warmup N                  Warmup frames to skip in analysis, default 30.\n"
				<< "  --no-render                 Skip scene rendering during benchmark.\n"
				<< "  --headless                  Hide the GLUT window and skip rendering.\n"
				<< "  --uncapped                  Benchmark without the 30 FPS timer cap.\n"
				<< "  --sync-gpu                  Call glFinish after benchmark swap buffers.\n"
				<< "  --disable-vsync             Try to disable WGL swap interval for benchmark.\n"
				<< "  --project-root PATH         Resolve config/shaders/textures from this GenPD root.\n"
				<< "  --output-dir PATH           Write benchmark/profile outputs to this directory.\n"
				<< "  --run-label NAME            Default output directory becomes results/NAME.\n"
                << "  --profile-gpu-queries       Read GL timer queries for GPU profile CSV fields.\n"
                << "  --profile-line-search-decisions  Trace Armijo decisions; diagnostic only.\n"
                << "  --force-cpu-state-roundtrip  Diagnostic: synchronize GPU-resident position/velocity state each frame.\n"
                << "  --energy-audit              Diagnostic: write CPU/GPU energy cross-checks; not performance data.\n"
                << "  --xpbd-fuse-apply-collision 0|1  Fuse XPBD vertex apply and collision passes (default 1).\n"
                << "  --xpbd-cached-pins 0|1     Cache hard-pin lookup instead of scanning CSR each XPBD iteration (default 1).\n"
                << "  --iterations-per-frame N    Override solver iterations for a reference run.\n"
                << "  --reference-export-dir PATH Export reference checkpoints to this directory.\n"
                << "  --quality-reference-dir PATH Compare quality metrics with checkpoints in this directory.\n"
                << "  --quality-checkpoint-stride N  Export/check checkpoints every N frames.\n"
                << "  --quality-metrics           Record quality metrics without a reference checkpoint.\n"
                << "  --timestep FLOAT            Override the simulation timestep for this run.\n"
                << "  --stretch-stiffness FLOAT   Override cloth stretch stiffness for this run.\n"
                << "  --bending-stiffness FLOAT   Override cloth bending stiffness for this run.\n"
                << "  --cloth-dimension N         Override square cloth resolution for this run.\n"
                << "  --cloth-width N             Override cloth grid width.\n"
                << "  --cloth-height N            Override cloth grid height.\n"
                << "  --batched-ls-k N            Set batched Armijo candidate count.\n"
                << "  --armijo-beta FLOAT         Set Armijo backtracking factor in (0,1).\n"
                << "  --adaptive-ls-history MODE  none | iteration | frame (default frame).\n"
                << "  --ncg-restart-mode MODE     none | periodic | non-descent.\n"
                << "  --ncg-restart-period N      Period for periodic NCG restart.\n"
                << "  --verify-cs-gradient PATH  Export CPU/edge/gather gradient diagnostics and exit.\n"
				<< "  --verify-adaptive-ls-history-resets PATH  Verify Reset/stiffness invalidation of adaptive history and exit.\n"
                << "  --scene PATH                Load a scene XML file relative to project root.\n"
                << "  --capture-frame N           Capture rendered benchmark frame N.\n"
                << "  --capture-output PATH       PNG output path for --capture-frame.\n"
                << "  --capture-resolution W H    Render capture at a fixed size.\n"
                << "  --render-resolution W H     Render benchmarks at a fixed viewport.\n"
				<< "  --solver-variant NAME       cpu-ncg | gpu-edge-scatter | gpu-gather-no-fusion |\n"
				<< "                              gpu-gather-fusion | gpu-gather-fusion-batched-ls |\n"
				<< "                              gpu-gather-fusion-serial-ls-persistent |\n"
				<< "                              gpu-gather-fusion-batched-ls-persistent |\n"
				<< "                              gpu-gather-fusion-adaptive-ls-persistent | gpu-xpbd-jacobi.\n"
				<< "  --print-paths               Print resolved project/output/executable paths.\n"
				<< "  --benchmark-swing-attachments\n"
				<< "                              Move attachment constraints during benchmark.\n";
			exit(EXIT_SUCCESS);
		}
	}

    if (g_cli_capture_pending)
    {
        g_benchmark_mode = true;
        g_benchmark_no_render = false;
        g_benchmark_hide_window = false;
    }
}

void maybe_capture_benchmark_frame(void)
{
    if (!g_cli_capture_pending || g_current_frame != g_cli_capture_frame + 1)
    {
        return;
    }

    std::string capture_path = g_cli_capture_output;
    if (capture_path.empty())
    {
        char capture_name[64];
        sprintf_s(capture_name, 64, "screenshots\\capture_frame_%06d.png", g_cli_capture_frame);
        capture_path = capture_name;
    }

    capture_path = GenPDResolveOutputPath(capture_path);
    GenPDEnsureDirectoryForFile(capture_path);
    grab_screen(capture_path.c_str());
    g_cli_capture_pending = false;
    std::cout << "Captured benchmark frame " << g_cli_capture_frame
        << ": " << capture_path << std::endl;
}

void maybe_finish_benchmark(void)
{
	if (!g_benchmark_mode)
	{
		return;
	}

	if (g_current_frame >= g_benchmark_warmup_frames + g_benchmark_frames)
	{
		if (should_use_uncapped_benchmark() && !g_benchmark_no_render)
		{
			g_benchmark_finish_after_display = true;
			return;
		}

		std::cout << "Benchmark complete after " << g_current_frame
			<< " frames. Profile: " << GenPDResolveOutputPath("frame_profile.csv") << std::endl;
		cleanup();
		exit(EXIT_SUCCESS);
	}
}

void mouse_click(int button, int state, int x, int y)
{
   // if (!TwEventMouseButtonGLUT(button, state, x, y))
   // {
   //     switch(state)
   //     {
   //     case GLUT_DOWN:
   //         if (glutGetModifiers() == GLUT_ACTIVE_ALT)
   //         {
   //             // left: 0. right: 2. middle: 1.
   //             g_button_mask |= 0x01 << button;
   //             g_mouse_old_x = x;
   //             g_mouse_old_y = y;
   //         }
   //         else if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
   //         {
   //             // ctrl: 3
   //             g_button_mask |= 0x01 << 3;
   //             g_mouse_old_x = x;
   //             g_mouse_old_y = y;
   //         }
 		//	else
			//{
			//	if (g_simulation->TryToToggleAttachmentConstraint(GLM2Eigen(g_camera->GetCameraPosition()), GLM2Eigen(g_camera->GetRaycastDirection(x, y))))
			//	{ // hit something
			//		g_simulation->SetReprefactorFlag();
			//	}
			//}
   //        break;
   //     case GLUT_UP:
   //         if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
   //         {// special case for ctrl
   //             button = 3;
   //         }

   //         g_simulation->UnselectAttachmentConstraint();

   //         unsigned char mask_not = ~g_button_mask;
   //         mask_not |= 0x01 << button;
   //         g_button_mask = ~mask_not;
   //         break;
   //     }
   // }
	if (!TwEventMouseButtonGLUT(button, state, x, y))
	{
		switch (state)
		{
		case GLUT_DOWN:
			// left: 2^0. right: 2^2. middle: 2^1.
			g_button_mask |= 0x01 << button;
			g_mouse_old_x = x;
			g_mouse_old_y = y;
			if (glutGetModifiers() != GLUT_ACTIVE_ALT)
			{
				switch (g_selection_tool->GetMode())
				{
				case GUI_MODE_SELECTION:
					// if selection mode
					// selection box start
					g_selection_tool->SelectFirstPoint(x, y, g_screen_width, g_screen_height, g_button_mask);
					break;
				case GUI_MODE_TRANSLATION:
					g_selection_tool->TranslateRotateFirstPoint();
					g_mouse_down = true;
					break;
				case GUI_MODE_ROTATION:
					g_selection_tool->TranslateRotateFirstPoint();
					g_mouse_down = true;
					break;
				}
			}
			break;
		case GLUT_UP:
			if (glutGetModifiers() != GLUT_ACTIVE_ALT)
			{
				g_selection_tool->SelectSecondPoint(x, y, g_screen_width, g_screen_height, true);
				switch (g_selection_tool->GetMode())
				{
				case GUI_MODE_SELECTION:
					// if selection mode
					// selection box end and selection
					g_mesh->Update();
					g_selection_tool->SelectVertices(g_mesh->m_positions, g_camera->GetMVP());
					g_simulation->SelectTetConstraints(g_selection_tool->SelectedIndices());
					g_simulation->GetPartialMaterialProperty();
					break;
				case GUI_MODE_TRANSLATION:
					g_simulation->MoveHandleFinalize();
					g_mouse_down = false;
					break;
				case GUI_MODE_ROTATION:
					g_simulation->RotateHandleFinalize();
					g_mouse_down = false;
					break;
				}
			}

			unsigned char mask_not = ~g_button_mask;
			mask_not |= 0x01 << button;
			g_button_mask = ~mask_not;
			break;
		}
	}

	glutPostRedisplay();
}

void mouse_motion(int x, int y)
{
	if (!TwEventMouseMotionGLUT(x, y))
	{
		float dx, dy;
		dx = (float)(x - g_mouse_old_x);
		dy = (float)(y - g_mouse_old_y);

		if (g_button_mask & 0x01)
		{// left button
			// alt + left button
			if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			{
				g_camera->MouseChangeHeadPitch(0.2f, dx, dy);
			}
			else
			{
				g_selection_tool->SelectSecondPoint(x, y, g_screen_width, g_screen_height);
				if (g_selection_tool->GetMode() == GUI_MODE_TRANSLATION)
				{
					g_simulation->MoveHandleTemporary(g_camera->GetCurrentTargetPoint(x, y));
				}
				else if (g_selection_tool->GetMode() == GUI_MODE_ROTATION)
				{
					ScalarType theta;
					glm::vec3 axis;
					g_camera->GetCurrentRotation(x, y, axis, theta);
					g_simulation->RotateHandleTemporary(axis, theta);
				}
			}
		}
		else if (g_button_mask & 0x02)
		{// middle button
			if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			{
				g_camera->MouseChangeLookat(0.01f, dx, dy);
			}
			else
			{
				g_selection_tool->SelectSecondPoint(x, y, g_screen_width, g_screen_height);
				if (g_selection_tool->GetMode() == GUI_MODE_TRANSLATION)
				{
					g_simulation->MoveHandleTemporary(g_camera->GetCurrentTargetPoint(x, y));
				}
				else if (g_selection_tool->GetMode() == GUI_MODE_ROTATION)
				{
					//ScalarType theta;
					//glm::vec3 axis;
					////g_camera->GetCurrentRotation(x, y, axis, theta);
					//g_simulation->RotateHandleTemporary(axis, theta);
				}
			}
		}
		else if (g_button_mask & 0x04)
		{// right button
			if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			{
				g_camera->MouseChangeDistance(0.05f, dx, dy);
			}
			else
			{
				g_selection_tool->SelectSecondPoint(x, y, g_screen_width, g_screen_height);
				if (g_selection_tool->GetMode() == GUI_MODE_TRANSLATION)
				{
					g_simulation->MoveHandleTemporary(g_camera->GetCurrentTargetPoint(x, y));
				}
				else if (g_selection_tool->GetMode() == GUI_MODE_ROTATION)
				{
					ScalarType theta;
					glm::vec3 axis;
					g_camera->GetCurrentRotation(x, y, axis, theta);
					g_simulation->RotateHandleTemporary(axis, theta);
				}
			}
		}
		//else if (g_button_mask & 0x08)
		//{// ctrl + button
		//	g_simulation->MoveSelectedAttachmentConstraintTo(GLM2Eigen(g_camera->GetCurrentTargetPoint(x, y)));
		//}

		g_mouse_old_x = x;
		g_mouse_old_y = y;
	}

	glutPostRedisplay();
}

void mouse_wheel(int button, int dir, int x, int y)
{
    if (!TwMouseWheel(g_mouse_wheel_pos+=dir))
    {
        g_camera->MouseChangeDistance(1.0f, 0, (ScalarType)(dir));
    }

    glutPostRedisplay();
}

void mouse_over(int x, int y)
{
	if (!TwEventMouseMotionGLUT(x, y))
	{
		if (g_button_mask == 0)
		{
			if (g_selection_tool->HoverSelectHandle(g_simulation, g_camera->CastRay(x, y), g_camera->GetMVP()))
			{
				g_camera->CacheLastSelectedPointLocalCoM(g_simulation->SelectedHandleLocalCoM());
				g_camera->CacheLastSelectedPointGlobalCoM(g_simulation->SelectedHandleCoM());
			}
		}
	}

	glutPostRedisplay();
}

void apply_cli_simulation_overrides()
{
	if (!g_mesh || !g_simulation)
	{
		return;
	}

	if (g_cli_cloth_dimension > 0)
	{
		g_mesh->m_dim[0] = static_cast<unsigned int>(g_cli_cloth_dimension);
		g_mesh->m_dim[1] = static_cast<unsigned int>(g_cli_cloth_dimension);
	}
	if (g_cli_cloth_width > 0)
	{
		g_mesh->m_dim[0] = static_cast<unsigned int>(g_cli_cloth_width);
	}
	if (g_cli_cloth_height > 0)
	{
		g_mesh->m_dim[1] = static_cast<unsigned int>(g_cli_cloth_height);
	}
	if (g_cli_timestep > 0)
	{
		g_simulation->SetTimestep(g_cli_timestep);
	}
	g_simulation->SetExperimentMaterialStiffness(g_cli_stretch_stiffness, g_cli_bending_stiffness);
}

void init()
{
    // glew init
    fprintf(stdout, "Initializing glew...\n");
    glewInit();
    if (!glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
        )) {
            std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
            exit(EXIT_FAILURE);
    }
	apply_benchmark_swap_interval();


    // config init
    fprintf(stdout, "Initializing AntTweakBar...\n");
    g_config_bar = new AntTweakBarWrapper();
if (g_benchmark_mode)
{
    g_config_bar->SetSaveSettingsOnDestroy(false);
}
    g_config_bar->ChangeTwBarWindowSize(g_screen_width, g_screen_height);

#ifdef ENABLE_MATLAB_DEBUGGING
	// debugger init
	g_debugger = new MatlabDebugger();
#endif // ENABLE_MATLAB_DEBUGGING

    // render wrapper init
    fprintf(stdout, "Initializing render wrapper...\n");
    g_renderer = new RenderWrapper();
    g_renderer->InitShader(DEFAULT_VERT_SHADER_FILE, DEFAULT_FRAG_SHADER_FILE);
    g_texture_load_succeed = g_renderer->InitTexture(DEFAULT_TEXTURE_FILE);

	// selection tool
	g_selection_tool = new SelectionTool();

    // camera init
    fprintf(stdout, "Initializing camera...\n");
    g_camera = new Camera();

    // scene init
    fprintf(stdout, "Initializing scene...\n");
    const std::string requested_scene = g_cli_scene.empty() ? std::string(DEFAULT_SCENE_FILE) : g_cli_scene;
    const std::string scene_path = GenPDResolveProjectPath(requested_scene);
    g_scene = new Scene(scene_path.c_str());

    // mesh init
    fprintf(stdout, "Initializing mesh...\n");
    g_mesh = new Mesh();

    // simulation init
    fprintf(stdout, "Initializing simulation...\n");
    g_simulation = new Simulation();

	// load or get default value
	g_config_bar->LoadSettings();

	reset_camera(NULL);
    reset_simulation(NULL);
}

void cleanup() // clean up in a reverse order
{
    //if (g_simulation)
    //    delete g_simulation;
    ////if (g_mesh)
    ////    delete g_mesh;
    if (g_scene)
        delete g_scene;
    if (g_camera)
        delete g_camera;
    if (g_renderer)
    {
        g_renderer->CleanupShader();
        delete g_renderer;
    }
	if (g_selection_tool)
	{
		delete g_selection_tool;
	}
#ifdef ENABLE_MATLAB_DEBUGGING
	if (g_debugger)
	{
		delete g_debugger;
	}
#endif // ENABLE_MATLAB_DEBUGGING
    if (g_cli_restore_iterations_per_frame && g_simulation)
{
    g_simulation->SetIterationsPerFrame(g_cli_original_iterations_per_frame);
}
if (g_config_bar)
    {
        delete g_config_bar;
    }
}

void TW_CALL set_handle(void*)
{
	// set handle
	if (!g_selection_tool->SelectedIndices().empty())
	{
		g_simulation->NewHandle(g_selection_tool->SelectedIndices(), g_handle_color);
		// change color
		if (g_random_handle_color)
		{
			glm::vec3 hsv_color;
			hsv_color[0] = rand() / (float)RAND_MAX * 360.0f;
			hsv_color[1] = 0.95f;
			hsv_color[2] = 1.0f;
			g_handle_color = glm::rgbColor(hsv_color);
		}
		g_simulation->SetReprefactorFlag();
	}
	g_selection_tool->Reset();
}
void TW_CALL save_handle(void*)
{
	g_simulation->SaveHandles(DEFAULT_CONFIG_HANDLE_FILE);
}
void TW_CALL load_handle(void*)
{
	g_simulation->LoadHandles(DEFAULT_CONFIG_HANDLE_FILE);
}
void TW_CALL reset_handle(void*)
{
	g_simulation->ResetHandles();
}

void TW_CALL reset_simulation(void*)
{
	// save current setting before reset
	if (!g_benchmark_mode)
{
    AntTweakBarWrapper::SaveSettings(g_config_bar);
}

	// reset frame#
	g_current_frame = 0;
	g_pause = true;

	switch(g_mesh->GetMeshType())
    {
	case MESH_TYPE_CLOTH:
		delete g_mesh;
		g_mesh = new ClothMesh();
        break;
	case MESH_TYPE_TET:
		delete g_mesh;
        g_mesh = new TetMesh();
        break;
    }
	g_config_bar->LoadSettings();
	apply_cli_simulation_overrides();
    g_mesh->Reset();

    // reset simulation
    g_simulation->SetMesh(g_mesh);
	g_simulation->ResetVisualizationMesh();
	g_simulation->SetVisualizationMesh();
	g_simulation->ResetVisualizationMeshHeight();
	g_simulation->SetScene(g_scene);

    g_simulation->Reset();

	// reset selection
	g_selection_tool->Reset();
	
	// reset config, (config bar is recommended to reset last)
	g_config_bar->Reset();

	// reset scene
	g_scene->Reset();

	// reset matlab debugger related stuff
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->InitVisualizationVector(g_mesh->GetNumberOfVertices());
#endif // ENABLE_MATLAB_DEBUGGING
}

void TW_CALL reset_camera(void*)
{
    // reset camera
	g_camera->Reset(g_screen_width, g_screen_height);
	// reset selection
	g_selection_tool->Reset();
}

void TW_CALL set_partial_material_property(void*)
{
	g_simulation->SetPartialMaterialProperty();
}

void TW_CALL step_through(void*)
{
    if(!g_pause)
    {
        g_pause = true;
    }

	g_scene->Update(g_simulation->Timestep(), g_current_frame);

	g_simulation->AnimateHandle(g_current_frame);

    // enable step mode
	g_simulation->SetStepMode(true);    // update cloth
    g_simulation->Update();
    g_simulation->LogFrameProfile(g_current_frame, g_fps_tracker.fpsAverage(), g_fps_tracker.fpsInstant());
    // disable step mode
	g_simulation->SetStepMode(false);

    g_current_frame++;
}

void grab_screen(void)
{
	char anim_name[64];
	sprintf_s(anim_name, 64, "Simulation%04d.png", g_current_frame);
	const std::string anim_filename = GenPDResolveOutputPath(anim_name);
	grab_screen(anim_filename.c_str());
}

void grab_screen(const char* filename)
{
	unsigned char* bitmapData = new unsigned char[3 * g_screen_width * g_screen_height];

    for (int i=0; i < g_screen_height; i++) 
    {
        glReadPixels(0, i, g_screen_width, 1, GL_RGB, GL_UNSIGNED_BYTE, 
            bitmapData + (g_screen_width * 3 * ((g_screen_height - 1) - i)));
    }

	GenPDEnsureDirectoryForFile(filename);
    stbi_write_png(filename, g_screen_width, g_screen_height, 3, bitmapData, g_screen_width * 3);

    delete [] bitmapData;
}

void draw_overlay()
{
    // Draw Overlay
    glColor4d(0.0, 0.0, 0.0, 1.0);
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRasterPos2d(0.03, 0.01);

    char overlay_char_from_simulation[255] = ""; 
    g_simulation->GetOverlayChar(overlay_char_from_simulation, 255);

    char info[1024];
	sprintf_s(info, "FPS: %3.1f | Frame#: %d%s", g_fps_tracker.fpsAverage(), g_current_frame, overlay_char_from_simulation);


	if (!g_pause) {
		cowhorse += g_fps_tracker.fpsAverage();
		niuma++;
		if (niuma >= max_t) {
			printf("Æ½¾ùÖ¡ÂÊ£º%.2f\n", cowhorse / max_t);
			cowhorse = 0;
			niuma = 0;
		}
	}
    for (unsigned int i = 0; i < strlen(info); i++)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, info[i]);
    }

    glPopAttrib();
}

// matlab calls
void TW_CALL matlab_reset_current_data(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->ResetCurrent();
#endif
}
void TW_CALL matlab_reset_all_data(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->Reset();
#endif
}
void TW_CALL matlab_set_converged_energy(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_simulation->SetConvergedEnergy();
#endif
}
void TW_CALL matlab_new_data(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->AddNewPlotData();
	g_debugger->ResetCurrent();
#endif
}
void TW_CALL matlab_remove_last_data(void*)
{
#ifdef  ENABLE_MATLAB_DEBUGGING
	g_debugger->RemoveLastData();
#endif //  ENABLE_MATLAB_DEBUGGING

}
void TW_CALL matlab_export_data(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->Export();
#endif
}
void TW_CALL matlab_import_data(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->Import();
#endif
}

void TW_CALL matlab_plot(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->Plot();
#endif
}

void TW_CALL matlab_plot_all(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->PlotAll();
#endif
}

void TW_CALL matlab_visualize_vector(void*)
{
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SetVisualizationVariableName();
#endif
}

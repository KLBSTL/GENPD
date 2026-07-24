#ifndef _GENPD_RUNTIME_PATHS_H_
#define _GENPD_RUNTIME_PATHS_H_

#include <string>

void GenPDInitializeRuntimePaths(
	const char* argv0,
	const std::string& requested_project_root,
	const std::string& requested_output_dir,
	const std::string& requested_run_label,
	bool profile_gpu_queries,
	bool print_paths);

const std::string& GenPDProjectRoot();
const std::string& GenPDOutputDir();
const std::string& GenPDRunLabel();
const std::string& GenPDExecutablePath();
const std::string& GenPDOriginalWorkingDirectory();

bool GenPDProfileGpuQueriesEnabled();
bool GenPDPrintPathsEnabled();

std::string GenPDResolveProjectPath(const std::string& path);
std::string GenPDResolveOutputPath(const std::string& path);
bool GenPDEnsureDirectory(const std::string& path);
bool GenPDEnsureDirectoryForFile(const std::string& path);

void GenPDWriteRunMetadata(
	int argc,
	char** argv,
	bool benchmark_mode,
	int benchmark_frames,
	int benchmark_warmup_frames,
	bool benchmark_no_render,
	bool benchmark_uncapped,
	bool benchmark_sync_gpu,
	bool benchmark_disable_vsync,
	bool benchmark_hide_window,
	const char* gl_vendor,
	const char* gl_renderer,
	const char* gl_version);

#endif

#include "runtime_paths.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace
{
	std::string g_project_root;
	std::string g_output_dir;
	std::string g_run_label = "default";
	std::string g_executable_path;
	std::string g_original_working_directory;
	bool g_profile_gpu_queries = false;
	bool g_print_paths = false;

	bool IsSlash(char c)
	{
		return c == '/' || c == '\\';
	}

	bool IsAbsolutePath(const std::string& path)
	{
		if (path.size() >= 2 && path[1] == ':')
		{
			return true;
		}
		return path.size() >= 2 && IsSlash(path[0]) && IsSlash(path[1]);
	}

	std::string NormalizeSlashes(std::string path)
	{
		std::replace(path.begin(), path.end(), '/', '\\');
		while (path.size() > 1 && IsSlash(path[path.size() - 1]))
		{
			if (path.size() == 3 && path[1] == ':')
			{
				break;
			}
			path.erase(path.size() - 1);
		}
		return path;
	}

	std::string NormalizePath(const std::string& path)
	{
		if (path.empty())
		{
			return path;
		}

		DWORD needed = GetFullPathNameA(path.c_str(), 0, NULL, NULL);
		if (needed == 0)
		{
			return NormalizeSlashes(path);
		}

		std::vector<char> buffer(needed + 1, '\0');
		DWORD written = GetFullPathNameA(path.c_str(), static_cast<DWORD>(buffer.size()), &buffer[0], NULL);
		if (written == 0 || written >= buffer.size())
		{
			return NormalizeSlashes(path);
		}
		return NormalizeSlashes(std::string(&buffer[0]));
	}

	std::string JoinPath(const std::string& base, const std::string& child)
	{
		if (child.empty())
		{
			return NormalizePath(base);
		}
		if (IsAbsolutePath(child))
		{
			return NormalizePath(child);
		}

		std::string trimmed = child;
		while (trimmed.size() >= 2 && trimmed[0] == '.' && IsSlash(trimmed[1]))
		{
			trimmed = trimmed.substr(2);
		}

		if (base.empty())
		{
			return NormalizePath(trimmed);
		}
		const char separator = IsSlash(base[base.size() - 1]) ? '\0' : '\\';
		return NormalizePath(separator == '\0' ? base + trimmed : base + separator + trimmed);
	}

	bool FileExists(const std::string& path)
	{
		DWORD attrs = GetFileAttributesA(path.c_str());
		return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
	}

	bool DirectoryExists(const std::string& path)
	{
		DWORD attrs = GetFileAttributesA(path.c_str());
		return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
	}

	std::string ParentPath(const std::string& path)
	{
		std::string normalized = NormalizeSlashes(path);
		std::size_t slash = normalized.find_last_of("\\/");
		if (slash == std::string::npos)
		{
			return std::string();
		}
		if (slash == 2 && normalized.size() >= 3 && normalized[1] == ':')
		{
			return normalized.substr(0, 3);
		}
		return normalized.substr(0, slash);
	}

	bool LooksLikeProjectRoot(const std::string& path)
	{
		const std::string root = NormalizePath(path);
		return DirectoryExists(root)
			&& FileExists(JoinPath(root, "source\\main.cpp"))
			&& FileExists(JoinPath(root, "shaders\\gradient.comp"))
			&& FileExists(JoinPath(root, "config\\config.txt"));
	}

	void AddCandidate(std::vector<std::string>& candidates, const std::string& candidate)
	{
		if (candidate.empty())
		{
			return;
		}
		candidates.push_back(NormalizePath(candidate));
		candidates.push_back(NormalizePath(JoinPath(candidate, "GenPD")));
	}

	std::string CurrentDirectory()
	{
		DWORD needed = GetCurrentDirectoryA(0, NULL);
		if (needed == 0)
		{
			return std::string();
		}
		std::vector<char> buffer(needed + 1, '\0');
		if (GetCurrentDirectoryA(static_cast<DWORD>(buffer.size()), &buffer[0]) == 0)
		{
			return std::string();
		}
		return NormalizeSlashes(std::string(&buffer[0]));
	}

	std::string ModulePath(const char* argv0)
	{
		char module_path[MAX_PATH] = { 0 };
		DWORD length = GetModuleFileNameA(NULL, module_path, MAX_PATH);
		if (length > 0 && length < MAX_PATH)
		{
			return NormalizePath(module_path);
		}
		return argv0 ? NormalizePath(argv0) : std::string();
	}

	std::string DetectProjectRoot(const std::string& requested_project_root, const std::string& executable_path)
	{
		if (!requested_project_root.empty())
		{
			return NormalizePath(requested_project_root);
		}

		std::vector<std::string> candidates;
		const std::string cwd = CurrentDirectory();
		const std::string exe_dir = ParentPath(executable_path);
		AddCandidate(candidates, cwd);
		AddCandidate(candidates, exe_dir);

		std::string walker = exe_dir;
		for (int i = 0; i < 5 && !walker.empty(); ++i)
		{
			AddCandidate(candidates, walker);
			walker = ParentPath(walker);
		}

		for (std::vector<std::string>::const_iterator it = candidates.begin(); it != candidates.end(); ++it)
		{
			if (LooksLikeProjectRoot(*it))
			{
				return NormalizePath(*it);
			}
		}
		return NormalizePath(cwd);
	}

	std::string SanitizeRunLabel(const std::string& label)
	{
		if (label.empty())
		{
			return "default";
		}

		std::string sanitized = label;
		for (std::string::iterator it = sanitized.begin(); it != sanitized.end(); ++it)
		{
			const bool ok = (*it >= 'a' && *it <= 'z')
				|| (*it >= 'A' && *it <= 'Z')
				|| (*it >= '0' && *it <= '9')
				|| *it == '-' || *it == '_';
			if (!ok)
			{
				*it = '_';
			}
		}
		return sanitized;
	}

	std::string JsonEscape(const std::string& value)
	{
		std::ostringstream out;
		for (std::string::const_iterator it = value.begin(); it != value.end(); ++it)
		{
			switch (*it)
			{
			case '\\': out << "\\\\"; break;
			case '"': out << "\\\""; break;
			case '\n': out << "\\n"; break;
			case '\r': out << "\\r"; break;
			case '\t': out << "\\t"; break;
			default: out << *it; break;
			}
		}
		return out.str();
	}

	std::string NowIsoLocal()
	{
		SYSTEMTIME now;
		GetLocalTime(&now);
		char buffer[64] = { 0 };
		sprintf_s(
			buffer,
			sizeof(buffer),
			"%04u-%02u-%02uT%02u:%02u:%02u",
			now.wYear,
			now.wMonth,
			now.wDay,
			now.wHour,
			now.wMinute,
			now.wSecond);
		return std::string(buffer);
	}

	std::string EnvValue(const char* name)
	{
		char* value = NULL;
		std::size_t length = 0;
		if (_dupenv_s(&value, &length, name) != 0 || value == NULL)
		{
			return std::string();
		}
		std::string result(value);
		free(value);
		return result;
	}
}

void GenPDInitializeRuntimePaths(
	const char* argv0,
	const std::string& requested_project_root,
	const std::string& requested_output_dir,
	const std::string& requested_run_label,
	bool profile_gpu_queries,
	bool print_paths)
{
	g_original_working_directory = CurrentDirectory();
	g_executable_path = ModulePath(argv0);
	g_run_label = SanitizeRunLabel(requested_run_label);
	g_profile_gpu_queries = profile_gpu_queries;
	g_print_paths = print_paths;
	g_project_root = DetectProjectRoot(requested_project_root, g_executable_path);

	if (!LooksLikeProjectRoot(g_project_root))
	{
		std::cerr << "Warning: project root does not contain expected GenPD inputs: " << g_project_root << std::endl;
	}

	if (requested_output_dir.empty())
	{
		g_output_dir = JoinPath(g_project_root, std::string("results\\") + g_run_label);
	}
	else if (IsAbsolutePath(requested_output_dir))
	{
		g_output_dir = NormalizePath(requested_output_dir);
	}
	else
	{
		g_output_dir = JoinPath(g_project_root, requested_output_dir);
	}

	GenPDEnsureDirectory(g_output_dir);
	SetCurrentDirectoryA(g_project_root.c_str());

	if (g_print_paths)
	{
		std::cout << "GenPD paths:" << std::endl
			<< "  project_root=" << g_project_root << std::endl
			<< "  output_dir=" << g_output_dir << std::endl
			<< "  run_label=" << g_run_label << std::endl
			<< "  executable=" << g_executable_path << std::endl
			<< "  original_cwd=" << g_original_working_directory << std::endl;
	}
}

const std::string& GenPDProjectRoot()
{
	return g_project_root;
}

const std::string& GenPDOutputDir()
{
	return g_output_dir;
}

const std::string& GenPDRunLabel()
{
	return g_run_label;
}

const std::string& GenPDExecutablePath()
{
	return g_executable_path;
}

const std::string& GenPDOriginalWorkingDirectory()
{
	return g_original_working_directory;
}

bool GenPDProfileGpuQueriesEnabled()
{
	return g_profile_gpu_queries;
}

bool GenPDPrintPathsEnabled()
{
	return g_print_paths;
}

std::string GenPDResolveProjectPath(const std::string& path)
{
	return IsAbsolutePath(path) ? NormalizePath(path) : JoinPath(g_project_root, path);
}

std::string GenPDResolveOutputPath(const std::string& path)
{
	return IsAbsolutePath(path) ? NormalizePath(path) : JoinPath(g_output_dir, path);
}

bool GenPDEnsureDirectory(const std::string& path)
{
	if (path.empty() || DirectoryExists(path))
	{
		return true;
	}

	const std::string parent = ParentPath(path);
	if (!parent.empty() && parent != path && !DirectoryExists(parent))
	{
		GenPDEnsureDirectory(parent);
	}

	if (CreateDirectoryA(path.c_str(), NULL) != 0 || GetLastError() == ERROR_ALREADY_EXISTS)
	{
		return true;
	}
	return DirectoryExists(path);
}

bool GenPDEnsureDirectoryForFile(const std::string& path)
{
	const std::string parent = ParentPath(path);
	return parent.empty() ? true : GenPDEnsureDirectory(parent);
}

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
	const char* gl_version)
{
	const std::string metadata_path = GenPDResolveOutputPath("run_metadata.json");
	GenPDEnsureDirectoryForFile(metadata_path);
	std::ofstream out(metadata_path.c_str(), std::ios::out | std::ios::trunc);
	if (!out)
	{
		std::cerr << "Warning: cannot write run metadata: " << metadata_path << std::endl;
		return;
	}

	out << "{\n";
	out << "  \"timestamp_local\": \"" << JsonEscape(NowIsoLocal()) << "\",\n";
	out << "  \"run_label\": \"" << JsonEscape(g_run_label) << "\",\n";
	out << "  \"project_root\": \"" << JsonEscape(g_project_root) << "\",\n";
	out << "  \"output_dir\": \"" << JsonEscape(g_output_dir) << "\",\n";
	out << "  \"executable\": \"" << JsonEscape(g_executable_path) << "\",\n";
	out << "  \"original_working_directory\": \"" << JsonEscape(g_original_working_directory) << "\",\n";
	out << "  \"git_commit\": \"" << JsonEscape(EnvValue("GENPD_GIT_COMMIT")) << "\",\n";
	out << "  \"gpu_name\": \"" << JsonEscape(EnvValue("GENPD_GPU_NAME")) << "\",\n";
	out << "  \"nvidia_driver_version\": \"" << JsonEscape(EnvValue("GENPD_NVIDIA_DRIVER_VERSION")) << "\",\n";
	out << "  \"solver_variant\": \"" << JsonEscape(EnvValue("GENPD_SOLVER_VARIANT")) << "\",\n";
	out << "  \"quality\": {\n";
	out << "    \"iterations_per_frame\": \"" << JsonEscape(EnvValue("GENPD_ITERATIONS_PER_FRAME")) << "\",\n";
	out << "    \"reference_export_dir\": \"" << JsonEscape(EnvValue("GENPD_REFERENCE_EXPORT_DIR")) << "\",\n";
	out << "    \"reference_dir\": \"" << JsonEscape(EnvValue("GENPD_QUALITY_REFERENCE_DIR")) << "\",\n";
	out << "    \"checkpoint_stride\": \"" << JsonEscape(EnvValue("GENPD_QUALITY_CHECKPOINT_STRIDE")) << "\",\n";
	out << "    \"metrics_enabled\": \"" << JsonEscape(EnvValue("GENPD_QUALITY_METRICS")) << "\"\n";
	out << "  },\n";
	out << "  \"experiment_overrides\": {\n";
	out << "    \"timestep\": \"" << JsonEscape(EnvValue("GENPD_TIMESTEP_OVERRIDE")) << "\",\n";
	out << "    \"stretch_stiffness\": \"" << JsonEscape(EnvValue("GENPD_STRETCH_STIFFNESS_OVERRIDE")) << "\",\n";
	out << "    \"bending_stiffness\": \"" << JsonEscape(EnvValue("GENPD_BENDING_STIFFNESS_OVERRIDE")) << "\",\n";
	out << "    \"cloth_dimension\": \"" << JsonEscape(EnvValue("GENPD_CLOTH_DIMENSION_OVERRIDE")) << "\",\n";
	out << "    \"scene\": \"" << JsonEscape(EnvValue("GENPD_SCENE")) << "\"\n";
	out << "  },\n";
	out << "  \"profile_gpu_queries\": " << (g_profile_gpu_queries ? "true" : "false") << ",\n";
	out << "  \"benchmark\": {\n";
	out << "    \"enabled\": " << (benchmark_mode ? "true" : "false") << ",\n";
	out << "    \"frames\": " << benchmark_frames << ",\n";
	out << "    \"warmup_frames\": " << benchmark_warmup_frames << ",\n";
	out << "    \"no_render\": " << (benchmark_no_render ? "true" : "false") << ",\n";
	out << "    \"uncapped\": " << (benchmark_uncapped ? "true" : "false") << ",\n";
	out << "    \"sync_gpu\": " << (benchmark_sync_gpu ? "true" : "false") << ",\n";
	out << "    \"disable_vsync\": " << (benchmark_disable_vsync ? "true" : "false") << ",\n";
	out << "    \"hide_window\": " << (benchmark_hide_window ? "true" : "false") << "\n";
	out << "  },\n";
	out << "  \"opengl\": {\n";
	out << "    \"vendor\": \"" << JsonEscape(gl_vendor ? gl_vendor : "") << "\",\n";
	out << "    \"renderer\": \"" << JsonEscape(gl_renderer ? gl_renderer : "") << "\",\n";
	out << "    \"version\": \"" << JsonEscape(gl_version ? gl_version : "") << "\"\n";
	out << "  },\n";
	out << "  \"argv\": [";
	for (int i = 0; i < argc; ++i)
	{
		if (i > 0)
		{
			out << ", ";
		}
		out << "\"" << JsonEscape(argv[i] ? argv[i] : "") << "\"";
	}
	out << "]\n";
	out << "}\n";
}

#include "quality_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

#include "runtime_paths.h"

namespace
{
	const char kReferenceMagic[8] = { 'G', 'P', 'D', 'Q', 'R', 'E', 'F', '1' };
	const std::uint32_t kReferenceVersion = 1u;

	bool WriteU32(std::ofstream& output, std::uint32_t value)
	{
		output.write(reinterpret_cast<const char*>(&value), sizeof(value));
		return static_cast<bool>(output);
	}

	bool ReadU32(std::ifstream& input, std::uint32_t& value)
	{
		input.read(reinterpret_cast<char*>(&value), sizeof(value));
		return static_cast<bool>(input);
	}
}

std::string GenPDReferenceCheckpointPath(const std::string& directory, unsigned int frame)
{
	char filename[64] = { 0 };
	sprintf_s(filename, sizeof(filename), "reference_state_%06u.bin", frame);
	if (directory.empty())
	{
		return std::string(filename);
	}

	const char last = directory[directory.size() - 1];
	return directory + ((last == '\\' || last == '/') ? "" : "\\") + filename;
}

bool GenPDWriteReferenceCheckpoint(
	const std::string& directory,
	unsigned int frame,
	const VectorX& positions,
	const VectorX& velocities)
{
	if (directory.empty() || positions.size() != velocities.size() || positions.size() <= 0)
	{
		return false;
	}

	if (!GenPDEnsureDirectory(directory))
	{
		return false;
	}

	std::ofstream output(GenPDReferenceCheckpointPath(directory, frame).c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (!output)
	{
		return false;
	}

	const std::uint32_t element_count = static_cast<std::uint32_t>(positions.size());
	output.write(kReferenceMagic, sizeof(kReferenceMagic));
	WriteU32(output, kReferenceVersion);
	WriteU32(output, frame);
	WriteU32(output, static_cast<std::uint32_t>(sizeof(ScalarType)));
	WriteU32(output, element_count);
	output.write(reinterpret_cast<const char*>(positions.data()), positions.size() * sizeof(ScalarType));
	output.write(reinterpret_cast<const char*>(velocities.data()), velocities.size() * sizeof(ScalarType));
	return static_cast<bool>(output);
}

bool GenPDReadReferenceCheckpoint(
	const std::string& directory,
	unsigned int frame,
	VectorX& positions,
	VectorX& velocities)
{
	std::ifstream input(GenPDReferenceCheckpointPath(directory, frame).c_str(), std::ios::in | std::ios::binary);
	if (!input)
	{
		return false;
	}

	char magic[sizeof(kReferenceMagic)] = { 0 };
	std::uint32_t version = 0;
	std::uint32_t stored_frame = 0;
	std::uint32_t scalar_bytes = 0;
	std::uint32_t element_count = 0;
	input.read(magic, sizeof(magic));
	if (!input || std::memcmp(magic, kReferenceMagic, sizeof(magic)) != 0
		|| !ReadU32(input, version) || !ReadU32(input, stored_frame)
		|| !ReadU32(input, scalar_bytes) || !ReadU32(input, element_count))
	{
		return false;
	}

	if (version != kReferenceVersion || stored_frame != frame
		|| scalar_bytes != sizeof(ScalarType) || element_count == 0u)
	{
		return false;
	}

	positions.resize(element_count);
	velocities.resize(element_count);
	input.read(reinterpret_cast<char*>(positions.data()), positions.size() * sizeof(ScalarType));
	input.read(reinterpret_cast<char*>(velocities.data()), velocities.size() * sizeof(ScalarType));
	return static_cast<bool>(input);
}

bool GenPDVectorIsFinite(const VectorX& values)
{
	for (int i = 0; i < values.size(); ++i)
	{
		if (!std::isfinite(values[i]))
		{
			return false;
		}
	}
	return true;
}

ScalarType GenPDRelativeL2(const VectorX& values, const VectorX& reference_values)
{
	if (values.size() != reference_values.size() || values.size() == 0)
	{
		return std::numeric_limits<ScalarType>::quiet_NaN();
	}

	const ScalarType reference_norm = reference_values.norm();
	return (values - reference_values).norm() / std::max(reference_norm, static_cast<ScalarType>(EPSILON));
}

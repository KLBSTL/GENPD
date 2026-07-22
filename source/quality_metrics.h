#ifndef _GENPD_QUALITY_METRICS_H_
#define _GENPD_QUALITY_METRICS_H_

#include <string>

#include "math_headers.h"

std::string GenPDReferenceCheckpointPath(const std::string& directory, unsigned int frame);

bool GenPDWriteReferenceCheckpoint(
	const std::string& directory,
	unsigned int frame,
	const VectorX& positions,
	const VectorX& velocities);

bool GenPDReadReferenceCheckpoint(
	const std::string& directory,
	unsigned int frame,
	VectorX& positions,
	VectorX& velocities);

bool GenPDVectorIsFinite(const VectorX& values);
ScalarType GenPDRelativeL2(const VectorX& values, const VectorX& reference_values);

#endif

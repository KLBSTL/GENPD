#ifndef _GENPD_EXPERIMENT_VARIANT_H_
#define _GENPD_EXPERIMENT_VARIANT_H_

#include <string>

enum GenPDExperimentVariant
{
	GENPD_VARIANT_CPU_NCG,
	GENPD_VARIANT_GPU_EDGE_SCATTER,
	GENPD_VARIANT_GPU_GATHER_NO_FUSION,
	GENPD_VARIANT_GPU_GATHER_FUSION,
	GENPD_VARIANT_GPU_GATHER_FUSION_SERIAL_LS_PERSISTENT,
	GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS,
	GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT,
	GENPD_VARIANT_GPU_GATHER_FUSION_ADAPTIVE_LS_PERSISTENT,
	GENPD_VARIANT_GPU_XPBD_JACOBI,
	GENPD_VARIANT_GPU_XPBD_VERTEX_GATHER
};

bool GenPDParseExperimentVariant(const std::string& value, GenPDExperimentVariant& variant);
void GenPDSetExperimentVariant(GenPDExperimentVariant variant);
GenPDExperimentVariant GenPDGetExperimentVariant();
const char* GenPDExperimentVariantName();

bool GenPDExperimentUsesCSNCG();
bool GenPDExperimentUsesEdgeScatter();
bool GenPDExperimentUsesFusedGradientStats();
bool GenPDExperimentUsesBatchedLineSearch();
bool GenPDExperimentUsesAdaptiveLineSearch();
bool GenPDExperimentUsesPersistentBuffers();
bool GenPDExperimentUsesGPUXPBD();
bool GenPDExperimentUsesGPUXPBDVertexGather();

#endif

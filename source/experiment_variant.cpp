#include "experiment_variant.h"

namespace
{
	GenPDExperimentVariant g_experiment_variant = GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT;
}

bool GenPDParseExperimentVariant(const std::string& value, GenPDExperimentVariant& variant)
{
	if (value == "cpu-ncg")
	{
		variant = GENPD_VARIANT_CPU_NCG;
	}
	else if (value == "gpu-edge-scatter")
	{
		variant = GENPD_VARIANT_GPU_EDGE_SCATTER;
	}
	else if (value == "gpu-gather-no-fusion")
	{
		variant = GENPD_VARIANT_GPU_GATHER_NO_FUSION;
	}
	else if (value == "gpu-gather-fusion")
	{
		variant = GENPD_VARIANT_GPU_GATHER_FUSION;
	}
	else if (value == "gpu-gather-fusion-batched-ls")
	{
		variant = GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS;
	}
	else if (value == "gpu-gather-fusion-batched-ls-persistent")
	{
		variant = GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT;
	}
	else
	{
		return false;
	}

	return true;
}

void GenPDSetExperimentVariant(GenPDExperimentVariant variant)
{
	g_experiment_variant = variant;
}

GenPDExperimentVariant GenPDGetExperimentVariant()
{
	return g_experiment_variant;
}

const char* GenPDExperimentVariantName()
{
	switch (g_experiment_variant)
	{
	case GENPD_VARIANT_CPU_NCG:
		return "cpu-ncg";
	case GENPD_VARIANT_GPU_EDGE_SCATTER:
		return "gpu-edge-scatter";
	case GENPD_VARIANT_GPU_GATHER_NO_FUSION:
		return "gpu-gather-no-fusion";
	case GENPD_VARIANT_GPU_GATHER_FUSION:
		return "gpu-gather-fusion";
	case GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS:
		return "gpu-gather-fusion-batched-ls";
	case GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT:
	default:
		return "gpu-gather-fusion-batched-ls-persistent";
	}
}

bool GenPDExperimentUsesCSNCG()
{
	return g_experiment_variant != GENPD_VARIANT_CPU_NCG;
}

bool GenPDExperimentUsesEdgeScatter()
{
	return g_experiment_variant == GENPD_VARIANT_GPU_EDGE_SCATTER;
}

bool GenPDExperimentUsesFusedGradientStats()
{
	return g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION
		|| g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS
		|| g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT;
}

bool GenPDExperimentUsesBatchedLineSearch()
{
	return g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS
		|| g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT;
}

bool GenPDExperimentUsesPersistentBuffers()
{
	return g_experiment_variant == GENPD_VARIANT_GPU_GATHER_FUSION_BATCHED_LS_PERSISTENT;
}

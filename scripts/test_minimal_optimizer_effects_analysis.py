#!/usr/bin/env python3
"""Unit contract for iteration-scaled optimizer-effects structural counters."""

import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).with_name("analyze_minimal_optimizer_effects.py")
SPEC = importlib.util.spec_from_file_location("minimal_optimizer_effects", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DispatchCounterTests(unittest.TestCase):
    def test_traversal_dispatches_scale_with_iterations(self):
        self.assertEqual(MODULE.expected_dispatch_counts("edge-scatter-k8", 1), (2.0, 1.0))
        self.assertEqual(MODULE.expected_dispatch_counts("edge-scatter-k8", 8), (16.0, 8.0))
        self.assertEqual(MODULE.expected_dispatch_counts("vertex-gather-k8", 8), (8.0, 8.0))
        self.assertEqual(MODULE.expected_dispatch_counts("gather-fusion-k8", 8), (8.0, 0.0))


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_marks_stress_drift_unqualified_without_rejecting_raw_evidence(self):
        self.assertTrue(MODULE.quality_gate_passes(1.0e-3, 1.0e-3))
        self.assertFalse(MODULE.quality_gate_passes(2.163e-3, 1.0e-3))


class DecisionTraceTests(unittest.TestCase):
    def test_requires_nonempty_trace_only_for_adaptive_solver(self):
        fixed_condition = {
            "decision_trace": True,
            "solver_variant": "gpu-gather-fusion-batched-ls-persistent",
        }
        adaptive_condition = {
            "decision_trace": True,
            "solver_variant": "gpu-gather-fusion-adaptive-ls-persistent",
        }
        self.assertFalse(MODULE.requires_nonempty_adaptive_trace(fixed_condition))
        self.assertTrue(MODULE.requires_nonempty_adaptive_trace(adaptive_condition))


if __name__ == "__main__":
    unittest.main()

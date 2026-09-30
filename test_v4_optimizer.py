"""Focused optimizer regressions; run with .venv/bin/python test_v4_optimizer.py.

Load function definitions without running the standalone script's simulation.
"""
import ast
from pathlib import Path
import time
import unittest

import numpy as np
from scipy.optimize import OptimizeResult, least_squares


SCRIPT = Path(__file__).with_name("multiparameter_trotter_v4_least_squares_over_time.py")
FIT_FUNCTIONS = {
    "w_estimate_linearized", "is_finite_fit", "w_estimate_staged",
    "w_estimate_start", "w_estimate_over_time",
}


def load_fit_functions():
    tree = ast.parse(SCRIPT.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in FIT_FUNCTIONS]
    namespace = {"np": np, "t": time, "least_squares": least_squares,
                 "t_stage_start": 250, "w_bound": 1,
                 "n_meas_cum": np.full(1001, 3)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def candidate(cost, x=None, calls=7, success=True):
    return OptimizeResult(cost=cost, x=np.zeros(3) if x is None else np.array(x),
                          residual_calls=calls, success=success,
                          status=1 if success else 0, optimality=0.1)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_fit_functions()
        self.route_starts = []

    def install_routes(self, staged, direct):
        staged, direct = iter(staged), iter(direct)

        def route(results, label):
            def fit(noise, expectation, w0, cutoff, controls, daggers):
                self.route_starts.append((label, cutoff, np.array(w0)))
                return next(results)
            return fit

        self.ns["w_estimate_staged"] = route(staged, "staged")
        self.ns["w_estimate_linearized"] = route(direct, "direct")

    def fit(self, starts, cutoffs=(1000,)):
        return self.ns["w_estimate_over_time"](
            np.zeros(3), np.zeros(3), np.asarray(starts), cutoffs, None, None)

    def test_in_band_anchor_does_not_suppress_better_start(self):
        self.install_routes([candidate(4), candidate(2), candidate(1, success=False)],
                            [candidate(3), candidate(5), candidate(6)])

        def forbidden_gate(*args):
            self.fail("The cost band must not control candidate evaluation")

        self.ns["cost_accept_threshold"] = forbidden_gate
        result, = self.fit([[[0.5, 0, 0], [-0.5, 0, 0]]])
        self.assertEqual(len(self.route_starts), 6)
        self.assertEqual(result.cost, 1)
        self.assertEqual(result.start_index, 2)
        self.assertFalse(result.success) #A finite lower cost remains eligible.
        self.assertFalse(result.direct)
        self.assertEqual(result.residual_calls, 42)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_exact_ties_keep_staged_and_earliest_start(self):
        self.install_routes([candidate(1), candidate(1)], [candidate(1), candidate(1)])
        result, = self.fit([[[0.5, 0, 0]]])
        self.assertFalse(result.direct)
        self.assertEqual(result.start_index, 0)
        self.assertEqual(result.residual_calls, 28)

    def test_nonfinite_candidates_do_not_poison_later_starts(self):
        self.install_routes([candidate(np.nan), candidate(2)],
                            [candidate(0, x=[np.nan, 0, 0]), candidate(1)])
        result, = self.fit([[[0.5, 0, 0]]])
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.direct)
        self.assertEqual(result.start_index, 1)
        self.assertEqual(result.residual_calls, 28)

    def test_direct_recovers_from_invalid_staged_candidate(self):
        self.install_routes([candidate(np.inf)], [candidate(2)])
        result, = self.fit(np.empty((1, 0, 3)))
        self.assertTrue(result.direct)
        self.assertEqual(result.cost, 2)

    def test_all_invalid_candidates_raise_clear_error(self):
        self.install_routes([candidate(np.nan)], [candidate(np.inf)])
        with self.assertRaisesRegex(RuntimeError, "cutoff 1000 after 1 starts"):
            self.fit(np.empty((1, 0, 3)))

    def test_zero_extra_starts_and_single_stage_cutoffs(self):
        self.install_routes([candidate(2), candidate(1)], [])
        results = self.fit(np.empty((2, 0, 3)), cutoffs=(10, 250))
        self.assertEqual(len(self.route_starts), 2) #No duplicate direct solves.
        for label, _, start in self.route_starts:
            self.assertEqual(label, "staged")
            np.testing.assert_array_equal(start, np.zeros(3)) #No cross-cutoff warm start.
        for result in results:
            self.assertEqual(result.start_index, 0)
            self.assertFalse(result.direct)
            self.assertEqual(result.residual_calls, 7)

    def test_invalid_prefix_is_not_used_as_next_initial_point(self):
        calls = []

        def local_fit(noise, expectation, w0, cutoff, controls, daggers):
            calls.append(cutoff)
            return candidate(np.nan)

        self.ns["w_estimate_linearized"] = local_fit
        result = self.ns["w_estimate_staged"](None, None, np.zeros(3), 1000, None, None)
        self.assertEqual(calls, [250])
        self.assertFalse(self.ns["is_finite_fit"](result))

    def test_actual_residual_calls_include_jacobians_and_all_candidates(self):
        calls = []

        def residual(w, measurements, cutoff, controls, daggers):
            calls.append(cutoff)
            return w - measurements

        self.ns["residuals"] = residual
        target = np.array([0.1, -0.2, 0.3])
        local = self.ns["w_estimate_linearized"](np.zeros(3), target, np.zeros(3), 250, None, None)
        self.assertEqual(local.residual_calls, len(calls))
        self.assertGreater(local.residual_calls, local.nfev)
        calls.clear()
        result, = self.ns["w_estimate_over_time"](
            np.zeros(3), target, np.array([[[0.5, 0.5, 0.5]]]), [1000], None, None)
        self.assertEqual(result.residual_calls, len(calls))
        self.assertEqual(set(calls), {250, 500, 1000})
        np.testing.assert_allclose(result.x, target, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

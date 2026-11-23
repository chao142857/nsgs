"""Unit and integration tests for the unified nsgs.baselines sampling framework."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from nsgs.baselines import (
    AVAILABLE_METHODS,
    DEFAULT_KERNEL,
    SHARP_KERNEL,
    AdaptiveTargetTracker,
    TaskPool,
    apply_one_sided_optimism,
    build_sampler,
    kernel_probabilities,
)
from nsgs.baselines.benchmark import make_synthetic_domain, run_single_benchmark


class KernelAndOptimismTests(unittest.TestCase):
    def test_slide3_five_goal_draw_probabilities(self) -> None:
        rates = np.array([0.0, 0.25, 0.50, 0.66, 0.95], dtype=np.float64)
        p_default = kernel_probabilities(rates, DEFAULT_KERNEL)
        p_sharp = kernel_probabilities(rates, SHARP_KERNEL)

        np.testing.assert_allclose(p_default.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(p_sharp.sum(), 1.0, atol=1e-12)
        # Matches the five-goal table on Slide 3 (within rounding tolerance).
        np.testing.assert_allclose(p_default, [0.038, 0.251, 0.270, 0.263, 0.178], atol=2e-3)
        np.testing.assert_allclose(p_sharp, [0.00019, 0.150, 0.334, 0.380, 0.136], atol=2e-3)

    def test_uniform_floor_prevents_starvation(self) -> None:
        rates = np.array([0.0, 0.0, 0.66, 0.66], dtype=np.float64)
        p = kernel_probabilities(rates, "sharp_safe")
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
        self.assertGreater(float(p[0]), 0.05 / 4.0)
        self.assertGreater(float(p[2]), float(p[0]))

    def test_one_sided_optimism_preserves_mastered_tasks(self) -> None:
        mean = np.array([0.05, 0.50, 0.66, 0.90], dtype=np.float64)
        std = np.array([0.20, 0.30, 0.25, 0.20], dtype=np.float64)
        eff = apply_one_sided_optimism(mean, std, target=0.66, beta=1.0)
        # Pessimistic tasks lifted toward target, capped at target.
        self.assertAlmostEqual(float(eff[0]), 0.25, places=12)
        self.assertAlmostEqual(float(eff[1]), 0.66, places=12)
        # Mastered tasks (mean >= target) strictly untouched.
        self.assertAlmostEqual(float(eff[2]), 0.66, places=12)
        self.assertAlmostEqual(float(eff[3]), 0.90, places=12)

    def test_adaptive_target_tracker_moves_toward_progress_bin(self) -> None:
        tracker = AdaptiveTargetTracker(initial_target=0.66, num_bins=5, ema=0.5, min_bin_count=10)
        p = np.concatenate([np.full(40, 0.30), np.full(40, 0.70)])
        y1 = np.concatenate([np.zeros(40), np.full(40, 0.70)])
        y2 = np.concatenate([np.full(40, 0.50), np.full(40, 0.70)])  # bin [0.2, 0.4) gained +0.50
        tracker.update(p, y1)
        t_after = tracker.update(p, y2)
        self.assertLess(t_after, 0.66)
        self.assertGreaterEqual(t_after, tracker.min_target)


class AllBaselinesContractTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        raw = rng.uniform(-1.0, 1.0, size=(256, 5))
        self.pool = TaskPool.from_raw(raw, bounds=[(-1.0, 1.0)] * 5, periodic_dims=(1,))

    def _small_kwargs(self, method: str) -> dict:
        if method in ("nsgs", "nsgs_v2", "neural_plr"):
            return dict(
                num_members=3,
                hidden_dims=(32, 32),
                history_limit=512,
                optimizer_steps=2,
                batch_size=64,
                cold_updates=1,
            )
        if method == "grid_sgs":
            return dict(dims=(0, 1, 2), bins=(4, 4, 4))
        if method == "knn_sgs":
            return dict(k=8, history_limit=512, max_anchors=256, query_chunk_size=64)
        return {}

    def test_all_methods_probability_simplex_diagnostics_and_rng_isolation(self) -> None:
        for method in AVAILABLE_METHODS:
            with self.subTest(method=method):
                np.random.seed(777)
                torch.manual_seed(777)
                np_before = np.random.get_state()[1].copy()
                torch_before = torch.get_rng_state().clone()

                sampler = build_sampler(method, self.pool, seed=42, device="cpu", **self._small_kwargs(method))
                for u in range(1, 4):
                    ids = sampler.sample(32)
                    self.assertEqual(ids.shape, (32,))
                    self.assertGreaterEqual(int(ids.min()), 0)
                    self.assertLess(int(ids.max()), self.pool.num_tasks)
                    succ = (ids % 3 == 0)
                    scores = np.where(succ, 0.2, 0.8)
                    sampler.record(ids, succ, update=u, scores=scores)
                    diag = sampler.step_update(u)
                    probs = sampler.probabilities()
                    self.assertEqual(probs.shape, (self.pool.num_tasks,))
                    self.assertTrue(np.all(np.isfinite(probs)))
                    self.assertTrue(np.all(probs >= 0.0))
                    self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)
                    self.assertEqual(diag["method"], method)
                    self.assertTrue(math.isfinite(diag["entropy"]))

                # Verify global numpy and torch RNG were never mutated during init or training.
                np.testing.assert_array_equal(np.random.get_state()[1], np_before)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_checkpoint_roundtrip_bit_exact_draws(self) -> None:
        for method in AVAILABLE_METHODS:
            with self.subTest(method=method):
                kw = self._small_kwargs(method)
                s1 = build_sampler(method, self.pool, seed=99, device="cpu", **kw)
                for u in range(1, 3):
                    ids = s1.sample(24)
                    s1.record(ids, ids < 128, update=u, scores=np.linspace(0.1, 0.9, 24))
                    s1.step_update(u)
                ckpt = s1.state_dict()
                expected_probs = s1.probabilities()
                expected_draws = s1.sample(40)

                s2 = build_sampler(method, self.pool, seed=9999, device="cpu", **kw)
                s2.load_state_dict(ckpt)
                np.testing.assert_allclose(s2.probabilities(), expected_probs, atol=1e-12)
                actual_draws = s2.sample(40)
                np.testing.assert_array_equal(actual_draws, expected_draws)

    def test_high_dimensional_sparse_grid_no_overflow(self) -> None:
        rng = np.random.default_rng(7)
        raw = rng.uniform(-1.0, 1.0, size=(512, 16))
        pool = TaskPool.from_raw(raw, bounds=[(-1.0, 1.0)] * 16)
        # 8^16 = 2.8e14 cells; sparse active-cell hashing must handle without overflow.
        gs = build_sampler("grid_sgs", pool, seed=1, bins=8)
        diag = gs.diagnostics()
        self.assertLessEqual(diag["occupied_cells"], 512)
        self.assertEqual(diag["total_grid_cells"], 8**16)

    def test_plr_and_neural_plr_reject_bad_scores_atomically(self) -> None:
        for method in ("plr", "neural_plr"):
            with self.subTest(method=method):
                sampler = build_sampler(method, self.pool, seed=17, device="cpu", **self._small_kwargs(method))
                ids = np.array([0, 1, 2], dtype=np.int64)
                succ = np.array([True, False, True], dtype=np.bool_)
                for bad_scores in (
                    np.array([0.5], dtype=np.float32),
                    np.array([0.1, float("nan"), 0.3], dtype=np.float32),
                    np.array([0.1, float("inf"), 0.3], dtype=np.float32),
                    np.ones((3, 1), dtype=np.float32),
                ):
                    with self.assertRaises(ValueError):
                        sampler.record(ids, succ, update=1, scores=bad_scores)
                    self.assertEqual(sampler.total_completed, 0)
                    self.assertEqual(int(sampler.completed_counts.sum()), 0)

    def test_adaptive_target_checkpoint_mid_update_preserves_pending_feedback(self) -> None:
        kw = dict(
            num_members=3,
            hidden_dims=(32, 32),
            history_limit=512,
            optimizer_steps=2,
            batch_size=64,
            cold_updates=1,
            adaptive_target=True,
        )
        s1 = build_sampler("nsgs_v2", self.pool, seed=31, device="cpu", **kw)
        s1.target_tracker.min_bin_count = 4
        # Prime tracker with update 1 and 2.
        for u in (1, 2):
            ids = s1.sample(48)
            s1.record(ids, ids < 128, update=u)
            s1.step_update(u)
        # Record feedback for update 3, then checkpoint BEFORE step_update(3).
        ids3 = s1.sample(64)
        s1.record(ids3, np.ones(64, dtype=np.bool_), update=3)
        ckpt = s1.state_dict()

        s2 = build_sampler("nsgs_v2", self.pool, seed=999, device="cpu", **kw)
        s2.target_tracker.min_bin_count = 4
        s2.load_state_dict(ckpt)

        d1 = s1.step_update(3)
        d2 = s2.step_update(3)
        self.assertAlmostEqual(d1["active_target"], d2["active_target"], places=12)
        np.testing.assert_allclose(s1.probabilities(), s2.probabilities(), atol=1e-12)
        np.testing.assert_array_equal(s1.sample(32), s2.sample(32))

    def test_neural_init_does_not_touch_cuda_rng(self) -> None:
        orig_cuda_seed_all = torch.cuda.manual_seed_all

        def _forbid_cuda_seed_all(_seed: int) -> None:
            raise AssertionError("Sampler init must not call torch.cuda.manual_seed_all")

        torch.cuda.manual_seed_all = _forbid_cuda_seed_all  # type: ignore[assignment]
        try:
            for method in ("nsgs", "nsgs_v2", "neural_plr"):
                build_sampler(method, self.pool, seed=55, device="cpu", **self._small_kwargs(method))
        finally:
            torch.cuda.manual_seed_all = orig_cuda_seed_all  # type: ignore[assignment]


class SyntheticBenchmarkSanityTests(unittest.TestCase):
    def test_high_dim_benchmark_runs_all_methods(self) -> None:
        domain = make_synthetic_domain(num_tasks=1024, total_dims=10, active_dims=3, seed=11)
        results = {}
        for method in AVAILABLE_METHODS:
            kw = {}
            if method in ("nsgs", "nsgs_v2", "neural_plr"):
                kw = dict(
                    num_members=3,
                    hidden_dims=(64, 64),
                    optimizer_steps=10,
                    batch_size=256,
                    cold_updates=3,
                    lr=1e-3,
                )
            elif method == "grid_sgs":
                kw = dict(bins=4)
            elif method == "knn_sgs":
                kw = dict(k=16, max_anchors=1024)
            res = run_single_benchmark(
                method,
                domain,
                seed=11,
                updates=40,
                batch_per_update=128,
                initial_capability=0.08,
                sharpness=20.0,
                learning_gain=0.05,
                sampler_kwargs=kw,
            )
            self.assertGreater(res["final_coverage"], 0.1)
            self.assertTrue(math.isfinite(res["coverage_auc"]))
            results[method] = res
        # NSGS v2 must outperform uniform on the high-D synthetic frontier benchmark.
        self.assertGreater(results["nsgs_v2"]["coverage_auc"], results["uniform"]["coverage_auc"])


if __name__ == "__main__":
    unittest.main()

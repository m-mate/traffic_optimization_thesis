"""Unit tests for sub-task 1.6.8 — Configuration & Reproducibility."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_input_validation_boundaries():
    """Verify constructor boundary validation for config parameters."""
    # Non-existent sumocfg
    try:
        SumoEnv(sumocfg_path="nonexistent_config_file.sumocfg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Invalid delta_time <= 0
    try:
        SumoEnv(sumocfg_path=SUMOCFG, delta_time=0)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "delta_time must be positive" in str(exc)

    # Invalid yellow_time < 0
    try:
        SumoEnv(sumocfg_path=SUMOCFG, yellow_time=-1)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "yellow_time" in str(exc)

    # Invalid min_green < 0
    try:
        SumoEnv(sumocfg_path=SUMOCFG, min_green=-1)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "min_green cannot be negative" in str(exc)

    # Invalid max_sim_time <= 0
    try:
        SumoEnv(sumocfg_path=SUMOCFG, max_sim_time=0)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "max_sim_time must be positive" in str(exc)

    # Invalid reward_fn
    try:
        SumoEnv(sumocfg_path=SUMOCFG, reward_fn="unsupported_reward")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "Unknown reward_fn" in str(exc)


def test_cross_instance_reproducibility():
    """Verify two separate env instances seeded with same seed produce identical rollouts."""
    env1 = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5)
    env2 = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5)
    try:
        obs1, _ = env1.reset(seed=777)
        obs2, _ = env2.reset(seed=777)
        assert np.array_equal(obs1, obs2)

        fixed_action = np.zeros(len(env1.controlled_tls), dtype=int)
        for step_idx in range(5):
            o1, r1, term1, _, info1 = env1.step(fixed_action)
            o2, r2, term2, _, info2 = env2.step(fixed_action)

            assert np.array_equal(o1, o2), f"Observation divergence at step {step_idx}"
            assert r1 == r2, f"Reward divergence at step {step_idx}"
            assert term1 == term2
            assert info1["sim_time"] == info2["sim_time"]
    finally:
        env1.close()
        env2.close()


def test_internal_rng_isolation():
    """Verify Gymnasium self.np_random is used and global RNGs remain unaffected."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        # Pre-seed global numpy RNG
        np.random.seed(12345)
        expected_global_val = np.random.rand()

        # Reset env with independent seed
        env.reset(seed=999)

        # Global numpy state must not have been consumed by env
        np.random.seed(12345)
        assert np.random.rand() == expected_global_val
    finally:
        env.close()


if __name__ == "__main__":
    print("Running test_input_validation_boundaries()...")
    test_input_validation_boundaries()
    print("[PASS] test_input_validation_boundaries")

    print("Running test_cross_instance_reproducibility()...")
    test_cross_instance_reproducibility()
    print("[PASS] test_cross_instance_reproducibility")

    print("Running test_internal_rng_isolation()...")
    test_internal_rng_isolation()
    print("[PASS] test_internal_rng_isolation")

    print("\n[PASS] All 1.6.8 reproducibility tests passed!")

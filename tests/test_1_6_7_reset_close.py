"""Unit tests for sub-task 1.6.7 — reset() and close() lifecycle."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_reset_return_contract():
    """Verify reset() conforms to Gymnasium API returning (obs, info)."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert "sim_time" in info
        assert info["sim_time"] == 0.0
    finally:
        env.close()


def test_seed_reproducibility():
    """Verify identical seeds produce identical trajectories (determinism)."""
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5)
    try:
        # Run 1: seed=42
        obs1_init, _ = env.reset(seed=42)
        trajectory1 = [obs1_init]
        fixed_action = np.zeros(len(env.controlled_tls), dtype=int)
        for _ in range(5):
            obs, _, _, _, _ = env.step(fixed_action)
            trajectory1.append(obs)

        # Run 2: seed=42 again
        obs2_init, _ = env.reset(seed=42)
        trajectory2 = [obs2_init]
        for _ in range(5):
            obs, _, _, _, _ = env.step(fixed_action)
            trajectory2.append(obs)

        # Observations must be identical step-for-step
        for step_idx, (o1, o2) in enumerate(zip(trajectory1, trajectory2)):
            assert np.array_equal(o1, o2), f"Trajectory diverged at step {step_idx} with identical seed"
    finally:
        env.close()


def test_seed_variation():
    """Verify different seeds produce divergent trajectories (seed affects simulation)."""
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5)
    try:
        fixed_action = np.zeros(len(env.controlled_tls), dtype=int)

        # Seed 42
        env.reset(seed=42)
        for _ in range(15):
            obs_seed42, _, _, _, _ = env.step(fixed_action)

        # Seed 9999
        env.reset(seed=9999)
        for _ in range(15):
            obs_seed9999, _, _, _, _ = env.step(fixed_action)

        # Trajectories should differ under different seeds
        assert not np.array_equal(obs_seed42, obs_seed9999), (
            "Observations under seeds 42 and 9999 were unexpectedly identical"
        )
    finally:
        env.close()


def test_idempotent_close():
    """Verify multiple consecutive close() calls do not crash."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    env.reset(seed=1)
    env.close()
    env.close()
    env.close()
    assert env._is_traci_connected is False


def test_consecutive_resets_no_leak():
    """Verify rapid consecutive resets do not crash or leak ports/processes."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        for i in range(5):
            obs, info = env.reset(seed=i)
            assert info["sim_time"] == 0.0
            assert env._is_traci_connected is True
    finally:
        env.close()


if __name__ == "__main__":
    print("Running test_reset_return_contract()...")
    test_reset_return_contract()
    print("[PASS] test_reset_return_contract")

    print("Running test_seed_reproducibility()...")
    test_seed_reproducibility()
    print("[PASS] test_seed_reproducibility")

    print("Running test_seed_variation()...")
    test_seed_variation()
    print("[PASS] test_seed_variation")

    print("Running test_idempotent_close()...")
    test_idempotent_close()
    print("[PASS] test_idempotent_close")

    print("Running test_consecutive_resets_no_leak()...")
    test_consecutive_resets_no_leak()
    print("[PASS] test_consecutive_resets_no_leak")

    print("\n[PASS] All 1.6.7 reset() and close() tests passed!")

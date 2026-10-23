"""Unit tests for sub-task 1.6.6 — Reward Function (v1)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_reward_type_and_reset():
    """Verify reward is float and previous_total_waiting resets to 0.0 on reset()."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        _, info = env.reset(seed=42)
        assert env.previous_total_waiting == 0.0
        assert info["total_waiting_time"] == 0.0

        _, reward, _, _, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert isinstance(info["total_waiting_time"], float)
        assert env.previous_total_waiting >= 0.0

        # After another reset, must be 0.0 again
        env.reset(seed=99)
        assert env.previous_total_waiting == 0.0
    finally:
        env.close()


def test_reward_matches_delta_waiting_time():
    """Verify reward equals previous_waiting - current_waiting across multiple steps."""
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5)
    try:
        env.reset(seed=123)

        for step_idx in range(10):
            prev_wait = env.previous_total_waiting
            _, reward, _, _, info = env.step(env.action_space.sample())
            curr_wait = env.previous_total_waiting

            expected_reward = prev_wait - curr_wait
            assert abs(reward - expected_reward) < 1e-6, (
                f"Step {step_idx}: expected reward {expected_reward}, got {reward}"
            )
            assert info["total_waiting_time"] == curr_wait
    finally:
        env.close()


def test_reward_when_disconnected():
    """Verify _compute_reward returns 0.0 when simulation is not connected."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    # env bootstrap closed after init
    assert env._compute_reward() == 0.0


if __name__ == "__main__":
    print("Running test_reward_type_and_reset()...")
    test_reward_type_and_reset()
    print("[PASS] test_reward_type_and_reset")

    print("Running test_reward_matches_delta_waiting_time()...")
    test_reward_matches_delta_waiting_time()
    print("[PASS] test_reward_matches_delta_waiting_time")

    print("Running test_reward_when_disconnected()...")
    test_reward_when_disconnected()
    print("[PASS] test_reward_when_disconnected")

    print("\n[PASS] All 1.6.6 reward function tests passed!")

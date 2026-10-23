"""Unit tests for sub-task 1.6.9 — Gymnasium & SB3 Compliance Check."""

import os
import sys
import time
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_gym_and_sb3_env_checkers():
    """Verify SumoEnv passes both Gymnasium and Stable-Baselines3 env checkers without errors."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        # Gymnasium official env checker
        gym_check_env(env.unwrapped)

        # Stable-Baselines3 official env checker
        sb3_check_env(env, warn=True)
    finally:
        env.close()


def test_full_episode_random_rollout():
    """Verify a full 3600s episode (720 steps at delta_time=5) completes without errors."""
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5, max_sim_time=3600)
    try:
        t0 = time.time()
        obs, info = env.reset(seed=42)
        assert info["sim_time"] == 0.0

        total_reward = 0.0
        step_count = 0

        while True:
            action = env.action_space.sample()
            assert env.action_space.contains(action), f"Action {action} not in action_space"

            obs, reward, terminated, truncated, info = env.step(action)

            assert obs.shape == env.observation_space.shape
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert isinstance(reward, float)
            assert np.isfinite(reward)

            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        wall_time = time.time() - t0
        # Exactly 3600s / 5s = 720 steps
        assert step_count == 720, f"Expected 720 steps, got {step_count}"
        assert info["sim_time"] == 3600.0, f"Expected final sim_time 3600.0, got {info['sim_time']}"
        assert terminated is True
        assert truncated is False

        print(f"  Full 3600s episode completed in {wall_time:.2f}s wall clock ({step_count} steps, {info['throughput']} vehicles arrived)")
    finally:
        env.close()


if __name__ == "__main__":
    print("Running test_gym_and_sb3_env_checkers()...")
    test_gym_and_sb3_env_checkers()
    print("[PASS] test_gym_and_sb3_env_checkers")

    print("Running test_full_episode_random_rollout()...")
    test_full_episode_random_rollout()
    print("[PASS] test_full_episode_random_rollout")

    print("\n[PASS] All 1.6.9 compliance and rollout tests passed!")

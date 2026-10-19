"""Unit tests for sub-task 1.6.3 — Observation Space Definition."""

import os
import sys
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_observation_space_structure():
    """Verify observation_space is a valid Box space with the exact expected shape."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.dtype == np.float32

        # 40 lanes across 4 junctions -> 40 * 4 + 4 * 2 = 168
        total_lanes = sum(len(env.tls_lanes[t]) for t in env.tls_ids)
        expected_dim = total_lanes * 4 + len(env.tls_ids) * 2
        assert expected_dim == 168
        assert env.observation_space.shape == (expected_dim,)

        # Sample from space should be within space
        sample = env.observation_space.sample()
        assert sample.shape == (expected_dim,)
        assert sample.dtype == np.float32
    finally:
        env.close()


def test_obs_returned_by_reset():
    """Verify reset() returns an observation adhering to observation_space."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"
        assert np.all(obs >= 0.0), "Observation contains negative values"
        assert env.observation_space.contains(obs), "Observation not contained in observation_space"
    finally:
        env.close()


def test_observation_space_subset():
    """Verify observation_space dynamically resizes when a subset of tls_ids is specified."""
    subset = ["cluster_12861150566_248948059_5243538191"]
    env = SumoEnv(sumocfg_path=SUMOCFG, tls_ids=subset)
    try:
        # 4 lanes * 4 + 1 junction * 2 = 18
        assert env.observation_space.shape == (18,)
        obs, info = env.reset(seed=42)
        assert obs.shape == (18,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_observations_change_over_time():
    """Verify observations evolve as simulation time advances (not constant)."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        obs_init, _ = env.reset(seed=123)

        # Advance SUMO directly via traci to let traffic flow and light timers tick
        import traci
        traci.switch(env._label)
        for _ in range(25):
            traci.simulationStep()

        obs_later = env._get_obs()
        assert obs_later.shape == obs_init.shape
        assert env.observation_space.contains(obs_later)

        # The observations must have changed (e.g. elapsed time, vehicle positions)
        diff = np.abs(obs_later - obs_init)
        assert np.any(diff > 0.0), "Observation remained completely static after 25 simulation steps"
    finally:
        env.close()


def test_obs_when_disconnected():
    """Verify _get_obs returns zero array matching space shape when traci is not connected."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    # env is not connected after __init__ (bootstrap closed)
    obs = env._get_obs()
    assert obs.shape == env.observation_space.shape
    assert np.all(obs == 0.0)


if __name__ == "__main__":
    print("Running test_observation_space_structure()...")
    test_observation_space_structure()
    print("[PASS] test_observation_space_structure")

    print("Running test_obs_returned_by_reset()...")
    test_obs_returned_by_reset()
    print("[PASS] test_obs_returned_by_reset")

    print("Running test_observation_space_subset()...")
    test_observation_space_subset()
    print("[PASS] test_observation_space_subset")

    print("Running test_observations_change_over_time()...")
    test_observations_change_over_time()
    print("[PASS] test_observations_change_over_time")

    print("Running test_obs_when_disconnected()...")
    test_obs_when_disconnected()
    print("[PASS] test_obs_when_disconnected")

    print("\n[PASS] All 1.6.3 observation space tests passed!")

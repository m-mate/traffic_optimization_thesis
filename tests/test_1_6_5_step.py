"""Unit tests for sub-task 1.6.5 — step() Logic."""

import os
import sys
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_step_returns_valid_5_tuple():
    """Verify step() returns (obs, reward, terminated, truncated, info) matching Gym API."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        step_result = env.step(action)

        assert isinstance(step_result, tuple), "step() must return a tuple"
        assert len(step_result) == 5, f"step() must return 5 elements, got {len(step_result)}"

        obs, reward, terminated, truncated, info = step_result

        assert isinstance(obs, np.ndarray), f"obs must be ndarray, got {type(obs)}"
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs), "obs not in observation_space"

        assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
        assert isinstance(terminated, bool), f"terminated must be bool, got {type(terminated)}"
        assert isinstance(truncated, bool), f"truncated must be bool, got {type(truncated)}"
        assert truncated is False

        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        expected_keys = {"sim_time", "total_waiting_time", "total_halting", "throughput", "departed"}
        assert expected_keys.issubset(info.keys()), f"info missing keys: {expected_keys - set(info.keys())}"
    finally:
        env.close()


def test_simulation_time_advances_by_delta_time():
    """Verify simulation time advances by exactly delta_time each step."""
    delta_time = 5
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=delta_time)
    try:
        _, info = env.reset(seed=10)
        assert info["sim_time"] == 0.0

        for step_idx in range(1, 11):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
            expected_time = float(step_idx * delta_time)
            assert info["sim_time"] == expected_time, (
                f"Step {step_idx}: expected sim_time {expected_time}, got {info['sim_time']}"
            )
    finally:
        env.close()


def test_min_green_constraint_prevents_flickering():
    """Verify min-green time prevents switching phases too quickly."""
    # delta_time = 5, yellow_time = 3, min_green = 5
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5, yellow_time=3, min_green=5)
    try:
        env.reset(seed=100)
        tls_id = env.controlled_tls[0]  # cluster_... has green phases [0, 2]

        # Initial green phase should be green_phase_indices[tls_id][0] == 0
        assert env.current_phase[tls_id] == 0
        assert env.current_phase_time[tls_id] == 0

        # Step 1: Request green phase 1 (SUMO phase 2).
        # Since current_phase_time is 0 (< min_green=5), the switch must be blocked!
        action = [1, 0]
        env.step(action)

        # After step 1: should STILL be in phase 0, but current_phase_time should now be 5
        assert env.current_phase[tls_id] == 0, (
            f"Phase switched before min_green elapsed! Got phase {env.current_phase[tls_id]}"
        )
        assert env.current_phase_time[tls_id] == 5

        # Step 2: Request green phase 1 again.
        # Now current_phase_time is 5 (>= min_green=5), so switch must occur!
        env.step(action)
        # Target green phase is SUMO phase 2
        assert env.current_phase[tls_id] == 2, (
            f"Phase did not switch after min_green elapsed! Expected 2, got {env.current_phase[tls_id]}"
        )
        # It had yellow for 3s, remaining green for 2s
        assert env.current_phase_time[tls_id] == 2
    finally:
        env.close()


def test_yellow_transition_inserted():
    """Verify yellow phase transition is executed before reaching target green phase."""
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=5, yellow_time=3, min_green=0)
    try:
        env.reset(seed=200)
        tls_id = env.controlled_tls[0]

        # Initial phase: 0 (green)
        assert env.current_phase[tls_id] == 0

        # Request green phase 1 (SUMO phase 2). With min_green=0, switch happens immediately.
        # Check that yellow phase index is 1
        expected_yellow = env.yellow_phase_indices[tls_id][0]
        assert expected_yellow == 1

        env.step([1, 0])
        # After step completes, phase is target green phase 2
        assert env.current_phase[tls_id] == 2
    finally:
        env.close()


def test_termination_at_max_sim_time():
    """Verify episode terminates when sim_time >= max_sim_time."""
    max_sim_time = 15
    delta_time = 5
    env = SumoEnv(sumocfg_path=SUMOCFG, delta_time=delta_time, max_sim_time=max_sim_time)
    try:
        env.reset(seed=300)

        # Step 1: sim_time = 5
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert terminated is False
        assert info["sim_time"] == 5.0

        # Step 2: sim_time = 10
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert terminated is False
        assert info["sim_time"] == 10.0

        # Step 3: sim_time = 15 -> terminated!
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert terminated is True
        assert truncated is False
        assert info["sim_time"] == 15.0
    finally:
        env.close()


def test_step_before_reset_raises_runtime_error():
    """Verify calling step() before reset() raises RuntimeError."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        env.step(env.action_space.sample())
        assert False, "Should have raised RuntimeError"
    except RuntimeError as exc:
        assert "reset() before step()" in str(exc)
    finally:
        env.close()


def test_invalid_yellow_time_raises_value_error():
    """Verify yellow_time > delta_time raises ValueError."""
    try:
        SumoEnv(sumocfg_path=SUMOCFG, delta_time=3, yellow_time=5)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "cannot be greater than delta_time" in str(exc)


if __name__ == "__main__":
    print("Running test_step_returns_valid_5_tuple()...")
    test_step_returns_valid_5_tuple()
    print("[PASS] test_step_returns_valid_5_tuple")

    print("Running test_simulation_time_advances_by_delta_time()...")
    test_simulation_time_advances_by_delta_time()
    print("[PASS] test_simulation_time_advances_by_delta_time")

    print("Running test_min_green_constraint_prevents_flickering()...")
    test_min_green_constraint_prevents_flickering()
    print("[PASS] test_min_green_constraint_prevents_flickering")

    print("Running test_yellow_transition_inserted()...")
    test_yellow_transition_inserted()
    print("[PASS] test_yellow_transition_inserted")

    print("Running test_termination_at_max_sim_time()...")
    test_termination_at_max_sim_time()
    print("[PASS] test_termination_at_max_sim_time")

    print("Running test_step_before_reset_raises_runtime_error()...")
    test_step_before_reset_raises_runtime_error()
    print("[PASS] test_step_before_reset_raises_runtime_error")

    print("Running test_invalid_yellow_time_raises_value_error()...")
    test_invalid_yellow_time_raises_value_error()
    print("[PASS] test_invalid_yellow_time_raises_value_error")

    print("\n[PASS] All 1.6.5 step() logic tests passed!")

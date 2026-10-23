"""Unit tests for sub-task 1.6.4 — Action Space Definition."""

import os
import sys
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_action_space_structure():
    """Verify action_space is MultiDiscrete([2, 4]) for the default Debrecen network."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        expected_nvec = np.array([2, 4], dtype=np.int64)
        assert np.array_equal(env.action_space.nvec, expected_nvec), (
            f"Expected nvec {expected_nvec}, got {env.action_space.nvec}"
        )

        # Sampling produces valid actions
        for _ in range(20):
            sample = env.action_space.sample()
            assert env.action_space.contains(sample), f"Sample {sample} not in action_space"
            assert sample.shape == (2,)
    finally:
        env.close()


def test_action_to_sumo_phases_mapping():
    """Verify discrete actions map correctly to SUMO phase indices."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        tls_0 = "cluster_12861150566_248948059_5243538191"
        tls_1 = "joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814"

        # Action [0, 0] -> first green phase for both (SUMO phase indices 0 and 0)
        phases_00 = env.action_to_sumo_phases([0, 0])
        assert phases_00 == {tls_0: 0, tls_1: 0}

        # Action [1, 3] -> second green phase for tls_0 (phase 2), 4th for tls_1 (phase 6)
        phases_13 = env.action_to_sumo_phases([1, 3])
        assert phases_13 == {tls_0: 2, tls_1: 6}

        # Action [1, 1] -> second green phase for tls_0 (phase 2), 2nd for tls_1 (phase 2)
        phases_11 = env.action_to_sumo_phases(np.array([1, 1]))
        assert phases_11 == {tls_0: 2, tls_1: 2}
    finally:
        env.close()


def test_action_bounds_and_validation():
    """Verify action_to_sumo_phases raises ValueError on invalid actions."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        # Out of bounds for junction 0 (only has 2 phases: 0, 1)
        try:
            env.action_to_sumo_phases([2, 0])
            assert False, "Should have raised ValueError for action index out of bounds"
        except ValueError as exc:
            assert "out of bounds" in str(exc)

        # Negative action index
        try:
            env.action_to_sumo_phases([-1, 0])
            assert False, "Should have raised ValueError for negative action index"
        except ValueError as exc:
            assert "out of bounds" in str(exc)

        # Wrong length: 1 instead of 2
        try:
            env.action_to_sumo_phases([0])
            assert False, "Should have raised ValueError for wrong action length"
        except ValueError as exc:
            assert "does not match number of controlled TLS" in str(exc)

        # Wrong length: 3 instead of 2
        try:
            env.action_to_sumo_phases([0, 1, 0])
            assert False, "Should have raised ValueError for wrong action length"
        except ValueError as exc:
            assert "does not match number of controlled TLS" in str(exc)
    finally:
        env.close()


def test_action_space_subset():
    """Verify action_space dynamically adapts when a subset of TLS is specified."""
    subset = ["cluster_12861150566_248948059_5243538191"]
    env = SumoEnv(sumocfg_path=SUMOCFG, tls_ids=subset)
    try:
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert np.array_equal(env.action_space.nvec, np.array([2]))

        mapping = env.action_to_sumo_phases([1])
        assert mapping == {subset[0]: 2}
    finally:
        env.close()


def test_no_controllable_tls_raises_error():
    """Verify passing only single-phase junctions raises ValueError if min_green_phases=2."""
    try:
        SumoEnv(sumocfg_path=SUMOCFG, tls_ids=["2528360813"], min_green_phases=2)
        assert False, "Should have raised ValueError when no controllable TLS exist"
    except ValueError as exc:
        assert "No controllable traffic lights found" in str(exc)


def test_single_phase_controllable_when_allowed():
    """Verify single-phase junction is controllable when min_green_phases=1."""
    env = SumoEnv(sumocfg_path=SUMOCFG, tls_ids=["2528360813"], min_green_phases=1)
    try:
        assert np.array_equal(env.action_space.nvec, np.array([1]))
        assert env.action_to_sumo_phases([0]) == {"2528360813": 0}
    finally:
        env.close()


if __name__ == "__main__":
    print("Running test_action_space_structure()...")
    test_action_space_structure()
    print("[PASS] test_action_space_structure")

    print("Running test_action_to_sumo_phases_mapping()...")
    test_action_to_sumo_phases_mapping()
    print("[PASS] test_action_to_sumo_phases_mapping")

    print("Running test_action_bounds_and_validation()...")
    test_action_bounds_and_validation()
    print("[PASS] test_action_bounds_and_validation")

    print("Running test_action_space_subset()...")
    test_action_space_subset()
    print("[PASS] test_action_space_subset")

    print("Running test_no_controllable_tls_raises_error()...")
    test_no_controllable_tls_raises_error()
    print("[PASS] test_no_controllable_tls_raises_error")

    print("Running test_single_phase_controllable_when_allowed()...")
    test_single_phase_controllable_when_allowed()
    print("[PASS] test_single_phase_controllable_when_allowed")

    print("\n[PASS] All 1.6.4 action space tests passed!")

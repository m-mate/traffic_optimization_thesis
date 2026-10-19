"""Unit tests for sub-task 1.6.2 — Network Introspection (TLS Discovery)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_auto_discovery_default():
    """Verify auto-discovery of all TLS junctions and their metadata when tls_ids=None."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    try:
        expected_tls = [
            "2528360813",
            "cluster_12861150566_248948059_5243538191",
            "joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814",
            "joinedS_1399579262_cluster_130364737_1509164855_3184608325",
        ]
        assert env.tls_ids == sorted(expected_tls), f"Expected {expected_tls}, got {env.tls_ids}"

        # Verify green phase counts
        assert env.tls_phases["2528360813"] == 1
        assert env.tls_phases["cluster_12861150566_248948059_5243538191"] == 2
        assert env.tls_phases["joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814"] == 4
        assert env.tls_phases["joinedS_1399579262_cluster_130364737_1509164855_3184608325"] == 1

        # Verify green phase indices in SUMO program
        assert env.green_phase_indices["2528360813"] == [0]
        assert env.green_phase_indices["cluster_12861150566_248948059_5243538191"] == [0, 2]
        assert env.green_phase_indices["joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814"] == [0, 2, 4, 6]
        assert env.green_phase_indices["joinedS_1399579262_cluster_130364737_1509164855_3184608325"] == [0]

        # Verify controlled incoming lanes
        assert len(env.tls_lanes["2528360813"]) == 4
        assert len(env.tls_lanes["cluster_12861150566_248948059_5243538191"]) == 4
        assert len(env.tls_lanes["joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814"]) == 24
        assert len(env.tls_lanes["joinedS_1399579262_cluster_130364737_1509164855_3184608325"]) == 8

        # Controlled TLS filtering (only >= 2 green phases)
        expected_controlled = [
            "cluster_12861150566_248948059_5243538191",
            "joinedS_130364740_2528360815_2528360817_cluster_130364741_2528360814",
        ]
        assert env.controlled_tls == expected_controlled
    finally:
        env.close()


def test_explicit_tls_ids_subset():
    """Verify specifying a subset of TLS IDs limits control and lanes to that subset."""
    subset = ["cluster_12861150566_248948059_5243538191"]
    env = SumoEnv(sumocfg_path=SUMOCFG, tls_ids=subset)
    try:
        assert env.tls_ids == subset
        assert env.controlled_tls == subset
        assert list(env.tls_lanes.keys()) == subset
        assert len(env.tls_lanes[subset[0]]) == 4
        assert env.tls_phases[subset[0]] == 2
        assert env.green_phase_indices[subset[0]] == [0, 2]
    finally:
        env.close()


def test_invalid_tls_id_raises_value_error():
    """Verify passing a non-existent TLS ID raises ValueError with informative message."""
    invalid_ids = ["nonexistent_junction_xyz"]
    try:
        SumoEnv(sumocfg_path=SUMOCFG, tls_ids=invalid_ids)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "not found in SUMO network" in str(exc)
        assert "nonexistent_junction_xyz" in str(exc)


def test_custom_min_green_phases():
    """Verify setting min_green_phases=1 includes single-phase junctions in controlled_tls."""
    env = SumoEnv(sumocfg_path=SUMOCFG, min_green_phases=1)
    try:
        assert len(env.controlled_tls) == 4
    finally:
        env.close()


if __name__ == "__main__":
    print("Running test_auto_discovery_default()...")
    test_auto_discovery_default()
    print("[PASS] test_auto_discovery_default")

    print("Running test_explicit_tls_ids_subset()...")
    test_explicit_tls_ids_subset()
    print("[PASS] test_explicit_tls_ids_subset")

    print("Running test_invalid_tls_id_raises_value_error()...")
    test_invalid_tls_id_raises_value_error()
    print("[PASS] test_invalid_tls_id_raises_value_error")

    print("Running test_custom_min_green_phases()...")
    test_custom_min_green_phases()
    print("[PASS] test_custom_min_green_phases")

    print("\n[PASS] All 1.6.2 network introspection tests passed!")

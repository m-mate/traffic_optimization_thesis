"""Test 2.6 — Configurable reward shaping with anti-flicker switch penalty."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_reward_shaping_and_penalty():
    print("=" * 60)
    print("TEST: Reward shaping with switch penalty")
    print("=" * 60)

    # Initialize env with shaped reward, switch_penalty=5.0
    env = SumoEnv(
        sumocfg_path=SUMOCFG,
        reward_fn="diff-waiting-time-shaped",
        switch_penalty=5.0,
        throughput_coef=10.0,
        min_green=0,  # allow immediate switch for test
    )

    obs, info = env.reset(seed=42)
    assert info["switches"] == 0, f"Expected 0 switches on reset, got {info['switches']}"

    # Step 1: Same phase -> no switches
    action_same = [0, 0]
    obs, r_no_switch, terminated, truncated, info = env.step(action_same)
    assert info["switches"] == 0
    print(f"[PASS] No-switch step: reward={r_no_switch:.2f}, switches={info['switches']}")

    # Step 2: Switch action -> 1 switch
    action_switch = [1, 0]
    obs, r_with_switch, terminated, truncated, info = env.step(action_switch)
    assert info["switches"] == 1, f"Expected 1 switch, got {info['switches']}"
    print(f"[PASS] With-switch step: reward={r_with_switch:.2f}, switches={info['switches']}")

    env.close()
    print("\n[ALL PASS] test_2_6_reward_shaping passed!")


if __name__ == "__main__":
    test_reward_shaping_and_penalty()

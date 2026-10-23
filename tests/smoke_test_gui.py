"""Interactive GUI smoke test for visual inspection in SUMO-GUI (Sub-task 1.6.10).

Usage
-----
Run with GUI visualization:
    python tests/smoke_test_gui.py --steps 50 --delay 0.05
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def run_gui_smoke_test(steps: int = 50, delay: float = 0.05, seed: int = 42):
    """Run an interactive visual rollout in SUMO-GUI.

    Parameters
    ----------
    steps : int
        Number of agent decision steps to execute.
    delay : float
        Sleep delay in seconds between steps to slow down simulation for visual inspection.
    seed : int
        Simulation random seed.
    """
    print("=" * 70)
    print("SUMO-GUI Visual Smoke Test (1.6.10)")
    print(f"Config: {SUMOCFG}")
    print(f"Steps: {steps} | Step Delay: {delay}s | Seed: {seed}")
    print("=" * 70)

    env = SumoEnv(
        sumocfg_path=SUMOCFG,
        use_gui=True,
        delta_time=5,
        yellow_time=3,
        min_green=5,
    )

    try:
        obs, info = env.reset(seed=seed)
        print(f"[OK] Simulation launched in sumo-gui! Initial sim_time: {info['sim_time']}")
        print(f"Controlled TLS junctions: {env.controlled_tls}")
        print(f"Action space: {env.action_space}")
        print("-" * 70)

        for step_idx in range(1, steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Print status every 5 steps or on first step
            if step_idx <= 5 or step_idx % 5 == 0:
                phases = {tls: env.current_phase.get(tls) for tls in env.controlled_tls}
                print(
                    f"Step {step_idx:3d}/{steps} | "
                    f"SimTime: {info['sim_time']:6.1f}s | "
                    f"Action: {action} | "
                    f"Phases: {phases} | "
                    f"Halting: {info['total_halting']:2d} | "
                    f"Reward: {reward:6.1f}"
                )

            if delay > 0:
                time.sleep(delay)

            if terminated or truncated:
                print(f"[INFO] Episode finished early at sim_time {info['sim_time']}s.")
                break

        print("-" * 70)
        print(f"[PASS] GUI smoke test completed successfully ({step_idx} steps, final sim_time: {info['sim_time']}s).")
    finally:
        env.close()
        print("[OK] sumo-gui closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO-GUI visual smoke test.")
    parser.add_argument("--steps", type=int, default=30, help="Number of decision steps (default: 30)")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps in seconds (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed (default: 42)")
    args = parser.parse_args()

    run_gui_smoke_test(steps=args.steps, delay=args.delay, seed=args.seed)

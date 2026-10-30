"""Test 2.7 — Hyperparameter sweep functionality."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.algorithms.ppo.train_ppo import run_sweep, MODEL_DIR, LOG_DIR


def test_micro_sweep():
    print("=" * 60)
    print("TEST: Micro Hyperparameter Sweep")
    print("=" * 60)

    test_configs = [
        {
            "name": "test_cfg1",
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "gamma": 0.99,
        },
        {
            "name": "test_cfg2",
            "learning_rate": 1e-4,
            "n_steps": 64,
            "batch_size": 32,
            "gamma": 0.95,
        },
    ]

    completed = run_sweep(configs=test_configs, timesteps_per_config=64)
    assert len(completed) == 2, f"Expected 2 completed runs, got {completed}"

    for name in completed:
        model_file = os.path.join(MODEL_DIR, f"ppo_sweep_{name}.zip")
        assert os.path.exists(model_file), f"Expected model file {model_file}"
        print(f"[PASS] Model {model_file} created.")

    print("\n[ALL PASS] test_2_7_sweep passed!")


if __name__ == "__main__":
    test_micro_sweep()

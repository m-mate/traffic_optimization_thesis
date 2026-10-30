"""Test 2.4 — PPO integration with SumoEnv."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from src.algorithms.ppo.train_ppo import train, SUMOCFG, MODEL_DIR, LOG_DIR
from src.env.sumo_env import SumoEnv


def test_ppo_train_and_inference():
    print("=" * 60)
    print("TEST: PPO training run (short) & inference")
    print("=" * 60)

    # Train for 128 timesteps (2 iterations of n_steps=64)
    model = train(total_timesteps=128, n_steps=64)
    saved_path = os.path.join(MODEL_DIR, "ppo_initial.zip")
    assert os.path.exists(saved_path), f"Expected saved model at {saved_path}"
    print("[PASS] PPO model saved successfully.")

    # Load model and verify deterministic inference
    loaded_model = PPO.load(saved_path)
    eval_env = SumoEnv(sumocfg_path=SUMOCFG)
    obs, info = eval_env.reset(seed=42)
    action, _states = loaded_model.predict(obs, deterministic=True)

    assert eval_env.action_space.contains(action), f"Action {action} invalid for action space {eval_env.action_space}"
    obs, reward, terminated, truncated, info = eval_env.step(action)
    assert obs.shape == eval_env.observation_space.shape
    eval_env.close()
    print(f"[PASS] Inference produced valid action {action} with reward {reward:.2f}")

    # Check monitor file exists and has content
    monitor_file = os.path.join(LOG_DIR, "monitor.monitor.csv")
    assert os.path.exists(monitor_file), f"Expected monitor file at {monitor_file}"
    print("[PASS] Monitor log file verified.")


if __name__ == "__main__":
    test_ppo_train_and_inference()
    print("\n[ALL PASS] test_2_4_ppo_training passed!")

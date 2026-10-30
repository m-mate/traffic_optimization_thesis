"""Test 2.9 — Final PPO model training and deterministic evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from src.algorithms.ppo.train_ppo import train_final, SUMOCFG, MODEL_DIR
from src.env.sumo_env import SumoEnv


def test_final_model_train_and_eval():
    print("=" * 60)
    print("TEST: Train final PPO model and run evaluation episode")
    print("=" * 60)

    # Train short run with final settings
    train_final(total_timesteps=128, n_steps=64)
    model_path = os.path.join(MODEL_DIR, "ppo_final.zip")
    assert os.path.exists(model_path), f"Expected final model at {model_path}"
    print(f"[PASS] Model saved at {model_path}")

    # Evaluate deterministically for 1 episode (short sim 300s for speed)
    model = PPO.load(model_path)
    env = SumoEnv(sumocfg_path=SUMOCFG, max_sim_time=300)
    obs, info = env.reset(seed=123)

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()
    print(f"[PASS] Evaluation episode completed: steps={steps}, total_reward={total_reward:.2f}, "
          f"arrived={info.get('throughput', 0)}, waiting_time={info.get('total_waiting_time', 0.0):.1f}s")
    assert steps > 0


if __name__ == "__main__":
    test_final_model_train_and_eval()
    print("\n[ALL PASS] test_2_9_final_eval passed!")

"""Train a PPO agent on the SUMO traffic environment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "debrecen.sumocfg")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "ppo")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


def make_env():
    """Create a Monitored SumoEnv instance."""
    env = SumoEnv(sumocfg_path=SUMOCFG)
    os.makedirs(LOG_DIR, exist_ok=True)
    return Monitor(env, filename=os.path.join(LOG_DIR, "monitor"))


def train(total_timesteps: int = 100_000, n_steps: int = 512, **ppo_kwargs):
    """Train PPO and save the model."""
    env = make_env()
    os.makedirs(os.path.join(LOG_DIR, "tb"), exist_ok=True)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        **ppo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "ppo_initial"))
    env.close()
    print(f"Model saved to {os.path.join(MODEL_DIR, 'ppo_initial.zip')}")
    return model


if __name__ == "__main__":
    train()

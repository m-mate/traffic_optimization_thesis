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


def make_env(
    reward_fn: str = "diff-waiting-time",
    switch_penalty: float = 0.0,
    throughput_coef: float = 0.0,
    log_name: str = "monitor",
):
    """Create a Monitored SumoEnv instance."""
    env = SumoEnv(
        sumocfg_path=SUMOCFG,
        reward_fn=reward_fn,
        switch_penalty=switch_penalty,
        throughput_coef=throughput_coef,
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    return Monitor(env, filename=os.path.join(LOG_DIR, log_name))


def train(
    total_timesteps: int = 100_000,
    n_steps: int = 512,
    reward_fn: str = "diff-waiting-time",
    switch_penalty: float = 0.0,
    throughput_coef: float = 0.0,
    tb_log_name: str = "PPO",
    model_name: str = "ppo_initial",
    **ppo_kwargs,
):
    """Train PPO and save the model."""
    env = make_env(
        reward_fn=reward_fn,
        switch_penalty=switch_penalty,
        throughput_coef=throughput_coef,
        log_name=f"{model_name}_monitor",
    )
    os.makedirs(os.path.join(LOG_DIR, "tb"), exist_ok=True)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        **ppo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, model_name)
    model.save(save_path)
    env.close()
    print(f"Model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    train()

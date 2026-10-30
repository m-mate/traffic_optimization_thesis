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


SWEEP_CONFIGS = [
    {
        "name": "baseline_lr3e-4_n512",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "policy_kwargs": dict(net_arch=[64, 64]),
    },
    {
        "name": "deep_lr1e-4_n1024",
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "gamma": 0.99,
        "ent_coef": 0.0,
        "policy_kwargs": dict(net_arch=[128, 128]),
    },
    {
        "name": "fast_lr5e-4_n256",
        "learning_rate": 5e-4,
        "n_steps": 256,
        "batch_size": 64,
        "gamma": 0.95,
        "ent_coef": 0.01,
        "policy_kwargs": dict(net_arch=[64, 64]),
    },
]


def run_sweep(configs: list[dict] | None = None, timesteps_per_config: int = 10_000):
    """Run a hyperparameter sweep over candidate configurations."""
    configs = configs or SWEEP_CONFIGS
    results = []
    for cfg in configs:
        cfg = dict(cfg)
        name = cfg.pop("name")
        print(f"\n=== Starting Sweep Run: {name} ===")
        model = train(
            total_timesteps=timesteps_per_config,
            tb_log_name=f"sweep_{name}",
            model_name=f"ppo_sweep_{name}",
            **cfg,
        )
        results.append(name)
    print(f"\nSweep completed for: {results}")
    return results


def train_final(total_timesteps: int = 100_000, **custom_kwargs):
    """Train the final PPO model with tuned hyperparameters and save to ppo_final.zip."""
    kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "reward_fn": "diff-waiting-time-shaped",
        "switch_penalty": 2.0,
        "throughput_coef": 5.0,
        "tb_log_name": "PPO_final",
        "model_name": "ppo_final",
    }
    kwargs.update(custom_kwargs)
    return train(total_timesteps=total_timesteps, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO traffic agent or run hyperparameter sweep.")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--final", action="store_true", help="Train final PPO model with tuned hyperparameters")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps to train")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(timesteps_per_config=args.timesteps if args.timesteps != 100_000 else 10_000)
    elif args.final:
        train_final(total_timesteps=args.timesteps)
    else:
        train(total_timesteps=args.timesteps)

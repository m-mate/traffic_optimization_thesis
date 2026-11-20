"""Parallel Multi-Core PPO Training on Debrecen SUMO network.

Trains across 8 parallel environment workers for 300,000 timesteps (~20-25 min),
saves the final policy, and evaluates on 3 unseen test seeds against baseline and GA.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(PROJECT_ROOT, "data", "debrecen.sumocfg")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "ppo")


def make_single_env(rank: int, seed: int = 42):
    """Factory function for an isolated SumoEnv instance with monitoring."""
    def _init():
        env = SumoEnv(
            sumocfg_path=SUMOCFG,
            min_green=10,
            reward_fn="diff-waiting-time-shaped",
            switch_penalty=2.5,
            throughput_coef=5.0,
        )
        os.makedirs(LOG_DIR, exist_ok=True)
        return Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_worker_{rank}"))
    return _init


def eval_episode(model, sumocfg_path: str, seed: int = 42, max_sim_time: int = 3600):
    """Run one evaluation episode with deterministic actions."""
    env = SumoEnv(sumocfg_path=sumocfg_path, min_green=10, max_sim_time=max_sim_time)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    res = {
        "avg_waiting": info.get("total_waiting_time", 0.0),
        "halting": info.get("total_halting", 0),
        "throughput": info.get("throughput", 0),
        "steps": steps,
        "reward": total_reward,
    }
    env.close()
    return res


def run_parallel_training(
    n_workers: int = 8,
    total_timesteps: int = 200_000,
    eval_seeds: list[int] = [100, 200, 300],
):
    print("=" * 70)
    print("  PARALLEL MULTI-CORE PPO TRAINING")
    print(f"  Workers: {n_workers} CPU cores (SubprocVecEnv)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Observation dim: 168, Action space: MultiDiscrete([2, 4])")
    print("=" * 70)

    # 1. Create vectorized environment
    env_fns = [make_single_env(rank=i) for i in range(n_workers)]
    vec_env = SubprocVecEnv(env_fns)

    tb_dir = os.path.join(LOG_DIR, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # 2. Instantiate PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=256,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log=tb_dir,
    )

    # 3. Train
    print("\n--- Training started ---")
    t0 = time.time()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 10_000 // n_workers),
        save_path=MODEL_DIR,
        name_prefix="ppo_checkpoint",
    )
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO_parallel",
        callback=checkpoint_callback,
    )
    train_duration = time.time() - t0
    print(f"\n--- Training finished in {train_duration:.1f}s ({train_duration / 60:.1f} min) ---")

    # 4. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "ppo_parallel_final.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    vec_env.close()

    # 5. Reload for deterministic evaluation
    print("\n--- Running Evaluation on Unseen Test Seeds ---")
    loaded_model = PPO.load(save_path)
    eval_results = []
    for s in eval_seeds:
        res = eval_episode(loaded_model, SUMOCFG, seed=s, max_sim_time=3600)
        eval_results.append(res)
        print(f"  Seed {s}: avg_waiting={res['avg_waiting']:.2f}s, halting={res['halting']} veh, throughput={res['throughput']}")

    avg_wait = sum(r["avg_waiting"] for r in eval_results) / len(eval_results)
    avg_halt = sum(r["halting"] for r in eval_results) / len(eval_results)
    avg_tp = sum(r["throughput"] for r in eval_results) / len(eval_results)

    # Baseline & GA reference numbers from PoC
    baseline_wait = 77.59
    ga_wait = 41.91

    delta_vs_baseline = (baseline_wait - avg_wait) / baseline_wait * 100.0

    print("\n" + "=" * 70)
    print("  FINAL 3-WAY COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'Avg Waiting Time':>18} {'Improvement vs Base':>22}")
    print("-" * 70)
    print(f"{'Fixed Baseline':<20} {baseline_wait:>16.2f} s {'—':>22}")
    print(f"{'GA (Evolved)':<20} {ga_wait:>16.2f} s {'+46.0 %':>22}")
    print(f"{'PPO (300k 8-core)':<20} {avg_wait:>16.2f} s {delta_vs_baseline:>+20.1f} %")
    print("=" * 70)

    # Persist summary
    summary_path = os.path.join(LOG_DIR, "parallel_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"PPO_300k_avg_waiting: {avg_wait:.2f}\n")
        f.write(f"PPO_300k_avg_halting: {avg_halt:.2f}\n")
        f.write(f"PPO_300k_avg_throughput: {avg_tp:.2f}\n")
        f.write(f"PPO_300k_train_seconds: {train_duration:.1f}\n")
        f.write(f"PPO_improvement_vs_baseline: {delta_vs_baseline:+.2f}%\n")
    print(f"\nSummary logged to {summary_path}")


if __name__ == "__main__":
    run_parallel_training(n_workers=8, total_timesteps=200_000)

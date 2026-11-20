"""Proof-of-concept runner: Baseline vs PPO vs GA on the Debrecen network.

Runs short training + evaluation for all three approaches and prints a
comparison table.  Designed to complete in ~15 minutes on a 16-core machine.
"""

import os
import pickle
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sumolib
import traci
from sumolib.miscutils import getFreeSocketPort

SUMOCFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "debrecen.sumocfg"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "poc"))


# ── Shared evaluation helper ─────────────────────────────────────────

def run_baseline_episode(sumocfg_path: str, seed: int = 42, max_sim_time: int = 3600):
    """Run one episode with SUMO's default fixed-time TLS programs (no agent)."""
    label = f"baseline_{uuid.uuid4().hex[:8]}"
    port = getFreeSocketPort()
    sumo_binary = sumolib.checkBinary("sumo")

    cmd = [
        sumo_binary, "-c", os.path.abspath(sumocfg_path),
        "--seed", str(seed), "--random", "false",
        "--no-warnings", "true", "--no-step-log", "true",
    ]
    traci.start(cmd, port=port, label=label)
    try:
        traci.switch(label)

        # Discover all lanes controlled by any TLS
        all_lanes = set()
        for tls_id in traci.trafficlight.getIDList():
            all_lanes.update(traci.trafficlight.getControlledLanes(tls_id))
        all_lanes = sorted(all_lanes)

        total_waiting = 0.0
        total_halting = 0
        steps = 0
        while traci.simulation.getTime() < max_sim_time:
            traci.simulationStep()
            total_waiting += sum(traci.lane.getWaitingTime(l) for l in all_lanes)
            total_halting += sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
            steps += 1

        arrived = traci.simulation.getArrivedNumber()
        return {
            "avg_waiting_per_step": total_waiting / max(steps, 1),
            "total_waiting": total_waiting,
            "avg_halting_per_step": total_halting / max(steps, 1),
            "throughput": arrived,
            "steps": steps,
        }
    finally:
        try:
            traci.switch(label)
            traci.close()
        except Exception:
            pass


def eval_ppo_episode(model, sumocfg_path: str, seed: int = 42, max_sim_time: int = 3600):
    """Run one evaluation episode with a trained PPO model."""
    from src.env.sumo_env import SumoEnv

    env = SumoEnv(sumocfg_path=sumocfg_path, max_sim_time=max_sim_time)
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

    result = {
        "avg_waiting_per_step": info.get("total_waiting_time", 0.0),
        "total_waiting": info.get("total_waiting_time", 0.0),
        "avg_halting_per_step": info.get("total_halting", 0),
        "throughput": info.get("throughput", 0),
        "steps": steps,
        "total_reward": total_reward,
    }
    env.close()
    return result


def eval_ga_episode(individual, network_info, sumocfg_path: str, seed: int = 42, max_sim_time: int = 3600):
    """Run one evaluation episode with the best GA chromosome."""
    from src.algorithms.ga.ga_agent import evaluate_individual

    # evaluate_individual returns (-avg_waiting,) tuple
    fitness = evaluate_individual(
        individual, sumocfg_path=sumocfg_path, network_info=network_info,
        seed=seed, max_sim_time=max_sim_time,
    )

    # Also run a detailed episode for metrics
    label = f"ga_eval_detail_{uuid.uuid4().hex[:8]}"
    port = getFreeSocketPort()
    sumo_binary = sumolib.checkBinary("sumo")
    cmd = [
        sumo_binary, "-c", os.path.abspath(sumocfg_path),
        "--seed", str(seed), "--random", "false",
        "--no-warnings", "true", "--no-step-log", "true",
    ]
    traci.start(cmd, port=port, label=label)
    try:
        traci.switch(label)

        from src.algorithms.ga.chromosome import decode
        durations = decode(individual, network_info)
        for tls_id, phase_durs in durations.items():
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            logic = logics[0]
            phases = list(logic.phases)
            green_indices = network_info["green_phase_indices"][tls_id]
            for i, g_idx in enumerate(green_indices):
                dur = float(phase_durs[i])
                phases[g_idx].duration = dur
                phases[g_idx].minDur = dur
                phases[g_idx].maxDur = dur
            new_logic = traci.trafficlight.Logic(logic.programID, logic.type, 0, phases)
            traci.trafficlight.setProgramLogic(tls_id, new_logic)
            traci.trafficlight.setProgram(tls_id, logic.programID)

        all_lanes = [lane for t in network_info["tls_ids"] for lane in network_info["tls_lanes"][t]]

        total_waiting = 0.0
        total_halting = 0
        steps = 0
        while traci.simulation.getTime() < max_sim_time:
            traci.simulationStep()
            total_waiting += sum(traci.lane.getWaitingTime(l) for l in all_lanes)
            total_halting += sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
            steps += 1

        arrived = traci.simulation.getArrivedNumber()
        return {
            "avg_waiting_per_step": total_waiting / max(steps, 1),
            "total_waiting": total_waiting,
            "avg_halting_per_step": total_halting / max(steps, 1),
            "throughput": arrived,
            "steps": steps,
            "fitness": fitness[0],
        }
    finally:
        try:
            traci.switch(label)
            traci.close()
        except Exception:
            pass


# ── Main PoC runner ──────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_seeds = [100, 200, 300]
    sim_time = 3600  # full 1-hour episodes

    print("=" * 70)
    print("  PROOF OF CONCEPT: Baseline vs PPO vs GA")
    print("  Network: Debrecen 4-junction, 1-hour episodes")
    print(f"  Evaluation seeds: {eval_seeds}")
    print("=" * 70)

    # ── 1. BASELINE ──────────────────────────────────────────────────
    print("\n[1/3] BASELINE — Running fixed-time TLS episodes...")
    t0 = time.time()
    baseline_results = []
    for seed in eval_seeds:
        r = run_baseline_episode(SUMOCFG, seed=seed, max_sim_time=sim_time)
        baseline_results.append(r)
        print(f"  Seed {seed}: avg_waiting={r['avg_waiting_per_step']:.2f}, throughput={r['throughput']}")
    baseline_time = time.time() - t0
    print(f"  Baseline completed in {baseline_time:.1f}s")

    # ── 2. PPO TRAINING + EVAL ───────────────────────────────────────
    print("\n[2/3] PPO — Training (10k timesteps, shaped reward)...")
    t0 = time.time()

    from src.algorithms.ppo.train_ppo import train
    from stable_baselines3 import PPO

    ppo_model = train(
        total_timesteps=10_000,
        n_steps=512,
        reward_fn="diff-waiting-time-shaped",
        switch_penalty=2.0,
        throughput_coef=5.0,
        tb_log_name="POC_PPO",
        model_name="ppo_poc",
    )
    ppo_train_time = time.time() - t0
    print(f"  PPO training completed in {ppo_train_time:.1f}s")

    # Reload and evaluate
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_poc.zip")
    ppo_model = PPO.load(model_path)

    print("  PPO — Evaluating on test seeds...")
    t0 = time.time()
    ppo_results = []
    for seed in eval_seeds:
        r = eval_ppo_episode(ppo_model, SUMOCFG, seed=seed, max_sim_time=sim_time)
        ppo_results.append(r)
        print(f"  Seed {seed}: avg_waiting={r['avg_waiting_per_step']:.2f}, throughput={r['throughput']}")
    ppo_eval_time = time.time() - t0
    print(f"  PPO evaluation completed in {ppo_eval_time:.1f}s")

    # ── 3. GA EVOLUTION + EVAL ───────────────────────────────────────
    print("\n[3/3] GA — Evolving (pop=20, gens=10, 8 workers)...")
    t0 = time.time()

    from src.algorithms.ga.train_ga import evolve
    from src.algorithms.ga.ga_agent import get_network_info

    best_individual, logbook = evolve(
        pop_size=20,
        n_generations=10,
        n_workers=8,
        seed=42,
        max_sim_time=sim_time,
    )
    ga_train_time = time.time() - t0
    print(f"  GA evolution completed in {ga_train_time:.1f}s")

    network_info = get_network_info(SUMOCFG)

    print("  GA — Evaluating best individual on test seeds...")
    t0 = time.time()
    ga_results = []
    for seed in eval_seeds:
        r = eval_ga_episode(list(best_individual), network_info, SUMOCFG, seed=seed, max_sim_time=sim_time)
        ga_results.append(r)
        print(f"  Seed {seed}: avg_waiting={r['avg_waiting_per_step']:.2f}, throughput={r['throughput']}")
    ga_eval_time = time.time() - t0
    print(f"  GA evaluation completed in {ga_eval_time:.1f}s")

    # ── COMPARISON TABLE ─────────────────────────────────────────────
    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':>12} {'PPO':>12} {'GA':>12}")
    print("-" * 70)
    print(f"{'Avg Waiting/Step (s)':<30} {avg(baseline_results, 'avg_waiting_per_step'):>12.2f} {avg(ppo_results, 'avg_waiting_per_step'):>12.2f} {avg(ga_results, 'avg_waiting_per_step'):>12.2f}")
    print(f"{'Avg Halting/Step (veh)':<30} {avg(baseline_results, 'avg_halting_per_step'):>12.2f} {avg(ppo_results, 'avg_halting_per_step'):>12.2f} {avg(ga_results, 'avg_halting_per_step'):>12.2f}")
    print(f"{'Throughput (veh arrived)':<30} {avg(baseline_results, 'throughput'):>12.1f} {avg(ppo_results, 'throughput'):>12.1f} {avg(ga_results, 'throughput'):>12.1f}")
    print(f"{'Training Time (s)':<30} {'N/A':>12} {ppo_train_time:>12.1f} {ga_train_time:>12.1f}")
    print("-" * 70)

    # Improvement over baseline
    bw = avg(baseline_results, 'avg_waiting_per_step')
    pw = avg(ppo_results, 'avg_waiting_per_step')
    gw = avg(ga_results, 'avg_waiting_per_step')
    if bw > 0:
        print(f"{'Waiting Δ vs Baseline':<30} {'—':>12} {((bw - pw) / bw * 100):>+11.1f}% {((bw - gw) / bw * 100):>+11.1f}%")
    print("=" * 70)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "poc_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Baseline avg_waiting: {avg(baseline_results, 'avg_waiting_per_step'):.4f}\n")
        f.write(f"PPO avg_waiting:      {avg(ppo_results, 'avg_waiting_per_step'):.4f}\n")
        f.write(f"GA avg_waiting:       {avg(ga_results, 'avg_waiting_per_step'):.4f}\n")
        f.write(f"Baseline throughput:  {avg(baseline_results, 'throughput'):.1f}\n")
        f.write(f"PPO throughput:       {avg(ppo_results, 'throughput'):.1f}\n")
        f.write(f"GA throughput:        {avg(ga_results, 'throughput'):.1f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

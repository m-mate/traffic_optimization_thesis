"""Test 3.6 — Full GA evolutionary loop, elitism, persistence, and convergence logging."""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.algorithms.ga.chromosome import GENE_MAX, GENE_MIN
from src.algorithms.ga.train_ga import LOG_DIR, MODEL_DIR, evolve


def test_micro_evolution():
    print("=" * 60)
    print("TEST: Micro GA evolution (pop=4, gens=2)")
    print("=" * 60)

    best, logbook = evolve(
        pop_size=4,
        n_generations=2,
        cxpb=0.7,
        mutpb=0.3,
        elite_frac=0.25,
        n_workers=1,
        seed=123,
        max_sim_time=30,
    )

    # 1. Best individual verification
    assert len(best) == 6, f"Expected 6 genes, got {len(best)}"
    for g in best:
        assert GENE_MIN <= g <= GENE_MAX, f"Gene {g} outside bounds [{GENE_MIN}, {GENE_MAX}]"
    assert best.fitness.valid
    assert best.fitness.values[0] <= 0.0
    print(f"[PASS] Best individual: genes={[round(g, 2) for g in best]}, fitness={best.fitness.values[0]:.2f}")

    # 2. Check saved pkl
    model_path = os.path.join(MODEL_DIR, "ga_best_individual.pkl")
    assert os.path.exists(model_path), f"File not found: {model_path}"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    assert "individual" in data
    assert "fitness" in data
    assert "network_info" in data
    assert "logbook" in data
    print(f"[PASS] Model artifact verified at {model_path}")

    # 3. Check convergence log
    log_path = os.path.join(LOG_DIR, "convergence.csv")
    assert os.path.exists(log_path), f"File not found: {log_path}"
    with open(log_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 4, f"Expected header + 3 generations (0, 1, 2), got {len(lines)} lines"
    assert lines[0].strip() == "generation,best,avg"
    print(f"[PASS] Convergence log verified at {log_path}")

    print("\n[ALL PASS] test_3_6_ga_evolution passed!")


if __name__ == "__main__":
    test_micro_evolution()

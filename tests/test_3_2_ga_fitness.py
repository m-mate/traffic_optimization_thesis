"""Test 3.2 — Chromosome encoding and GA fitness evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.algorithms.ga.chromosome import (
    GENE_MIN,
    GENE_MAX,
    create_random_gene,
    clamp_individual,
    chromosome_length,
    decode,
)
from src.algorithms.ga.ga_agent import get_network_info, evaluate_individual

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")


def test_chromosome_functions():
    print("=" * 60)
    print("TEST: Chromosome encoding, clamping, and decoding")
    print("=" * 60)

    # 1. Random gene is within [5.0, 60.0]
    for _ in range(50):
        g = create_random_gene()
        assert GENE_MIN <= g <= GENE_MAX

    # 2. Clamping
    ind = [1.0, 100.0, -10.0, 25.0]
    clamped = clamp_individual(ind)
    assert clamped[0] == GENE_MIN
    assert clamped[1] == GENE_MAX
    assert clamped[2] == GENE_MIN
    assert clamped[3] == 25.0

    # 3. Network info and chromosome length
    info = get_network_info(SUMOCFG)
    assert len(info["controlled_tls"]) == 2
    n_genes = chromosome_length(info)
    assert n_genes == 6, f"Expected 6 genes (2 for TLS 1, 4 for TLS 2), got {n_genes}"

    # 4. Decoding
    sample_genes = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    decoded = decode(sample_genes, info)
    tls1, tls2 = info["controlled_tls"]
    assert len(decoded[tls1]) == 2
    assert decoded[tls1] == [10.0, 20.0]
    assert len(decoded[tls2]) == 4
    assert decoded[tls2] == [30.0, 40.0, 50.0, 60.0]
    print("[PASS] Chromosome functions verified.")
    return info


def test_fitness_evaluation(info):
    print("=" * 60)
    print("TEST: GA fitness evaluation via raw TraCI")
    print("=" * 60)

    sample_individual = [15.0, 30.0, 20.0, 25.0, 35.0, 40.0]
    # Run a short 60s episode
    fitness = evaluate_individual(
        sample_individual,
        sumocfg_path=SUMOCFG,
        network_info=info,
        seed=42,
        max_sim_time=60,
    )
    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert isinstance(fitness[0], float)
    assert fitness[0] <= 0.0, f"Fitness should be <= 0 (negative avg waiting time), got {fitness[0]}"
    print(f"[PASS] Fitness evaluation completed with fitness={fitness[0]:.2f}")


if __name__ == "__main__":
    net_info = test_chromosome_functions()
    test_fitness_evaluation(net_info)
    print("\n[ALL PASS] test_3_2_ga_fitness passed!")

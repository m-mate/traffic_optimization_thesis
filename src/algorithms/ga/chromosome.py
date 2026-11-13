"""Chromosome encoding for GA-based traffic signal optimization."""

import random

# Gene bounds: phase durations in seconds
GENE_MIN = 5.0
GENE_MAX = 60.0


def create_random_gene() -> float:
    """Return a single random phase duration in [GENE_MIN, GENE_MAX]."""
    return random.uniform(GENE_MIN, GENE_MAX)


def clamp_individual(individual):
    """Clamp all genes to valid bounds (in-place). Used after mutation."""
    for i in range(len(individual)):
        individual[i] = max(GENE_MIN, min(GENE_MAX, float(individual[i])))
    return individual


def chromosome_length(network_info: dict) -> int:
    """Return total gene count from network introspection info."""
    return sum(
        len(network_info["green_phase_indices"][t])
        for t in network_info["controlled_tls"]
    )


def decode(individual, network_info: dict) -> dict[str, list[float]]:
    """Decode a flat chromosome into per-TLS phase duration dicts.

    Returns {tls_id: [dur_phase0, dur_phase1, ...]} for controlled TLS.
    """
    result = {}
    offset = 0
    for tls_id in network_info["controlled_tls"]:
        n = len(network_info["green_phase_indices"][tls_id])
        result[tls_id] = [float(val) for val in individual[offset:offset + n]]
        offset += n
    return result

"""Train a GA agent to optimize traffic signal timing via DEAP."""

import argparse
import multiprocessing
import os
import pickle
import random
import sys
from functools import partial

# Add project root to sys.path so subprocesses can resolve 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deap import base, creator, tools

from src.algorithms.ga.chromosome import (
    GENE_MAX,
    GENE_MIN,
    clamp_individual,
    chromosome_length,
    create_random_gene,
)
from src.algorithms.ga.ga_agent import evaluate_individual, get_network_info

SUMOCFG = os.path.join(PROJECT_ROOT, "data", "debrecen.sumocfg")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "ga")


def setup_deap(n_genes: int):
    """Register DEAP types and operators."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("gene", create_random_gene)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=n_genes,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        eta=20.0,
        low=GENE_MIN,
        up=GENE_MAX,
    )
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def evolve(
    pop_size: int = 50,
    n_generations: int = 100,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    elite_frac: float = 0.05,
    n_workers: int | None = None,
    seed: int = 42,
    max_sim_time: int = 3600,
):
    """Run the evolutionary loop and save the best individual."""
    random.seed(seed)
    network_info = get_network_info(SUMOCFG)
    n_genes = chromosome_length(network_info)
    toolbox = setup_deap(n_genes)

    # Fitness evaluation wrapper
    fit_fn = partial(
        evaluate_individual,
        sumocfg_path=SUMOCFG,
        network_info=network_info,
        seed=seed,
        max_sim_time=max_sim_time,
    )
    toolbox.register("evaluate", fit_fn)

    # Setup parallel pool if requested
    pool = None
    if n_workers is not None and n_workers > 1:
        pool = multiprocessing.Pool(processes=n_workers)
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    try:
        # Statistics and Hall of Fame
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("best", max)
        stats.register("avg", lambda x: sum(x) / len(x))
        hof = tools.HallOfFame(1)

        pop = toolbox.population(n=pop_size)
        logbook = tools.Logbook()
        n_elite = max(1, int(pop_size * elite_frac))

        # Evaluate initial population
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, **record)
        print(f"Gen 0: best={record['best']:.2f}, avg={record['avg']:.2f}")

        for gen in range(1, n_generations + 1):
            elite = [toolbox.clone(ind) for ind in tools.selBest(pop, n_elite)]
            offspring = toolbox.select(pop, pop_size - n_elite)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            # Mutation
            for mut in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mut)
                    clamp_individual(mut)
                    del mut.fitness.values

            # Evaluate offspring whose fitness was invalidated
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            if invalids:
                invalids_fitness = toolbox.map(toolbox.evaluate, invalids)
                for ind, fit in zip(invalids, invalids_fitness):
                    ind.fitness.values = fit

            pop[:] = elite + offspring
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            print(f"Gen {gen}: best={record['best']:.2f}, avg={record['avg']:.2f}")

        # Save best individual
        os.makedirs(MODEL_DIR, exist_ok=True)
        best = hof[0]
        save_data = {
            "individual": list(best),
            "fitness": best.fitness.values,
            "network_info": network_info,
            "logbook": logbook,
        }
        save_path = os.path.join(MODEL_DIR, "ga_best_individual.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Best individual saved to {save_path}")
        print(f"Best fitness: {best.fitness.values[0]:.2f}")

        # Save convergence log
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "convergence.csv")
        with open(log_path, "w") as f:
            f.write("generation,best,avg\n")
            for record in logbook:
                f.write(f"{record['gen']},{record['best']:.4f},{record['avg']:.4f}\n")
        print(f"Convergence log saved to {log_path}")

        return best, logbook

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve traffic signal timing with GA.")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--max-sim-time", type=int, default=3600, help="Simulation time per episode")
    args = parser.parse_args()

    evolve(
        pop_size=args.pop,
        n_generations=args.gens,
        n_workers=args.workers,
        max_sim_time=args.max_sim_time,
    )

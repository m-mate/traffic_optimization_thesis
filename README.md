# Traffic Optimization Thesis

This project contains the codebase for optimizing traffic using Reinforcement Learning (PPO) and Genetic Algorithms (GA) with the SUMO traffic simulator.

## Project Structure

- `data/`: Contains the map and configuration files for the SUMO simulation (e.g., `.osm`, `.net.xml`, `.rou.xml`, `.sumocfg`).
- `models/`: Directory to store trained models and checkpoints.
- `results/`: Directory to store evaluation results, metrics, and generated plots.
- `src/`: Source code directory.
  - `src/algorithms/`: Implementations of optimization algorithms.
    - `src/algorithms/ga/`: Genetic Algorithm implementation.
    - `src/algorithms/ppo/`: Proximal Policy Optimization implementation.
  - `src/env/`: SUMO environment wrappers for integration with algorithms.
  - `src/evaluation/`: Scripts for evaluating algorithms and computing metrics.
  - `src/utils/`: Utility scripts for tasks like converting OSM to SUMO format and visualizing data.
- `tests/`: Unit tests and integration tests for the project.

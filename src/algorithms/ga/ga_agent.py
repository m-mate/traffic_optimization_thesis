"""GA fitness evaluation against a SUMO simulation."""

import os
import uuid

import sumolib
import traci
from sumolib.miscutils import getFreeSocketPort

from src.algorithms.ga.chromosome import decode
from src.env.sumo_env import _ensure_sumo_home

_ensure_sumo_home()


def get_network_info(sumocfg_path: str) -> dict:
    """Introspect the SUMO network via a temporary SumoEnv instance."""
    from src.env.sumo_env import SumoEnv

    env = SumoEnv(sumocfg_path=sumocfg_path)
    info = {
        "controlled_tls": list(env.controlled_tls),
        "green_phase_indices": {
            t: list(env.green_phase_indices[t]) for t in env.controlled_tls
        },
        "tls_ids": list(env.tls_ids),
        "tls_lanes": {t: list(env.tls_lanes[t]) for t in env.tls_ids},
    }
    env.close()
    return info


def evaluate_individual(
    individual,
    sumocfg_path: str,
    network_info: dict,
    seed: int = 42,
    max_sim_time: int = 3600,
) -> tuple[float]:
    """Run one SUMO episode with fixed phase durations and return fitness.

    Returns (-average_waiting_time_per_step,) as a DEAP fitness tuple.
    """
    label = f"ga_eval_{uuid.uuid4().hex[:8]}"
    port = getFreeSocketPort()
    sumo_binary = sumolib.checkBinary("sumo")

    cmd = [
        sumo_binary,
        "-c",
        os.path.abspath(sumocfg_path),
        "--seed",
        str(seed),
        "--random",
        "false",
        "--no-warnings",
        "true",
        "--no-step-log",
        "true",
    ]
    traci.start(cmd, port=port, label=label)
    try:
        traci.switch(label)

        # Decode chromosome and apply phase durations to TLS programs
        durations = decode(individual, network_info)
        for tls_id, phase_durs in durations.items():
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                continue
            logic = logics[0]
            phases = list(logic.phases)
            green_indices = network_info["green_phase_indices"][tls_id]
            for i, g_idx in enumerate(green_indices):
                dur = float(phase_durs[i])
                phases[g_idx].duration = dur
                phases[g_idx].minDur = dur
                phases[g_idx].maxDur = dur
            new_logic = traci.trafficlight.Logic(
                logic.programID, logic.type, 0, phases
            )
            traci.trafficlight.setProgramLogic(tls_id, new_logic)
            traci.trafficlight.setProgram(tls_id, logic.programID)

        # Collect all lanes for waiting time measurement
        all_lanes = [
            lane
            for tls_id in network_info["tls_ids"]
            for lane in network_info["tls_lanes"][tls_id]
        ]

        total_waiting = 0.0
        steps = 0
        while traci.simulation.getTime() < max_sim_time:
            traci.simulationStep()
            total_waiting += sum(
                float(traci.lane.getWaitingTime(lane)) for lane in all_lanes
            )
            steps += 1

        avg_waiting = total_waiting / max(steps, 1)
        return (-float(avg_waiting),)
    finally:
        try:
            traci.switch(label)
            traci.close()
        except Exception:
            pass

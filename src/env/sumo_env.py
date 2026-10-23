"""
Gymnasium wrapper for SUMO traffic light control via TraCI.

This environment exposes traffic signal phase selection as a standard
Gymnasium reset() / step(action) / close() loop.  It is used by both
the PPO agent (Phase 2) and the GA fitness evaluator (Phase 3).
"""

from __future__ import annotations

import os
import uuid

import gymnasium
import numpy as np
import traci


def _ensure_sumo_home() -> str:
    """Return *SUMO_HOME* — auto-detect and set it if missing."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and os.path.isdir(sumo_home):
        return sumo_home

    # Common Windows install locations
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "SUMO"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Eclipse", "Sumo"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Eclipse", "Sumo"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "bin", "sumo.exe")):
            os.environ["SUMO_HOME"] = path
            return path

    raise EnvironmentError(
        "SUMO_HOME is not set and SUMO could not be found in the "
        "standard install locations.  Set the SUMO_HOME environment "
        "variable to your SUMO installation directory."
    )


# Auto-detect on import so sumolib/traci can find SUMO tools
_ensure_sumo_home()

import sumolib  # noqa: E402 — must come after SUMO_HOME is set
from sumolib.miscutils import getFreeSocketPort  # noqa: E402


class SumoEnv(gymnasium.Env):
    """SUMO Traffic-Light Control Environment.

    Parameters
    ----------
    sumocfg_path : str
        Path to the ``.sumocfg`` file (absolute or relative to cwd).
    tls_ids : list[str] | None
        Explicit list of traffic-light IDs to control.  If *None* the
        environment auto-discovers all TLS junctions from the network
        after starting TraCI.
    delta_time : int
        Seconds of simulation time between agent decisions.
    yellow_time : int
        Duration (s) of the yellow transition inserted when switching
        green phases.
    min_green : int
        Minimum green-phase duration (s) before a phase switch is
        allowed.  Actions requesting an earlier switch are ignored.
    max_sim_time : int
        Episode length in simulation-seconds.  The episode terminates
        (``terminated = True``) once this time is reached.
    use_gui : bool
        If *True*, launch ``sumo-gui`` instead of headless ``sumo``.
    reward_fn : str
        Name of the reward function variant.  Currently only
        ``"diff-waiting-time"`` is implemented.
    min_green_phases : int
        Minimum number of green phases required for a junction to be
        considered controllable (default 2). Junctions with fewer green
        phases have no phase-switching decisions and are excluded from
        ``controlled_tls``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sumocfg_path: str,
        tls_ids: list[str] | None = None,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 5,
        max_sim_time: int = 3600,
        use_gui: bool = False,
        reward_fn: str = "diff-waiting-time",
        min_green_phases: int = 2,
    ):
        super().__init__()

        # ── Store config ──────────────────────────────────────────────
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        if not os.path.isfile(self.sumocfg_path):
            raise FileNotFoundError(f"SUMO config file not found at '{self.sumocfg_path}'.")

        self._requested_tls_ids = tls_ids  # may be None → auto-discover
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_sim_time = max_sim_time
        self.use_gui = use_gui
        self.reward_fn = reward_fn
        self.min_green_phases = min_green_phases

        if self.delta_time <= 0:
            raise ValueError(f"delta_time must be positive, got {self.delta_time}.")
        if self.yellow_time < 0 or self.yellow_time > self.delta_time:
            raise ValueError(
                f"yellow_time ({self.yellow_time}) cannot be greater than delta_time ({self.delta_time}) or negative."
            )
        if self.min_green < 0:
            raise ValueError(f"min_green cannot be negative, got {self.min_green}.")
        if self.max_sim_time <= 0:
            raise ValueError(f"max_sim_time must be positive, got {self.max_sim_time}.")
        if self.reward_fn != "diff-waiting-time":
            raise ValueError(
                f"Unknown reward_fn '{self.reward_fn}'. Currently only 'diff-waiting-time' is supported."
            )

        # ── TraCI connection bookkeeping ──────────────────────────────
        # Each env instance gets a unique label so that multiple SumoEnv
        # objects can coexist (traci uses the label to namespace calls).
        self._label = f"sumo_env_{uuid.uuid4().hex[:8]}"
        self._is_traci_connected = False

        # ── Bootstrap: start SUMO once to introspect the network ──────
        self.tls_ids: list[str] = []
        self.tls_phases: dict[str, int] = {}
        self.tls_lanes: dict[str, list[str]] = {}
        self.controlled_tls: list[str] = []
        self.green_phase_indices: dict[str, list[int]] = {}
        self.yellow_phase_indices: dict[str, dict[int, int]] = {}
        self.default_programs: dict[str, str] = {}

        self._start_sumo(seed=0)
        try:
            self._introspect_network()
        finally:
            self._close_sumo()

        # ── Spaces (1.6.3 / 1.6.4) ────────────────────────────────────
        total_lanes = sum(len(self.tls_lanes[t]) for t in self.tls_ids)
        obs_dim = total_lanes * 4 + len(self.tls_ids) * 2
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        if not self.controlled_tls:
            raise ValueError(
                f"No controllable traffic lights found among {self.tls_ids}. "
                f"All junctions have fewer than min_green_phases={self.min_green_phases} green phases."
            )

        self.action_space = gymnasium.spaces.MultiDiscrete(
            [self.tls_phases[tls_id] for tls_id in self.controlled_tls]
        )

        # ── Internal state reset every episode ────────────────────────
        self.previous_total_waiting: float = 0.0
        self.current_phase: dict[str, int] = {}
        self.current_phase_time: dict[str, int] = {}
        self.phase_is_yellow: dict[str, bool] = {}
        self.sim_seed: int = 0

    # ================================================================
    #  Network introspection  (1.6.2)
    # ================================================================

    def _introspect_network(self) -> None:
        """Introspect TLS programs and controlled lanes from the running SUMO simulation."""
        traci.switch(self._label)
        available_tls = set(traci.trafficlight.getIDList())

        if self._requested_tls_ids is None:
            self.tls_ids = sorted(available_tls)
        else:
            missing = [t for t in self._requested_tls_ids if t not in available_tls]
            if missing:
                raise ValueError(
                    f"Requested TLS ID(s) {missing} not found in SUMO network. "
                    f"Available TLS IDs: {sorted(available_tls)}"
                )
            self.tls_ids = list(self._requested_tls_ids)

        self.tls_phases = {}
        self.green_phase_indices = {}
        self.yellow_phase_indices = {}
        self.default_programs = {}
        self.tls_lanes = {}

        for tls_id in self.tls_ids:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                raise ValueError(f"No traffic light logic found for TLS ID '{tls_id}'.")
            logic = logics[0]
            self.default_programs[tls_id] = logic.programID

            # Pure green phases: contains green ('G' or 'g') and no yellow transition ('y' or 'Y')
            green_indices = [
                i
                for i, p in enumerate(logic.phases)
                if ("G" in p.state or "g" in p.state)
                and "y" not in p.state
                and "Y" not in p.state
            ]
            self.green_phase_indices[tls_id] = green_indices
            self.tls_phases[tls_id] = len(green_indices)
            self.tls_lanes[tls_id] = sorted(
                set(traci.trafficlight.getControlledLanes(tls_id))
            )

            # Map each green phase to its transition yellow phase if defined in program
            yellow_map = {}
            for g_idx in green_indices:
                next_idx = (g_idx + 1) % len(logic.phases)
                next_phase = logic.phases[next_idx]
                if "y" in next_phase.state or "Y" in next_phase.state:
                    yellow_map[g_idx] = next_idx
            self.yellow_phase_indices[tls_id] = yellow_map

        self.controlled_tls = [
            tls_id
            for tls_id in self.tls_ids
            if self.tls_phases[tls_id] >= self.min_green_phases
        ]

    # ================================================================
    #  TraCI lifecycle  (1.6.1)
    # ================================================================

    def _start_sumo(self, seed: int = 0) -> None:
        """Start a new SUMO simulation instance via TraCI.

        Parameters
        ----------
        seed : int
            Random seed forwarded to SUMO (``--seed``).
        """
        if self._is_traci_connected:
            self._close_sumo()

        sumo_binary = sumolib.checkBinary(
            "sumo-gui" if self.use_gui else "sumo"
        )

        cmd = [
            sumo_binary,
            "-c", self.sumocfg_path,
            "--seed", str(seed),
            "--random", "false",
            "--no-warnings", "true",
            "--no-step-log", "true",
        ]

        port = getFreeSocketPort()
        traci.start(cmd, port=port, label=self._label)
        self._is_traci_connected = True

    def _close_sumo(self) -> None:
        """Shut down the current TraCI connection (idempotent)."""
        if not getattr(self, "_is_traci_connected", False):
            return
        try:
            traci.switch(self._label)
            traci.close()
        except traci.exceptions.FatalTraCIError:
            # Connection was already lost (e.g. SUMO crashed) — nothing
            # to close.
            pass
        finally:
            self._is_traci_connected = False

    # ================================================================
    #  Gymnasium API  (stubs — filled in 1.6.5 / 1.6.7)
    # ================================================================

    def reset(self, seed=None, options=None):  # noqa: D401
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        # Determine simulation seed
        if seed is not None:
            self.sim_seed = seed
        else:
            self.sim_seed = int(self.np_random.integers(0, 2**31))

        # (Re-)start SUMO
        self._start_sumo(seed=self.sim_seed)
        traci.switch(self._label)

        # Reset bookkeeping and initialize controlled junctions in their first green phase
        self.current_phase = {}
        self.current_phase_time = {}
        self.phase_is_yellow = {}
        for tls_id in self.controlled_tls:
            initial_green = self.green_phase_indices[tls_id][0]
            self._set_green_phase(tls_id, initial_green)
            self.current_phase[tls_id] = initial_green
            self.current_phase_time[tls_id] = 0
            self.phase_is_yellow[tls_id] = False

        self.previous_total_waiting = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ================================================================
    #  Signal control helpers
    # ================================================================

    def _set_yellow_phase(self, tls_id: str, current_phase_idx: int) -> None:
        """Transition a traffic light from green to yellow."""
        yellow_idx = self.yellow_phase_indices.get(tls_id, {}).get(current_phase_idx)
        if yellow_idx is not None:
            if traci.trafficlight.getProgram(tls_id) != self.default_programs[tls_id]:
                traci.trafficlight.setProgram(tls_id, self.default_programs[tls_id])
            traci.trafficlight.setPhase(tls_id, yellow_idx)
        else:
            current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            yellow_state = current_state.replace("G", "y").replace("g", "y")
            traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)

    def _set_green_phase(self, tls_id: str, target_phase_idx: int) -> None:
        """Set the traffic light to the target green phase."""
        if traci.trafficlight.getProgram(tls_id) != self.default_programs[tls_id]:
            traci.trafficlight.setProgram(tls_id, self.default_programs[tls_id])
        traci.trafficlight.setPhase(tls_id, target_phase_idx)

    # ================================================================
    #  Action mapping  (1.6.4)
    # ================================================================

    def action_to_sumo_phases(
        self, action: np.ndarray | list[int] | tuple[int, ...]
    ) -> dict[str, int]:
        """Map discrete green-phase action indices to actual SUMO phase indices.

        Parameters
        ----------
        action : np.ndarray | list[int] | tuple[int, ...]
            Discrete action vector where ``action[i]`` selects the green phase
            index for ``controlled_tls[i]``.

        Returns
        -------
        dict[str, int]
            Mapping of ``tls_id -> sumo_phase_index``.

        Raises
        ------
        ValueError
            If action length or values are invalid.
        """
        if len(action) != len(self.controlled_tls):
            raise ValueError(
                f"Action length {len(action)} does not match number of "
                f"controlled TLS {len(self.controlled_tls)} ({self.controlled_tls})."
            )

        sumo_phases = {}
        for i, tls_id in enumerate(self.controlled_tls):
            act = int(action[i])
            num_phases = self.tls_phases[tls_id]
            if act < 0 or act >= num_phases:
                raise ValueError(
                    f"Action index {act} out of bounds for TLS '{tls_id}' "
                    f"which has {num_phases} green phases (valid: 0 to {num_phases - 1})."
                )
            sumo_phases[tls_id] = self.green_phase_indices[tls_id][act]

        return sumo_phases

    def step(self, action: np.ndarray | list[int] | tuple[int, ...]):
        """Execute one agent decision and advance the simulation by delta_time.

        Parameters
        ----------
        action : np.ndarray | list[int] | tuple[int, ...]
            Discrete action vector where ``action[i]`` selects the green phase
            for ``controlled_tls[i]``.

        Returns
        -------
        tuple
            (obs, reward, terminated, truncated, info)
        """
        if not self._is_traci_connected:
            raise RuntimeError("Simulation is not running. Call reset() before step().")

        traci.switch(self._label)

        # 1. Decode desired green phase for each controlled TLS
        desired_phases = self.action_to_sumo_phases(action)

        # 2. Check which junctions need and are allowed to switch
        to_switch = {}
        for tls_id in self.controlled_tls:
            desired = desired_phases[tls_id]
            current = self.current_phase.get(tls_id, traci.trafficlight.getPhase(tls_id))
            if desired != current:
                # Minimum green time constraint: ignore switch if elapsed time < min_green
                if self.current_phase_time.get(tls_id, 0) >= self.min_green:
                    to_switch[tls_id] = desired

        # 3. Advance simulation
        if to_switch:
            # Transition switching junctions to yellow
            for tls_id in to_switch:
                self._set_yellow_phase(tls_id, self.current_phase[tls_id])
                self.phase_is_yellow[tls_id] = True

            # Run yellow phase duration
            for _ in range(self.yellow_time):
                traci.simulationStep()

            # Non-switching junctions accrued yellow_time in their green phase
            for tls_id in self.controlled_tls:
                if tls_id not in to_switch:
                    self.current_phase_time[tls_id] += self.yellow_time

            # Switch junctions to the target green phase
            for tls_id, desired in to_switch.items():
                self._set_green_phase(tls_id, desired)
                self.current_phase[tls_id] = desired
                self.current_phase_time[tls_id] = 0
                self.phase_is_yellow[tls_id] = False

            # Run remaining green time
            remaining_time = self.delta_time - self.yellow_time
            for _ in range(remaining_time):
                traci.simulationStep()

            # Accrue remaining green time
            for tls_id in self.controlled_tls:
                self.current_phase_time[tls_id] += remaining_time
        else:
            # No junctions switched; advance full delta_time
            for _ in range(self.delta_time):
                traci.simulationStep()

            for tls_id in self.controlled_tls:
                self.current_phase_time[tls_id] += self.delta_time

        # 4. Gather results
        obs = self._get_obs()
        reward = self._compute_reward()
        sim_time = traci.simulation.getTime()
        terminated = bool(sim_time >= self.max_sim_time)
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up the SUMO process."""
        self._close_sumo()

    def __del__(self):
        """Ensure SUMO process is terminated on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    # ================================================================
    #  Helpers  (stubs — filled in 1.6.3 / 1.6.6)
    # ================================================================

    def _get_obs(self) -> np.ndarray:
        """Gather the current observation vector from SUMO.

        The observation vector is a 1D float32 array concatenating:
        - For each junction in ``tls_ids``:
          - For each controlled incoming lane in ``tls_lanes[tls_id]``:
            - Queue length (halting vehicles, speed < 0.1 m/s)
            - Total vehicle count on the lane
            - Average speed on the lane (m/s)
            - Cumulative waiting time on the lane (s)
          - Current phase index
          - Elapsed time in the current phase (s)

        Returns
        -------
        np.ndarray
            Observation vector of shape ``(obs_dim,)``.
        """
        if not self._is_traci_connected:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        traci.switch(self._label)
        obs: list[float] = []

        for tls_id in self.tls_ids:
            for lane_id in self.tls_lanes[tls_id]:
                obs.append(float(traci.lane.getLastStepHaltingNumber(lane_id)))
                obs.append(float(traci.lane.getLastStepVehicleNumber(lane_id)))
                obs.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
                obs.append(float(traci.lane.getWaitingTime(lane_id)))

            obs.append(float(traci.trafficlight.getPhase(tls_id)))
            spent = traci.trafficlight.getSpentDuration(tls_id)
            obs.append(float(max(0.0, spent)))

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute the reward for the current step.

        Uses the negative change in total waiting time across controlled lanes:
        ``reward = previous_waiting - current_waiting``.
        """
        if not self._is_traci_connected:
            return 0.0

        traci.switch(self._label)
        current_waiting = sum(
            float(traci.lane.getWaitingTime(lane))
            for tls_id in self.tls_ids
            for lane in self.tls_lanes[tls_id]
        )
        reward = self.previous_total_waiting - current_waiting
        self.previous_total_waiting = current_waiting
        return float(reward)

    def _get_info(self) -> dict:
        """Return auxiliary per-step metrics."""
        if not self._is_traci_connected:
            return {"sim_time": 0.0}

        traci.switch(self._label)
        total_halting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for tls_id in self.tls_ids
            for lane in self.tls_lanes[tls_id]
        )
        return {
            "sim_time": float(traci.simulation.getTime()),
            "total_waiting_time": float(self.previous_total_waiting),
            "total_halting": int(total_halting),
            "throughput": int(traci.simulation.getArrivedNumber()),
            "departed": int(traci.simulation.getDepartedNumber()),
        }

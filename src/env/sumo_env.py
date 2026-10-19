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
        self._requested_tls_ids = tls_ids  # may be None → auto-discover
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_sim_time = max_sim_time
        self.use_gui = use_gui
        self.reward_fn = reward_fn
        self.min_green_phases = min_green_phases

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

        self._start_sumo(seed=0)
        try:
            self._introspect_network()
        finally:
            self._close_sumo()

        # ── Spaces (1.6.3 / 1.6.4 — stubs) ───────────────────────────
        # Placeholder — will be replaced once 1.6.3 / 1.6.4 are built.
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.MultiDiscrete([2])

        # ── Internal state reset every episode ────────────────────────
        self.previous_total_waiting: float = 0.0
        self.phase_is_yellow: dict[str, bool] = {}
        self.current_phase_time: dict[str, int] = {}
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
        self.tls_lanes = {}

        for tls_id in self.tls_ids:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                raise ValueError(f"No traffic light logic found for TLS ID '{tls_id}'.")
            logic = logics[0]

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
        if not self._is_traci_connected:
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

        # Reset bookkeeping
        self.previous_total_waiting = 0.0
        self.phase_is_yellow = {tls: False for tls in self.controlled_tls}
        self.current_phase_time = {tls: 0 for tls in self.controlled_tls}

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """Execute one agent decision and advance the simulation."""
        # TODO: implement in 1.6.5
        obs = self._get_obs()
        reward = 0.0
        terminated = traci.simulation.getTime() >= self.max_sim_time
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up the SUMO process."""
        self._close_sumo()

    # ================================================================
    #  Helpers  (stubs — filled in 1.6.3 / 1.6.6)
    # ================================================================

    def _get_obs(self) -> np.ndarray:
        """Gather the current observation vector."""
        # TODO: implement in 1.6.3
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute the reward for the current step."""
        # TODO: implement in 1.6.6
        return 0.0

    def _get_info(self) -> dict:
        """Return auxiliary per-step metrics."""
        # TODO: implement properly; basic stub for now
        return {"sim_time": traci.simulation.getTime()}

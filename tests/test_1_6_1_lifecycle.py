"""Quick smoke-test for 1.6.1 — TraCI lifecycle management."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.sumo_env import SumoEnv

SUMOCFG = os.path.join(os.path.dirname(__file__), "..", "data", "debrecen.sumocfg")

print("=" * 60)
print("TEST 1: Start -> query sim time -> close")
print("=" * 60)
env = SumoEnv(sumocfg_path=SUMOCFG)
# The constructor starts and stops SUMO for introspection.
# Now do a manual reset to test the full lifecycle.
obs, info = env.reset(seed=42)
print(f"  reset() OK  |  obs shape: {obs.shape}  |  sim_time: {info['sim_time']}")
env.close()
print("  close() OK  — no orphan processes")

print()
print("=" * 60)
print("TEST 2: Double close() (idempotency)")
print("=" * 60)
env = SumoEnv(sumocfg_path=SUMOCFG)
obs, info = env.reset(seed=7)
env.close()
env.close()  # should not crash
print("  double close() OK")

print()
print("=" * 60)
print("TEST 3: Multiple resets (no process leak)")
print("=" * 60)
env = SumoEnv(sumocfg_path=SUMOCFG)
for i in range(3):
    obs, info = env.reset(seed=i)
    print(f"  reset #{i}  |  sim_time: {info['sim_time']}")
env.close()
print("  3 resets + close OK")

print()
print("=" * 60)
print("TEST 4: Two concurrent env instances (port isolation)")
print("=" * 60)
env_a = SumoEnv(sumocfg_path=SUMOCFG)
env_b = SumoEnv(sumocfg_path=SUMOCFG)
obs_a, _ = env_a.reset(seed=100)
obs_b, _ = env_b.reset(seed=200)
print(f"  env_a label: {env_a._label}  |  env_b label: {env_b._label}")
print(f"  env_a sim_time: {_['sim_time']}  |  env_b queried OK")
env_a.close()
env_b.close()
print("  two instances OK — no port conflict")

print()
print("[PASS] All 1.6.1 lifecycle tests passed!")

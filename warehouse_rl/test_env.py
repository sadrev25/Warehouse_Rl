from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from warehouse_rl_env import WarehouseRLEnv

print("=" * 55)
print("WarehouseRLEnv — connection test")
print("=" * 55)

env = WarehouseRLEnv(n_robots=12, seed=0)

# ── Test 1: reset ─────────────────────────────────────────────────────
print("\n[1] reset()...")
obs, _ = env.reset()
assert obs.shape == (11,), f"Wrong obs shape: {obs.shape}"
print(f"    obs shape : {obs.shape}  OK")
print(f"    obs values: {np.round(obs, 3)}")

# Confirm antagonist was found
from robots.deployment_robot import AntagonisticWarehouseRobot
assert isinstance(env._antagonist, AntagonisticWarehouseRobot), \
    "Antagonist robot not found in swarm!"
print(f"    antagonist robot id: {env._antagonist.id}  OK")

# ── Test 2: neutral action (no disruption) ────────────────────────────
print("\n[2] step() — neutral action [0, 0, 0]...")
obs, reward, done, trunc, info = env.step(np.array([0.0, 0.0, 0.0]))
print(f"    tasks completed : {info['tasks_completed']} / {env.max_tasks}")
print(f"    runtime         : {info['runtime']:.1f}s")
print(f"    throughput      : {info['throughput']:.4f} tasks/sec")
print(f"    reward          : {reward:.4f}  (near 0 expected)")

# ── Test 3: max disruption action ─────────────────────────────────────
print("\n[3] reset() + step() — max disruption [10, 1, -6]...")
obs, _ = env.reset()
obs, reward, done, trunc, info = env.step(np.array([10.0, 1.0, -3.0]))
print(f"    tasks completed : {info['tasks_completed']} / {env.max_tasks}")
print(f"    runtime         : {info['runtime']:.1f}s")
print(f"    throughput      : {info['throughput']:.4f} tasks/sec")
print(f"    reward          : {reward:.4f}  (positive = disruption achieved)")

# ── Test 4: action space samples ─────────────────────────────────────
print("\n[4] action_space.sample() × 3...")
for i in range(3):
    a = env.action_space.sample()
    print(f"    {i}: {np.round(a, 3)}")

print("\n" + "=" * 55)
print("All checks passed — ready for train_ppo.py")
print("=" * 55)
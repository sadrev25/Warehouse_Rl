from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from warehouse_rl_env import WarehouseRLEnv


# ── Environment ───────────────────────────────────────────────────────
env = Monitor(
    WarehouseRLEnv(n_robots=12, seed=9),
    filename="./logs/monitor",
    info_keywords=("tasks_completed", "runtime", "throughput"),
)

# ── Directories ───────────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("tb_logs",     exist_ok=True)

# ── Callbacks ─────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq   = 5,
    save_path   = "./checkpoints/",
    name_prefix = "ppo_antagonist",
    verbose     = 1,
)

# ── PPO model ─────────────────────────────────────────────────────────
# n_steps=1 because each env.step() IS one complete simulation run.
# PPO collects n_steps rollouts before updating — with episodic sims
# this must be 1. batch_size must equal n_steps * n_envs = 1.
model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    verbose         = 1,
    learning_rate   = 3e-4,
    n_steps         = 8,      # ← fix: 1 step per episode
    batch_size      = 8,      # ← fix: must equal n_steps
    n_epochs        = 10,
    gamma           = 0.99,
    ent_coef        = 0.05,
    clip_range      = 0.2,
    tensorboard_log = "./tb_logs/",
)

print("=" * 60)
print("PPO Antagonist Training")
print(f"  n_robots:            {env.unwrapped.n_robots}")
print(f"  max_tasks/episode:   {env.unwrapped.max_tasks}")
print(f"  baseline throughput: {WarehouseRLEnv.BASELINE_MEAN_THROUGHPUT:.4f} tasks/sec")
print(f"  est. time/episode:   ~{WarehouseRLEnv.BASELINE_MEAN_RUNTIME:.0f}s")
print(f"  total_timesteps=50 → est. total time: ~{50 * 276.5 / 3600:.1f} hours")
print("=" * 60)

model.learn(
    total_timesteps = 50,
    callback        = checkpoint_cb,
    progress_bar    = True,
)

model.save("ppo_antagonist_final")
print("\nDone. Model saved → ppo_antagonist_final.zip")
print("Run tensorboard with: tensorboard --logdir ./tb_logs/")
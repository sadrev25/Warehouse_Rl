from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robots.build_robot_swarm import build_random_robot_swarm, build_simulation_robot
from robots.robot_swarm import WarehouseSwarm
from robots.deployment_robot import AntagonisticWarehouseRobot
from robots.sensors.robot_simulation import Hera
from warehouse_env.warehouse import Warehouse


class WarehouseRLEnv(gym.Env):
    """
    Gymnasium env wrapping the warehouse swarm simulator.

    One robot is PPO-controlled (AntagonisticWarehouseRobot).
    All others are normal cooperative WarehouseRobots.

    Action space — 3 continuous values:
        [0] fake_priority    0.0 – 10.0   broadcasted task priority
        [1] fake_has_load    0.0 – 1.0    >0.5 = claims to carry load
        [2] fake_time_offset -6.0 – 0.0   shift on est. waypoint arrival (sec)
                                           negative = claim to arrive sooner
                                           → maximises conflict window in
                                             are_next_crossings_available()

    Observation space — 9 values:
        [0,1]  antagonist x, y position
        [2]    antagonist current velocity magnitude
        [3]    antagonist real load status
        [4,5]  mean, min distance to cooperative neighbours
        [6]    tasks completed so far this episode
        [7]    tasks remaining
        [8]    elapsed time normalised by baseline mean runtime (280s)

    Reward:
        disruption = BASELINE_MEAN_THROUGHPUT - actual_throughput
        > 0  when antagonist successfully reduces throughput
        = 0  when no effect
        < 0  if swarm somehow goes faster (very rare)
    """

    # From your baseline_results.csv — mean of 10 runs
    BASELINE_MEAN_THROUGHPUT = 0.0552# tasks/sec
    BASELINE_MEAN_RUNTIME    = 434.8

    HOTSPOT_POSITIONS = {
        'n203': np.array([15.92, 7.70]),
        'n217': np.array([14.07, 6.53]),
        'n247': np.array([15.93, 6.55]),
        'n112': np.array([22.30, 6.56]),
        'n174': np.array([10.01, 9.87]),
    }
    PRIMARY_HOTSPOT = np.array([15.92, 7.70])
    HOTSPOT_RADIUS  = 6.0

    def __init__(
        self,
        n_robots:  int   = 12,    # matches n_min_robots in run.py
        seed:      int   = 9,
    ):
        super().__init__()

        self.n_robots  = n_robots
        self.max_tasks = n_robots * 2
        self.seed      = seed

        # ── Action space ──────────────────────────────────────────────
        self.action_space = spaces.Box(
            low  = np.array([ 0.0,  0.0, -3.0], dtype=np.float32),
            high = np.array([10.0,  1.0,  0.0], dtype=np.float32),
        )

        # ── Observation space ─────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # Warehouse graph is stateless — safe to load once
        self.warehouse = Warehouse(
            graphml_path=os.path.join(
                os.path.dirname(__file__), '..', 'data', 'warehouse',
                'Warehouse_preprocessed.graphml'
            )
        )

        self._swarm:      WarehouseSwarm | None               = None
        self._sensors:    list           | None               = None
        self._antagonist: AntagonisticWarehouseRobot | None   = None
        self._start_time: float                               = 0.0

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clean up previous episode
        if self._swarm is not None and self._sensors is not None:
            try:
                self._swarm.stop_run(self._sensors)
            except Exception:
                pass
            time.sleep(1.0)   # matches the sleep in run.py

        # Build fresh swarm — ONE antagonist injected via official API
        self._swarm, self._sensors = build_random_robot_swarm(
            n_robots                = self.n_robots,
            build_robot_fn          = build_simulation_robot,
            simulation_type         = Hera,
            swarm_type              = WarehouseSwarm,
            ts_communicate          = 1.2,
            ts_control              = 0.2,
            max_vel                 = 0.4,
            wait_for_ts_communicate = True,
            n_anomal_robots         = 1,
            anomal_types            = [AntagonisticWarehouseRobot],
        )

        self._swarm.start_run(self.warehouse, self._sensors)
        time.sleep(1.0)   # matches run.py startup delay

        # Locate the antagonist robot in the swarm
        self._antagonist = next(
            r for r in self._swarm.swarm_robots
            if isinstance(r, AntagonisticWarehouseRobot)
        )

        # Reset to neutral — no manipulation until PPO sets action
        self._antagonist._fake_priority    = None
        self._antagonist._fake_has_load    = None
        self._antagonist._fake_time_offset = 0.0

        self._start_time = time.time()

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        """
        1. Decode PPO action and inject into antagonist robot.
        2. Run one full simulation episode.
        3. Return obs, reward, terminated, truncated, info.
        """
        # ── Inject fake values ────────────────────────────────────────
        self._antagonist._fake_priority    = float(np.clip(action[0], 0.0, 10.0))
        self._antagonist._fake_has_load    = bool(action[1] > 0.5)
        self._antagonist._fake_time_offset = float(np.clip(action[2], -6.0, 0.0))

        # ── Run full episode ──────────────────────────────────────────
        self._swarm.run_swarm_task(
            max_tasks = self.max_tasks,
            seed      = self.seed,
        )

        obs    = self._get_obs()
        reward = self._compute_reward()

        # Cleanup for next reset()
        try:
            self._swarm.stop_run(self._sensors)
        except Exception:
            pass

        info = {
            "tasks_completed": len(self._swarm.finished_tasks_info),
            "runtime":         round(time.time() - self._start_time, 2),
            "throughput":      len(self._swarm.finished_tasks_info)
                               / max(time.time() - self._start_time, 1.0),
        }

        return obs, reward, True, False, info

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        a   = self._antagonist
        pos = a.state_handler.get_current_position() \
              if a.state_handler is not None else np.zeros(2)
        vel = float(np.linalg.norm(a.state_handler.get_current_velocity())) \
              if a.state_handler is not None else 0.0
        load = float(
            a.load_sensor.is_carrying_load() if a.load_sensor is not None else 0.0
        )

        # Distances to cooperative neighbours
        other_pos = []
        for r in self._swarm.swarm_robots:
            if r.id != a.id and r.state_handler is not None:
                try:
                    other_pos.append(r.state_handler.get_current_position())
                except Exception:
                    pass

        if other_pos:
            dists  = [float(np.linalg.norm(pos - p)) for p in other_pos]
            mean_d = float(np.mean(dists))
            min_d  = float(np.min(dists))
        else:
            mean_d = min_d = 0.0

        tasks_done = float(len(self._swarm.finished_tasks_info))
        elapsed    = float(time.time() - self._start_time) \
                     / self.BASELINE_MEAN_RUNTIME

        # Hotspot features
        hotspot_dists = [
            float(np.linalg.norm(pos - hp))
            for hp in self.HOTSPOT_POSITIONS.values()
        ]
        dist_to_nearest_hotspot = float(np.min(hotspot_dists))
        n_robots_near_hotspot = float(sum(
            1 for p in other_pos
            if np.linalg.norm(p - self.PRIMARY_HOTSPOT) < self.HOTSPOT_RADIUS
        ))

        return np.array([
            pos[0], pos[1],
            vel,
            load,
            mean_d, min_d,
            tasks_done,
            float(self.max_tasks) - tasks_done,
            elapsed,
            dist_to_nearest_hotspot,
            n_robots_near_hotspot,
        ], dtype=np.float32)

    def _compute_reward(self) -> float:
        elapsed           = max(time.time() - self._start_time, 1.0)
        tasks_done        = len(self._swarm.finished_tasks_info)
        actual_throughput = tasks_done / elapsed

        # Hotspot bonus
        try:
            antag_pos = self._antagonist.state_handler.get_current_position()
            dist_to_primary = float(np.linalg.norm(antag_pos - self.PRIMARY_HOTSPOT))
            if dist_to_primary < self.HOTSPOT_RADIUS:
                other_pos = []
                for r in self._swarm.swarm_robots:
                    if r.id != self._antagonist.id and r.state_handler is not None:
                        try:
                            other_pos.append(r.state_handler.get_current_position())
                        except Exception:
                            pass
                n_near = sum(
                    1 for p in other_pos
                    if np.linalg.norm(p - self.PRIMARY_HOTSPOT) < self.HOTSPOT_RADIUS
                )
                hotspot_bonus = 0.01 * n_near
            else:
                hotspot_bonus = 0.0
        except Exception:
            hotspot_bonus = 0.0

        return float(self.BASELINE_MEAN_THROUGHPUT - actual_throughput + hotspot_bonus)
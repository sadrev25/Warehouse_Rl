from __future__ import annotations

import gzip
import pickle
from typing import TYPE_CHECKING, Callable, List

import numpy as np

if TYPE_CHECKING:
    from robots.robot_swarm import RobotSwarm


def save_robot_swarm(robot_swarm, filepath: str) -> None:

    robot_swarm.communicated_states_history = dict(
        robot_swarm.communicated_states_history
    )
    robot_swarm.finished_tasks_info = list(robot_swarm.finished_tasks_info)
    robot_swarm.last_communicated_states = None
    robot_swarm.state_mgr = None
    robot_swarm._state_lock = None

    for robot in robot_swarm.swarm_robots:
        robot.state_history = list(robot.state_history)
        robot.state_mgr = None
        robot._state_lock = None
        for key, item in vars(robot).items():
            if hasattr(item, "__dict__"):
                vars(robot)[key] = str(item.__class__)

    for monitor in robot_swarm.external_state_monitors:
        monitor.simulated_robot = None
        monitor.state_mgr = None
        monitor.recorded_positions = list(monitor.recorded_positions)
        monitor.time_stamps = list(monitor.time_stamps)
        monitor._state_lock = None
        monitor.monitor_process = None
        monitor.position = np.array(monitor.position)
        monitor.velocity = np.array(monitor.velocity)

    with gzip.open(filepath, "wb") as fp:
        pickle.dump(robot_swarm, fp)


# def save_robot_swarm(robot_swarm: RobotSwarm, filepath: str) -> None:

#     robot_swarm.__dict__ = {}
#     # robot_swarm.swarm_robots = None
#     # robot_swarm.robot_mgr = None
#     # robot_swarm.available_robots = None
#     # robot_swarm.stop_event = None

#     # for robot in robot_swarm.swarm_robots:
#     #     robot.deployment_area = None
#     #     robot.load_sensor = None
#     #     robot.swarm_communication_handler = None
#     #     robot.move_handler = None
#     #     robot.state_handler = None

#     # for sensor in robot_swarm.external_state_monitors:
#     #     sensor.monitor_process = None
#     #     sensor.state_mgr = None
#     #     sensor.simulated_robot = None

#     with gzip.open(filepath, "wb") as fp:
#         pickle.dump(robot_swarm, fp, protocol=pickle.HIGHEST_PROTOCOL)

#     return


def load_robot_swarm(filepath: str) -> RobotSwarm:

    with gzip.open(filepath, "rb") as fp:
        robot_swarm = pickle.load(fp)

    return robot_swarm

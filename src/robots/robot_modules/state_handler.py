from __future__ import annotations

import abc
import logging
import multiprocessing
import threading
import time
from multiprocessing import Array, Lock
from typing import TYPE_CHECKING

import numpy as np
from helper_functions import l2_norm
from robots.sensors.position_monitor import PositionMonitor, SimulatedPositionSensor

# pyright: reportAttributeAccessIssue=false
if TYPE_CHECKING:
    from robots.sensors.lidar import Lidar
    from robots.sensors.load_sensor import LoadSensor


class RobotState:

    def __init__(self) -> None:
        self.time_stamp = 0.0
        self.position = None
        self.velocity = np.array([0.0, 0.0])
        self.has_load = False
        self.next_waypoints = np.array([])
        self.time_to_next_waypoints = np.array([])
        self.priority = 0.0
        self.lidar_info_obj = []
        self.robot_info = []
        return

    def set_final_target_position(self, final_target_position: np.ndarray):
        self.final_target_position = final_target_position

    def set_lidar_info(self, object_info: list, robot_info: list):
        if len(object_info) > 0:
            self.lidar_info_obj = np.vstack(object_info)
        else:
            self.lidar_info_obj = np.array([])
        if len(robot_info) > 0:
            self.robot_info = np.vstack(robot_info)
        else:
            self.robot_info = np.array([])

    def set_load_info(self, has_load: bool):
        self.has_load = has_load

    def set_motion_info(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity

    def set_current_target_position(self, current_target_pos: np.ndarray):
        self.current_target_pos = current_target_pos

    def set_adapted_target_position(self, adapted_target_pos: np.ndarray):
        self.adapted_target_pos = adapted_target_pos

    def set_path_info(
        self, next_waypoints: np.ndarray, time_to_next_waypoints: np.ndarray
    ):
        self.next_waypoints = next_waypoints
        self.time_to_next_waypoints = time_to_next_waypoints

    def set_priority(self, task_priority: float):
        self.priority = task_priority

    def set_time_stamp(self, time_stamp: float):
        self.time_stamp = time_stamp


class StateHandler(metaclass=abc.ABCMeta):
    """Keeps track of the robot's current state and state history.
    The state can contain the robot's position, velocity, energy level, etc.
    """

    def __init__(self, ts_control: float, **kwargs) -> None:
        self.ts_control = ts_control
        self.current_robot_state = RobotState()

    def get_current_state(self) -> RobotState:
        # logging.warning(f"get current state {self.current_robot_state}")
        return self.current_robot_state

    def update_task_info(
        self,
        task_priority,
        final_target_position,
        next_waypoints,
        time_to_next_waypoints,
    ) -> None:
        self.current_robot_state.set_priority(task_priority)
        self.current_robot_state.set_final_target_position(final_target_position)
        self.current_robot_state.set_path_info(next_waypoints, time_to_next_waypoints)
        self.current_robot_state.set_load_info(
            self.load_sensor.is_carrying_load() if self.load_sensor else False
        )
        # logging.warning(f"update task info {self.current_robot_state.__dict__}")

    def update_lidar_info(
        self,
        shelf_positions_in_range: list,
        robot_positions_in_range: list,
        current_target: np.ndarray,
        adapted_target_position: np.ndarray,
    ):

        self.current_robot_state.set_lidar_info(
            shelf_positions_in_range, robot_positions_in_range
        )
        self.current_robot_state.set_current_target_position(current_target)
        self.current_robot_state.set_adapted_target_position(adapted_target_position)
        # logging.warning(f"update lidar info {self.current_robot_state.__dict__}")

    @abc.abstractmethod
    def get_current_position(self) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_velocity(self) -> np.ndarray:
        raise NotImplementedError()

    def get_time_stamp(self) -> float:
        return self.current_robot_state.time_stamp

    @abc.abstractmethod
    def _update_ts_and_position(self) -> None:
        """Receive the robot's current position and velocity and save it to the state. Additionally, the time stamp is updated."""
        raise NotImplementedError()

    def set_load_sensor(self, load_sensor: LoadSensor):
        self.load_sensor = load_sensor

    def set_lidar_sensor(self, lidar_sensor: Lidar):
        self.lidar_sensor = lidar_sensor

    def start(self) -> None:
        self.start_time = time.time()
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        self.start_time = time.time()
        return


class BasicStateHandler(StateHandler):
    """Has access to a state monitor that monitors the robot's state."""

    def __init__(
        self, state_monitor: PositionMonitor, ts_control: float = 0.2, **kwargs
    ) -> None:
        super().__init__(ts_control)
        self.state_monitor = state_monitor

    def get_current_position(self) -> np.ndarray:
        # logging.warning(f"get position {self.current_robot_state.__dict__}")
        self._update_ts_and_position()
        assert self.current_robot_state.position is not None
        return self.current_robot_state.position

    def get_current_velocity(self) -> np.ndarray:
        # logging.warning(f"get velocity {self.current_robot_state.__dict__}")
        return self.current_robot_state.velocity

    def _update_ts_and_position(self) -> None:
        """Get the current state from the state monitor."""
        assert type(self.state_monitor) is SimulatedPositionSensor
        current_position = self.state_monitor.get_current_position()
        current_velocity = self.state_monitor.get_current_velocity()
        assert current_velocity is not None

        time_stamp = np.round(time.time() - self.start_time, decimals=1)

        self.current_robot_state.set_time_stamp(time_stamp=time_stamp)
        self.current_robot_state.set_motion_info(
            position=current_position, velocity=current_velocity
        )
        return

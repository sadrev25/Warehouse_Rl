from __future__ import annotations

import abc
import multiprocessing
import threading
import time
from multiprocessing import Array, Lock
from typing import TYPE_CHECKING

import lcm
import numpy as np
from helper_functions import l2_norm
from lcm_types.itmessage import vector_t
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
        self.next_waypoints = []
        self.estimated_time_to_waypoints = []
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
        self,
        next_waypoints: np.ndarray,
        distance_to_waypoints: np.ndarray,
        max_vel: float,
    ):
        self.next_waypoints = next_waypoints
        self.estimated_time_to_waypoints = (
            distance_to_waypoints / max_vel
        )  # assume that the robot can quickly reach the maximum velocity even if idle

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
        self._update_state()
        self.current_robot_state.set_load_info(
            self.load_sensor.is_carrying_load() if self.load_sensor else False
        )

        return self.current_robot_state

    @abc.abstractmethod
    def get_current_position(self) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_velocity(self) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _update_state(self) -> None:
        """Receive the robot's current state and save it to the state history."""
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
        self._update_state()
        assert self.current_robot_state.position is not None
        return self.current_robot_state.position

    def get_current_velocity(self) -> np.ndarray:
        self._update_state()
        return self.current_robot_state.velocity

    def _update_state(self) -> None:
        """Get the current state from the state monitor."""
        assert type(self.state_monitor) is SimulatedPositionSensor
        current_position = self.state_monitor.get_current_position()
        current_velocity = self.state_monitor.get_current_velocity()

        time_stamp = np.round(time.time() - self.start_time, decimals=1)

        self.current_robot_state.set_time_stamp(time_stamp=time_stamp)
        self.current_robot_state.set_motion_info(
            position=current_position, velocity=current_velocity
        )
        return


class LCM_StateHandler(StateHandler):
    """Uses LCM to receive the robot state sent by a state monitor."""

    def __init__(
        self,
        communication_id: int,
        ts_control: float = 0.2,
        ttl: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(ts_control)

        # The communication id refers to the robot id used for robot-specific communication. When using hardware robots, the communication id might differ from the robot id.
        self.communication_id = communication_id
        self.pause_between_updating = self.ts_control / 3
        # self.update_thread = threading.Thread(target=self._update_state, daemon=True)
        self.position = Array("f", [0, 0], lock=True)
        self._lock = self.position.get_lock()

        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number_pos = 0
        self.lc = lcm.LCM(f"udpm://239.255.76.67:7667?ttl={ttl}")
        lcs = self.lc.subscribe(
            f"/robot{self.communication_id}/euler", self._lcm_handler
        )
        lcs.set_queue_capacity(1)

        self.update_process = multiprocessing.Process(
            target=self._update_state,
            args=(),
            daemon=True,
        )

    def get_current_position(self) -> np.ndarray:
        with self._lock:
            current_position = np.array(self.position)
        return current_position

    def get_current_velocity(self) -> np.ndarray:
        return self.current_robot_state.velocity

    def _update_state(self) -> None:
        """Listen to LCM messages that contain the robot state."""
        while self.is_updating:
            # timeout in ms
            self.lc.handle_timeout(int(3 * self.ts_control * 1000))
            time_stamp = np.round(time.time() - self.start_time, decimals=1)
            # self.current_robot_state = RobotState(
            #     time_stamp=time_stamp,
            #     position=np.array(self.position),
            # TODO: does not work, state handler does not have access to this
            # TODO: communication of load, speed (+ multiprocessing for robot state?)
            # )
            time.sleep(self.pause_between_updating)
        print("state_handler: stop listening")

    def _lcm_handler(self, _: str, state: bytes) -> None:
        """Read the LCM message containing the robot state that was sent by a state monitor.

        Args:
            state (vector_t): message
        """
        assert type(state) == vector_t
        msg = vector_t.decode(state)
        with self._lock:
            if msg.seq_number > self.seq_number_pos:
                self.seq_number_pos = msg.seq_number
                current_position = np.array(msg.value)[:2]
                self.position[:] = current_position
        return

    def start(self) -> None:
        super().start()
        self.update_process.start()
        return

    def stop(self) -> None:
        super().stop()
        while self.update_process.is_alive():
            self.update_process.terminate()

    def reset(self) -> None:
        super().reset()
        self.position[:] = np.zeros(2)

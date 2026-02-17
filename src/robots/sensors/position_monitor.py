from __future__ import annotations

import abc
import multiprocessing
import threading
import time
from multiprocessing import Array, Manager, Value
from typing import TYPE_CHECKING

import lcm
import numpy as np
from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from robots.sensors.robot_simulation import RobotSimulation
    from src.deployment_area.polygon import PolygonWrapper


class PositionMonitor(metaclass=abc.ABCMeta):
    """Represents a type of sensor that monitors the position, rotation and/or velocity of a robot."""

    def __init__(
        self,
        robot_id: int,
        simulated_robot: RobotSimulation,
        ts_control: float = 0.2,
        **kwargs,
    ) -> None:
        self.simulated_robot = simulated_robot
        self.ts_control = ts_control
        self.monitored_robot_id = robot_id
        self.pause_between_monitoring = self.ts_control / 2

        self.state_mgr = Manager()
        self.recorded_positions = self.state_mgr.list()
        self.time_stamps = self.state_mgr.list()
        self._state_lock = self.state_mgr.Lock()

        self.monitor_process = multiprocessing.Process(
            target=self._monitor_state,
            args=(),
            daemon=True,
        )
        self.velocity = Array("f", size_or_initializer=[0, 0], lock=True)
        self.position = Array("f", size_or_initializer=[0, 0], lock=True)

        return

    @abc.abstractmethod
    def _monitor_state(self) -> None:
        """Implements the monitoring function that is continuously called by the monitor thread."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_position(self) -> np.ndarray | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_velocity(self) -> float | None:
        raise NotImplementedError()

    def get_recorded_positions(self) -> list[np.ndarray]:
        """Get the recorded positions of the robot."""
        return list(self.recorded_positions)

    def start(
        self,
        start_area: object,
        swarm_size: int,
        formation_center: np.ndarray | None = None,
        use_formation: bool = False,
    ) -> None:
        if use_formation:
            assert formation_center is not None
            self.simulated_robot.initialize_robot_position_in_formation(
                formation_center, swarm_size
            )
        else:
            self.simulated_robot.initialize_robot_position_randomly(
                start_area, swarm_size
            )
        self.simulated_robot.start()
        time.sleep(0.1)

        self.start_time = time.time()
        self.monitor_process.start()
        assert self.monitor_process.is_alive()
        return

    def stop(self) -> None:
        while self.monitor_process.is_alive():
            self.monitor_process.terminate()
        return

    def reset(self) -> None:
        self.recorded_positions = self.state_mgr.list()
        self.time_stamps = self.state_mgr.list()
        self.start_time = time.time()
        return


class SimulatedPositionSensor(PositionMonitor):
    """Simulates the behaviour of a position sensor that is directly attached to the robot."""

    def __init__(
        self,
        robot_id: int,
        simulated_robot: RobotSimulation,
        ts_control: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(robot_id, simulated_robot, ts_control)
        return

    def _monitor_state(self) -> None:
        """Get the robot state from the simulated robot."""
        while True:
            position_info = self.simulated_robot.get_robot_state()[-2:]
            velocity_info = self.simulated_robot.get_current_velocity()

            # with self._lock:
            #     self.current_position[:] = position_info
            with self._state_lock:
                time_stamp = np.round(time.time() - self.start_time, decimals=1)
                self.time_stamps.append(time_stamp)
                self.recorded_positions.append(position_info)
            with self.position.get_lock():
                self.position[:] = position_info
            with self.velocity.get_lock():
                self.velocity[:] = velocity_info

            time.sleep(self.pause_between_monitoring)

    def get_current_position(self) -> np.ndarray:
        """Get the current position of the robot."""
        with self.position.get_lock():
            position = np.array(self.position)
        return position

    def get_current_velocity(self) -> np.ndarray:
        """Get the current velocity of the robot."""
        with self.velocity.get_lock():
            velocity = np.array(self.velocity)
        return velocity


class SimulatedOptitrack(PositionMonitor):
    """Simulates the behaviour of an optitrack (camera) system that monitors the robot
    positions and communicates them via LCM."""

    def __init__(
        self,
        robot_id: int,
        simulated_robot: RobotSimulation,
        ts_control: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(robot_id, simulated_robot, ts_control)

        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number_pos = 0
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        return

    def _monitor_state(self) -> None:
        """Get the robot state from the simulated robot and send it via LCM to the state monitor."""
        while True:
            position_info = self.simulated_robot.get_robot_state()[-2:]
            velocity_info = self.simulated_robot.get_current_velocity()
            with self._state_lock:
                time_stamp = np.round(time.time() - self.start_time, decimals=1)
                self.time_stamps.append(time_stamp)
                self.recorded_positions.append(position_info)
                self.position[:] = position_info
                self.velocity[:] = velocity_info
            self._send_position(position_info)
            # TODO: send velocity and load
            time.sleep(self.pause_between_monitoring)

    def _send_position(self, position: np.ndarray) -> None:
        """Sends a LCM message containing the id of the monitored robot and its current state."""
        msg_content = np.zeros(shape=(6,))
        msg_content[:2] = position
        self.seq_number_pos += 1

        state_msg = vector_t()
        state_msg.length = 6
        state_msg.id_sender = self.monitored_robot_id
        state_msg.seq_number = self.seq_number_pos
        state_msg.value = list(msg_content)
        # print(state_msg.value)
        self.lc.publish(f"/robot{self.monitored_robot_id}/euler", state_msg.encode())
        return

    def get_current_position(self) -> None:
        """This is not necessary when the classes are used correctly, since the position is sent via LCM."""
        return None

    def get_current_velocity(self) -> None:
        """This is not necessary when the classes are used correctly, since the velocity is sent via LCM."""
        return None

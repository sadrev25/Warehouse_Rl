from __future__ import annotations

import abc
import asyncio
import logging
import threading
import time
from multiprocessing import Manager
from typing import TYPE_CHECKING

import lcm
import numpy as np
from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from robots.deployment_robot import Robot
    from robots.robot_modules.state_handler import RobotState, StateHandler
    from robots.robot_swarm import RobotSwarm
    from robots.sensors.position_monitor import PositionMonitor


class CommunicatedRobotState:
    def __init__(self, robot_state: RobotState) -> None:
        self.time_stamp = robot_state.time_stamp
        self.position = robot_state.position
        self.has_load = robot_state.has_load
        self.next_waypoints = robot_state.next_waypoints
        self.time_to_next_waypoints = robot_state.time_to_next_waypoints
        self.priority = robot_state.priority


class SwarmCommunicationHandler(metaclass=abc.ABCMeta):
    """Communicates with other entities of the robot swarm, e.g. in order to send and receive robot states."""

    def __init__(self, id: int, ts_communicate: float, **kwargs) -> None:
        self.id = id
        self.ts_communicate = ts_communicate
        self.communicated_position = np.zeros(2)
        self.state_mgr = Manager()

        self.listen_thread = threading.Thread(target=self._listen_to_swarm, daemon=True)
        self.send_thread = threading.Thread(
            target=self._send_info_to_swarm, daemon=True
        )
        return

    def get_swarm_info(self) -> dict:
        """Returns the most recent swarm information.

        Returns:
            dict: most recent information received by other agents within the swarm
        """
        return dict(self.swarm_info)

    def get_communicated_position(self) -> np.ndarray:
        return self.communicated_position

    @abc.abstractmethod
    def _send_info_to_swarm(self) -> None:
        """Sends information about its own state to the swarm.

        Args:
            own_state (RobotState): information about the robot's current state, e.g. the current position that is sent to the swarm
            robot_swarm (RobotSwarm): robot swarm
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _listen_to_swarm(self) -> None:
        raise NotImplementedError

    def start(self, robot: Robot, robot_swarm: RobotSwarm) -> None:

        self.robot_swarm = robot_swarm
        self.robot = robot
        self.swarm_info = self.state_mgr.dict({})

        if hasattr(self, "listen_thread") and not self.listen_thread.is_alive():
            print("communication_handler: start listening and sending")
            self.is_listening = True
            self.is_sending = True
            self.listen_thread.start()
            self.send_thread.start()

        return

    def stop(self) -> None:
        if hasattr(self, "listen_thread") and self.listen_thread.is_alive():
            self.is_listening = False
            # create a new thread for the next run
            self.listen_thread = threading.Thread(target=self._listen_to_swarm, daemon=True)  # type: ignore

        if hasattr(self, "send_thread") and self.send_thread.is_alive():
            self.is_sending = False
            # create a new thread for the next run
            self.send_thread = threading.Thread(target=self._send_info_to_swarm, daemon=True)  # type: ignore
        return

    def reset(self) -> None:
        self.swarm_info = self.state_mgr.dict({})
        return


class BasicCommunicationHandler(SwarmCommunicationHandler):
    """Uses the reference to the robot swarm to gather information about the swarm state."""

    def __init__(self, id: int, ts_communicate: float, **kwargs) -> None:
        super().__init__(id, ts_communicate)
        self.listening_interval = 0.2

    def _send_info_to_swarm(
        self,
    ) -> None:
        """Sends information about its own state to the swarm."""
        while self.is_sending:
            own_state = self.robot.get_last_recorded_state()
            state_to_communicate = CommunicatedRobotState(own_state)
            self.robot_swarm.add_communicated_state(self.id, state_to_communicate)
            self.communicated_position = state_to_communicate.position

            time.sleep(self.ts_communicate)
        return

    def _listen_to_swarm(self) -> None:
        while self.is_listening:
            time.sleep(self.listening_interval)
            self.swarm_info.update(self.robot_swarm.get_last_communicated_states())

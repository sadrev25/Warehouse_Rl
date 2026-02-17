from __future__ import annotations

import abc
import asyncio
import threading
import time
from typing import TYPE_CHECKING

import lcm
import numpy as np
from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from robots.robot_modules.state_handler import RobotState
    from robots.robot_swarm import RobotSwarm
    from robots.sensors.position_monitor import PositionMonitor


class CommunicatedRobotState:
    def __init__(self, robot_state: RobotState) -> None:
        self.time_stamp = robot_state.time_stamp
        self.position = robot_state.position
        self.has_load = robot_state.has_load
        self.next_waypoints = robot_state.next_waypoints
        self.estimated_time_to_waypoints = robot_state.estimated_time_to_waypoints
        self.priority = robot_state.priority


class SwarmCommunicationHandler(metaclass=abc.ABCMeta):
    """Communicates with other entities of the robot swarm, e.g. in order to send and receive robot states."""

    def __init__(self, id: int, ts_communicate: float, **kwargs) -> None:
        self.id = id
        self.ts_communicate = ts_communicate
        self.communicated_position = np.zeros(2)
        return

    @abc.abstractmethod
    def send_info_to_swarm(
        self, own_state: RobotState, robot_swarm: RobotSwarm
    ) -> None:
        """Sends information about its own state to the swarm.

        Args:
            own_state (RobotState): information about the robot's current state, e.g. the current position that is sent to the swarm
            robot_swarm (RobotSwarm): robot swarm
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def gather_swarm_info(self, robot_swarm: RobotSwarm) -> dict:
        """Waits for a short period of time while receiving the messages from other agents, before returning the swarm information.

        Returns:
            dict: most recent information received by other agents within the swarm
        """
        raise NotImplementedError()

    def get_communicated_position(self) -> np.ndarray:
        return self.communicated_position

    def start(self) -> None:

        self.swarm_info = {}

        if hasattr(self, "listen_thread") and not self.listen_thread.is_alive():
            print("communication_handler: start listening")
            self.is_listening = True
            self.listen_thread.start()

        return

    def stop(self) -> None:
        if hasattr(self, "listen_thread") and self.listen_thread.is_alive():
            self.is_listening = False
            # create a new thread for the next run
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)  # type: ignore
        return

    def reset(self) -> None:
        self.swarm_info = {}
        return


class BasicCommunicationHandler(SwarmCommunicationHandler):
    """Uses the reference to the robot swarm to gather information about the swarm state."""

    def __init__(
        self, id: int, ts_communicate: float, state_monitor: PositionMonitor, **kwargs
    ) -> None:
        super().__init__(id, ts_communicate)
        self.state_monitor = state_monitor
        self.wait_for_swarm_info = 0.1

    def send_info_to_swarm(
        self, own_state: RobotState, robot_swarm: RobotSwarm
    ) -> None:
        """Sends information about its own state to the swarm.

        Args:
            own_state (RobotState): information about the robot's current state, e.g. the current position that is sent to the swarm
            robot_swarm (RobotSwarm): robot swarm
        """
        # communicate own state to swarm
        state_to_communicate = CommunicatedRobotState(own_state)
        robot_swarm.add_communicated_state(self.id, state_to_communicate)
        self.communicated_position = state_to_communicate.position
        return

    def gather_swarm_info(self, robot_swarm: RobotSwarm) -> dict:
        """Waits for a short period of time while receiving the messages from other agents, before returning the swarm information.

        Returns:
            dict: most recent information received by other agents within the swarm
        """
        # wait for swarm info to arrive
        time.sleep(self.wait_for_swarm_info)
        communicated_states = robot_swarm.get_last_communicated_states()
        return communicated_states


class LCM_CommunicationHandler(SwarmCommunicationHandler):
    """Uses LCM to send and receive messages to and from other entities in the robot swarm."""

    def __init__(
        self,
        id: int,
        ts_communicate: float,
        ttl: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(id, ts_communicate)

        self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        self.wait_for_swarm_info = 0.1
        self.lc = lcm.LCM(f"udpm://239.255.76.67:7667?ttl={ttl}")
        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number = 0
        lcs = self.lc.subscribe(f"/robot/euler", self._lcm_handler)
        lcs.set_queue_capacity(30)

    def send_info_to_swarm(
        self, own_state: RobotState, robot_swarm: RobotSwarm
    ) -> None:
        """Sends information about its own state to the swarm.

        Args:
            own_state (RobotState): information about the robot's current state, e.g. the current position that is sent to the swarm
            robot_swarm (RobotSwarm): robot swarm
        """
        # communicate own state to swarm
        state_to_communicate = CommunicatedRobotState(own_state)
        robot_swarm.add_communicated_state(self.id, state_to_communicate)
        assert state_to_communicate.position is not None
        self.communicated_position = state_to_communicate.position
        self._communicate_to_swarm(self.communicated_position)
        return

    def gather_swarm_info(self, robot_swarm: RobotSwarm) -> dict:
        """Waits for a short period of time while receiving the messages from other agents, before returning the swarm information.

        Returns:
            dict: most recent information received by other agents within the swarm
        """
        # wait for swarm info to arrive
        time.sleep(self.wait_for_swarm_info)
        return self.swarm_info

    def _communicate_to_swarm(self, position: np.ndarray) -> None:
        """Send a LCM message containing the robot state to the swarm.

        Args:
            position (np.array): robot position
        """
        msg_content = list(np.zeros(shape=(6,)))
        msg_content[:2] = position
        self.seq_number += 1

        state_msg = vector_t()
        state_msg.length = 6
        state_msg.id_sender = self.id
        state_msg.seq_number = self.seq_number
        state_msg.value = msg_content

        self.lc.publish(f"/robot/euler", state_msg.encode())

    def _listen(self) -> None:
        while self.is_listening:
            self.lc.handle_timeout(int(2 * self.ts_communicate * 1000))
        print("communication_handler: stop listening")

    def _lcm_handler(self, _: str, state: bytes) -> None:
        """Decode LCM messages sent by the other swarm agents.

        Args:
            state (vector_t): message
        """
        assert type(state) == vector_t
        msg = vector_t.decode(state)
        # 6 dim: 1st x, 2nd y, 4th rotation
        if msg.seq_number >= self.seq_number:
            current_position = np.array(msg.value)[:2]
            self.swarm_info[msg.id_sender] = current_position

        return

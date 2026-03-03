from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from anomaly_detectors.anomaly_detector import AnomalyDetector
    from src.deployment_area.polygon import PolygonWrapper
    from robots.robot_modules.move_handler import MoveHandler, SimulationMoveHandler
    from robots.robot_modules.state_handler import StateHandler
    from robots.robot_modules.swarm_communication_handler import (
        SwarmCommunicationHandler,
    )
    from robots.robot_swarm import RobotSwarm

from helper_functions import l2_norm
from robots.robot_modules.state_handler import RobotState


class Robot(object, metaclass=abc.ABCMeta):

    def __init__(
        self,
        robot_id: int,
        color: str,
        ts_communicate: float,
        ts_control: float,
        max_vel: float,
        state_handler: StateHandler,
        move_handler: MoveHandler,
        swarm_communication_handler: SwarmCommunicationHandler,
        **args,
    ) -> None:
        self.id = robot_id
        self.color = color
        self.label = -1
        self.label_text = ""

        self.state_handler = state_handler
        self.move_handler = move_handler
        self.swarm_communication_handler = swarm_communication_handler

        self.set_ts_communicate(ts_communicate)
        self.set_ts_control(ts_control)
        self.set_max_vel(max_vel)

        self.state_history = []
        self.current_target = None

    @abc.abstractmethod
    def update_path_planning(self, swarm_info: dict, **kwargs) -> None:
        """
        Compute the path of the robot while considering the swarm's currently known state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_position(self, **kwargs) -> None:
        """
        Update the position by moving the robot.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_last_recorded_state(self) -> RobotState:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_deployment_area(self, area: object) -> None:
        raise NotImplementedError()

    def start(self, robot_swarm: RobotSwarm) -> None:
        """Start the robot modules."""
        self.state_handler.start()
        self.move_handler.start()
        self.swarm_communication_handler.start(self, robot_swarm)
        return

    def stop(self) -> None:
        """Record the current state of the robot and stop the robot modules."""
        self.move_handler.stop()
        self.state_handler.stop()
        self.swarm_communication_handler.stop()
        return

    def reset(self) -> None:
        """Clear the information collected by different robot modules. Does not restart the robot modules."""
        self.target_history = []
        self.move_handler.reset()
        self.state_handler.reset()
        self.swarm_communication_handler.reset()
        self.anomaly_detector.reset()
        return

    def trigger_movement(self, vel: np.ndarray, adapted_vel: float) -> np.ndarray:
        max_abs_vel = np.max(np.abs(vel))
        if max_abs_vel > adapted_vel:
            vel = vel * (adapted_vel / max_abs_vel)
        # vel = np.clip(vel, a_min=-adapted_vel, a_max=adapted_vel)
        self.move_handler.trigger_movement(vel)
        return vel

    def get_communicated_position(self) -> np.ndarray:
        return self.swarm_communication_handler.get_communicated_position()

    def _get_current_position(self) -> np.ndarray:
        return self.state_handler.get_current_position()

    def _get_current_velocity(self) -> np.ndarray:
        return self.state_handler.get_current_velocity()

    def get_vel_history(self) -> list:
        return self.move_handler.velocity_history

    def set_max_vel(self, max_vel: float) -> None:
        self.max_vel = max_vel
        return

    def set_ts_control(self, ts_control: float) -> None:
        self.ts_control = ts_control
        return

    def set_ts_communicate(self, ts_communicate: float) -> None:
        self.ts_communicate = ts_communicate
        return

    def set_anomaly_detector(self, anomaly_detector: AnomalyDetector) -> None:
        self.anomaly_detector = anomaly_detector
        return

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robots.sensors.robot_simulation import RobotSimulation


class MoveHandler(metaclass=abc.ABCMeta):
    """Sends the current velocity to the robot and saves the velocity history."""

    def __init__(self, ts_control: float = 0.2, **kwargs) -> None:
        self.ts_control = ts_control
        return

    @abc.abstractmethod
    def trigger_movement(self, vel: np.ndarray) -> None:
        """Trigger a movement of target velocity vel in the corresponding robot.

        Args:
            vel (np.array): Target velocity.
        """
        raise NotImplementedError()

    def start(self) -> None:
        self.velocity_history = []
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        self.velocity_history = []
        return


class SimulationMoveHandler(MoveHandler):
    """Holds a reference to a robot simulation monitored by a state_monitor and triggers the movement of this robot simulation."""

    def __init__(
        self, simulated_robot: RobotSimulation, ts_control: float = 0.2, **kwargs
    ) -> None:
        super().__init__(ts_control)
        self.simulated_robot = simulated_robot

    def trigger_movement(self, vel: np.ndarray) -> None:
        """Passes the target velocity to the robot simulation.

        Args:
            vel (np.array): Target velocity.
        """
        self.simulated_robot.trigger_movement(vel)
        return

from robots.sensors.robot_simulation import RobotSimulation
import numpy as np


class LoadSensor:

    def __init__(
        self,
        simulated_robot: RobotSimulation,
        **kwargs,
    ) -> None:
        self.load = 0
        self.simulated_robot = simulated_robot
        return

    def add_load(self) -> None:
        """Select object load randomly and add the load to the robot simulation."""
        self.load = np.random.uniform(5.0, 30.0)
        self.simulated_robot.set_load(self.load)
        return

    def remove_load(self) -> None:
        self.load = 0
        self.simulated_robot.set_load(self.load)
        return

    def is_carrying_load(self) -> bool:
        return self.load > 0
